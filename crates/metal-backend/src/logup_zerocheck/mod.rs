use std::{
    cmp::max,
    iter::{self, zip},
    sync::Arc,
};

use itertools::{izip, Itertools};
use openvm_metal_common::{copy::MemCopyH2D, d_buffer::MetalBuffer};
use openvm_stark_backend::{
    air_builders::symbolic::SymbolicConstraints,
    calculate_n_logup,
    dft::Radix2BowersSerial,
    p3_matrix::dense::RowMajorMatrix,
    poly_common::{
        eq_sharp_uni_poly, eq_uni_poly, eval_eq_mle, eval_eq_sharp_uni, eval_eq_uni,
        eval_eq_uni_at_one, UnivariatePoly,
    },
    proof::{column_openings_by_rot, BatchConstraintProof, GkrProof},
    prover::{
        fractional_sumcheck_gkr::Frac, stacked_pcs::StackedLayout, sumcheck::sumcheck_round0_deg,
        ColMajorMatrix, DeviceMultiStarkProvingKey, MatrixDimensions, ProvingContext,
    },
    FiatShamirTranscript,
};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use rustc_hash::FxHashMap;
use tracing::{debug, info, info_span, instrument};

use crate::{
    base::MetalMatrix,
    data_transporter::transport_matrix_d2h_col_major,
    logup_zerocheck::{
        batch_mle::evaluate_zerocheck_batched, fold_ple::fold_ple_evals_rotate,
        gkr_input::TraceInteractionMeta, round0::evaluate_round0_interactions_metal,
    },
    metal::logup_zerocheck::{
        fold_selectors_round0, interpolate_columns_gpu, ColumnPtr, MainMatrixPtrs,
    },
    metal::sumcheck::fold_mle_matrix,
    poly::EqEvalLayers,
    prelude::{EF, F, SC},
    sponge::DuplexSpongeMetal,
    utils::compute_barycentric_inv_lagrange_denoms,
    MetalBackend,
};

pub(crate) mod batch_mle;
pub(crate) mod batch_mle_monomial;
mod errors;
pub(crate) mod fold_ple;
/// Fraction sumcheck via GKR
mod fractional;
/// Logup interaction evaluations for GKR input
mod gkr_input;
mod mle_round;
mod round0;
pub(crate) mod rules;

use batch_mle::{evaluate_logup_batched, TraceCtx};
use batch_mle_monomial::{
    compute_lambda_combinations, compute_logup_combinations, get_num_monomials,
    get_zerocheck_rules_len, trace_has_monomials, LogupCombinations, LogupMonomialBatch,
    ZerocheckMonomialBatch, ZerocheckMonomialParYBatch,
};
pub use errors::*;
pub use fractional::fractional_sumcheck_metal;
use gkr_input::{collect_trace_interactions, log_gkr_input_evals};
use round0::evaluate_round0_constraints_metal;
use rules::codec::Codec;

/// When `num_monomials >= DAG_FALLBACK_MONOMIAL_RATIO * rules_len`, use DAG evaluation
/// instead of the monomial kernel for high num_y traces.
/// This ratio can be tuned. Currently it is set to prefer the monomial kernel except for the
/// Poseidon2Air where the DAG node size is much smaller than number of monomials.
const DAG_FALLBACK_MONOMIAL_RATIO: usize = 2;

#[inline]
fn frac_buffer_to_vec(buf: &MetalBuffer<Frac<EF>>) -> Vec<Frac<EF>> {
    let len = buf.len();
    if len == 0 {
        return Vec::new();
    }
    let p_ptr = buf.as_ptr() as *const EF;
    let q_ptr = unsafe { p_ptr.add(len) };
    (0..len)
        .map(|i| unsafe { Frac::new(*p_ptr.add(i), *q_ptr.add(i)) })
        .collect()
}

#[inline]
pub(crate) fn air_width_for_mat(need_rot: bool, mat_width: usize) -> u32 {
    if need_rot {
        debug_assert_eq!(mat_width % 2, 0, "rotated matrices should have even width");
        (mat_width / 2) as u32
    } else {
        mat_width as u32
    }
}

#[instrument(level = "info", skip_all)]
pub fn prove_zerocheck_and_logup_metal(
    transcript: &mut DuplexSpongeMetal,
    mpk: &DeviceMultiStarkProvingKey<MetalBackend>,
    ctx: &ProvingContext<MetalBackend>,
    save_memory: bool,
    monomial_num_y_threshold: u32,
    sm_count: u32,
) -> (GkrProof<SC>, BatchConstraintProof<SC>, Vec<EF>) {
    let logup_gkr_span = info_span!("prover.rap_constraints.logup_gkr", phase = "prover").entered();
    let l_skip = mpk.params.l_skip;
    let constraint_degree = mpk.max_constraint_degree;
    let num_traces = ctx.per_trace.len();

    // Traces are sorted
    let n_max = log2_strict_usize(ctx.per_trace[0].1.common_main.height()).saturating_sub(l_skip);
    // Gather interactions metadata, including interactions stacked layout which depends on trace
    // heights
    let mut total_interactions = 0u64;
    let interactions_meta: Vec<_> = ctx
        .per_trace
        .iter()
        .map(|(air_idx, air_ctx)| {
            let pk = &mpk.per_air[*air_idx];

            let num_interactions = pk.vk.symbolic_constraints.interactions.len();
            let height = air_ctx.common_main.height();
            let log_height = log2_strict_usize(height);
            let log_lifted_height = log_height.max(l_skip);
            total_interactions += (num_interactions as u64) << log_lifted_height;
            (num_interactions, log_lifted_height)
        })
        .collect();
    // Implicitly, the width of this stacking should be 1
    let n_logup = calculate_n_logup(l_skip, total_interactions);
    // There's no stride threshold for `interactions_layout` because there's no univariate skip for
    // GKR
    let interactions_layout = StackedLayout::new(0, l_skip + n_logup, interactions_meta);

    // Grind to increase soundness of random sampling for LogUp
    let logup_pow_witness = transcript.grind_gpu(mpk.params.logup.pow_bits).unwrap();
    let alpha_logup = transcript.sample_ext();
    let beta_logup = transcript.sample_ext();
    debug!(%alpha_logup, %beta_logup);

    let has_interactions = !interactions_layout.sorted_cols.is_empty();
    let mut prover = LogupZerocheckMetal::new(
        mpk,
        ctx,
        n_logup,
        interactions_layout,
        alpha_logup,
        beta_logup,
        save_memory,
        monomial_num_y_threshold,
        sm_count,
    );
    let n_global = prover.n_global;

    let total_leaves = 1 << (l_skip + n_logup);
    let inputs = if has_interactions {
        log_gkr_input_evals(
            &prover.trace_interactions,
            mpk,
            ctx,
            l_skip,
            alpha_logup,
            &prover.d_challenges,
            total_leaves,
        )
        .expect("failed to evaluate interactions on device")
    } else {
        MetalBuffer::with_capacity(0)
    };
    // Set memory limit for batch MLE based on inputs buffer size
    prover.gkr_mem_contribution = inputs.len() * std::mem::size_of::<Frac<EF>>();
    prover.memory_limit_bytes = prover.gkr_mem_contribution;
    if !prover.save_memory {
        const DEFAULT_MEMORY_LIMIT: usize = 5 << 30; // 5GiB
        if prover.memory_limit_bytes > DEFAULT_MEMORY_LIMIT {
            tracing::warn!(
                "prover.memory_limit_bytes {} already exceeds 5GiB",
                prover.memory_limit_bytes
            );
        }
        prover.memory_limit_bytes = DEFAULT_MEMORY_LIMIT;
    }

    let (frac_sum_proof, mut xi) = fractional_sumcheck_metal(transcript, inputs, true)
        .expect("failed to run fractional sumcheck on Metal");
    while xi.len() != l_skip + n_global {
        xi.push(transcript.sample_ext());
    }
    debug!(?xi);
    prover.xi = xi;

    logup_gkr_span.exit();

    // Note: this span includes ple_fold, but that function has no metal synchronization so it does
    // not include the kernel times for the actual folding
    let round0_span = info_span!("prover.rap_constraints.round0", phase = "prover").entered();
    // begin batch sumcheck
    let mut sumcheck_round_polys = Vec::with_capacity(n_max);
    let mut r = Vec::with_capacity(n_max + 1);
    // batching randomness
    let lambda = transcript.sample_ext();
    debug!(%lambda);

    let sp_0_polys = prover.sumcheck_uni_round0_polys(ctx, lambda);
    let s_0_cpu_span = info_span!("s'_0 -> s_0 cpu interpolations").entered();
    let sp_0_deg = sumcheck_round0_deg(l_skip, constraint_degree);
    let s_deg = constraint_degree + 1;
    let s_0_deg = sumcheck_round0_deg(l_skip, s_deg);
    let large_uni_domain = (s_0_deg + 1).next_power_of_two();
    let dft = Radix2BowersSerial;
    let s_0_logup_polys = {
        let eq_sharp_uni = eq_sharp_uni_poly(&prover.xi[..l_skip]);
        let mut eq_coeffs = eq_sharp_uni.into_coeffs();
        eq_coeffs.resize(large_uni_domain, EF::ZERO);
        let eq_evals = dft.dft(eq_coeffs);

        let width = 2 * num_traces;
        let mut sp_coeffs_mat = EF::zero_vec(width * large_uni_domain);
        for (i, coeffs) in sp_0_polys[..2 * num_traces].iter().enumerate() {
            for (j, &c_j) in coeffs.coeffs().iter().enumerate().take(sp_0_deg + 1) {
                // SAFETY:
                // - coeffs length is <= sp_0_deg + 1 <= s_0_deg < large_uni_domain
                // - sp_coeffs_mat allocated for width
                unsafe {
                    *sp_coeffs_mat.get_unchecked_mut(j * width + i) = c_j;
                }
            }
        }
        let mut s_evals = dft.dft_batch(RowMajorMatrix::new(sp_coeffs_mat, width));
        for (eq, row) in zip(eq_evals, s_evals.values.chunks_mut(width)) {
            for x in row {
                *x *= eq;
            }
        }
        dft.idft_batch(s_evals)
    };
    let skip_domain_size = F::from_usize(1 << l_skip);
    // logup sum claims (sum_{\hat p}, sum_{\hat q}) per present AIR
    let (numerator_term_per_air, denominator_term_per_air): (Vec<_>, Vec<_>) = (0..num_traces)
        .map(|trace_idx| {
            let [sum_claim_p, sum_claim_q] = [0, 1].map(|is_denom| {
                // Compute sum over D of s_0(Z) to get the sum claim
                (0..=s_0_deg)
                    .step_by(1 << l_skip)
                    .map(|j| unsafe {
                        // SAFETY: matrix is 2 * num_trace x large_uni_domain, s_0_deg <
                        // large_uni_domain
                        *s_0_logup_polys
                            .values
                            .get_unchecked(j * 2 * num_traces + 2 * trace_idx + is_denom)
                    })
                    .sum::<EF>()
                    * skip_domain_size
            });
            transcript.observe_ext(sum_claim_p);
            transcript.observe_ext(sum_claim_q);

            (sum_claim_p, sum_claim_q)
        })
        .unzip();

    let mu = transcript.sample_ext();
    debug!(%mu);
    let mu_pows = mu.powers().take(3 * num_traces).collect_vec();

    let s_0_zc_poly = {
        let eq_uni = eq_uni_poly::<F, _>(l_skip, prover.xi[0]);
        let mut eq_coeffs = eq_uni.into_coeffs();
        eq_coeffs.resize(large_uni_domain, EF::ZERO);
        let eq_evals = dft.dft(eq_coeffs);

        // Algebraically batch
        let mut sp_coeffs = EF::zero_vec(large_uni_domain);
        let mus = &mu_pows[2 * num_traces..];
        let polys = &sp_0_polys[2 * num_traces..];
        for (j, batch_coeff) in sp_coeffs.iter_mut().enumerate().take(sp_0_deg + 1) {
            for (&mu, poly) in zip(mus, polys) {
                *batch_coeff += mu * *poly.coeffs().get(j).unwrap_or(&EF::ZERO);
            }
        }
        let mut s_evals = dft.dft(sp_coeffs);
        for (eq, x) in zip(eq_evals, &mut s_evals) {
            *x *= eq;
        }
        dft.idft(s_evals)
    };

    // Algebraically batch
    let s_0_poly = UnivariatePoly::new(
        zip(
            s_0_logup_polys.values.chunks_exact(2 * num_traces),
            s_0_zc_poly,
        )
        .take(s_0_deg + 1)
        .map(|(logup_row, batched_zc)| {
            let coeff = batched_zc
                + zip(&mu_pows, logup_row)
                    .map(|(&mu_j, &x)| mu_j * x)
                    .sum::<EF>();
            transcript.observe_ext(coeff);
            coeff
        })
        .collect(),
    );
    drop(s_0_cpu_span);

    let r_0 = transcript.sample_ext();
    r.push(r_0);
    debug!(round = 0, r_round = %r_0);
    prover.prev_s_eval = s_0_poly.eval_at_point(r_0);
    debug!("s_0(r_0) = {}", prover.prev_s_eval);

    prover.fold_ple_evals(ctx, r_0);
    drop(round0_span);

    // Sumcheck rounds:
    // - each round the prover needs to compute univariate polynomial `s_round`. This poly is linear
    //   since we are taking MLE of `evals`.
    // - at end of each round, sample random `r_round` in `EF`
    //
    // `s_round` is degree `s_deg` so we evaluate it at `0, ..., =s_deg`. The prover skips
    // evaluation at `0` because the verifier can infer it from the previous round's
    // `s_{round-1}(r)` claim. The degree is constraint_degree + 1, where + 1 is from eq term
    let mle_rounds_span =
        info_span!("prover.rap_constraints.mle_rounds", phase = "prover").entered();
    debug!(%s_deg);
    for round in 1..=n_max {
        let sp_round_evals = prover.sumcheck_polys_batch_eval(round, r[round - 1]);
        let batch_s = prover.compute_batch_s_poly(sp_round_evals, num_traces, round, &mu_pows);
        let batch_s_evals = (1..=s_deg)
            .map(|i| batch_s.eval_at_point(EF::from_usize(i)))
            .collect_vec();
        for &eval in &batch_s_evals {
            transcript.observe_ext(eval);
        }
        sumcheck_round_polys.push(batch_s_evals);

        let r_round = transcript.sample_ext();
        debug!(%round, %r_round);
        r.push(r_round);
        prover.prev_s_eval = batch_s.eval_at_point(r_round);

        prover.fold_mle_evals(round, r_round);
    }
    assert_eq!(r.len(), n_max + 1);

    let column_openings = prover.into_column_openings();

    let need_rot_per_trace = ctx
        .per_trace
        .iter()
        .map(|(air_idx, _)| mpk.per_air[*air_idx].vk.params.need_rot)
        .collect::<Vec<_>>();
    // Observe common main openings first, and then preprocessed/cached
    for (need_rot, openings) in need_rot_per_trace.iter().zip(column_openings.iter()) {
        for (claim, claim_rot) in column_openings_by_rot(&openings[0], *need_rot) {
            transcript.observe_ext(claim);
            transcript.observe_ext(claim_rot);
        }
    }
    for (need_rot, openings) in need_rot_per_trace.iter().zip(column_openings.iter()) {
        for part in openings.iter().skip(1) {
            for (claim, claim_rot) in column_openings_by_rot(part, *need_rot) {
                transcript.observe_ext(claim);
                transcript.observe_ext(claim_rot);
            }
        }
    }
    drop(mle_rounds_span);

    let batch_constraint_proof = BatchConstraintProof {
        numerator_term_per_air,
        denominator_term_per_air,
        univariate_round_coeffs: s_0_poly.into_coeffs(),
        sumcheck_round_polys,
        column_openings,
    };
    let gkr_proof = GkrProof {
        logup_pow_witness,
        q0_claim: frac_sum_proof.fractional_sum.1,
        claims_per_layer: frac_sum_proof.claims_per_layer,
        sumcheck_polys: frac_sum_proof.sumcheck_polys,
    };
    (gkr_proof, batch_constraint_proof, r)
}

pub struct LogupZerocheckMetal<'a> {
    pub alpha_logup: EF,
    pub beta_pows: Vec<EF>,
    // [alpha, beta^0, beta^1, .., beta^max_interaction_len]
    pub d_challenges: MetalBuffer<EF>,

    pub l_skip: usize,
    n_logup: usize,
    n_global: usize,

    pub omega_skip: F,
    pub omega_skip_pows: Vec<F>,
    d_omega_skip_pows: MetalBuffer<F>,

    pub interactions_layout: StackedLayout,
    pub constraint_degree: usize,
    n_per_trace: Vec<isize>,
    max_num_constraints: usize,
    pub monomial_num_y_threshold: u32,
    sm_count: u32,
    // Available after GKR:
    pub xi: Vec<EF>,
    pub lambda_pows: Option<MetalBuffer<EF>>,
    /// Precomputed lambda combinations per AIR (indexed by air_idx). Set when lambda is sampled.
    lambda_combinations: Vec<Option<MetalBuffer<EF>>>,
    /// Beta powers on device for logup MLE rounds.
    d_beta_pows: MetalBuffer<EF>,
    /// Precomputed logup combinations per trace (indexed by trace_idx). Set when xi is sampled.
    logup_combinations: Vec<Option<LogupCombinations>>,

    // n_T => segment tree of eq(xi[j..1+n_T]) for j=1..={n_T-round+1} in _reverse_ layout
    eq_xis: FxHashMap<usize, EqEvalLayers<EF>>,
    eq_3b_per_trace: Vec<Vec<EF>>,
    d_eq_3b_per_trace: Vec<MetalBuffer<EF>>,
    // Evaluations on hypercube only, for round 0
    sels_per_trace_base: Vec<MetalMatrix<F>>,
    // After univariate round 0:
    mat_evals_per_trace: Vec<Vec<MetalMatrix<EF>>>,
    sels_per_trace: Vec<MetalMatrix<EF>>,
    // Store public_values per trace (similar to CPU's EvalHelper)
    public_values_per_trace: Vec<MetalBuffer<F>>,
    air_indices_per_trace: Vec<usize>,
    zerocheck_tilde_evals: Vec<EF>,
    logup_tilde_evals: Vec<[EF; 2]>,
    needs_next_per_trace: Vec<bool>,

    trace_interactions: Vec<Option<TraceInteractionMeta>>,
    pk: &'a DeviceMultiStarkProvingKey<MetalBackend>,

    // In round `j`, contains `s_{j-1}(r_{j-1})`
    pub(crate) prev_s_eval: EF,
    pub(crate) eq_ns: Vec<EF>,
    pub(crate) eq_sharp_ns: Vec<EF>,

    save_memory: bool,

    gkr_mem_contribution: usize,
    /// Memory limit for batch MLE intermediate buffers (set after GKR input eval)
    memory_limit_bytes: usize,
}

impl<'a> LogupZerocheckMetal<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        pk: &'a DeviceMultiStarkProvingKey<MetalBackend>,
        ctx: &ProvingContext<MetalBackend>,
        n_logup: usize,
        interactions_layout: StackedLayout,
        alpha_logup: EF,
        beta_logup: EF,
        save_memory: bool,
        monomial_num_y_threshold: u32,
        sm_count: u32,
    ) -> Self {
        let l_skip = pk.params.l_skip;
        let omega_skip = F::two_adic_generator(l_skip);
        let omega_skip_pows = omega_skip.powers().take(1 << l_skip).collect_vec();
        let d_omega_skip_pows = omega_skip_pows.to_device();
        let num_airs_present = ctx.per_trace.len();

        let constraint_degree = pk.max_constraint_degree;

        let max_interaction_length = pk
            .per_air
            .iter()
            .map(|air_pk| air_pk.other_data.interaction_rules.max_fields_len)
            .max()
            .unwrap_or(0);
        let beta_pows = beta_logup
            .powers()
            .take(max_interaction_length + 1)
            .collect_vec();
        let challenges = [&[alpha_logup], &beta_pows[..]].concat();
        let d_challenges = challenges.to_device();
        let d_beta_pows = beta_pows.to_device();

        let n_per_trace: Vec<isize> = ctx
            .common_main_traces()
            .map(|(_, t)| log2_strict_usize(t.height()) as isize - l_skip as isize)
            .collect();
        let n_max = n_per_trace[0].max(0) as usize;
        let n_global = max(n_max, n_logup);
        info!(%n_global, %n_logup);

        let max_num_constraints = pk
            .per_air
            .iter()
            .map(|air_pk| {
                air_pk
                    .vk
                    .symbolic_constraints
                    .constraints
                    .constraint_idx
                    .len()
            })
            .max()
            .unwrap_or(0);

        // Collect interaction metadata for Metal execution
        let trace_interactions = collect_trace_interactions(pk, ctx, &interactions_layout);

        let needs_next_per_trace = ctx
            .per_trace
            .iter()
            .map(|(air_idx, _)| pk.per_air[*air_idx].vk.params.need_rot)
            .collect::<Vec<_>>();

        Self {
            alpha_logup,
            beta_pows,
            d_challenges,
            l_skip,
            n_logup,
            n_global,
            omega_skip,
            omega_skip_pows,
            d_omega_skip_pows,
            interactions_layout,
            constraint_degree,
            n_per_trace,
            max_num_constraints,
            sm_count,
            xi: vec![],
            lambda_pows: None,
            lambda_combinations: (0..pk.per_air.len()).map(|_| None).collect(),
            d_beta_pows,
            logup_combinations: (0..num_airs_present).map(|_| None).collect(),
            eq_xis: FxHashMap::default(),
            eq_3b_per_trace: vec![],
            d_eq_3b_per_trace: vec![],
            sels_per_trace_base: vec![],
            mat_evals_per_trace: vec![],
            sels_per_trace: vec![],
            public_values_per_trace: ctx
                .per_trace
                .iter()
                .map(|(_, air_ctx)| {
                    if air_ctx.public_values.is_empty() {
                        MetalBuffer::with_capacity(0)
                    } else {
                        air_ctx.public_values.to_device()
                    }
                })
                .collect_vec(),
            air_indices_per_trace: ctx
                .per_trace
                .iter()
                .map(|(air_idx, _)| *air_idx)
                .collect_vec(),
            zerocheck_tilde_evals: vec![EF::ZERO; num_airs_present],
            logup_tilde_evals: vec![[EF::ZERO; 2]; num_airs_present],
            needs_next_per_trace,
            trace_interactions,
            pk,
            prev_s_eval: EF::ZERO,
            eq_ns: Vec::with_capacity(n_max + 1),
            eq_sharp_ns: Vec::with_capacity(n_max + 1),
            save_memory,
            gkr_mem_contribution: 0,
            memory_limit_bytes: 0, // Set after GKR input eval
            monomial_num_y_threshold,
        }
    }

    // PERF[jpw]: we could return evals and batch zerocheck poly by degree before interpolating.
    // Cannot do it for logup because we need to calculate the sum claims.
    #[instrument(name = "prover.rap_constraints.ple_round0", level = "info", skip_all)]
    fn sumcheck_uni_round0_polys(
        &mut self,
        ctx: &ProvingContext<MetalBackend>,
        lambda: EF,
    ) -> Vec<UnivariatePoly<EF>> {
        let n_logup = self.n_logup;
        let l_skip = self.l_skip;
        let xi = &self.xi;
        let h_lambda_pows = lambda.powers().take(self.max_num_constraints).collect_vec();
        self.lambda_pows = Some(if !h_lambda_pows.is_empty() {
            h_lambda_pows.to_device()
        } else {
            MetalBuffer::with_capacity(0)
        });
        // Precompute lambda combinations for all AIRs with monomials
        let lambda_pows_ref = self.lambda_pows.as_ref().unwrap();
        for (air_idx, air_pk) in self.pk.per_air.iter().enumerate() {
            if air_pk.other_data.zerocheck_monomials.is_some() {
                self.lambda_combinations[air_idx] = Some(compute_lambda_combinations(
                    self.pk,
                    air_idx,
                    lambda_pows_ref,
                ));
            }
        }
        let num_present_airs = ctx.per_trace.len();
        debug_assert_eq!(num_present_airs, self.n_per_trace.len());

        self.eq_3b_per_trace = ctx
            .per_trace
            .iter()
            .enumerate()
            .map(|(trace_idx, (air_idx, _))| {
                let vk = &self.pk.per_air[*air_idx].vk;
                let num_interactions = vk.num_interactions();
                if num_interactions > 0 {
                    let n = self.n_per_trace[trace_idx];
                    let n_lift = n.max(0) as usize;
                    let mut b_vec = vec![F::ZERO; n_logup - n_lift];
                    let mut weights = Vec::with_capacity(num_interactions);
                    for interaction_idx in 0..num_interactions {
                        let stacked_idx = self
                            .interactions_layout
                            .get(trace_idx, interaction_idx)
                            .unwrap()
                            .row_idx;
                        let mut b_int = stacked_idx >> (l_skip + n_lift);
                        for bit in &mut b_vec {
                            *bit = F::from_bool(b_int & 1 == 1);
                            b_int >>= 1;
                        }
                        let weight =
                            eval_eq_mle(&self.xi[l_skip + n_lift..l_skip + n_logup], &b_vec);
                        weights.push(weight);
                    }
                    weights
                } else {
                    vec![]
                }
            })
            .collect_vec();
        self.d_eq_3b_per_trace = self
            .eq_3b_per_trace
            .iter()
            .map(|eq_3bs| {
                if eq_3bs.is_empty() {
                    MetalBuffer::with_capacity(0)
                } else {
                    eq_3bs.to_device()
                }
            })
            .collect();

        // Precompute logup combinations for all traces with interaction monomials
        for (trace_idx, (air_idx, _)) in ctx.per_trace.iter().enumerate() {
            let air_pk = &self.pk.per_air[*air_idx];
            if air_pk.other_data.interaction_monomials.is_some()
                && !self.eq_3b_per_trace[trace_idx].is_empty()
            {
                self.logup_combinations[trace_idx] = Some(compute_logup_combinations(
                    self.pk,
                    *air_idx,
                    &self.d_beta_pows,
                    &self.d_eq_3b_per_trace[trace_idx],
                    &self.eq_3b_per_trace[trace_idx],
                    &self.beta_pows,
                ));
            }
        }

        // PERF[jpw]: we could also build the layers for different n in a transposed way using
        // eq_nonoverlapping_stage_ext, which is more memory efficient
        for &n in &self.n_per_trace {
            let n_lift = n.max(0) as usize;
            self.eq_xis.entry(n_lift).or_insert_with(|| {
                EqEvalLayers::new_rev(n_lift, xi[l_skip..l_skip + n_lift].iter().rev())
                    .expect("failed to compute eq_xis on device")
            });
        }

        self.sels_per_trace_base = self
            .n_per_trace
            .iter()
            .map(|&n| {
                let n_lift = n.max(0) as usize;
                let height = 1 << n_lift;
                let mut cols = F::zero_vec(3 * height);
                cols[height..2 * height - 1].fill(F::ONE); // is_transition
                cols[0] = F::ONE; // is_first
                cols[2 * height + height - 1] = F::ONE; // is_last
                let d_cols = cols.to_device();
                MetalMatrix::new(Arc::new(d_cols), height, 3)
            })
            .collect_vec();

        let selectors_base = self.sels_per_trace_base.clone();

        // All (numer, denom) pairs per present AIR for logup, followed by 1 zerocheck poly per
        // present AIR
        let mut batch_sp_poly = vec![UnivariatePoly::new(vec![]); 3 * num_present_airs];
        let d_lambda_pows = self
            .lambda_pows
            .as_ref()
            .expect("lambda powers must be set before round-0 evaluation");

        // Loop through one AIR at a time; it is more efficient to do everything for one AIR
        // together
        for (trace_idx, ((air_idx, air_ctx), &n, selectors_cube, public_values, eq_3bs)) in izip!(
            &ctx.per_trace,
            &self.n_per_trace,
            &selectors_base,
            &self.public_values_per_trace,
            &self.eq_3b_per_trace,
        )
        .enumerate()
        {
            debug!("starting batch constraints for air_idx={air_idx} (trace_idx={trace_idx})");
            let single_pk = &self.pk.per_air[*air_idx];
            // Includes both plain AIR constraints and symbolic interactions
            let single_air_constraints =
                SymbolicConstraints::from(&single_pk.vk.symbolic_constraints);
            let local_constraint_deg = single_pk.vk.max_constraint_degree as usize;
            debug_assert_eq!(
                single_air_constraints.max_constraint_degree(),
                local_constraint_deg
            );
            assert!(
                local_constraint_deg <= self.constraint_degree,
                "Max constraint degree ({local_constraint_deg}) of AIR {air_idx} exceeds the global maximum {}",
                self.constraint_degree
            );

            let log_large_domain = log2_ceil_usize(local_constraint_deg << l_skip);
            let omega_root = F::two_adic_generator(log_large_domain);

            assert!(!xi.is_empty(), "xi vector must not be empty");

            let height = air_ctx.common_main.height();
            let mut main_parts = Vec::with_capacity(air_ctx.cached_mains.len() + 1);
            for committed in &air_ctx.cached_mains {
                main_parts.push(committed.trace.buffer().as_device_ptr());
            }
            main_parts.push(air_ctx.common_main.buffer().as_device_ptr());
            let d_main_parts = main_parts.to_device();

            let n_lift = n.max(0) as usize;
            let eq_xi_tree = &self.eq_xis[&n_lift];
            let max_temp_bytes = self.memory_limit_bytes;
            // local_constraint_deg = 0 means no constraints. The only way that linear constraints
            // on trace polynomials could vanish on 2^l_skip points is if the constraint polynomial
            // is identically zero. Thus for local_constraint_deg = 0 or 1, we must have `s'_0 = 0`.
            let num_cosets_zc = local_constraint_deg.saturating_sub(1);
            let sum_buffer = evaluate_round0_constraints_metal(
                single_pk,
                selectors_cube.buffer(),
                &d_main_parts,
                public_values,
                eq_xi_tree.get_ptr(n_lift),
                d_lambda_pows,
                1 << l_skip,
                1 << n_lift,
                height as u32,
                num_cosets_zc as u32,
                omega_root,
                max_temp_bytes,
            )
            .expect("failed to evaluate round-0 constraints on device");
            if !sum_buffer.is_empty() {
                let q_evals = sum_buffer.to_vec();
                debug!(
                    "round0 zerocheck q_evals | trace_idx: {} | values: {:?}",
                    trace_idx, q_evals
                );
                let q = {
                    // Make q_evals row-major, with columns <> cosets
                    let mut values = EF::zero_vec(num_cosets_zc << l_skip);
                    for coset_idx in 0..num_cosets_zc {
                        for i in 0..1 << l_skip {
                            values[i * num_cosets_zc + coset_idx] =
                                q_evals[(coset_idx << l_skip) + i];
                        }
                    }
                    UnivariatePoly::from_geometric_cosets_evals_idft(
                        RowMajorMatrix::new(values, num_cosets_zc),
                        omega_root,
                        omega_root,
                    )
                };
                // sp_0 = (Z^{2^l_skip} - 1) * q
                let sp_0_deg = sumcheck_round0_deg(l_skip, local_constraint_deg);
                let coeffs = (0..=sp_0_deg)
                    .map(|i| {
                        let mut c = -*q.coeffs().get(i).unwrap_or(&EF::ZERO);
                        if i >= 1 << l_skip {
                            c += q.coeffs()[i - (1 << l_skip)];
                        }
                        c
                    })
                    .collect_vec();
                debug_assert_eq!(
                    coeffs.iter().step_by(1 << l_skip).copied().sum::<EF>(),
                    EF::ZERO,
                    "Zerocheck sum is not zero for air_id: {}",
                    ctx.per_trace[trace_idx].0
                );

                batch_sp_poly[2 * num_present_airs + trace_idx] = UnivariatePoly::new(coeffs);
            }

            // PERF: we could use an interaction-specific constraint degree here
            let num_cosets_logup = local_constraint_deg;
            let sum = evaluate_round0_interactions_metal(
                single_pk,
                &single_air_constraints,
                selectors_cube.buffer(),
                &d_main_parts,
                public_values,
                eq_xi_tree.get_ptr(n_lift),
                &self.beta_pows,
                eq_3bs,
                1 << l_skip,
                1 << n_lift,
                height as u32,
                num_cosets_logup as u32,
                omega_root,
                max_temp_bytes,
            )
            .expect("failed to evaluate round-0 interactions on device");
            if !sum.is_empty() {
                let evals = frac_buffer_to_vec(&sum);
                let (mut numer, denom): (Vec<EF>, Vec<EF>) =
                    evals.into_iter().map(|frac| (frac.p, frac.q)).unzip();
                if n.is_negative() {
                    // normalize for lifting
                    let norm_factor = F::from_u32(1 << n.unsigned_abs()).inverse();
                    for s in &mut numer {
                        *s *= norm_factor;
                    }
                }
                let mut numer_values = EF::zero_vec(num_cosets_logup << l_skip);
                let mut denom_values = EF::zero_vec(num_cosets_logup << l_skip);
                for coset_idx in 0..num_cosets_logup {
                    for i in 0..1 << l_skip {
                        let src = (coset_idx << l_skip) + i;
                        let dst = i * num_cosets_logup + coset_idx;
                        numer_values[dst] = numer[src];
                        denom_values[dst] = denom[src];
                    }
                }
                // Logup uses cosets 1, g^1, g^2, ... (init = 1, shift = omega_root)
                batch_sp_poly[2 * trace_idx] = UnivariatePoly::from_geometric_cosets_evals_idft(
                    RowMajorMatrix::new(numer_values, num_cosets_logup),
                    omega_root,
                    F::ONE, // init = 1 for identity coset
                );
                batch_sp_poly[2 * trace_idx + 1] = UnivariatePoly::from_geometric_cosets_evals_idft(
                    RowMajorMatrix::new(denom_values, num_cosets_logup),
                    omega_root,
                    F::ONE, // init = 1 for identity coset
                );
            }
        }
        batch_sp_poly
    }

    // Note: there are no gpu sync points in this function, so span does not indicate kernel times
    #[instrument(name = "LogupZerocheck::fold_ple_evals", level = "debug", skip_all)]
    fn fold_ple_evals(&mut self, ctx: &ProvingContext<MetalBackend>, r_0: EF) {
        let l_skip = self.l_skip;
        let inv_lagrange_denoms_r0 =
            compute_barycentric_inv_lagrange_denoms(l_skip, &self.omega_skip_pows, r_0);
        if self.n_per_trace.iter().any(|&n| n <= 0) {
            let ones_check = self
                .omega_skip_pows
                .iter()
                .zip(inv_lagrange_denoms_r0.iter())
                .fold(EF::ZERO, |acc, (&w, &inv)| acc + inv * w);
            debug!(
                l_skip,
                inv_denoms = ?inv_lagrange_denoms_r0,
                ones_check = %ones_check,
                "inv_lagrange_denoms_round0"
            );
        }
        let d_inv_lagrange_denoms_r0 = inv_lagrange_denoms_r0.to_device();

        let mut mem_limit = self.gkr_mem_contribution;
        // Metal folding for mat_evals_per_trace
        self.mat_evals_per_trace = ctx
            .per_trace
            .iter()
            .enumerate()
            .map(|(trace_idx, (air_idx, air_ctx))| {
                let air_pk = &self.pk.per_air[*air_idx];
                let need_rot = air_pk.vk.params.need_rot;
                let mut results: Vec<MetalMatrix<EF>> = Vec::new();
                let mut mat_idx = 0usize;

                // Preprocessed (if exists)
                if let Some(committed) = &air_pk.preprocessed_data {
                    let trace = &committed.trace;
                    let folded = fold_ple_evals_rotate(
                        l_skip,
                        &self.d_omega_skip_pows,
                        trace,
                        &d_inv_lagrange_denoms_r0,
                        need_rot,
                    )
                    .unwrap();
                    #[cfg(debug_assertions)]
                    if trace_idx == 0 {
                        let trace_host = transport_matrix_d2h_col_major(trace);
                        let plain = openvm_stark_backend::prover::sumcheck::fold_ple_evals(
                            l_skip,
                            trace_host.as_view().into(),
                            false,
                            r_0,
                        );
                        let expected = if need_rot {
                            let rot = openvm_stark_backend::prover::sumcheck::fold_ple_evals(
                                l_skip,
                                trace_host.as_view().into(),
                                true,
                                r_0,
                            );
                            [plain.values, rot.values].concat()
                        } else {
                            plain.values
                        };
                        let got = folded.buffer().to_vec();
                        if got != expected {
                            let first = got
                                .iter()
                                .zip(expected.iter())
                                .position(|(a, b)| a != b)
                                .unwrap_or(0);
                            panic!(
                                "fold_ple_round0_mismatch trace_idx={} mat_idx={} first={} got={:?} expected={:?}",
                                trace_idx,
                                mat_idx,
                                first,
                                got[first],
                                expected[first]
                            );
                        }
                    }
                    results.push(folded);
                    mat_idx += 1;
                }

                // Cached mains
                for committed in &air_ctx.cached_mains {
                    let trace = &committed.trace;
                    let folded = fold_ple_evals_rotate(
                        l_skip,
                        &self.d_omega_skip_pows,
                        trace,
                        &d_inv_lagrange_denoms_r0,
                        need_rot,
                    )
                    .unwrap();
                    #[cfg(debug_assertions)]
                    if trace_idx == 0 {
                        let trace_host = transport_matrix_d2h_col_major(trace);
                        let plain = openvm_stark_backend::prover::sumcheck::fold_ple_evals(
                            l_skip,
                            trace_host.as_view().into(),
                            false,
                            r_0,
                        );
                        let expected = if need_rot {
                            let rot = openvm_stark_backend::prover::sumcheck::fold_ple_evals(
                                l_skip,
                                trace_host.as_view().into(),
                                true,
                                r_0,
                            );
                            [plain.values, rot.values].concat()
                        } else {
                            plain.values
                        };
                        let got = folded.buffer().to_vec();
                        if got != expected {
                            let first = got
                                .iter()
                                .zip(expected.iter())
                                .position(|(a, b)| a != b)
                                .unwrap_or(0);
                            panic!(
                                "fold_ple_round0_mismatch trace_idx={} mat_idx={} first={} got={:?} expected={:?}",
                                trace_idx,
                                mat_idx,
                                first,
                                got[first],
                                expected[first]
                            );
                        }
                    }
                    results.push(folded);
                    mat_idx += 1;
                }

                // Common main
                let trace = &air_ctx.common_main;
                let folded = fold_ple_evals_rotate(
                    l_skip,
                    &self.d_omega_skip_pows,
                    trace,
                    &d_inv_lagrange_denoms_r0,
                    need_rot,
                )
                .unwrap();
                #[cfg(debug_assertions)]
                if trace_idx == 0 {
                    let trace_host = transport_matrix_d2h_col_major(trace);
                    let plain = openvm_stark_backend::prover::sumcheck::fold_ple_evals(
                        l_skip,
                        trace_host.as_view().into(),
                        false,
                        r_0,
                    );
                    let expected = if need_rot {
                        let rot = openvm_stark_backend::prover::sumcheck::fold_ple_evals(
                            l_skip,
                            trace_host.as_view().into(),
                            true,
                            r_0,
                        );
                        [plain.values, rot.values].concat()
                    } else {
                        plain.values
                    };
                    let got = folded.buffer().to_vec();
                    if got != expected {
                        let first = got
                            .iter()
                            .zip(expected.iter())
                            .position(|(a, b)| a != b)
                            .unwrap_or(0);
                        panic!(
                            "fold_ple_round0_mismatch trace_idx={} mat_idx={} first={} got={:?} expected={:?}",
                            trace_idx,
                            mat_idx,
                            first,
                            got[first],
                            expected[first]
                        );
                    }
                }
                mem_limit = mem_limit.saturating_sub(folded.buffer().len() * size_of::<EF>());
                results.push(folded);

                results
            })
            .collect();
        if tracing::enabled!(tracing::Level::DEBUG) {
            for (trace_idx, mats) in self.mat_evals_per_trace.iter().enumerate() {
                if trace_idx > 0 {
                    continue;
                }
                for (mat_idx, mat) in mats.iter().enumerate() {
                    if mat.height() <= 8 {
                        debug!(
                            trace_idx,
                            mat_idx,
                            width = mat.width(),
                            height = mat.height(),
                            values = ?mat.buffer().to_vec(),
                            "metal_fold_ple_mat_evals_trace"
                        );
                    }
                }
            }
        }
        if self.save_memory {
            self.memory_limit_bytes = mem_limit;
        }

        // Metal folding for sels_per_trace (rotate=false, only need offset=0)
        self.sels_per_trace = std::mem::take(&mut self.sels_per_trace_base)
            .into_iter()
            .enumerate()
            .map(|(trace_idx, selectors_cube)| {
                let n = self.n_per_trace[trace_idx];
                let num_x = selectors_cube.height();
                debug_assert_eq!(num_x, 1 << n.max(0));
                debug_assert_eq!(selectors_cube.width(), 3);
                let (l, r) = if n.is_negative() {
                    (
                        l_skip.wrapping_add_signed(n),
                        r_0.exp_power_of_2(-n as usize),
                    )
                } else {
                    (l_skip, r_0)
                };
                let omega = F::two_adic_generator(l);
                let is_first = eval_eq_uni_at_one(l, r);
                let is_last = eval_eq_uni_at_one(l, r * omega);
                let folded_buf = MetalBuffer::<EF>::with_capacity(num_x * 3);
                unsafe {
                    fold_selectors_round0(
                        &folded_buf,
                        selectors_cube.buffer(),
                        is_first,
                        is_last,
                        num_x,
                    )
                    .unwrap();
                }
                MetalMatrix::new(Arc::new(folded_buf), num_x, 3)
            })
            .collect();
        if tracing::enabled!(tracing::Level::DEBUG) {
            for (trace_idx, sels) in self.sels_per_trace.iter().enumerate() {
                if trace_idx > 0 {
                    continue;
                }
                if sels.height() <= 8 {
                    debug!(
                        trace_idx,
                        width = sels.width(),
                        height = sels.height(),
                        values = ?sels.buffer().to_vec(),
                        "metal_fold_ple_selectors_trace"
                    );
                }
            }
        }

        // Compute scalars on CPU (small computation)
        let eq_r0 = eval_eq_uni(l_skip, self.xi[0], r_0);
        let eq_sharp_r0 = eval_eq_sharp_uni(&self.omega_skip_pows, &self.xi[..l_skip], r_0);
        self.eq_ns.push(eq_r0);
        self.eq_sharp_ns.push(eq_sharp_r0);
        for tree in self.eq_xis.values_mut() {
            // trim the back (which corresponds to r_{j-1}) because we don't need it anymore
            if tree.layers.len() > 1 {
                tree.layers.pop();
            }
        }
        if tracing::enabled!(tracing::Level::DEBUG) {
            for (trace_idx, &n) in self.n_per_trace.iter().enumerate() {
                if trace_idx > 0 {
                    continue;
                }
                let n_lift = n.max(0) as usize;
                if let Some(tree) = self.eq_xis.get(&n_lift) {
                    if tree.layers.len() > 1 {
                        debug!(
                            trace_idx,
                            n_lift,
                            eq_layer = ?tree.layers[1].to_vec(),
                            "metal_eq_xi_after_round0_fold"
                        );
                    }
                }
            }
        }
    }

    #[instrument(
        name = "LogupZerocheck::sumcheck_polys_batch_eval",
        level = "info",
        skip_all,
        fields(round = round)
    )]
    fn sumcheck_polys_batch_eval(&mut self, round: usize, r_prev: EF) -> Vec<Vec<EF>> {
        let sp_deg = self.constraint_degree;

        // Per-trace outputs (filled as we go)
        let mut zc_out: Vec<Vec<EF>> = vec![vec![EF::ZERO; sp_deg]; self.n_per_trace.len()];
        let mut logup_out: Vec<[Vec<EF>; 2]> =
            vec![[vec![EF::ZERO; sp_deg], vec![EF::ZERO; sp_deg]]; self.n_per_trace.len()];

        // Keep early interpolations alive for duration of kernels
        let mut _keepalive_interpolated: Vec<MetalMatrix<EF>> = Vec::new();

        let mut late_eval: Vec<TraceCtx> = Vec::new(); // round == n_lift + 1
        let mut early_eval: Vec<TraceCtx> = Vec::new(); // round <= n_lift

        // First, handle traces in original order and split into cases
        for (trace_idx, (&n, mats, sels, eq_3bs, public_vals, &air_idx)) in izip!(
            self.n_per_trace.iter(),
            self.mat_evals_per_trace.iter(),
            self.sels_per_trace.iter(),
            self.d_eq_3b_per_trace.iter(),
            self.public_values_per_trace.iter(),
            self.air_indices_per_trace.iter()
        )
        .enumerate()
        {
            let pk = &self.pk.per_air[air_idx];
            let dag = &pk.vk.symbolic_constraints;
            let has_constraints = dag.constraints.num_constraints() > 0;
            let has_interactions = !dag.interactions.is_empty();
            if !has_constraints && !has_interactions {
                continue;
            }

            let n_lift = n.max(0) as usize;
            let norm_factor_denom = 1 << (-n).max(0);
            let norm_factor = F::from_usize(norm_factor_denom).inverse();
            let has_preprocessed = pk.preprocessed_data.is_some();
            let need_rot = pk.vk.params.need_rot;
            let first_main_idx = usize::from(has_preprocessed);
            let eq_xi_tree = &self.eq_xis[&n_lift];

            if round > n_lift {
                // Case A
                if round == n_lift + 1 {
                    // A.1: evaluate directly at (num_x=1, num_y=1)
                    let prep_ptr = if has_preprocessed {
                        MainMatrixPtrs {
                            data: mats[0].buffer().as_device_ptr(),
                            air_width: air_width_for_mat(need_rot, mats[0].width()),
                        }
                    } else {
                        MainMatrixPtrs {
                            data: std::ptr::null(),
                            air_width: 0,
                        }
                    };
                    let main_ptrs: Vec<MainMatrixPtrs<EF>> = mats[first_main_idx..]
                        .iter()
                        .map(|m| MainMatrixPtrs {
                            data: m.buffer().as_device_ptr(),
                            air_width: air_width_for_mat(need_rot, m.width()),
                        })
                        .collect_vec();
                    let main_ptrs_dev = main_ptrs.to_device();

                    late_eval.push(TraceCtx {
                        trace_idx,
                        air_idx,
                        n_lift,
                        num_y: 1,
                        has_constraints,
                        has_interactions,
                        norm_factor,
                        eq_xi_ptr: eq_xi_tree.get_device_ptr(0),
                        sels_ptr: sels.buffer().as_device_ptr(),
                        prep_ptr,
                        main_ptrs_dev,
                        public_ptr: public_vals.as_device_ptr(),
                        eq_3bs_ptr: eq_3bs.as_device_ptr(),
                    });
                } else {
                    // A.2: scale tilde evals only
                    if has_constraints {
                        let tilde_eval = &mut self.zerocheck_tilde_evals[trace_idx];
                        *tilde_eval *= r_prev;
                        // zc_out not set, will be handled directly from tilde eval in
                        // compute_batch_s
                    }
                    if has_interactions {
                        for x in self.logup_tilde_evals[trace_idx].iter_mut() {
                            *x *= r_prev;
                        }
                        // logup_out not set, will be handled directly from tilde eval in
                        // compute_batch_s
                    }
                }
            } else {
                // Case B: interpolate columns and evaluate (num_x = s_deg, num_y = height/2)
                let log_num_y = n_lift - round;
                let num_y = 1 << log_num_y;
                let height = 2 * num_y;
                debug_assert_eq!(height, mats[0].height());

                let mut columns: Vec<ColumnPtr<EF>> = Vec::new();
                columns.extend(iter::once(sels).chain(mats.iter()).flat_map(|m| {
                    assert_eq!(m.height(), height);
                    (0..m.width()).map(|col| ColumnPtr {
                        data: m.buffer().as_device_ptr().wrapping_add(col * m.height()),
                    })
                }));
                let interpolated = MetalMatrix::<EF>::with_capacity(sp_deg * num_y, columns.len());
                let d_columns = columns.to_device();
                unsafe {
                    interpolate_columns_gpu(interpolated.buffer(), &d_columns, sp_deg, num_y)
                        .expect("failed to interpolate columns on Metal");
                }
                #[cfg(debug_assertions)]
                if tracing::enabled!(tracing::Level::DEBUG) && trace_idx == 0 {
                    let got = interpolated.buffer().to_vec();
                    let mut expected = vec![EF::ZERO; got.len()];
                    let mut out_col = 0usize;
                    for m in iter::once(sels).chain(mats.iter()) {
                        let src = m.buffer().to_vec();
                        for col in 0..m.width() {
                            let base = col * height;
                            for x in 0..sp_deg {
                                for y in 0..num_y {
                                    let t0 = src[base + (y << 1)];
                                    let t1 = src[base + ((y << 1) | 1)];
                                    let out_idx = out_col * sp_deg * num_y + x * num_y + y;
                                    expected[out_idx] = t0 + (t1 - t0) * F::from_usize(x + 1);
                                }
                            }
                            out_col += 1;
                        }
                    }
                    if got != expected {
                        let first = got
                            .iter()
                            .zip(expected.iter())
                            .position(|(a, b)| a != b)
                            .unwrap_or(0);
                        let plane = sp_deg * num_y;
                        let col = first / plane;
                        let rem = first % plane;
                        let x = rem / num_y;
                        let y = rem % num_y;
                        debug!(
                            round,
                            trace_idx,
                            num_y,
                            sp_deg,
                            first,
                            col,
                            x,
                            y,
                            got = ?got[first],
                            expected = ?expected[first],
                            "interpolate_columns_full_mismatch"
                        );
                    }
                    if round == 1 {
                        let rules_host = pk.other_data.zerocheck_mle.inner.d_rules.to_vec();
                        let used_nodes = pk.other_data.zerocheck_mle.inner.d_used_nodes.to_vec();
                        let interpolated_height = sp_deg * num_y;
                        let decoded_rules_all: Vec<_> = rules_host
                            .iter()
                            .map(|&r| crate::logup_zerocheck::rules::RuleWithFlag::<F>::decode(r))
                            .collect();
                        let decoded_rules: Vec<_> =
                            decoded_rules_all.iter().take(10).cloned().collect();
                        debug!(
                            round,
                            trace_idx,
                            rules_len = rules_host.len(),
                            used_nodes_len = used_nodes.len(),
                            ?used_nodes,
                            ?decoded_rules,
                            "zerocheck_mle_rules_preview"
                        );

                        let lambda_pows = self
                            .lambda_pows
                            .as_ref()
                            .expect("lambda_pows must exist after round0")
                            .to_vec();
                        let eq_xi_vals = self.eq_xis[&n_lift].layers[log_num_y].to_vec();
                        let public_vals_host = public_vals.to_vec();
                        let mut main_part_base_cols = Vec::new();
                        let mut main_part_air_widths = Vec::new();
                        let mut cursor =
                            3usize + if has_preprocessed { mats[0].width() } else { 0 };
                        for m in mats[first_main_idx..].iter() {
                            main_part_base_cols.push(cursor);
                            main_part_air_widths
                                .push(air_width_for_mat(need_rot, m.width()) as usize);
                            cursor += m.width();
                        }

                        let mut cpu_sim = vec![EF::ZERO; sp_deg];
                        for x_int in 0..sp_deg {
                            let mut x_sum = EF::ZERO;
                            for y in 0..num_y {
                                let row = x_int * num_y + y;
                                let mut inter = vec![
                                    EF::ZERO;
                                    pk.other_data.zerocheck_mle.inner.buffer_size
                                        as usize
                                ];
                                let mut sum = EF::ZERO;
                                let mut lambda_idx = 0usize;
                                for (node_idx, rule) in decoded_rules_all.iter().enumerate() {
                                    let eval_source =
                                        |src: &crate::logup_zerocheck::rules::Source<F>,
                                         inter: &Vec<EF>| {
                                            match src {
                                            crate::logup_zerocheck::rules::Source::Intermediate(i) => inter[*i],
                                            crate::logup_zerocheck::rules::Source::TerminalIntermediate => EF::ZERO,
                                            crate::logup_zerocheck::rules::Source::Var(v) => {
                                                match v.entry {
                                                    openvm_stark_backend::air_builders::symbolic::symbolic_variable::Entry::Main { part_index, offset } => {
                                                        let part = part_index;
                                                        let base_col = main_part_base_cols[part];
                                                        let air_w = main_part_air_widths[part];
                                                        let col = base_col + offset * air_w + v.index;
                                                        got[col * interpolated_height + row]
                                                    }
                                                    openvm_stark_backend::air_builders::symbolic::symbolic_variable::Entry::Preprocessed { offset } => {
                                                        let base_col = 3 + offset * mats[0].width();
                                                        let col = base_col + v.index;
                                                        got[col * interpolated_height + row]
                                                    }
                                                    openvm_stark_backend::air_builders::symbolic::symbolic_variable::Entry::Public => {
                                                        EF::from(public_vals_host[v.index])
                                                    }
                                                    _ => EF::ZERO,
                                                }
                                            }
                                            crate::logup_zerocheck::rules::Source::IsFirst => {
                                                got[row]
                                            }
                                            crate::logup_zerocheck::rules::Source::IsTransition => {
                                                got[interpolated_height + row]
                                            }
                                            crate::logup_zerocheck::rules::Source::IsLast => {
                                                got[2 * interpolated_height + row]
                                            }
                                            crate::logup_zerocheck::rules::Source::Constant(c) => EF::from(*c),
                                        }
                                        };
                                    let result = match &rule.inner {
                                        crate::logup_zerocheck::rules::Rule::Add(x, y_src, z) => {
                                            let r =
                                                eval_source(x, &inter) + eval_source(y_src, &inter);
                                            if let crate::logup_zerocheck::rules::Source::Intermediate(i) = z {
                                                inter[*i] = r;
                                            }
                                            r
                                        }
                                        crate::logup_zerocheck::rules::Rule::Sub(x, y_src, z) => {
                                            let r =
                                                eval_source(x, &inter) - eval_source(y_src, &inter);
                                            if let crate::logup_zerocheck::rules::Source::Intermediate(i) = z {
                                                inter[*i] = r;
                                            }
                                            r
                                        }
                                        crate::logup_zerocheck::rules::Rule::Mul(x, y_src, z) => {
                                            let r =
                                                eval_source(x, &inter) * eval_source(y_src, &inter);
                                            if let crate::logup_zerocheck::rules::Source::Intermediate(i) = z {
                                                inter[*i] = r;
                                            }
                                            r
                                        }
                                        crate::logup_zerocheck::rules::Rule::Neg(x, z) => {
                                            let r = -eval_source(x, &inter);
                                            if let crate::logup_zerocheck::rules::Source::Intermediate(i) = z {
                                                inter[*i] = r;
                                            }
                                            r
                                        }
                                        crate::logup_zerocheck::rules::Rule::Variable(x) => {
                                            eval_source(x, &inter)
                                        }
                                        crate::logup_zerocheck::rules::Rule::BufferVar(x, z) => {
                                            let r = eval_source(x, &inter);
                                            if let crate::logup_zerocheck::rules::Source::Intermediate(i) = z {
                                                inter[*i] = r;
                                            }
                                            r
                                        }
                                    };
                                    if rule.need_accumulate {
                                        while lambda_idx < used_nodes.len()
                                            && used_nodes[lambda_idx] == node_idx
                                        {
                                            sum += lambda_pows[lambda_idx] * result;
                                            lambda_idx += 1;
                                        }
                                    }
                                }
                                x_sum += sum * eq_xi_vals[y];
                            }
                            cpu_sim[x_int] = x_sum;
                        }
                        debug!(round, trace_idx, ?cpu_sim, "zerocheck_mle_cpu_sim");
                    }
                }

                let interpolated_height = interpolated.height();
                let mut widths_so_far = 0usize;
                let sels_ptr = interpolated
                    .buffer()
                    .as_device_ptr()
                    .wrapping_add(widths_so_far * interpolated_height);
                widths_so_far += 3;
                let prep_ptr = if has_preprocessed {
                    MainMatrixPtrs {
                        data: interpolated
                            .buffer()
                            .as_device_ptr()
                            .wrapping_add(widths_so_far * interpolated_height),
                        air_width: air_width_for_mat(need_rot, mats[0].width()),
                    }
                } else {
                    MainMatrixPtrs {
                        data: std::ptr::null(),
                        air_width: 0,
                    }
                };
                if has_preprocessed {
                    widths_so_far += mats[0].width();
                }
                let main_ptrs: Vec<MainMatrixPtrs<EF>> = mats[first_main_idx..]
                    .iter()
                    .map(|m| {
                        let main_ptr = MainMatrixPtrs {
                            data: interpolated
                                .buffer()
                                .as_device_ptr()
                                .wrapping_add(widths_so_far * interpolated_height),
                            air_width: air_width_for_mat(need_rot, m.width()),
                        };
                        widths_so_far += m.width();
                        main_ptr
                    })
                    .collect_vec();
                debug_assert_eq!(widths_so_far, interpolated.width());
                let main_ptrs_dev = main_ptrs.to_device();

                _keepalive_interpolated.push(interpolated);
                let eq_xi_ptr = eq_xi_tree.get_device_ptr(log_num_y);

                early_eval.push(TraceCtx {
                    trace_idx,
                    air_idx,
                    n_lift,
                    num_y: num_y as u32,
                    has_constraints,
                    has_interactions,
                    norm_factor,
                    eq_xi_ptr,
                    sels_ptr,
                    prep_ptr,
                    main_ptrs_dev,
                    public_ptr: public_vals.as_device_ptr(),
                    eq_3bs_ptr: eq_3bs.as_device_ptr(),
                });
                debug!(
                    round,
                    trace_idx, n_lift, log_num_y, num_y, "sumcheck_early_trace_shape"
                );
            }
        }

        let d_challenges_ptr = self.d_challenges.as_device_ptr();

        // Late traces (num_y=1): always use monomial
        let late_logup_traces: Vec<_> = late_eval.iter().filter(|t| t.has_interactions).collect();
        if !late_logup_traces.is_empty() {
            let logup_combs: Vec<_> = late_logup_traces
                .iter()
                .map(|t| {
                    self.logup_combinations[t.trace_idx]
                        .as_ref()
                        .expect("missing logup monomial combinations for late trace")
                })
                .collect();
            let batch =
                LogupMonomialBatch::new(late_logup_traces.iter().copied(), self.pk, &logup_combs);
            let out = batch.evaluate(1);
            let host = frac_buffer_to_vec(&out);
            for (i, trace_idx) in batch.trace_indices().enumerate() {
                self.logup_tilde_evals[trace_idx][0] = host[i].p * late_logup_traces[i].norm_factor;
                self.logup_tilde_evals[trace_idx][1] = host[i].q;
            }
        }
        let late_mono_traces: Vec<_> = late_eval
            .iter()
            .filter(|t| trace_has_monomials(t, self.pk))
            .collect();
        if !late_mono_traces.is_empty() {
            let lambda_combs: Vec<_> = late_mono_traces
                .iter()
                .map(|t| self.lambda_combinations[t.air_idx].as_ref().unwrap())
                .collect();
            let batch = ZerocheckMonomialBatch::new(late_mono_traces, self.pk, &lambda_combs);
            let out = batch.evaluate(1);
            let host = out.to_vec();
            for (i, trace_idx) in batch.trace_indices().enumerate() {
                self.zerocheck_tilde_evals[trace_idx] = host[i];
                // zc_out not set for num_x=1, handled from tilde_eval in compute_batch_s
            }
        }

        // Logup for early traces: partition by num_y threshold
        if !early_eval.is_empty() {
            evaluate_logup_batched(
                &early_eval,
                self.pk,
                d_challenges_ptr,
                sp_deg as u32,
                self.monomial_num_y_threshold,
                &self.logup_combinations,
                &mut logup_out,
                &mut self.logup_tilde_evals,
                self.memory_limit_bytes,
            );
        }

        // Early traces (num_y>1): partition by threshold for zerocheck path
        let (low_early, high_early): (Vec<&TraceCtx>, Vec<&TraceCtx>) = early_eval
            .iter()
            .filter(|t| t.has_constraints)
            .partition(|t| t.num_y <= self.monomial_num_y_threshold);

        // Partition high num_y traces by monomial-to-rules ratio
        // (traces without monomials are skipped - they contribute zero)
        let (high_dag_traces, high_mono_traces): (Vec<&TraceCtx>, Vec<&TraceCtx>) =
            high_early.iter().partition(|t| {
                let num_monomials = get_num_monomials(t, self.pk);
                let rules_len = get_zerocheck_rules_len(t, self.pk);
                // Use DAG when monomial expansion significantly increased the term count.
                num_monomials as usize >= DAG_FALLBACK_MONOMIAL_RATIO * rules_len
            });

        // DAG evaluation for high num_y traces with high monomial-to-rules ratio
        if !high_dag_traces.is_empty() {
            let lambda_pows = self.lambda_pows.as_ref().unwrap();
            evaluate_zerocheck_batched(
                high_dag_traces,
                self.pk,
                lambda_pows,
                sp_deg as u32,
                &mut zc_out,
                self.memory_limit_bytes,
            );
        }

        // Par-Y monomial kernel for high num_y traces
        if !high_mono_traces.is_empty() {
            let lambda_combs: Vec<_> = high_mono_traces
                .iter()
                .map(|t| self.lambda_combinations[t.air_idx].as_ref().unwrap())
                .collect();
            let batch = ZerocheckMonomialParYBatch::new(
                high_mono_traces,
                self.pk,
                &lambda_combs,
                self.sm_count,
                sp_deg as u32,
                None,
            );
            let out = batch.evaluate(sp_deg as u32);
            let host = out.to_vec();
            for (i, trace_idx) in batch.trace_indices().enumerate() {
                zc_out[trace_idx].copy_from_slice(&host[(i * sp_deg)..((i + 1) * sp_deg)]);
            }
        }

        // Monomial zerocheck for low num_y traces
        let low_mono_traces = low_early;
        if !low_mono_traces.is_empty() {
            let lambda_combs: Vec<_> = low_mono_traces
                .iter()
                .map(|t| self.lambda_combinations[t.air_idx].as_ref().unwrap())
                .collect();
            let batch = ZerocheckMonomialBatch::new(
                low_mono_traces.iter().copied(),
                self.pk,
                &lambda_combs,
            );
            let out = batch.evaluate(sp_deg as u32);
            let host = out.to_vec();
            for (i, trace_idx) in batch.trace_indices().enumerate() {
                zc_out[trace_idx].copy_from_slice(&host[(i * sp_deg)..((i + 1) * sp_deg)]);
            }

            #[cfg(debug_assertions)]
            if tracing::enabled!(tracing::Level::DEBUG) {
                let mut dag_cmp_out: Vec<Vec<EF>> =
                    vec![vec![EF::ZERO; sp_deg]; self.n_per_trace.len()];
                evaluate_zerocheck_batched(
                    low_mono_traces.iter().copied(),
                    self.pk,
                    self.lambda_pows
                        .as_ref()
                        .expect("lambda_pows must exist for zerocheck eval"),
                    sp_deg as u32,
                    &mut dag_cmp_out,
                    self.memory_limit_bytes,
                );
                for t in low_mono_traces.iter().copied() {
                    let mono_vals = &zc_out[t.trace_idx];
                    let dag_vals = &dag_cmp_out[t.trace_idx];
                    if mono_vals != dag_vals {
                        debug!(
                            round,
                            trace_idx = t.trace_idx,
                            ?mono_vals,
                            ?dag_vals,
                            "zerocheck_low_mono_vs_dag_mismatch"
                        );
                    }
                }
            }
        }

        #[cfg(debug_assertions)]
        if tracing::enabled!(tracing::Level::DEBUG) {
            for (trace_idx, vals) in zc_out.iter().enumerate() {
                debug!(round, trace_idx, ?vals, "metal_zc_round_evals");
            }
        }

        logup_out.into_iter().flatten().chain(zc_out).collect()
    }

    #[instrument(level = "debug", skip_all, fields(round = round))]
    fn compute_batch_s_poly(
        &mut self,
        sp_round_evals: Vec<Vec<EF>>,
        num_traces: usize,
        round: usize,
        mu_pows: &[EF],
    ) -> UnivariatePoly<EF> {
        debug_assert_eq!(sp_round_evals.len(), 3 * num_traces);
        debug_assert_eq!(sp_round_evals.len(), mu_pows.len());
        let constraint_degree = self.constraint_degree;
        let mut sp_head_zc = vec![EF::ZERO; constraint_degree];
        let mut sp_head_logup = vec![EF::ZERO; constraint_degree];
        let mut sp_tail = EF::ZERO;
        for (trace_idx, &n) in self.n_per_trace.iter().enumerate() {
            let n_lift = n.max(0) as usize;
            let zc_idx = 2 * num_traces + trace_idx;
            let numer_idx = 2 * trace_idx;
            let denom_idx = numer_idx + 1;
            if round == n_lift + 1 {
                let eq_r_acc = *self.eq_ns.last().unwrap();
                let eq_sharp_r_acc = *self.eq_sharp_ns.last().unwrap();
                self.zerocheck_tilde_evals[trace_idx] *= eq_r_acc;
                self.logup_tilde_evals[trace_idx][0] *= eq_sharp_r_acc;
                self.logup_tilde_evals[trace_idx][1] *= eq_sharp_r_acc;
            }
            if round <= n_lift {
                for i in 0..constraint_degree {
                    sp_head_zc[i] += mu_pows[zc_idx] * sp_round_evals[zc_idx][i];
                    sp_head_logup[i] += mu_pows[numer_idx] * sp_round_evals[numer_idx][i]
                        + mu_pows[denom_idx] * sp_round_evals[denom_idx][i];
                }
            } else {
                sp_tail += mu_pows[zc_idx] * self.zerocheck_tilde_evals[trace_idx]
                    + mu_pows[numer_idx] * self.logup_tilde_evals[trace_idx][0]
                    + mu_pows[denom_idx] * self.logup_tilde_evals[trace_idx][1];
            }
        }
        let s_deg = constraint_degree + 1;
        let l_skip = self.l_skip;
        // With eq(xi,r) contributions
        let mut sp_head_evals = vec![EF::ZERO; s_deg];
        for i in 0..constraint_degree {
            sp_head_evals[i + 1] = self.eq_ns[round - 1] * sp_head_zc[i]
                + self.eq_sharp_ns[round - 1] * sp_head_logup[i];
        }
        // We need to derive s'(0).
        // We use that s_j(0) + s_j(1) = s_{j-1}(r_{j-1})
        let xi_cur = self.xi[l_skip + round - 1];
        {
            let eq_xi_0 = EF::ONE - xi_cur;
            let eq_xi_1 = xi_cur;
            sp_head_evals[0] =
                (self.prev_s_eval - eq_xi_1 * sp_head_evals[1] - sp_tail) * eq_xi_0.inverse();
        }
        // s' has degree s_deg - 1
        let sp_head = UnivariatePoly::lagrange_interpolate(
            &(0..s_deg).map(F::from_usize).collect_vec(),
            &sp_head_evals,
        );
        // eq(xi, X) = (2 * xi - 1) * X + (1 - xi)
        // Compute s(X) = eq(xi, X) * s'_head(X) + s'_tail * X (s'_head now contains eq(..,r))
        // s(X) has degree s_deg
        let mut coeffs = sp_head.into_coeffs();
        coeffs.push(EF::ZERO);
        let b = EF::ONE - xi_cur;
        let a = xi_cur - b;
        for i in (0..s_deg).rev() {
            coeffs[i + 1] = a * coeffs[i] + b * coeffs[i + 1];
        }
        coeffs[0] *= b;
        coeffs[1] += sp_tail;
        UnivariatePoly::new(coeffs)
    }

    #[instrument(name = "LogupZerocheck::fold_mle_evals", level = "debug", skip_all, fields(round = round))]
    fn fold_mle_evals(&mut self, round: usize, r_round: EF) {
        // Assumes that input_mats are sorted by height
        let batch_fold = |input_mats: Vec<MetalMatrix<EF>>| {
            let num_matrices = input_mats.partition_point(|mat| mat.height() > 1);
            let mut output_mats: Vec<_> = input_mats
                .iter()
                .take(num_matrices)
                .map(|mat| {
                    let height = mat.height();
                    let width = mat.width();
                    let output_height = height >> 1;
                    MetalMatrix::<EF>::with_capacity(output_height, width)
                })
                .collect();

            for (input_mat, output_mat) in
                input_mats.iter().take(num_matrices).zip(output_mats.iter())
            {
                let output_height = (input_mat.height() >> 1) as u32;
                let width = input_mat.width() as u32;
                unsafe {
                    fold_mle_matrix(
                        input_mat.buffer(),
                        output_mat.buffer(),
                        width,
                        output_height,
                        r_round,
                    )
                    .expect("failed to fold MLE on Metal");
                }
            }
            #[cfg(debug_assertions)]
            {
                for (mat_idx, (input_mat, output_mat)) in input_mats
                    .iter()
                    .zip(output_mats.iter())
                    .take(num_matrices)
                    .enumerate()
                {
                    let height = input_mat.height();
                    let width = input_mat.width();
                    let output_height = height >> 1;
                    let input = input_mat.buffer().to_vec();
                    let output = output_mat.buffer().to_vec();
                    let mut mismatch = 0usize;
                    let mut first_mismatch = None;
                    for col in 0..width {
                        let in_col = col * height;
                        let out_col = col * output_height;
                        for row in 0..output_height {
                            let t0 = input[in_col + (row << 1)];
                            let t1 = input[in_col + (row << 1) + 1];
                            let expected = t0 + r_round * (t1 - t0);
                            let got = output[out_col + row];
                            if got != expected {
                                mismatch += 1;
                                if first_mismatch.is_none() {
                                    first_mismatch = Some((col, row, t0, t1, expected, got));
                                }
                            }
                        }
                    }
                    if mismatch > 0 {
                        let (first_col, first_row, t0, t1, expected, got) = first_mismatch.unwrap();
                        panic!(
                            "fold_mle_mismatch round={} mat_idx={} h={} w={} out_h={} mismatch={} first_col={} first_row={} t0={:?} t1={:?} expected={:?} got={:?}",
                            round,
                            mat_idx,
                            height,
                            width,
                            output_height,
                            mismatch,
                            first_col,
                            first_row,
                            t0,
                            t1,
                            expected,
                            got
                        );
                    }
                }
            }
            output_mats.extend_from_slice(&input_mats[num_matrices..]);
            output_mats
        };

        // Fold mat_evals_per_trace: Vec<Vec<MetalMatrix<EF>>>
        self.mat_evals_per_trace = {
            let lengths = self
                .mat_evals_per_trace
                .iter()
                .map(|v| v.len())
                .collect_vec();
            let input_mats = std::mem::take(&mut self.mat_evals_per_trace)
                .into_iter()
                .flatten()
                .collect_vec();
            let mut output_mats = batch_fold(input_mats).into_iter();
            lengths
                .into_iter()
                .map(|len| output_mats.by_ref().take(len).collect())
                .collect()
        };
        if self.save_memory {
            self.memory_limit_bytes = self.gkr_mem_contribution.saturating_sub(
                self.mat_evals_per_trace
                    .iter()
                    .flatten()
                    .map(|m| m.buffer().len() * size_of::<EF>())
                    .sum(),
            );
        }

        // Fold sels_per_trace: Vec<MetalMatrix<EF>>
        self.sels_per_trace = batch_fold(std::mem::take(&mut self.sels_per_trace));

        for tree in self.eq_xis.values_mut() {
            // trim the back (which corresponds to r_{j-1}) because we don't need it anymore
            if tree.layers.len() > 1 {
                tree.layers.pop();
            }
        }
        let xi = self.xi[self.l_skip + round - 1];
        let eq_r = eval_eq_mle(&[xi], &[r_round]);
        self.eq_ns.push(self.eq_ns[round - 1] * eq_r);
        self.eq_sharp_ns.push(self.eq_sharp_ns[round - 1] * eq_r);

        #[cfg(debug_assertions)]
        if tracing::enabled!(tracing::Level::DEBUG)
            && self
                .sels_per_trace
                .iter()
                .all(|m| m.height() == 1 && m.width() == 3)
        {
            for (trace_idx, sels) in self.sels_per_trace.iter().enumerate() {
                let vals = sels.buffer().to_vec();
                debug!(
                    round,
                    trace_idx,
                    is_first = %vals[0],
                    is_transition = %vals[1],
                    is_last = %vals[2],
                    "selector_eval"
                );
            }
        }
    }

    #[instrument(
        name = "LogupZerocheck::into_column_openings",
        level = "debug",
        skip_all
    )]
    fn into_column_openings(mut self) -> Vec<Vec<Vec<EF>>> {
        let num_airs_present = self.mat_evals_per_trace.len();
        let mut column_openings = Vec::with_capacity(num_airs_present);

        // At the end, we've folded all MLEs so they only have one row equal to evaluation at `\vec
        // r`.
        for (&need_rot, mat_evals) in self
            .needs_next_per_trace
            .iter()
            .zip(std::mem::take(&mut self.mat_evals_per_trace))
        {
            // Metal matrices are doubled-width (original + rotated), so we need to split them
            // First, copy all matrices to host and split them
            let mut split_mats: Vec<Option<ColMajorMatrix<EF>>> = mat_evals
                .into_iter()
                .flat_map(|mat| {
                    let mat_host = transport_matrix_d2h_col_major(&mat);
                    let width = mat_host.width();
                    let height = mat_host.height();
                    debug_assert_eq!(height, 1, "Matrices should have height=1 after folding");
                    let air_width = if need_rot {
                        debug_assert_eq!(
                            width % 2,
                            0,
                            "Metal matrices should have doubled width (original + rotated)"
                        );
                        width / 2
                    } else {
                        width
                    };

                    // Split doubled-width matrix into original and rotated parts
                    let values = &mat_host.values;
                    let orig: Vec<EF> = (0..air_width)
                        .map(|col| values[col * height]) // height=1, so values[col]
                        .collect();
                    let rot: Option<Vec<EF>> = if need_rot {
                        Some(
                            (air_width..width)
                                .map(|col| values[col * height]) // height=1, so values[col]
                                .collect(),
                        )
                    } else {
                        None
                    };

                    vec![
                        Some(ColMajorMatrix::new(orig, air_width)),
                        rot.map(|mat| ColMajorMatrix::new(mat, air_width)),
                    ]
                })
                .collect();

            // Order of mats after splitting is:
            // - preprocessed (if has_preprocessed),
            // - preprocessed_rot (if has_preprocessed),
            // - cached(0), cached(0)_rot, ...
            // - common_main
            // - common_main_rot
            // For column openings, we pop common_main, common_main_rot and put it at the front
            assert_eq!(
                split_mats.len() % 2,
                0,
                "Should have even number of matrices after splitting"
            );
            let common_main_rot = split_mats.pop().unwrap();
            let common_main = split_mats.pop().unwrap();

            let openings_of_air = iter::once(&[common_main, common_main_rot] as &[_])
                .chain(split_mats.chunks_exact(2))
                .map(|pair| {
                    let plains = pair[0].as_ref().unwrap();
                    if let Some(rots) = pair[1].as_ref() {
                        std::iter::zip(plains.columns(), rots.columns())
                            .flat_map(|(claim, claim_rot)| {
                                assert_eq!(claim.len(), 1);
                                assert_eq!(claim_rot.len(), 1);
                                [claim[0], claim_rot[0]]
                            })
                            .collect_vec()
                    } else {
                        plains
                            .columns()
                            .map(|claim| {
                                assert_eq!(claim.len(), 1);
                                claim[0]
                            })
                            .collect_vec()
                    }
                })
                .collect_vec();
            column_openings.push(openings_of_air);
        }
        column_openings
    }
}
