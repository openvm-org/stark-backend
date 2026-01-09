use std::{
    cmp::max,
    iter::{self, zip},
    sync::Arc,
};

use itertools::{Itertools, izip};
use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    memory_manager::MemTracker,
};
use openvm_stark_backend::{
    air_builders::symbolic::SymbolicConstraints, p3_matrix::dense::RowMajorMatrix,
    prover::MatrixDimensions,
};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{Field, FieldAlgebra, TwoAdicField};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use rustc_hash::FxHashMap;
use stark_backend_v2::{
    calculate_n_logup,
    dft::Radix2BowersSerial,
    poly_common::{
        UnivariatePoly, eq_sharp_uni_poly, eq_uni_poly, eval_eq_mle, eval_eq_sharp_uni,
        eval_eq_uni, eval_eq_uni_at_one,
    },
    poseidon2::sponge::FiatShamirTranscript,
    proof::{BatchConstraintProof, GkrProof},
    prover::{
        ColMajorMatrix, DeviceMultiStarkProvingKeyV2, ProvingContextV2,
        fractional_sumcheck_gkr::Frac, stacked_pcs::StackedLayout, sumcheck::sumcheck_round0_deg,
    },
    utils::batch_multiplicative_inverse_serial,
};
use tracing::{debug, info, info_span, instrument};

use crate::{
    EF, F, GpuBackendV2,
    cuda::{
        logup_zerocheck::{MainMatrixPtrs, fold_selectors_round0, interpolate_columns_gpu},
        sumcheck::batch_fold_mle,
    },
    gpu_backend::transport_matrix_d2h_col_major,
    logup_zerocheck::{
        fold_ple::fold_ple_evals_rotate, gkr_input::TraceInteractionMeta,
        round0::evaluate_round0_interactions_gpu,
    },
    poly::EqEvalLayers,
    sponge::DuplexSpongeGpu,
    utils::compute_barycentric_inv_lagrange_denoms,
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

use batch_mle::{
    TraceCtx, ZerocheckMleBatchBuilder, evaluate_logup_batched, evaluate_zerocheck_batched,
};
use batch_mle_monomial::{
    ZerocheckMonomialBatch, compute_lambda_combinations, trace_has_monomials,
};
pub use errors::*;
use fractional::fractional_sumcheck_gpu;
use gkr_input::{collect_trace_interactions, log_gkr_input_evals};
use mle_round::{evaluate_mle_constraints_gpu, evaluate_mle_interactions_gpu};
use round0::evaluate_round0_constraints_gpu;

#[instrument(level = "info", skip_all)]
pub fn prove_zerocheck_and_logup_gpu(
    transcript: &mut DuplexSpongeGpu,
    mpk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
    ctx: &ProvingContextV2<GpuBackendV2>,
    save_memory: bool,
    monomial_num_y_threshold: u32,
) -> (GkrProof, BatchConstraintProof, Vec<EF>) {
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
    let mut prover = LogupZerocheckGpu::new(
        mpk,
        ctx,
        n_logup,
        interactions_layout,
        alpha_logup,
        beta_logup,
        save_memory,
        monomial_num_y_threshold,
    );
    let n_global = prover.n_global;

    let total_leaves = 1 << (l_skip + n_logup);
    prover
        .mem
        .emit_metrics_with_label("prover.before_gkr_input_evals");
    prover.mem.reset_peak();
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
        DeviceBuffer::new()
    };
    // Set memory limit for batch MLE based on inputs buffer size
    prover.memory_limit_bytes = inputs.len() * std::mem::size_of::<Frac<EF>>();
    prover.mem.emit_metrics_with_label("prover.gkr_input_evals");

    let (frac_sum_proof, mut xi) =
        fractional_sumcheck_gpu(transcript, inputs, true, &mut prover.mem)
            .expect("failed to run fractional sumcheck on GPU");
    while xi.len() != l_skip + n_global {
        xi.push(transcript.sample_ext());
    }
    debug!(?xi);
    prover.xi = xi;

    logup_gkr_span.exit();

    // Note: this span includes ple_fold, but that function has no cuda synchronization so it does
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
            debug_assert!(coeffs.coeffs().len() <= sp_0_deg + 1);
            for (j, &c_j) in coeffs.coeffs().iter().enumerate() {
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
    let skip_domain_size = F::from_canonical_usize(1 << l_skip);
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
        let sp_round_evals = if round > 1 {
            prover.sumcheck_polys_batch_eval(round, r[round - 1])
        } else {
            prover.sumcheck_polys_eval(round, r[round - 1])
        };
        let batch_s = prover.compute_batch_s_poly(sp_round_evals, num_traces, round, &mu_pows);
        let batch_s_evals = (1..=s_deg)
            .map(|i| batch_s.eval_at_point(EF::from_canonical_usize(i)))
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

    // Observe common main openings first, and then preprocessed/cached
    for openings in &column_openings {
        for (claim, claim_rot) in &openings[0] {
            transcript.observe_ext(*claim);
            transcript.observe_ext(*claim_rot);
        }
    }
    for openings in &column_openings {
        for part in openings.iter().skip(1) {
            for (claim, claim_rot) in part {
                transcript.observe_ext(*claim);
                transcript.observe_ext(*claim_rot);
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

pub struct LogupZerocheckGpu<'a> {
    pub alpha_logup: EF,
    pub beta_pows: Vec<EF>,
    // [alpha, beta^0, beta^1, .., beta^max_interaction_len]
    pub d_challenges: DeviceBuffer<EF>,

    pub l_skip: usize,
    n_logup: usize,
    n_global: usize,

    pub omega_skip: F,
    pub omega_skip_pows: Vec<F>,
    d_omega_skip_pows: DeviceBuffer<F>,

    pub interactions_layout: StackedLayout,
    pub constraint_degree: usize,
    n_per_trace: Vec<isize>,
    max_num_constraints: usize,
    pub monomial_num_y_threshold: u32,
    // Available after GKR:
    pub xi: Vec<EF>,
    pub lambda_pows: Option<DeviceBuffer<EF>>,
    /// Precomputed lambda combinations per AIR (indexed by air_idx). Set when lambda is sampled.
    lambda_combinations: Vec<Option<DeviceBuffer<EF>>>,

    // n_T => segment tree of eq(xi[j..1+n_T]) for j=1..={n_T-round+1} in _reverse_ layout
    eq_xis: FxHashMap<usize, EqEvalLayers<EF>>,
    eq_3b_per_trace: Vec<Vec<EF>>,
    d_eq_3b_per_trace: Vec<DeviceBuffer<EF>>,
    // Evaluations on hypercube only, for round 0
    sels_per_trace_base: Vec<DeviceMatrix<F>>,
    // After univariate round 0:
    mat_evals_per_trace: Vec<Vec<DeviceMatrix<EF>>>,
    sels_per_trace: Vec<DeviceMatrix<EF>>,
    // Store public_values per trace (similar to CPU's EvalHelper)
    public_values_per_trace: Vec<DeviceBuffer<F>>,
    air_indices_per_trace: Vec<usize>,
    zerocheck_tilde_evals: Vec<EF>,
    logup_tilde_evals: Vec<[EF; 2]>,

    trace_interactions: Vec<Option<TraceInteractionMeta>>,
    // round0: Round0Buffers,
    pk: &'a DeviceMultiStarkProvingKeyV2<GpuBackendV2>,

    // In round `j`, contains `s_{j-1}(r_{j-1})`
    pub(crate) prev_s_eval: EF,
    pub(crate) eq_ns: Vec<EF>,
    pub(crate) eq_sharp_ns: Vec<EF>,

    mem: MemTracker,
    save_memory: bool,

    /// Memory limit for batch MLE intermediate buffers (set after GKR input eval)
    memory_limit_bytes: usize,
}

impl<'a> LogupZerocheckGpu<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        pk: &'a DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
        ctx: &ProvingContextV2<GpuBackendV2>,
        n_logup: usize,
        interactions_layout: StackedLayout,
        alpha_logup: EF,
        beta_logup: EF,
        save_memory: bool,
        monomial_num_y_threshold: u32,
    ) -> Self {
        let mem = MemTracker::start("prover.logup_zerocheck_prover");
        let l_skip = pk.params.l_skip;
        let omega_skip = F::two_adic_generator(l_skip);
        let omega_skip_pows = omega_skip.powers().take(1 << l_skip).collect_vec();
        let d_omega_skip_pows = omega_skip_pows.to_device().unwrap();
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
        let d_challenges = challenges.to_device().unwrap();

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

        // Collect interaction metadata for GPU execution (evaluations still run on CPU for now).
        let trace_interactions = collect_trace_interactions(pk, ctx, &interactions_layout);

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
            xi: vec![],
            lambda_pows: None,
            lambda_combinations: (0..pk.per_air.len()).map(|_| None).collect(),
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
                        DeviceBuffer::new()
                    } else {
                        air_ctx
                            .public_values
                            .to_device()
                            .expect("failed to copy public values to device")
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
            trace_interactions,
            pk,
            prev_s_eval: EF::ZERO,
            eq_ns: Vec::with_capacity(n_max + 1),
            eq_sharp_ns: Vec::with_capacity(n_max + 1),
            mem,
            save_memory,
            memory_limit_bytes: 0, // Set after GKR input eval
            monomial_num_y_threshold,
        }
    }

    // PERF[jpw]: we could return evals and batch zerocheck poly by degree before interpolating.
    // Cannot do it for logup because we need to calculate the sum claims.
    #[instrument(name = "prover.rap_constraints.ple_round0", level = "info", skip_all)]
    fn sumcheck_uni_round0_polys(
        &mut self,
        ctx: &ProvingContextV2<GpuBackendV2>,
        lambda: EF,
    ) -> Vec<UnivariatePoly<EF>> {
        self.mem
            .emit_metrics_with_label("prover.batch_constraints.before_round0");
        self.mem.reset_peak();
        let n_logup = self.n_logup;
        let l_skip = self.l_skip;
        let xi = &self.xi;
        let h_lambda_pows = lambda.powers().take(self.max_num_constraints).collect_vec();
        self.lambda_pows = Some(if !h_lambda_pows.is_empty() {
            h_lambda_pows
                .to_device()
                .expect("failed to copy lambda powers to device")
        } else {
            DeviceBuffer::new()
        });
        // Precompute lambda combinations for all AIRs with monomials
        let lambda_pows_ref = self.lambda_pows.as_ref().unwrap();
        for (air_idx, air_pk) in self.pk.per_air.iter().enumerate() {
            if air_pk.other_data.zerocheck_monomials.is_some() {
                self.lambda_combinations[air_idx] =
                    Some(compute_lambda_combinations(self.pk, air_idx, lambda_pows_ref).unwrap());
            }
        }
        let global_sp_0_deg = sumcheck_round0_deg(l_skip, self.constraint_degree);
        let num_present_airs = ctx.per_trace.len();
        debug_assert_eq!(num_present_airs, self.n_per_trace.len());

        let total_main_cells = ctx
            .per_trace
            .iter()
            .map(|(_, air_ctx)| air_ctx.common_main.buffer().len())
            .sum::<usize>();

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
                    DeviceBuffer::new()
                } else {
                    eq_3bs.to_device().unwrap()
                }
            })
            .collect();

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
                let d_cols = cols.to_device().unwrap();
                DeviceMatrix::new(Arc::new(d_cols), height, 3)
            })
            .collect_vec();

        let selectors_base = self.sels_per_trace_base.clone();

        let log_glob_large_domain = log2_ceil_usize(global_sp_0_deg + 1);
        // Length is global_large_domain * 2^l_skip, so it is easier to do on CPU
        let glob_inv_lagrange_denoms: Vec<F> = info_span!("inv_lagrange_denoms").in_scope(|| {
            let omega = F::two_adic_generator(log_glob_large_domain);
            let omega_pows = omega
                .powers()
                .take(1 << log_glob_large_domain)
                .collect_vec();
            let denoms = omega_pows
                .iter()
                .flat_map(|&z| {
                    self.omega_skip_pows.iter().map(move |&w_i| {
                        let denom = z - w_i;
                        if denom.is_zero() { F::ONE } else { denom }
                    })
                })
                .collect_vec();
            let mut inv_denoms = batch_multiplicative_inverse_serial(&denoms);
            let inv_weight = F::ONE.halve().exp_u64(l_skip as u64);
            for (z, inv_denoms_z) in omega_pows.iter().zip(inv_denoms.chunks_mut(1 << l_skip)) {
                let zerofier = z.exp_power_of_2(l_skip) - F::ONE;
                let scale_factor = zerofier * inv_weight;
                for v in inv_denoms_z {
                    *v *= scale_factor;
                }
            }
            inv_denoms
        });

        // All (numer, denom) pairs per present AIR for logup, followed by 1 zerocheck poly per
        // present AIR
        let mut batch_sp_evals = vec![vec![]; 3 * num_present_airs];
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
            let local_sp_deg = local_constraint_deg;
            let local_sp_0_deg = sumcheck_round0_deg(l_skip, local_sp_deg);

            let log_large_domain = log2_ceil_usize(local_sp_0_deg + 1);
            // NOTE: we barycentric evaluate, so we don't need the full DFT domain
            let large_domain = local_sp_0_deg + 1;

            assert!(!xi.is_empty(), "xi vector must not be empty");
            let inv_lagrange_denoms: Vec<F> = glob_inv_lagrange_denoms
                .chunks(1 << l_skip)
                .step_by(1 << (log_glob_large_domain - log_large_domain))
                .flatten()
                .copied()
                .collect();
            let d_inv_lagrange_denoms = inv_lagrange_denoms.to_device().unwrap();

            let height = air_ctx.common_main.height();
            let mut main_parts = Vec::with_capacity(air_ctx.cached_mains.len() + 1);
            for committed in &air_ctx.cached_mains {
                main_parts.push(committed.trace.buffer().as_ptr());
            }
            main_parts.push(air_ctx.common_main.buffer().as_ptr());
            let d_main_parts = main_parts.to_device().unwrap();

            let n_lift = n.max(0) as usize;
            let eq_xi_tree = &self.eq_xis[&n_lift];
            // TODO[jpw] We currently set `save_memory = true` even when we cache the RS-codewords
            // matrix, so the calculation of max_temp_bytes needs to be revisited.
            let max_temp_bytes = if self.save_memory {
                (total_main_cells << self.pk.params.log_blowup) * size_of::<F>()
            } else {
                1usize << 31 // 2gb
            };
            let sum_buffer = evaluate_round0_constraints_gpu(
                single_pk,
                selectors_cube.buffer(),
                &d_main_parts,
                public_values,
                &self.d_omega_skip_pows,
                &d_inv_lagrange_denoms,
                eq_xi_tree.get_ptr(n_lift),
                d_lambda_pows,
                large_domain as u32,
                1 << l_skip,
                1 << n_lift,
                height as u32,
                max_temp_bytes,
            )
            .expect("failed to evaluate round-0 constraints on device");
            if !sum_buffer.is_empty() {
                let host_sums = sum_buffer
                    .to_host()
                    .expect("failed to copy aggregated constraint sums back to host");
                batch_sp_evals[2 * num_present_airs + trace_idx] = host_sums;
            }

            let sum = evaluate_round0_interactions_gpu(
                single_pk,
                &single_air_constraints,
                selectors_cube.buffer(),
                &d_main_parts,
                public_values,
                &self.d_omega_skip_pows,
                &d_inv_lagrange_denoms,
                eq_xi_tree.get_ptr(n_lift),
                &self.beta_pows,
                eq_3bs,
                large_domain as u32,
                1 << l_skip,
                1 << n_lift,
                height as u32,
                max_temp_bytes,
            )
            .expect("failed to evaluate round-0 interactions on device");
            if !sum.is_empty() {
                let evals = sum
                    .to_host()
                    .expect("failed to copy interaction sums back to host");
                let (mut numer, denom): (Vec<EF>, Vec<EF>) =
                    evals.into_iter().map(|frac| (frac.p, frac.q)).unzip();
                if n.is_negative() {
                    // normalize for lifting
                    let norm_factor = F::from_canonical_u32(1 << n.unsigned_abs()).inverse();
                    for s in &mut numer {
                        *s *= norm_factor;
                    }
                }
                batch_sp_evals[2 * trace_idx] = numer;
                batch_sp_evals[2 * trace_idx + 1] = denom;
            }
        }
        self.mem
            .emit_metrics_with_label("prover.batch_constraints.round0");
        info_span!("chirp_z_transform").in_scope(|| {
            batch_sp_evals
                .into_iter()
                .map(|sp_evals| {
                    if sp_evals.is_empty() {
                        return UnivariatePoly::new(vec![]);
                    }
                    let poly = UnivariatePoly::from_evals(&sp_evals);
                    #[cfg(debug_assertions)]
                    if poly.coeffs().len() > global_sp_0_deg + 1 {
                        assert!(
                            poly.coeffs()[global_sp_0_deg + 1..]
                                .iter()
                                .all(|&coeff| coeff == EF::ZERO)
                        );
                    }
                    poly
                })
                .collect()
        })
    }

    // Note: there are no gpu sync points in this function, so span does not indicate kernel times
    #[instrument(name = "LogupZerocheck::fold_ple_evals", level = "debug", skip_all)]
    fn fold_ple_evals(&mut self, ctx: &ProvingContextV2<GpuBackendV2>, r_0: EF) {
        let l_skip = self.l_skip;
        let inv_lagrange_denoms_r0 =
            compute_barycentric_inv_lagrange_denoms(l_skip, &self.omega_skip_pows, r_0);
        let d_inv_lagrange_denoms_r0 = inv_lagrange_denoms_r0.to_device().unwrap();

        // GPU folding for mat_evals_per_trace
        self.mat_evals_per_trace = ctx
            .per_trace
            .iter()
            .map(|(air_idx, air_ctx)| {
                let air_pk = &self.pk.per_air[*air_idx];
                let mut results: Vec<DeviceMatrix<EF>> = Vec::new();

                // Preprocessed (if exists)
                if let Some(committed) = &air_pk.preprocessed_data {
                    let trace = &committed.trace;
                    let folded = fold_ple_evals_rotate(
                        l_skip,
                        &self.d_omega_skip_pows,
                        trace,
                        &d_inv_lagrange_denoms_r0,
                    )
                    .unwrap();
                    results.push(folded);
                }

                // Cached mains
                for committed in &air_ctx.cached_mains {
                    let trace = &committed.trace;
                    let folded = fold_ple_evals_rotate(
                        l_skip,
                        &self.d_omega_skip_pows,
                        trace,
                        &d_inv_lagrange_denoms_r0,
                    )
                    .unwrap();
                    results.push(folded);
                }

                // Common main
                let trace = &air_ctx.common_main;
                let folded = fold_ple_evals_rotate(
                    l_skip,
                    &self.d_omega_skip_pows,
                    trace,
                    &d_inv_lagrange_denoms_r0,
                )
                .unwrap();
                results.push(folded);

                results
            })
            .collect();

        // GPU folding for sels_per_trace (rotate=false, only need offset=0)
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
                let folded_buf = DeviceBuffer::<EF>::with_capacity(num_x * 3);
                unsafe {
                    fold_selectors_round0(
                        folded_buf.as_mut_ptr(),
                        selectors_cube.buffer().as_ptr(),
                        is_first,
                        is_last,
                        num_x,
                    )
                    .unwrap();
                }
                DeviceMatrix::new(Arc::new(folded_buf), num_x, 3)
            })
            .collect();

        // GPU scalar multiplication for eq_xi_per_trace and eq_sharp_per_trace
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

        self.mem
            .emit_metrics_with_label("prover.batch_constraints.fold_ple_evals");
    }

    #[instrument(
        name = "LogupZerocheck::sumcheck_polys_eval",
        level = "debug",
        skip_all,
        fields(round = round)
    )]
    fn sumcheck_polys_eval(&mut self, round: usize, r_prev: EF) -> Vec<Vec<EF>> {
        // sp = s'
        let sp_deg = self.constraint_degree;
        let lambda_pows = self.lambda_pows.as_ref().expect("lambda_pows must be set");

        let mut s_logup_evals = Vec::new();
        let mut s_zerocheck_evals = Vec::new();
        // Process zerocheck and logup together per trace
        izip!(
            self.n_per_trace.iter(),
            self.zerocheck_tilde_evals.iter_mut(),
            self.logup_tilde_evals.iter_mut(),
            self.mat_evals_per_trace.iter(),
            self.sels_per_trace.iter(),
            self.d_eq_3b_per_trace.iter(),
            self.public_values_per_trace.iter(),
            self.air_indices_per_trace.iter()
        )
        .for_each(
            |(&n, zc_tilde_eval, logup_tilde_eval, mats, sels, eq_3bs, public_vals, &air_idx)| {
                let mut results = vec![vec![EF::ZERO; sp_deg]; 3]; // logup numer, logup denom, zerocheck

                let pk = &self.pk.per_air[air_idx];
                let dag = &pk.vk.symbolic_constraints;
                let has_constraints = dag.constraints.num_constraints() > 0;
                let has_interactions = !dag.interactions.is_empty();

                if has_interactions || has_constraints {
                    let n_lift = n.max(0) as usize;
                    let norm_factor_denom = 1 << (-n).max(0);
                    let norm_factor = F::from_canonical_usize(norm_factor_denom).inverse();
                    let has_preprocessed = self.pk.per_air[air_idx].preprocessed_data.is_some();
                    let eq_xi_tree = &self.eq_xis[&n_lift];

                    if round > n_lift {
                        // Case A.1 round == n_lift + 1
                        if round == n_lift + 1 {
                            let prep_ptr = if has_preprocessed {
                                MainMatrixPtrs {
                                    data: mats[0].buffer().as_ptr(),
                                    air_width: mats[0].width() as u32 / 2,
                                }
                            } else {
                                MainMatrixPtrs {
                                    data: std::ptr::null(),
                                    air_width: 0,
                                }
                            };
                            let first_main_idx = usize::from(has_preprocessed);
                            let main_ptrs: Vec<MainMatrixPtrs<EF>> = mats[first_main_idx..]
                                .iter()
                                .map(|m| MainMatrixPtrs {
                                    data: m.buffer().as_ptr(),
                                    air_width: m.width() as u32 / 2,
                                })
                                .collect_vec();
                            let d_main_ptrs = main_ptrs
                                .to_device()
                                .expect("failed to copy main_ptrs to device");
                            let d_zc_evals = if has_constraints {
                                evaluate_mle_constraints_gpu(
                                    eq_xi_tree.get_ptr(0),
                                    sels.buffer().as_ptr(),
                                    prep_ptr,
                                    &d_main_ptrs,
                                    public_vals.as_ptr(),
                                    lambda_pows,
                                    &pk.other_data.zerocheck_mle,
                                    1,
                                    1,
                                )
                            } else {
                                DeviceBuffer::new()
                            };
                            let d_interactions_evals = if has_interactions {
                                Some(evaluate_mle_interactions_gpu(
                                    eq_xi_tree.get_ptr(0),
                                    sels.buffer().as_ptr(),
                                    prep_ptr,
                                    &d_main_ptrs,
                                    public_vals.as_ptr(),
                                    self.d_challenges.as_ptr(),
                                    eq_3bs.as_ptr(),
                                    &pk.other_data.interaction_rules,
                                    1,
                                    1,
                                ))
                            } else {
                                None
                            };
                            if !d_zc_evals.is_empty() {
                                let zc_evals = d_zc_evals
                                    .to_host()
                                    .expect("failed to copy reduction result to host");
                                assert_eq!(zc_evals.len(), 1);
                                // NOTE: eq_r_acc will be multiplied in after the function call
                                *zc_tilde_eval = zc_evals[0];
                                // zc_out not set, will be handled directly from tilde eval in
                                // compute_batch_s
                            }
                            if let Some(d_evals) = d_interactions_evals {
                                let logup_evals =
                                    d_evals.to_host().expect("failed to copy result to host");
                                assert_eq!(logup_evals.len(), 1);
                                logup_tilde_eval[0] = logup_evals[0].p * norm_factor;
                                logup_tilde_eval[1] = logup_evals[0].q;
                                // logup_out not set, will be handled directly from tilde eval in
                                // compute_batch_s
                            }
                        } else {
                            // Case A.2 round > n_lift + 1
                            if has_constraints {
                                *zc_tilde_eval *= r_prev;
                            }
                            if has_interactions {
                                for x in logup_tilde_eval.iter_mut() {
                                    *x *= r_prev;
                                }
                            }
                        }
                    } else {
                        // Case B: round <= n_lift
                        let log_num_y = n_lift - round;
                        let num_y = 1 << log_num_y;
                        let height = 2 * num_y;
                        debug_assert_eq!(height, mats[0].height());
                        let mut columns: Vec<*const EF> = Vec::new();
                        columns.extend(
                            iter::once(sels)
                                .chain(mats.iter())
                                .flat_map(|m| {
                                    assert_eq!(m.height(), height);
                                    (0..m.width()).map(|col| {
                                        m.buffer().as_ptr().wrapping_add(col * m.height())
                                    })
                                })
                                .collect_vec(),
                        );
                        let num_columns = columns.len();
                        let interpolated =
                            DeviceMatrix::<EF>::with_capacity(sp_deg * num_y, num_columns);
                        let d_columns = columns
                            .to_device()
                            .expect("failed to copy column ptrs to device");
                        unsafe {
                            interpolate_columns_gpu(
                                interpolated.buffer(),
                                &d_columns,
                                sp_deg,
                                num_y,
                            )
                            .expect("failed to interpolate columns on GPU");
                        }
                        // interpolated columns layout:
                        // [sels (x3), prep? (x1), main[i] (x1)]

                        // EVALUATION:
                        let interpolated_height = interpolated.height();
                        let mut widths_so_far = 0;
                        let eq_xi_ptr = eq_xi_tree.get_ptr(log_num_y);
                        let sels_ptr = interpolated
                            .buffer()
                            .as_ptr()
                            .wrapping_add(widths_so_far * interpolated_height);
                        widths_so_far += 3;
                        // Reminder: all widths include rotations
                        let prep_ptr = if has_preprocessed {
                            MainMatrixPtrs {
                                data: interpolated
                                    .buffer()
                                    .as_ptr()
                                    .wrapping_add(widths_so_far * interpolated_height),
                                air_width: mats[0].width() as u32 / 2,
                            }
                        } else {
                            MainMatrixPtrs {
                                data: std::ptr::null(),
                                air_width: 0,
                            }
                        };
                        widths_so_far += 2 * prep_ptr.air_width as usize;
                        let first_main_idx = usize::from(has_preprocessed);
                        let main_ptrs: Vec<MainMatrixPtrs<EF>> = mats[first_main_idx..]
                            .iter()
                            .map(|m| {
                                let main_ptr = MainMatrixPtrs {
                                    data: interpolated
                                        .buffer()
                                        .as_ptr()
                                        .wrapping_add(widths_so_far * interpolated_height),
                                    air_width: m.width() as u32 / 2,
                                };
                                widths_so_far += m.width();
                                main_ptr
                            })
                            .collect_vec();
                        debug_assert_eq!(widths_so_far, interpolated.width());
                        let d_main_ptrs = main_ptrs
                            .to_device()
                            .expect("failed to copy main_ptrs to device");
                        let d_constraints_eval = has_constraints.then(|| {
                            evaluate_mle_constraints_gpu(
                                eq_xi_ptr,
                                sels_ptr,
                                prep_ptr,
                                &d_main_ptrs,
                                public_vals.as_ptr(),
                                lambda_pows,
                                &pk.other_data.zerocheck_mle,
                                num_y as u32,
                                sp_deg as u32,
                            )
                        });
                        let d_interactions_eval = has_interactions.then(|| {
                            evaluate_mle_interactions_gpu(
                                eq_xi_ptr,
                                sels_ptr,
                                prep_ptr,
                                &d_main_ptrs,
                                public_vals.as_ptr(),
                                self.d_challenges.as_ptr(),
                                eq_3bs.as_ptr(),
                                &pk.other_data.interaction_rules,
                                num_y as u32,
                                sp_deg as u32,
                            )
                        });
                        if let Some(evals) = d_constraints_eval {
                            results[2] = evals
                                .to_host()
                                .expect("failed to copy reduction result to host");
                        }
                        if let Some(evals) = d_interactions_eval {
                            let logup_evals =
                                evals.to_host().expect("failed to copy result to host");
                            let (numer_evals, denom_evals): (Vec<_>, Vec<_>) =
                                logup_evals.into_iter().map(|f| (f.p, f.q)).unzip();

                            // Apply normalization to numer only (same as CPU)
                            let mut numer_normalized = numer_evals;
                            for p in numer_normalized.iter_mut() {
                                *p *= norm_factor;
                            }
                            results[0] = numer_normalized;
                            results[1] = denom_evals;
                        }
                    }
                }
                s_logup_evals.push(results[0].clone());
                s_logup_evals.push(results[1].clone());
                s_zerocheck_evals.push(results[2].clone());
            },
        );

        s_logup_evals.into_iter().chain(s_zerocheck_evals).collect()
    }

    #[instrument(
        name = "LogupZerocheck::sumcheck_polys_batch_eval",
        level = "info",
        skip_all,
        fields(round = round)
    )]
    fn sumcheck_polys_batch_eval(&mut self, round: usize, r_prev: EF) -> Vec<Vec<EF>> {
        let sp_deg = self.constraint_degree;
        let lambda_pows = self.lambda_pows.as_ref().expect("lambda_pows must be set");

        // Per-trace outputs (filled as we go)
        let mut zc_out: Vec<Vec<EF>> = vec![vec![EF::ZERO; sp_deg]; self.n_per_trace.len()];
        let mut logup_out: Vec<[Vec<EF>; 2]> =
            vec![[vec![EF::ZERO; sp_deg], vec![EF::ZERO; sp_deg]]; self.n_per_trace.len()];

        // Keep early interpolations alive for duration of kernels
        let mut _keepalive_interpolated: Vec<DeviceMatrix<EF>> = Vec::new();

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
            let norm_factor = F::from_canonical_usize(norm_factor_denom).inverse();
            let has_preprocessed = pk.preprocessed_data.is_some();
            let first_main_idx = usize::from(has_preprocessed);
            let eq_xi_tree = &self.eq_xis[&n_lift];

            if round > n_lift {
                // Case A
                if round == n_lift + 1 {
                    // A.1: evaluate directly at (num_x=1, num_y=1)
                    let prep_ptr = if has_preprocessed {
                        MainMatrixPtrs {
                            data: mats[0].buffer().as_ptr(),
                            air_width: mats[0].width() as u32 / 2,
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
                            data: m.buffer().as_ptr(),
                            air_width: m.width() as u32 / 2,
                        })
                        .collect_vec();
                    let main_ptrs_dev = main_ptrs
                        .to_device()
                        .expect("failed to copy main_ptrs to device");

                    late_eval.push(TraceCtx {
                        trace_idx,
                        air_idx,
                        n_lift,
                        num_y: 1,
                        has_constraints,
                        has_interactions,
                        norm_factor,
                        eq_xi_ptr: eq_xi_tree.get_ptr(0),
                        sels_ptr: sels.buffer().as_ptr(),
                        prep_ptr,
                        main_ptrs_dev,
                        public_ptr: public_vals.as_ptr(),
                        eq_3bs_ptr: eq_3bs.as_ptr(),
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

                let mut columns: Vec<*const EF> = Vec::new();
                columns.extend(
                    iter::once(sels)
                        .chain(mats.iter())
                        .flat_map(|m| {
                            assert_eq!(m.height(), height);
                            (0..m.width())
                                .map(|col| m.buffer().as_ptr().wrapping_add(col * m.height()))
                        })
                        .collect_vec(),
                );
                let interpolated = DeviceMatrix::<EF>::with_capacity(sp_deg * num_y, columns.len());
                let d_columns = columns
                    .to_device()
                    .expect("failed to copy column ptrs to device");
                unsafe {
                    interpolate_columns_gpu(interpolated.buffer(), &d_columns, sp_deg, num_y)
                        .expect("failed to interpolate columns on GPU");
                }

                let interpolated_height = interpolated.height();
                let mut widths_so_far = 0usize;
                let sels_ptr = interpolated
                    .buffer()
                    .as_ptr()
                    .wrapping_add(widths_so_far * interpolated_height);
                widths_so_far += 3;
                let prep_ptr = if has_preprocessed {
                    MainMatrixPtrs {
                        data: interpolated
                            .buffer()
                            .as_ptr()
                            .wrapping_add(widths_so_far * interpolated_height),
                        air_width: mats[0].width() as u32 / 2,
                    }
                } else {
                    MainMatrixPtrs {
                        data: std::ptr::null(),
                        air_width: 0,
                    }
                };
                widths_so_far += 2 * prep_ptr.air_width as usize;
                let main_ptrs: Vec<MainMatrixPtrs<EF>> = mats[first_main_idx..]
                    .iter()
                    .map(|m| {
                        let main_ptr = MainMatrixPtrs {
                            data: interpolated
                                .buffer()
                                .as_ptr()
                                .wrapping_add(widths_so_far * interpolated_height),
                            air_width: m.width() as u32 / 2,
                        };
                        widths_so_far += m.width();
                        main_ptr
                    })
                    .collect_vec();
                debug_assert_eq!(widths_so_far, interpolated.width());
                let main_ptrs_dev = main_ptrs
                    .to_device()
                    .expect("failed to copy interpolated main ptrs");

                _keepalive_interpolated.push(interpolated);
                let eq_xi_ptr = eq_xi_tree.get_ptr(log_num_y);

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
                    public_ptr: public_vals.as_ptr(),
                    eq_3bs_ptr: eq_3bs.as_ptr(),
                });
            }
        }

        let d_challenges_ptr = self.d_challenges.as_ptr();

        // Late traces (num_y=1): logup via DAG, zerocheck via monomial
        evaluate_logup_batched(
            &late_eval,
            self.pk,
            d_challenges_ptr,
            1,
            &mut logup_out,
            &mut self.logup_tilde_evals,
            self.memory_limit_bytes,
        );
        // Late traces (num_y=1) always use monomial for zerocheck
        let late_mono_traces: Vec<_> = late_eval
            .iter()
            .filter(|t| trace_has_monomials(t, self.pk))
            .collect();
        if !late_mono_traces.is_empty() {
            let lambda_combs: Vec<_> = late_mono_traces
                .iter()
                .map(|t| self.lambda_combinations[t.air_idx].as_ref().unwrap())
                .collect();
            let batch = ZerocheckMonomialBatch::new(
                late_mono_traces.iter().copied(),
                self.pk,
                &lambda_combs,
            );
            let out = batch.evaluate(1);
            let host = out.to_host().expect("copy monomial output");
            for (i, trace_idx) in batch.trace_indices().enumerate() {
                self.zerocheck_tilde_evals[trace_idx] = host[i];
                // zc_out not set for num_x=1, handled from tilde_eval in compute_batch_s
            }
        }

        // Early traces (num_y>1): partition by threshold for zerocheck path
        let (low_early, high_early): (Vec<&TraceCtx>, Vec<&TraceCtx>) = early_eval
            .iter()
            .partition(|t| t.num_y <= self.monomial_num_y_threshold);

        if low_early.is_empty() {
            // All early traces have high num_y: zerocheck + logup together via DAG
            evaluate_zerocheck_batched(
                &early_eval,
                self.pk,
                lambda_pows,
                sp_deg as u32,
                &mut zc_out,
                &mut self.zerocheck_tilde_evals,
                self.memory_limit_bytes,
            );
            evaluate_logup_batched(
                &early_eval,
                self.pk,
                d_challenges_ptr,
                sp_deg as u32,
                &mut logup_out,
                &mut self.logup_tilde_evals,
                self.memory_limit_bytes,
            );
        } else {
            // Mixed: logup via DAG for all, zerocheck split by threshold
            evaluate_logup_batched(
                &early_eval,
                self.pk,
                d_challenges_ptr,
                sp_deg as u32,
                &mut logup_out,
                &mut self.logup_tilde_evals,
                self.memory_limit_bytes,
            );
            // DAG zerocheck for high num_y traces
            let zc_builder =
                ZerocheckMleBatchBuilder::new(high_early.iter().copied(), self.pk, sp_deg as u32);
            if !zc_builder.is_empty() {
                let out = zc_builder.evaluate(lambda_pows, sp_deg as u32);
                let host = out.to_host().expect("copy zc output");
                for (i, trace_idx) in zc_builder.trace_indices().enumerate() {
                    zc_out[trace_idx].copy_from_slice(&host[(i * sp_deg)..((i + 1) * sp_deg)]);
                }
            }
            // Monomial zerocheck for low num_y traces
            let low_mono_traces: Vec<_> = low_early
                .iter()
                .filter(|t| trace_has_monomials(t, self.pk))
                .copied()
                .collect();
            if !low_mono_traces.is_empty() {
                let lambda_combs: Vec<_> = low_mono_traces
                    .iter()
                    .map(|t| self.lambda_combinations[t.air_idx].as_ref().unwrap())
                    .collect();
                let batch = ZerocheckMonomialBatch::new(
                    low_mono_traces.into_iter(),
                    self.pk,
                    &lambda_combs,
                );
                let out = batch.evaluate(sp_deg as u32);
                let host = out.to_host().expect("copy monomial output");
                for (i, trace_idx) in batch.trace_indices().enumerate() {
                    zc_out[trace_idx].copy_from_slice(&host[(i * sp_deg)..((i + 1) * sp_deg)]);
                }
            }
        }

        // Emit final output in the same order as original
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
            &(0..s_deg).map(F::from_canonical_usize).collect_vec(),
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
        let batch_fold = |input_mats: Vec<DeviceMatrix<EF>>| {
            let num_matrices = input_mats.partition_point(|mat| mat.height() > 1);
            let mut max_output_cells = 0;
            let (log_output_heights, widths, mut output_mats): (Vec<_>, Vec<_>, Vec<_>) =
                input_mats
                    .iter()
                    .take(num_matrices)
                    .map(|mat| {
                        let height = mat.height();
                        let width = mat.width();
                        let output_height = height >> 1;
                        max_output_cells = max(max_output_cells, output_height * width);
                        let output_mat = DeviceMatrix::<EF>::with_capacity(output_height, width);
                        (output_height.ilog2() as u8, width as u32, output_mat)
                    })
                    .multiunzip();

            let input_ptrs = input_mats
                .iter()
                .take(num_matrices)
                .map(|mat| mat.buffer().as_ptr())
                .collect_vec();
            let output_ptrs = output_mats
                .iter()
                .map(|mat| mat.buffer().as_mut_ptr())
                .collect_vec();

            let d_input_ptrs = input_ptrs
                .to_device()
                .expect("failed to copy input ptrs to device");
            let d_output_ptrs = output_ptrs
                .to_device()
                .expect("failed to copy output ptrs to device");
            let d_log_output_heights = log_output_heights
                .to_device()
                .expect("failed to copy heights to device");
            let d_widths = widths.to_device().expect("failed to copy widths to device");

            unsafe {
                batch_fold_mle(
                    &d_input_ptrs,
                    &d_output_ptrs,
                    &d_widths,
                    num_matrices.try_into().unwrap(),
                    &d_log_output_heights,
                    max_output_cells.try_into().unwrap(),
                    r_round,
                )
                .expect("failed to fold MLE selectors on GPU");
            }
            output_mats.extend_from_slice(&input_mats[num_matrices..]);
            output_mats
        };

        // Fold mat_evals_per_trace: Vec<Vec<DeviceMatrix<EF>>>
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

        // Fold sels_per_trace: Vec<DeviceMatrix<EF>>
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
    }

    #[instrument(
        name = "LogupZerocheck::into_column_openings",
        level = "debug",
        skip_all
    )]
    fn into_column_openings(mut self) -> Vec<Vec<Vec<(EF, EF)>>> {
        let num_airs_present = self.mat_evals_per_trace.len();
        let mut column_openings = Vec::with_capacity(num_airs_present);

        // At the end, we've folded all MLEs so they only have one row equal to evaluation at `\vec
        // r`.
        for mat_evals in std::mem::take(&mut self.mat_evals_per_trace) {
            // GPU matrices are doubled-width (original + rotated), so we need to split them
            // First, copy all matrices to host and split them
            let mut split_mats: Vec<ColMajorMatrix<EF>> = mat_evals
                .into_iter()
                .flat_map(|mat| {
                    let mat_host = transport_matrix_d2h_col_major(&mat)
                        .expect("failed to copy GPU matrix to host");
                    let width = mat_host.width();
                    let height = mat_host.height();
                    debug_assert_eq!(height, 1, "Matrices should have height=1 after folding");
                    debug_assert_eq!(
                        width % 2,
                        0,
                        "GPU matrices should have doubled width (original + rotated)"
                    );
                    let air_width = width / 2;

                    // Split doubled-width matrix into original and rotated parts
                    let values = &mat_host.values;
                    let orig: Vec<EF> = (0..air_width)
                        .map(|col| values[col * height]) // height=1, so values[col]
                        .collect();
                    let rot: Vec<EF> = (air_width..width)
                        .map(|col| values[col * height]) // height=1, so values[col]
                        .collect();

                    vec![
                        ColMajorMatrix::new(orig, air_width),
                        ColMajorMatrix::new(rot, air_width),
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
                    std::iter::zip(pair[0].columns(), pair[1].columns())
                        .map(|(claim, claim_rot)| {
                            assert_eq!(claim.len(), 1);
                            assert_eq!(claim_rot.len(), 1);
                            (claim[0], claim_rot[0])
                        })
                        .collect_vec()
                })
                .collect_vec();
            column_openings.push(openings_of_air);
        }
        column_openings
    }
}
