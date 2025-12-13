use std::{
    cmp::max,
    iter::{self},
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
    air_builders::symbolic::SymbolicConstraints, p3_maybe_rayon::prelude::*,
    prover::MatrixDimensions,
};
use p3_field::{Field, FieldAlgebra, TwoAdicField};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use stark_backend_v2::{
    calculate_n_logup,
    poly_common::{
        UnivariatePoly, eval_eq_mle, eval_eq_sharp_uni, eval_eq_uni, eval_eq_uni_at_one,
    },
    poseidon2::sponge::FiatShamirTranscript,
    proof::{BatchConstraintProof, GkrProof},
    prover::{
        ColMajorMatrix, DeviceMultiStarkProvingKeyV2, ProvingContextV2, stacked_pcs::StackedLayout,
        sumcheck::sumcheck_round0_deg,
    },
    utils::batch_multiplicative_inverse_serial,
};
use tracing::{debug, info, info_span, instrument};

use crate::{
    EF, F, GpuBackendV2,
    cuda::{
        logup_zerocheck::{MainMatrixPtrs, fold_selectors_round0, interpolate_columns_gpu},
        sumcheck::{batch_fold_mle, triangular_fold_mle},
    },
    gpu_backend::transport_matrix_d2h_col_major,
    logup_zerocheck::{
        fold_ple::fold_ple_evals_rotate, gkr_input::TraceInteractionMeta,
        round0::evaluate_round0_interactions_gpu,
    },
    poly::EqEvalSegments,
    utils::compute_barycentric_inv_lagrange_denoms,
};

mod errors;
mod fold_ple;
/// Fraction sumcheck via GKR
mod fractional;
/// Logup interaction evaluations for GKR input
mod gkr_input;
mod mle_round;
mod round0;
mod rules;

pub use errors::*;
use fold_ple::compute_eq_sharp_gpu;
use fractional::fractional_sumcheck_gpu;
use gkr_input::{collect_trace_interactions, log_gkr_input_evals};
use mle_round::{evaluate_mle_constraints_gpu, evaluate_mle_interactions_gpu};
use round0::evaluate_round0_constraints_gpu;

#[instrument(level = "info", skip_all)]
pub fn prove_zerocheck_and_logup_gpu<TS>(
    transcript: &mut TS,
    mpk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
    ctx: &ProvingContextV2<GpuBackendV2>,
) -> (GkrProof, BatchConstraintProof, Vec<EF>)
where
    TS: FiatShamirTranscript,
{
    let logup_gkr_span = info_span!("prover.rap_constraints.logup_gkr", phase = "prover").entered();
    let l_skip = mpk.params.l_skip;
    let constraint_degree = mpk.max_constraint_degree;
    let num_airs_present = ctx.per_trace.len();

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
    let logup_pow_witness = transcript.grind(mpk.params.logup.pow_bits);
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
            &prover.beta_pows,
            total_leaves,
        )
        .expect("failed to evaluate interactions on device")
    } else {
        DeviceBuffer::new()
    };
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

    let s_0_polys = prover.sumcheck_uni_round0_polys(ctx, lambda);
    // logup sum claims (sum_{\hat p}, sum_{\hat q}) per present AIR
    let (numerator_term_per_air, denominator_term_per_air): (Vec<_>, Vec<_>) = s_0_polys
        [..2 * num_airs_present]
        .chunks_exact(2)
        .map(|frac| {
            let [sum_claim_p, sum_claim_q] = [&frac[0], &frac[1]].map(|s_0| {
                s_0.coeffs()
                    .iter()
                    .step_by(1 << l_skip)
                    .copied()
                    .sum::<EF>()
                    * F::from_canonical_usize(1 << l_skip)
            });
            transcript.observe_ext(sum_claim_p);
            transcript.observe_ext(sum_claim_q);

            (sum_claim_p, sum_claim_q)
        })
        .unzip();

    let mu = transcript.sample_ext();
    debug!(%mu);

    let s_deg = constraint_degree + 1;
    let s_0_deg = sumcheck_round0_deg(l_skip, s_deg);
    let mu_pows = mu.powers().take(3 * num_airs_present).collect_vec();
    let univariate_round_coeffs = (0..=s_0_deg)
        .map(|i| {
            let coeff = s_0_polys
                .iter()
                .enumerate()
                .map(|(j, s_0)| mu_pows[j] * *s_0.coeffs().get(i).unwrap_or(&EF::ZERO))
                .sum::<EF>();
            transcript.observe_ext(coeff);
            coeff
        })
        .collect_vec();

    let r_0 = transcript.sample_ext();
    r.push(r_0);
    debug!(round = 0, r_round = %r_0);

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
        let s_round_evals = prover.sumcheck_polys_eval(round, r[round - 1]);

        let batch_s_evals = (0..s_deg)
            .map(|i| {
                s_round_evals
                    .iter()
                    .enumerate()
                    .map(|(j, evals)| mu_pows[j] * *evals.get(i).unwrap_or(&EF::ZERO))
                    .sum::<EF>()
            })
            .collect_vec();
        for &eval in &batch_s_evals {
            transcript.observe_ext(eval);
        }
        sumcheck_round_polys.push(batch_s_evals);

        let r_round = transcript.sample_ext();
        debug!(%round, %r_round);
        r.push(r_round);

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
        univariate_round_coeffs,
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
    s_deg: usize,
    // Available after GKR:
    pub xi: Vec<EF>,
    pub lambda_pows: Option<DeviceBuffer<EF>>,

    eq_xis: EqEvalSegments<EF>,
    eq_sharps: EqEvalSegments<EF>,
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

    mem: MemTracker,
}

impl<'a> LogupZerocheckGpu<'a> {
    fn new(
        pk: &'a DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
        ctx: &ProvingContextV2<GpuBackendV2>,
        n_logup: usize,
        interactions_layout: StackedLayout,
        alpha_logup: EF,
        beta_logup: EF,
    ) -> Self {
        let mem = MemTracker::start("prover.logup_zerocheck_prover");
        let l_skip = pk.params.l_skip;
        let omega_skip = F::two_adic_generator(l_skip);
        let omega_skip_pows = omega_skip.powers().take(1 << l_skip).collect_vec();
        let d_omega_skip_pows = omega_skip_pows.to_device().unwrap();
        let num_airs_present = ctx.per_trace.len();

        let constraint_degree = pk.max_constraint_degree;
        let s_deg = constraint_degree + 1;

        let max_interaction_length = pk
            .per_air
            .iter()
            .flat_map(|air_pk| air_pk.vk.symbolic_constraints.interactions.iter())
            .map(|interaction| interaction.message.len())
            .max()
            .unwrap_or(0);
        let beta_pows = beta_logup
            .powers()
            .take(max_interaction_length + 1)
            .collect_vec();

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
            s_deg,
            xi: vec![],
            lambda_pows: None,
            eq_xis: EqEvalSegments::new(&[]).unwrap(),
            eq_sharps: EqEvalSegments::new(&[]).unwrap(),
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
            mem,
        }
    }

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
        let global_s0_deg = sumcheck_round0_deg(l_skip, self.s_deg);
        let num_present_airs = ctx.per_trace.len();
        debug_assert_eq!(num_present_airs, self.n_per_trace.len());

        self.eq_3b_per_trace = self
            .trace_interactions
            .iter()
            .enumerate()
            .map(|(trace_idx, maybe_meta)| {
                if let Some(meta) = maybe_meta {
                    let n = self.n_per_trace[trace_idx];
                    let n_lift = n.max(0) as usize;
                    let mut b_vec = vec![F::ZERO; n_logup - n_lift];
                    let mut weights = Vec::with_capacity(meta.interactions.len());
                    for (interaction_idx, _) in meta.interactions.iter().enumerate() {
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

        self.eq_xis =
            EqEvalSegments::new(&xi[l_skip..]).expect("failed to compute eq_xis on device");

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

        let log_glob_large_domain = log2_ceil_usize(global_s0_deg + 1);
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
        let mut batch_s_evals = vec![None; 3 * num_present_airs];
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
            let local_s_deg = local_constraint_deg + 1;
            assert!(
                local_s_deg <= self.s_deg,
                "Max constraint degree ({local_constraint_deg}) of AIR {air_idx} exceeds the global maximum {}",
                self.s_deg - 1
            );
            let local_s0_deg = sumcheck_round0_deg(l_skip, local_s_deg);

            let log_large_domain = log2_ceil_usize(local_s0_deg + 1);
            // NOTE: we barycentric evaluate, so we don't need the full DFT domain
            let large_domain = local_s0_deg + 1;

            assert!(!xi.is_empty(), "xi vector must not be empty");
            let omega = F::two_adic_generator(log_large_domain);
            let omega_pows = omega.powers().take(large_domain).collect::<Vec<_>>();
            let eq_z_host: Vec<EF> = omega_pows
                .iter()
                .map(|&z| eval_eq_uni(l_skip, xi[0], z.into()))
                .collect();
            let d_eq_z = eq_z_host.to_device().unwrap();
            // Precompute eq_sharp_z (using eq_sharp instead of eq_z)
            let eq_sharp_z_host: Vec<EF> = omega_pows
                .iter()
                .map(|&z| eval_eq_sharp_uni(&self.omega_skip_pows, &xi[..l_skip], z.into()))
                .collect();
            let d_eq_sharp_z = eq_sharp_z_host.to_device().unwrap();
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
            let sum_buffer = evaluate_round0_constraints_gpu(
                single_pk,
                selectors_cube.buffer(),
                &d_main_parts,
                public_values,
                &self.d_omega_skip_pows,
                &d_inv_lagrange_denoms,
                &d_eq_z,
                self.eq_xis.get_ptr(n_lift),
                d_lambda_pows,
                large_domain as u32,
                1 << l_skip,
                1 << n_lift,
                height as u32,
            )
            .expect("failed to evaluate round-0 constraints on device");
            if !sum_buffer.is_empty() {
                let host_sums = sum_buffer
                    .to_host()
                    .expect("failed to copy aggregated constraint sums back to host");
                batch_s_evals[2 * num_present_airs + trace_idx] = Some(host_sums);
            }

            let sum = evaluate_round0_interactions_gpu(
                single_pk,
                &single_air_constraints,
                selectors_cube.buffer(),
                &d_main_parts,
                public_values,
                &self.d_omega_skip_pows,
                &d_inv_lagrange_denoms,
                &d_eq_sharp_z,
                self.eq_xis.get_ptr(n_lift),
                &self.beta_pows,
                eq_3bs,
                large_domain as u32,
                1 << l_skip,
                1 << n_lift,
                height as u32,
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
                batch_s_evals[2 * trace_idx] = Some(numer);
                batch_s_evals[2 * trace_idx + 1] = Some(denom);
            }
        }
        self.mem
            .emit_metrics_with_label("prover.batch_constraints.round0");
        info_span!("chirp_z_transform").in_scope(|| {
            batch_s_evals
                .into_par_iter()
                .map(|s_evals| {
                    if let Some(s_evals) = s_evals {
                        let mut poly = UnivariatePoly::from_evals(&s_evals);
                        #[cfg(debug_assertions)]
                        if poly.coeffs().len() > global_s0_deg + 1 {
                            assert!(
                                poly.coeffs()[global_s0_deg + 1..]
                                    .iter()
                                    .all(|&coeff| coeff == EF::ZERO)
                            );
                        }
                        poly.coeffs_mut().resize(global_s0_deg + 1, EF::ZERO);
                        poly
                    } else {
                        UnivariatePoly::new(vec![EF::ZERO; global_s0_deg + 1])
                    }
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

        // Mutate eq_xi_per_trace in-place and compute eq_sharp_per_trace in a single kernel call
        // This matches CPU behavior: eq_xi *= eq_r0, eq_sharp = original_eq_xi * eq_sharp_r0
        self.eq_sharps = unsafe {
            EqEvalSegments::from_raw_parts(
                compute_eq_sharp_gpu(&mut self.eq_xis.buffer, eq_r0, eq_sharp_r0)
                    .expect("failed to multiply eq_xi and compute eq_sharp on GPU"),
                self.xi.len() - l_skip,
            )
        };
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
        let s_deg = self.s_deg;
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
                let mut results = vec![vec![EF::ZERO; s_deg]; 3]; // logup numer, logup denom, zerocheck

                let constraints_dag = &self.pk.per_air[air_idx].vk.symbolic_constraints;
                let constraints = SymbolicConstraints::from(constraints_dag);
                let has_constraints = !constraints.constraints.is_empty();
                let has_interactions = !constraints.interactions.is_empty();

                if has_interactions || has_constraints {
                    let n_lift = n.max(0) as usize;
                    let norm_factor_denom = 1 << (-n).max(0);
                    let norm_factor = F::from_canonical_usize(norm_factor_denom).inverse();
                    let has_preprocessed = self.pk.per_air[air_idx].preprocessed_data.is_some();

                    if round > n_lift {
                        // Case A.1 round == n_lift + 1
                        if round == n_lift + 1 {
                            let (prep_ptr, first_main_idx) = if has_preprocessed {
                                (
                                    MainMatrixPtrs {
                                        data: mats[0].buffer().as_ptr(),
                                        air_width: mats[0].width() as u32 / 2,
                                    },
                                    1,
                                )
                            } else {
                                (
                                    MainMatrixPtrs {
                                        data: std::ptr::null(),
                                        air_width: 0,
                                    },
                                    0,
                                )
                            };
                            let main_ptrs: Vec<MainMatrixPtrs<EF>> = mats[first_main_idx..]
                                .iter()
                                .map(|m| MainMatrixPtrs {
                                    data: m.buffer().as_ptr(),
                                    air_width: m.width() as u32 / 2,
                                })
                                .collect_vec();
                            let d_zc_evals = if has_constraints {
                                evaluate_mle_constraints_gpu(
                                    self.eq_xis.get_ptr(0),
                                    sels.buffer().as_ptr(),
                                    prep_ptr,
                                    &main_ptrs,
                                    public_vals,
                                    lambda_pows,
                                    constraints_dag,
                                    1,
                                    1,
                                )
                            } else {
                                DeviceBuffer::new()
                            };
                            let d_interactions_evals = if has_interactions {
                                Some(evaluate_mle_interactions_gpu(
                                    self.eq_sharps.get_ptr(0),
                                    sels.buffer().as_ptr(),
                                    prep_ptr,
                                    &main_ptrs,
                                    public_vals,
                                    &self.beta_pows,
                                    eq_3bs,
                                    constraints_dag,
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
                                *zc_tilde_eval = zc_evals[0];
                            }
                            if let Some([numer_evals, denom_evals]) = d_interactions_evals {
                                let logup_numer_evals = numer_evals
                                    .to_host()
                                    .expect("failed to copy numer result to host");
                                let logup_denom_evals = denom_evals
                                    .to_host()
                                    .expect("failed to copy denom result to host");
                                assert_eq!(logup_numer_evals.len(), 1);
                                assert_eq!(logup_denom_evals.len(), 1);
                                logup_tilde_eval[0] = logup_numer_evals[0] * norm_factor;
                                logup_tilde_eval[1] = logup_denom_evals[0];
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
                        if has_constraints {
                            results[2] = (1..=s_deg)
                                .map(|x| *zc_tilde_eval * F::from_canonical_usize(x))
                                .collect();
                        }
                        if has_interactions {
                            let [numer, denom] = logup_tilde_eval.map(|tilde_eval| {
                                (1..=s_deg)
                                    .map(|x| tilde_eval * F::from_canonical_usize(x))
                                    .collect()
                            });
                            results[0] = numer;
                            results[1] = denom;
                        }
                    } else {
                        // Case B: round <= n_lift
                        let n = n_lift.saturating_sub(round - 1);
                        let height = 1 << n;
                        let mut columns: Vec<usize> = Vec::new();
                        if has_constraints {
                            columns.push(self.eq_xis.get_ptr(n) as usize);
                        }
                        if has_interactions {
                            columns.push(self.eq_sharps.get_ptr(n) as usize);
                        }
                        columns.extend(
                            iter::once(sels)
                                .chain(mats.iter())
                                .flat_map(|m| {
                                    assert_eq!(m.height(), height);
                                    (0..m.width()).map(|col| {
                                        m.buffer().as_ptr().wrapping_add(col * m.height()) as usize
                                    })
                                })
                                .collect_vec(),
                        );
                        let num_columns = columns.len();
                        let num_y = height / 2;
                        let interpolated =
                            DeviceMatrix::<EF>::with_capacity(s_deg * num_y, num_columns);
                        let d_columns = columns
                            .to_device()
                            .expect("failed to copy column ptrs to device");
                        unsafe {
                            interpolate_columns_gpu(
                                interpolated.buffer(),
                                &d_columns,
                                s_deg,
                                num_y,
                            )
                            .expect("failed to interpolate columns on GPU");
                        }
                        // interpolated columns layout:
                        // [eq_xi (x1), eq_sharp (x1), sels (x3), prep? (x1), main[i] (x1)]

                        // EVALUATION:
                        let interpolated_height = interpolated.height();
                        let mats_widths: Vec<usize> = mats.iter().map(|m| m.width() / 2).collect();
                        let mut widths_so_far = 0;
                        let eq_xi_ptr = if has_constraints {
                            let ptr = interpolated
                                .buffer()
                                .as_ptr()
                                .wrapping_add(widths_so_far * interpolated_height);
                            widths_so_far += 1;
                            ptr
                        } else {
                            std::ptr::null()
                        };
                        let eq_sharp_ptr = if has_interactions {
                            let ptr = interpolated
                                .buffer()
                                .as_ptr()
                                .wrapping_add(widths_so_far * interpolated_height);
                            widths_so_far += 1;
                            ptr
                        } else {
                            std::ptr::null()
                        };
                        let sels_ptr = interpolated
                            .buffer()
                            .as_ptr()
                            .wrapping_add(widths_so_far * interpolated_height);
                        widths_so_far += 3;
                        let (prep_ptr, first_main_idx) = if has_preprocessed {
                            (
                                MainMatrixPtrs {
                                    data: interpolated
                                        .buffer()
                                        .as_ptr()
                                        .wrapping_add(widths_so_far * interpolated_height),
                                    air_width: mats_widths[0] as u32,
                                },
                                1,
                            )
                        } else {
                            (
                                MainMatrixPtrs {
                                    data: std::ptr::null(),
                                    air_width: 0,
                                },
                                0,
                            )
                        };
                        widths_so_far += 2 * prep_ptr.air_width as usize;
                        let main_ptrs: Vec<MainMatrixPtrs<EF>> = mats_widths[first_main_idx..]
                            .iter()
                            .map(|m_width| {
                                let main_ptr = MainMatrixPtrs {
                                    data: interpolated
                                        .buffer()
                                        .as_ptr()
                                        .wrapping_add(widths_so_far * interpolated_height),
                                    air_width: *m_width as u32,
                                };
                                widths_so_far += m_width * 2;
                                main_ptr
                            })
                            .collect_vec();
                        debug_assert_eq!(widths_so_far, interpolated.width());
                        let d_constraints_eval = if has_constraints {
                            evaluate_mle_constraints_gpu(
                                eq_xi_ptr,
                                sels_ptr,
                                prep_ptr,
                                &main_ptrs,
                                public_vals,
                                lambda_pows,
                                constraints_dag,
                                interpolated_height,
                                s_deg,
                            )
                        } else {
                            DeviceBuffer::new()
                        };
                        let d_interactions_eval = if has_interactions {
                            Some(evaluate_mle_interactions_gpu(
                                eq_sharp_ptr,
                                sels_ptr,
                                prep_ptr,
                                &main_ptrs,
                                public_vals,
                                &self.beta_pows,
                                eq_3bs,
                                constraints_dag,
                                interpolated_height,
                                s_deg,
                            ))
                        } else {
                            None
                        };
                        if !d_constraints_eval.is_empty() {
                            results[2] = d_constraints_eval
                                .to_host()
                                .expect("failed to copy reduction result to host");
                        }
                        if let Some([numer_evals, denom_evals]) = d_interactions_eval {
                            let numer_evals = numer_evals
                                .to_host()
                                .expect("failed to copy numer result to host");
                            let denom_evals = denom_evals
                                .to_host()
                                .expect("failed to copy denom result to host");

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

    #[instrument(name = "LogupZerocheck::fold_mle_evals", level = "debug", skip_all, fields(round = _round))]
    fn fold_mle_evals(&mut self, _round: usize, r_round: EF) {
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

        let n = self.eq_xis.buffer.len().ilog2() as usize;
        if n >= 2 {
            let n = n - 2;
            {
                let tmp_buf = DeviceBuffer::with_capacity(2 << n);
                let mut tmp_segs = unsafe { EqEvalSegments::from_raw_parts(tmp_buf, n) };
                unsafe {
                    triangular_fold_mle(&mut tmp_segs, &self.eq_xis, r_round, n)
                        .expect("failed to fold MLE eq_xi on GPU");
                }
                self.eq_xis = tmp_segs;
            }
            {
                let tmp_buf = DeviceBuffer::with_capacity(2 << n);
                let mut tmp_segs = unsafe { EqEvalSegments::from_raw_parts(tmp_buf, n) };
                unsafe {
                    triangular_fold_mle(&mut tmp_segs, &self.eq_sharps, r_round, n)
                        .expect("failed to fold MLE eq_sharp on GPU");
                }
                self.eq_sharps = tmp_segs;
            }
        }
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
