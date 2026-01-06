//! Batch sumcheck for ZeroCheck constraints and sumcheck for LogUp input layer MLEs

use std::cmp::max;

use itertools::Itertools;
use openvm_stark_backend::prover::MatrixDimensions;
use p3_field::{Field, FieldAlgebra};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::{debug, info_span, instrument};

use crate::{
    calculate_n_logup,
    poseidon2::sponge::FiatShamirTranscript,
    proof::{BatchConstraintProof, GkrProof},
    prover::{
        fractional_sumcheck_gkr::{fractional_sumcheck, Frac},
        stacked_pcs::StackedLayout,
        sumcheck::sumcheck_round0_deg,
        CpuBackendV2, DeviceMultiStarkProvingKeyV2, MatrixView, ProvingContextV2,
    },
    EF, F,
};

mod cpu;
mod evaluator;
pub mod fractional_sumcheck_gkr;
mod single;

pub use cpu::LogupZerocheckCpu;
pub use single::*;

#[instrument(level = "info", skip_all)]
pub fn prove_zerocheck_and_logup<'a, TS>(
    transcript: &mut TS,
    mpk: &'a DeviceMultiStarkProvingKeyV2<CpuBackendV2>,
    ctx: &ProvingContextV2<CpuBackendV2>,
) -> (GkrProof, BatchConstraintProof, Vec<EF>)
where
    TS: FiatShamirTranscript,
{
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
    debug!(%n_logup);
    // There's no stride threshold for `interactions_layout` because there's no univariate skip for
    // GKR
    let interactions_layout = StackedLayout::new(0, l_skip + n_logup, interactions_meta);

    // Grind to increase soundness of random sampling for LogUp
    let logup_pow_witness = transcript.grind(mpk.params.logup.pow_bits);
    let alpha_logup = transcript.sample_ext();
    let beta_logup = transcript.sample_ext();
    debug!(%alpha_logup, %beta_logup);

    let mut prover = LogupZerocheckCpu::new(
        mpk,
        ctx,
        n_logup,
        interactions_layout,
        alpha_logup,
        beta_logup,
    );
    // GKR
    // Compute logup input layer: these are the evaluations of \hat{p}, \hat{q} on the hypercube
    // `H_{l_skip + n_logup}`
    let has_interactions = !prover.interactions_layout.sorted_cols.is_empty();
    let gkr_input_evals = if !has_interactions {
        vec![]
    } else {
        // Per trace, a row major matrix of interaction evaluations
        // NOTE: these are the evaluations _without_ lifting
        // PERF[jpw]: we should write directly to the stacked `evals` in memory below
        let unstacked_interaction_evals = prover
            .eval_helpers
            .par_iter()
            .enumerate()
            .map(|(trace_idx, helper)| {
                let trace_ctx = &ctx.per_trace[trace_idx].1;
                let mats = helper.view_mats(trace_ctx);
                let height = trace_ctx.common_main.height();
                (0..height)
                    .into_par_iter()
                    .map(|i| {
                        let mut row_parts = Vec::with_capacity(mats.len() + 1);
                        let is_first = F::from_bool(i == 0);
                        let is_transition = F::from_bool(i != height - 1);
                        let is_last = F::from_bool(i == height - 1);
                        let sels = vec![is_first, is_transition, is_last];
                        row_parts.push(sels);
                        for (mat, is_rot) in &mats {
                            let offset = usize::from(*is_rot);
                            row_parts.push(
                                // SAFETY: %height ensures we never go out of bounds
                                (0..mat.width())
                                    .map(|j| unsafe {
                                        *mat.get_unchecked((i + offset) % height, j)
                                    })
                                    .collect_vec(),
                            );
                        }
                        helper.eval_interactions(&row_parts, &prover.beta_pows)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut evals = vec![Frac::default(); 1 << (l_skip + n_logup)];
        for (trace_idx, interaction_idx, s) in
            prover.interactions_layout.sorted_cols.iter().copied()
        {
            let pq_evals = &unstacked_interaction_evals[trace_idx];
            let height = pq_evals.len();
            debug_assert_eq!(s.col_idx, 0);
            // the interactions layout has internal striding threshold=0
            debug_assert_eq!(1 << s.log_height(), s.len(0));
            debug_assert_eq!(s.len(0) % height, 0);
            let norm_factor_denom = s.len(0) / height;
            let norm_factor = F::from_canonical_usize(norm_factor_denom).inverse();
            // We need to fill `evals` with the logup evaluations on the lifted trace, which is
            // the same as cyclic repeating of the unlifted evaluations
            evals[s.row_idx..s.row_idx + s.len(0)]
                .chunks_exact_mut(height)
                .for_each(|evals| {
                    evals
                        .par_iter_mut()
                        .zip(pq_evals)
                        .for_each(|(pq_eval, evals_at_z)| {
                            let (mut numer, denom) = evals_at_z[interaction_idx];
                            numer *= norm_factor;
                            *pq_eval = Frac::new(numer.into(), denom);
                        });
                });
        }
        // Prevent division by zero:
        evals.par_iter_mut().for_each(|frac| frac.q += alpha_logup);
        evals
    };

    let (frac_sum_proof, mut xi) = fractional_sumcheck(transcript, &gkr_input_evals, true);

    // Sample more for `\xi` in the edge case that some AIRs don't have interactions
    let n_global = max(n_max, n_logup);
    debug!(%n_global);
    while xi.len() != l_skip + n_global {
        xi.push(transcript.sample_ext());
    }
    debug!(?xi);
    prover.xi = xi;
    // we now have full \xi vector

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

    // Sumcheck rounds:
    // - each round the prover needs to compute univariate polynomial `s_round`. This poly is linear
    //   since we are taking MLE of `evals`.
    // - at end of each round, sample random `r_round` in `EF`
    //
    // `s_round` is degree `s_deg` so we evaluate it at `0, ..., =s_deg`. The prover skips
    // evaluation at `0` because the verifier can infer it from the previous round's
    // `s_{round-1}(r)` claim. The degree is constraint_degree + 1, where + 1 is from eq term
    let _mle_rounds_span =
        info_span!("prover.batch_constraints.mle_rounds", phase = "prover").entered();
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
    drop(_mle_rounds_span);
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
