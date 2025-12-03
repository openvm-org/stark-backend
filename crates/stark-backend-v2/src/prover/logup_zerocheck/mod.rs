//! Batch sumcheck for ZeroCheck constraints and sumcheck for LogUp input layer MLEs

use itertools::Itertools;
use openvm_stark_backend::prover::MatrixDimensions;
use p3_field::FieldAlgebra;
use p3_util::log2_strict_usize;
use tracing::{debug, info_span, instrument};

use crate::{
    calculate_n_logup,
    poly_common::UnivariatePoly,
    poseidon2::sponge::FiatShamirTranscript,
    proof::{BatchConstraintProof, GkrProof},
    prover::{
        fractional_sumcheck_gkr::FracSumcheckProof, logup_zerocheck::single::EvalHelper,
        stacked_pcs::StackedLayout, sumcheck::sumcheck_round0_deg, DeviceMultiStarkProvingKeyV2,
        ProverBackendV2, ProvingContextV2,
    },
    EF, F,
};

mod cpu;
mod evaluator;
pub mod fractional_sumcheck_gkr;
mod single;

pub use cpu::LogupZerocheckCpu;

/// Helper trait for implementing [MultiRapProver] by performing Logup GKR to reduce interaction bus
/// balancing to an _input layer sumcheck_ which may be viewed as a stacking reduction from the GKR
/// leaf input layer to column evaluations of the trace. The input layer sumcheck is then batched
/// together with ZeroCheck in one large _batch constraints sumcheck_.
///
/// This trait is intended to be implemented on a stateful struct that holds state between the
/// stages of proving. The constructor is given by [`LogupZerocheckProver::prove_logup_gkr`] and the
/// trait is generic in `PD` which represents the `ProverDevice`.
pub trait LogupZerocheckProver<'a, PB: ProverBackendV2, PD, TS>: Sized {
    /// From trace matrices, evaluates the symbolic interactions to get the GKR input layer
    /// evaluations. These are stacked into a single matrix of `(\hat{p}(x), \hat{q}(x))` pairs. It
    /// is recommended to store the evaluations as part of a segment tree to aid in the GKR layer
    /// sum computation, although memory saving techniques may take precedence.
    ///
    /// This function both proves the LogUp GKR, without the input layer sumcheck, and provides the
    /// constructor for subsequent steps.
    ///
    /// Returns `self`, fractional sumcheck GKR proof.
    #[allow(clippy::too_many_arguments)]
    fn prove_logup_gkr(
        device: &'a PD,
        transcript: &mut TS,
        pk: &'a DeviceMultiStarkProvingKeyV2<PB>,
        ctx: &ProvingContextV2<PB>,
        common_main_pcs_data: &'a PB::PcsData,
        n_logup: usize,
        interactions_layout: StackedLayout,
        alpha_logup: PB::Challenge,
        beta_logup: PB::Challenge,
    ) -> (Self, FracSumcheckProof<PB::Challenge>);

    /// Returns the `s_0` polynomials in coefficient form. There should be exactly `num_airs_present
    /// \* 3` polynomials, in the order `(s_0)_{p,T}, (s_0)_{q,T}, (s_0)_{zerocheck,T}` per trace
    /// `T`. This is computed _before_ sampling batching randomness `mu` because the result is
    /// used to observe the sum claims `sum_{p,T}, sum_{q,T}`. The `s_0` polynomials could be
    /// returned in either coefficient or evaluation form, but we return them all in coefficient
    /// form for uniformity and debugging since this interpolation is inexpensive.
    fn sumcheck_uni_round0_polys(
        &mut self,
        ctx: &ProvingContextV2<PB>,
        lambda: PB::Challenge,
    ) -> Vec<UnivariatePoly<PB::Challenge>>;

    /// After univariate sumcheck round 0, fold prismalinear evaluations using randomness `r_0`.
    /// Folding _could_ directly mutate inplace the trace matrices in `ctx` as they will not be
    /// needed after this.
    fn fold_ple_evals(&mut self, ctx: &ProvingContextV2<PB>, r_0: PB::Challenge);

    /// Returns length `3 * num_airs_present` polynomials, each evaluated at `1..=s_deg`.
    fn sumcheck_polys_eval(
        &mut self,
        round: usize,
        r_prev: PB::Challenge,
    ) -> Vec<Vec<PB::Challenge>>;

    fn fold_mle_evals(&mut self, round: usize, r_round: PB::Challenge);

    #[allow(clippy::type_complexity)]
    fn into_column_openings(self) -> Vec<Vec<Vec<(PB::Challenge, PB::Challenge)>>>;
}

#[instrument(level = "info", skip_all)]
pub fn prove_zerocheck_and_logup<'a, PB, PD, TS, LZP>(
    device: &'a PD,
    transcript: &mut TS,
    mpk: &'a DeviceMultiStarkProvingKeyV2<PB>,
    ctx: &ProvingContextV2<PB>,
    common_main_pcs_data: &'a PB::PcsData,
) -> (GkrProof, BatchConstraintProof, Vec<PB::Challenge>)
where
    PB: ProverBackendV2<Val = F, Challenge = EF>,
    TS: FiatShamirTranscript,
    LZP: LogupZerocheckProver<'a, PB, PD, TS>,
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

    let (mut prover, frac_sum_proof) = LZP::prove_logup_gkr(
        device,
        transcript,
        mpk,
        ctx,
        common_main_pcs_data,
        n_logup,
        interactions_layout,
        alpha_logup,
        beta_logup,
    );

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
