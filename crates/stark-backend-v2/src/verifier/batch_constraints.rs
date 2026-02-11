use std::{
    iter::{self, zip},
    slice,
};

use itertools::Itertools;
use openvm_stark_backend::air_builders::symbolic::{
    symbolic_expression::SymbolicEvaluator, SymbolicConstraints,
};
use p3_field::{batch_multiplicative_inverse, Field, PrimeCharacteristicRing};
use thiserror::Error;
use tracing::{debug, instrument};

use crate::{
    calculate_n_logup,
    keygen::types::MultiStarkVerifyingKey0V2,
    poly_common::{eval_eq_mle, eval_eq_sharp_uni, eval_eq_uni, UnivariatePoly},
    poseidon2::sponge::FiatShamirTranscript,
    proof::{column_openings_by_rot, BatchConstraintProof, GkrProof},
    verifier::{
        evaluator::VerifierConstraintEvaluator,
        fractional_sumcheck_gkr::{verify_gkr, GkrVerificationError},
    },
    EF, F,
};

#[derive(Error, Debug, PartialEq, Eq)]
pub enum BatchConstraintError {
    #[error("Invalid logup_pow_witness")]
    InvalidLogupPowWitness,

    #[error("GKR verification failed: {0}")]
    GkrVerificationFailed(#[from] GkrVerificationError),

    #[error("GKR numerator evaluation claim {claim} does not match")]
    GkrNumeratorMismatch { claim: EF },

    #[error("GKR denominator evaluation claim {claim} does not match")]
    GkrDenominatorMismatch { claim: EF },

    #[error(
        "`sum_claim` does not equal the sum of `s_0` at all the roots of unity: {sum_claim} != {sum_univ_domain_s_0}"
    )]
    SumClaimMismatch {
        sum_claim: EF,
        sum_univ_domain_s_0: EF,
    },

    #[error("Claims are inconsistent")]
    InconsistentClaims,
}

/// `public_values` should be in vkey (air_idx) order, including non-present AIRs.
#[allow(clippy::too_many_arguments)]
#[instrument(level = "debug", skip_all)]
pub fn verify_zerocheck_and_logup<TS: FiatShamirTranscript>(
    transcript: &mut TS,
    mvk: &MultiStarkVerifyingKey0V2,
    public_values: &[Vec<F>],
    gkr_proof: &GkrProof,
    batch_proof: &BatchConstraintProof,
    trace_id_to_air_id: &[usize],
    n_per_trace: &[isize],
    omega_skip_pows: &[F],
) -> Result<Vec<EF>, BatchConstraintError> {
    let l_skip = mvk.params.l_skip;
    // let num_airs_present = mvk.per_air.len();
    let BatchConstraintProof {
        numerator_term_per_air,
        denominator_term_per_air,
        univariate_round_coeffs,
        sumcheck_round_polys,
        column_openings,
    } = batch_proof;

    // 1. Check GKR witness
    if !transcript.check_witness(mvk.params.logup.pow_bits, gkr_proof.logup_pow_witness) {
        return Err(BatchConstraintError::InvalidLogupPowWitness);
    }

    // 2. Sample alpha and beta, receive xi, sample lambda
    let alpha_logup = transcript.sample_ext();
    let beta_logup = transcript.sample_ext();
    debug!(%alpha_logup, %beta_logup);
    let total_interactions = zip(trace_id_to_air_id, n_per_trace)
        .map(|(&air_idx, &n)| {
            let n_lift = n.max(0) as usize;
            let num_interactions = mvk.per_air[air_idx].symbolic_constraints.interactions.len();
            (num_interactions as u64) << (l_skip + n_lift)
        })
        .sum::<u64>();
    let n_logup: usize = calculate_n_logup(l_skip, total_interactions);
    debug!(%n_logup);

    let total_gkr_rounds = if total_interactions > 0 {
        l_skip + n_logup
    } else {
        0
    };
    let mut xi = Vec::new();
    let mut p_xi_claim = EF::ZERO;
    let mut q_xi_claim = alpha_logup;
    if total_interactions > 0 {
        (p_xi_claim, q_xi_claim, xi) =
            verify_gkr(gkr_proof, transcript, total_gkr_rounds, mvk.params.n_logup_grid)?;
        debug_assert_eq!(xi.len(), l_skip + n_logup);
    }

    let n_max = n_per_trace.iter().copied().max().unwrap().max(0) as usize;
    let n_global = n_max.max(n_logup);
    while xi.len() != l_skip + n_global {
        xi.push(transcript.sample_ext());
    }
    debug!(%n_max);
    debug!(?xi);

    let lambda = transcript.sample_ext();
    debug!(%lambda);

    // 3. Observe everything from numerator_per_air and denominator_per_air, compute its sum
    for (&sum_claim_p, &sum_claim_q) in zip(numerator_term_per_air, denominator_term_per_air) {
        p_xi_claim -= sum_claim_p;
        q_xi_claim -= sum_claim_q;
        transcript.observe_ext(sum_claim_p);
        transcript.observe_ext(sum_claim_q);
    }
    if p_xi_claim != EF::ZERO {
        return Err(BatchConstraintError::GkrNumeratorMismatch { claim: p_xi_claim });
    }
    if q_xi_claim != alpha_logup {
        return Err(BatchConstraintError::GkrDenominatorMismatch { claim: q_xi_claim });
    }

    // 4. Sample mu, compute the mu-hash of interleave of numerator_per_air and denominator_per_air
    let mu = transcript.sample_ext();
    debug!(%mu);

    let mut sum_claim = EF::ZERO;
    let mut cur_mu_pow = EF::ONE;
    for (&sum_claim_p, &sum_claim_q) in zip(numerator_term_per_air, denominator_term_per_air) {
        sum_claim += sum_claim_p * cur_mu_pow;
        cur_mu_pow *= mu;
        sum_claim += sum_claim_q * cur_mu_pow;
        cur_mu_pow *= mu;
    }

    // 5. Univariate sumcheck round
    for &coeff in univariate_round_coeffs {
        transcript.observe_ext(coeff);
    }

    let s_deg = mvk.params.max_constraint_degree + 1;
    let r_0 = transcript.sample_ext();
    debug!(round = 0, r_round = %r_0);
    assert_eq!(
        univariate_round_coeffs.len(),
        (mvk.max_constraint_degree() + 1) * ((1 << l_skip) - 1) + 1
    );
    let s_0 = UnivariatePoly::new(univariate_round_coeffs.clone());
    let sum_univ_domain_s_0 = s_0
        .coeffs()
        .iter()
        .step_by(1 << l_skip)
        .copied()
        .sum::<EF>()
        * EF::from_usize(1 << l_skip);
    if sum_claim != sum_univ_domain_s_0 {
        return Err(BatchConstraintError::SumClaimMismatch {
            sum_claim,
            sum_univ_domain_s_0,
        });
    }
    let mut cur_sum = s_0.eval_at_point(r_0);
    let mut rs = vec![r_0];

    // 6. Multilinear sumcheck rounds
    #[allow(clippy::needless_range_loop)]
    for round in 0..n_max {
        debug!(sumcheck_round = round, sum_claim = %cur_sum, "batch_constraint_sumcheck");
        let batch_s_evals = &sumcheck_round_polys[round];
        for &eval in batch_s_evals.iter() {
            transcript.observe_ext(eval);
        }
        let s_1 = batch_s_evals[0];
        let s_0 = cur_sum - s_1;
        let batch_s_evals = iter::once(&s_0).chain(batch_s_evals).collect_vec();

        let mut factorials = vec![F::ONE; s_deg + 1];
        for i in 1..=s_deg {
            factorials[i] = factorials[i - 1] * F::from_usize(i);
        }
        let invfact = batch_multiplicative_inverse(&factorials);

        let r = transcript.sample_ext();
        let mut pref_product = vec![EF::ONE; s_deg + 1];
        let mut suf_product = vec![EF::ONE; s_deg + 1];
        for i in 0..s_deg {
            pref_product[i + 1] = pref_product[i] * (r - EF::from_usize(i));
            suf_product[i + 1] = suf_product[i] * (EF::from_usize(s_deg - i) - r);
        }
        cur_sum = (0..=s_deg)
            .map(|i| {
                *batch_s_evals[i]
                    * pref_product[i]
                    * suf_product[s_deg - i]
                    * invfact[i]
                    * invfact[s_deg - i]
            })
            .sum::<EF>();

        debug!(round = round + 1, r_round = %r);
        rs.push(r);
    }

    // 7. Compute `eq_3b_per_trace`
    let mut stacked_idx = 0usize;
    let eq_3b_per_trace = n_per_trace
        .iter()
        .enumerate()
        .map(|(trace_idx, &n)| {
            let air_idx = trace_id_to_air_id[trace_idx];
            let interactions = &mvk.per_air[air_idx].symbolic_constraints.interactions;
            if interactions.is_empty() {
                return vec![];
            }
            let n_lift = n.max(0) as usize;
            let mut b_vec = vec![F::ZERO; n_logup - n_lift];
            (0..interactions.len())
                .map(|_| {
                    debug_assert!(stacked_idx < 1 << (l_skip + n_logup));
                    debug_assert!(stacked_idx.trailing_zeros() as usize >= l_skip + n_lift);
                    let mut b_int = stacked_idx >> (l_skip + n_lift);
                    for b in &mut b_vec {
                        *b = F::from_bool(b_int & 1 == 1);
                        b_int >>= 1;
                    }
                    stacked_idx += 1 << (l_skip + n_lift);
                    eval_eq_mle(&xi[l_skip + n_lift..l_skip + n_logup], &b_vec)
                })
                .collect_vec()
        })
        .collect_vec();

    // 8. Compute `eq_ns` and `eq_sharp_ns`
    let mut eq_ns = vec![EF::ONE; n_max + 1];
    let mut eq_sharp_ns = vec![EF::ONE; n_max + 1];
    eq_ns[0] = eval_eq_uni(l_skip, xi[0], r_0);
    eq_sharp_ns[0] = eval_eq_sharp_uni(omega_skip_pows, &xi[..l_skip], r_0);
    debug_assert_eq!(rs.len(), n_max + 1);
    for (i, r) in rs.iter().enumerate().skip(1) {
        let eq_mle = eval_eq_mle(&[xi[l_skip + i - 1]], slice::from_ref(r));
        eq_ns[i] = eq_ns[i - 1] * eq_mle;
        eq_sharp_ns[i] = eq_sharp_ns[i - 1] * eq_mle;
    }
    let mut r_rev_prod = rs[n_max];
    // Product with r_i's to account for \hat{f} vs \tilde{f} for different n's in front-loaded
    // batch sumcheck.
    for i in (0..n_max).rev() {
        eq_ns[i] *= r_rev_prod;
        eq_sharp_ns[i] *= r_rev_prod;
        r_rev_prod *= rs[i];
    }

    // 9. Compute the interaction/constraint evals and their hash
    let mut interactions_evals = Vec::new();
    let mut constraints_evals = Vec::new();
    let need_rot_per_trace = trace_id_to_air_id
        .iter()
        .map(|&air_idx| mvk.per_air[air_idx].params.need_rot)
        .collect_vec();

    // Observe common main openings first, and then preprocessed/cached
    for (trace_idx, air_openings) in column_openings.iter().enumerate() {
        let need_rot = need_rot_per_trace[trace_idx];
        for (claim, claim_rot) in column_openings_by_rot(&air_openings[0], need_rot) {
            transcript.observe_ext(claim);
            transcript.observe_ext(claim_rot);
        }
    }

    for (trace_idx, air_openings) in column_openings.iter().enumerate() {
        let air_idx = trace_id_to_air_id[trace_idx];
        let vk = &mvk.per_air[air_idx];
        let n = n_per_trace[trace_idx];
        let n_lift = n.max(0) as usize;
        let need_rot = need_rot_per_trace[trace_idx];

        // claim lengths are checked in proof shape
        for claims in air_openings.iter().skip(1) {
            for (claim, claim_rot) in column_openings_by_rot(claims, need_rot) {
                transcript.observe_ext(claim);
                transcript.observe_ext(claim_rot);
            }
        }

        let has_preprocessed = vk.preprocessed_data.is_some();
        let common_main = column_openings_by_rot(&air_openings[0], need_rot).collect::<Vec<_>>();
        let preprocessed = has_preprocessed
            .then(|| column_openings_by_rot(&air_openings[1], need_rot).collect::<Vec<_>>());
        let cached_idx = 1 + has_preprocessed as usize;
        let mut partitioned_main: Vec<_> = air_openings[cached_idx..]
            .iter()
            .map(|opening| column_openings_by_rot(opening, need_rot).collect::<Vec<_>>())
            .collect();
        partitioned_main.push(common_main);
        let part_main_slices = partitioned_main
            .iter()
            .map(|x| x.as_slice())
            .collect::<Vec<_>>();

        // We are evaluating the lift, which is the same as evaluating the original with domain
        // D^{(2^{n})}
        let (l, rs_n, norm_factor) = if n.is_negative() {
            (
                l_skip.wrapping_add_signed(n),
                &[rs[0].exp_power_of_2(-n as usize)] as &[_],
                F::from_usize(1 << n.unsigned_abs()).inverse(),
            )
        } else {
            (l_skip, &rs[..=(n as usize)], F::ONE)
        };
        let evaluator = VerifierConstraintEvaluator::<F, EF>::new(
            preprocessed.as_deref(),
            &part_main_slices,
            &public_values[air_idx],
            rs_n,
            l,
        );

        let constraints = &vk.symbolic_constraints.constraints;
        let nodes = evaluator.eval_nodes(&constraints.nodes);
        let expr = zip(lambda.powers(), &constraints.constraint_idx)
            .map(|(lambda_pow, idx)| nodes[*idx] * lambda_pow)
            .sum::<EF>();
        debug!(%trace_idx, %expr, %air_idx, "constraints_eval");
        let eq_xi_r = eq_ns[n_lift];
        debug!(%trace_idx, %eq_xi_r);
        constraints_evals.push(eq_xi_r * expr);

        let symbolic_constraints = SymbolicConstraints::from(&vk.symbolic_constraints);
        let interactions = &symbolic_constraints.interactions;
        let cur_interactions_evals = interactions
            .iter()
            .map(|interaction| {
                let num = evaluator.eval_expr(&interaction.count);
                let denom = interaction
                    .message
                    .iter()
                    .map(|expr| evaluator.eval_expr(expr))
                    .chain(std::iter::once(EF::from_u16(interaction.bus_index + 1)))
                    .zip(beta_logup.powers())
                    .fold(EF::ZERO, |acc, (x, y)| acc + x * y);
                (num, denom)
            })
            .collect_vec();
        let eq_3bs = &eq_3b_per_trace[trace_idx];
        let mut num = EF::ZERO;
        let mut denom = EF::ZERO;
        for (&eq_3b, (n, d)) in eq_3bs.iter().zip_eq(cur_interactions_evals.iter()) {
            num += eq_3b * *n;
            denom += eq_3b * *d;
        }
        debug!(%trace_idx, %num, %denom, %air_idx, "interactions_eval");
        interactions_evals.push(num * norm_factor * eq_sharp_ns[n_lift]);
        interactions_evals.push(denom * eq_sharp_ns[n_lift]);
    }
    let evaluated_claim = interactions_evals
        .iter()
        .chain(constraints_evals.iter())
        .zip(mu.powers())
        .map(|(x, y)| *x * y)
        .sum::<EF>();
    if cur_sum != evaluated_claim {
        return Err(BatchConstraintError::InconsistentClaims);
    }

    Ok(rs)
}
