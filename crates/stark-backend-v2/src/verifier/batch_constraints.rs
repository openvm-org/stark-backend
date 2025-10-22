use std::{
    iter::{self, zip},
    slice,
};

use itertools::Itertools;
use openvm_stark_backend::air_builders::symbolic::{
    SymbolicConstraints, symbolic_expression::SymbolicEvaluator,
};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, batch_multiplicative_inverse};
use p3_util::log2_ceil_u64;
use thiserror::Error;
use tracing::{debug, instrument};

use crate::{
    EF, F,
    keygen::types::MultiStarkVerifyingKey0V2,
    poly_common::{UnivariatePoly, eval_eq_mle, eval_eq_sharp_uni, eval_eq_uni},
    poseidon2::sponge::FiatShamirTranscript,
    proof::{BatchConstraintProof, GkrProof},
    prover::stacked_pcs::StackedLayout,
    verifier::{
        evaluator::VerifierConstraintEvaluator,
        fractional_sumcheck_gkr::{GkrVerificationError, verify_gkr},
    },
};

#[derive(Error, Debug)]
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
    n_per_trace: &[usize],
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

    if !transcript.check_witness(mvk.params.logup_pow_bits, gkr_proof.logup_pow_witness) {
        return Err(BatchConstraintError::InvalidLogupPowWitness);
    }
    let alpha_logup = transcript.sample_ext();
    let beta_logup = transcript.sample_ext();
    debug!(%alpha_logup, %beta_logup);
    let total_interaction_wt = zip(trace_id_to_air_id, n_per_trace)
        .map(|(&air_idx, &n)| {
            let num_interactions = mvk.per_air[air_idx].symbolic_constraints.interactions.len();
            (num_interactions as u64) << n
        })
        .sum::<u64>();
    let n_logup: usize = log2_ceil_u64(total_interaction_wt).try_into().unwrap();
    debug!(%n_logup);

    let mut xi = Vec::new();
    let mut p_xi_claim = EF::ZERO;
    let mut q_xi_claim = alpha_logup;
    if total_interaction_wt > 0 {
        (p_xi_claim, q_xi_claim, xi) = verify_gkr(gkr_proof, transcript, l_skip + n_logup)?;
        debug_assert_eq!(xi.len(), l_skip + n_logup);
    }

    let mut n_global = n_per_trace.iter().copied().max().unwrap();
    if n_global < n_logup {
        n_global = n_logup;
    } else {
        while xi.len() != l_skip + n_global {
            xi.push(transcript.sample_ext());
        }
    }
    debug!(%n_global);
    debug!(?xi);

    let lambda = transcript.sample_ext();
    debug!(%lambda);

    let max_num_constraints = mvk
        .per_air
        .iter()
        .map(|vk| vk.symbolic_constraints.constraints.constraint_idx.len())
        .max()
        .unwrap_or(0);
    let lambda_pows = lambda.powers().take(max_num_constraints).collect_vec();

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
    for &coeff in univariate_round_coeffs {
        transcript.observe_ext(coeff);
    }

    let s_deg = mvk.max_constraint_degree + 1;
    let r_0 = transcript.sample_ext();
    debug!(round = 0, r_round = %r_0);
    assert_eq!(
        univariate_round_coeffs.len(),
        (mvk.max_constraint_degree + 1) * ((1 << l_skip) - 1) + 1
    );
    let s_0 = UnivariatePoly::new(univariate_round_coeffs.clone());
    let sum_univ_domain_s_0 = s_0
        .coeffs()
        .iter()
        .step_by(1 << l_skip)
        .copied()
        .sum::<EF>()
        * EF::from_canonical_usize(1 << l_skip);
    if sum_claim != sum_univ_domain_s_0 {
        return Err(BatchConstraintError::SumClaimMismatch {
            sum_claim,
            sum_univ_domain_s_0,
        });
    }
    let mut cur_sum = s_0.eval_at_point(r_0);
    let mut rs = vec![r_0];

    #[allow(clippy::needless_range_loop)]
    for round in 0..n_global {
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
            factorials[i] = factorials[i - 1] * F::from_canonical_usize(i);
        }
        let invfact = batch_multiplicative_inverse(&factorials);

        let r = transcript.sample_ext();
        let mut pref_product = vec![EF::ONE; s_deg + 1];
        let mut suf_product = vec![EF::ONE; s_deg + 1];
        for i in 0..s_deg {
            pref_product[i + 1] = pref_product[i] * (r - EF::from_canonical_usize(i));
            suf_product[i + 1] = suf_product[i] * (EF::from_canonical_usize(s_deg - i) - r);
        }
        cur_sum = (0..=s_deg)
            .map(|i| {
                *batch_s_evals[i]
                    * EF::from_base(pref_product[i] * suf_product[s_deg - i])
                    * invfact[i]
                    * invfact[s_deg - i]
            })
            .sum::<EF>();

        debug!(round = round + 1, r_round = %r);
        rs.push(r);
    }

    let interactions_meta = (0..n_per_trace.len())
        .map(|trace_idx| {
            (
                trace_idx,
                mvk.per_air[trace_id_to_air_id[trace_idx]]
                    .symbolic_constraints
                    .interactions
                    .len(),
                l_skip + n_per_trace[trace_idx],
            )
        })
        .collect_vec();
    let interactions_layout = StackedLayout::new(l_skip + n_logup, interactions_meta);
    let eq_3b_per_trace = (0..n_per_trace.len())
        .map(|trace_idx| {
            let air_idx = trace_id_to_air_id[trace_idx];
            let interactions = &mvk.per_air[air_idx].symbolic_constraints.interactions;
            if interactions.is_empty() {
                return vec![];
            }
            let n = n_per_trace[trace_idx];
            let mut b_vec = vec![F::ZERO; n_logup - n];
            // CLEAN[AG]: I copypasted this from prover
            (0..interactions.len())
                .map(|i| {
                    let stacked_idx = interactions_layout.get(trace_idx, i).unwrap().row_idx;
                    debug_assert!(stacked_idx.trailing_zeros() as usize >= l_skip + n);
                    let mut b_int = stacked_idx >> (l_skip + n);
                    for b in &mut b_vec {
                        *b = F::from_bool(b_int & 1 == 1);
                        b_int >>= 1;
                    }
                    eval_eq_mle(&xi[l_skip + n..l_skip + n_logup], &b_vec)
                })
                .collect_vec()
        })
        .collect_vec();

    let mut eq_ns = vec![EF::ONE; n_global + 1];
    let mut eq_sharp_ns = vec![EF::ONE; n_global + 1];
    eq_ns[0] = eval_eq_uni(l_skip, xi[0], r_0);
    eq_sharp_ns[0] = eval_eq_sharp_uni(omega_skip_pows, &xi[..l_skip], r_0);
    debug_assert_eq!(rs.len(), n_global + 1);
    for (i, r) in rs.iter().enumerate().skip(1) {
        let eq_mle = eval_eq_mle(&[xi[l_skip + i - 1]], slice::from_ref(r));
        eq_ns[i] = eq_ns[i - 1] * eq_mle;
        eq_sharp_ns[i] = eq_sharp_ns[i - 1] * eq_mle;
    }
    let mut r_rev_prod = rs[n_global];
    // Product with r_i's to account for \hat{f} vs \tilde{f} for different n's in front-loaded
    // batch sumcheck.
    for i in (0..n_global).rev() {
        eq_ns[i] *= r_rev_prod;
        eq_sharp_ns[i] *= r_rev_prod;
        r_rev_prod *= rs[i];
    }
    let mut interactions_evals = Vec::new();
    let mut constraints_evals = Vec::new();

    // Observe common main openings first, and then preprocessed/cached
    for air_openings in column_openings.iter() {
        for &(claim, claim_rot) in &air_openings[0] {
            transcript.observe_ext(claim);
            transcript.observe_ext(claim_rot);
        }
    }

    for (trace_idx, air_openings) in column_openings.iter().enumerate() {
        let air_idx = trace_id_to_air_id[trace_idx];
        let vk = &mvk.per_air[air_idx];
        let n = n_per_trace[trace_idx];

        // claim lengths are checked in proof shape
        for claims in air_openings.iter().skip(1) {
            for &(claim, claim_rot) in claims.iter() {
                transcript.observe_ext(claim);
                transcript.observe_ext(claim_rot);
            }
        }

        let has_preprocessed = vk.preprocessed_data.is_some();
        let common_main = air_openings[0].as_slice();
        let preprocessed = has_preprocessed.then(|| air_openings[1].as_slice());
        let cached_idx = 1 + has_preprocessed as usize;
        let mut partitioned_main: Vec<_> = air_openings[cached_idx..]
            .iter()
            .map(|opening| opening.as_slice())
            .collect();
        partitioned_main.push(common_main);

        let evaluator = VerifierConstraintEvaluator::<F, EF>::new(
            preprocessed,
            &partitioned_main,
            &public_values[air_idx],
            &rs[..=n],
            l_skip,
        );

        let constraints = &vk.symbolic_constraints.constraints;
        let nodes = evaluator.eval_nodes(&constraints.nodes);
        let expr = zip(&lambda_pows, &constraints.constraint_idx)
            .map(|(&lambda_pow, idx)| nodes[*idx] * lambda_pow)
            .sum::<EF>();
        debug!(%trace_idx, %expr, %air_idx, "constraints_eval");
        let eq_xi_r = eq_ns[n];
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
                    .chain(std::iter::once(EF::from_canonical_u16(
                        interaction.bus_index + 1,
                    )))
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
        interactions_evals.push(num * eq_sharp_ns[n_per_trace[trace_idx]]);
        interactions_evals.push(denom * eq_sharp_ns[n_per_trace[trace_idx]]);
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
