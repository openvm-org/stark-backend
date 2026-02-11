use p3_field::PrimeCharacteristicRing;
use thiserror::Error;
use tracing::debug;

use crate::{
    poly_common::{eval_eq_mle, interpolate_cubic_at_0123, interpolate_linear_at_01},
    proof::{GkrLayerClaims, GkrProof},
    FiatShamirTranscript, StarkProtocolConfig,
};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GkrVerificationError<EF: core::fmt::Debug + core::fmt::Display + PartialEq + Eq> {
    #[error("Zero-round proof: q0_claim should be 1, got {actual}")]
    InvalidZeroRoundValue { actual: EF },
    #[error("Zero-check failed: numerator at root should be zero, got {actual}")]
    ZeroCheckFailed { actual: EF },
    #[error("Denominator consistency check failed at root: expected {expected}, got {actual}")]
    RootConsistencyCheckFailed { expected: EF, actual: EF },
    #[error("Layer consistency check failed at round {round}: expected {expected}, got {actual}")]
    LayerConsistencyCheckFailed {
        round: usize,
        expected: EF,
        actual: EF,
    },
    // TODO(ayush): remove these errors and make them debug asserts once proof shape verifier is
    // implemented
    #[error(
        "Zero-round proof should have empty layers and sumcheck, got {claims_len} and {sumcheck_len}"
    )]
    InvalidZeroRoundShape {
        claims_len: usize,
        sumcheck_len: usize,
    },
    #[error("Expected {expected} layers, got {actual}")]
    IncorrectLayerCount { expected: usize, actual: usize },
    #[error("Expected {expected} sumcheck polynomial entries, got {actual}")]
    IncorrectSumcheckPolyCount { expected: usize, actual: usize },
    #[error("Round {round} expected {expected} sumcheck sub-rounds, got {actual}")]
    IncorrectSubroundCount {
        round: usize,
        expected: usize,
        actual: usize,
    },
}

/// Verifies the GKR protocol for fractional sumcheck.
///
/// Reduces the fractional sum ∑_{y ∈ H_{ℓ+n_logup}} p̂(y)/q̂(y) = 0 to evaluation claims
/// on the input layer polynomials p̂(ξ) and q̂(ξ) at a random point ξ.
///
/// The argument `total_rounds` must equal `ℓ+n_logup`.
///
/// # Returns
/// `(p̂(ξ), q̂(ξ), ξ)` where ξ ∈ F_ext^{ℓ+n_logup} is the random evaluation point.
pub fn verify_gkr<SC: StarkProtocolConfig, TS: FiatShamirTranscript<SC>>(
    proof: &GkrProof<SC>,
    transcript: &mut TS,
    total_rounds: usize,
) -> Result<(SC::EF, SC::EF, Vec<SC::EF>), GkrVerificationError<SC::EF>> {
    if total_rounds == 0 {
        // Check proof shape
        if !proof.claims_per_layer.is_empty() || !proof.sumcheck_polys.is_empty() {
            return Err(GkrVerificationError::InvalidZeroRoundShape {
                claims_len: proof.claims_per_layer.len(),
                sumcheck_len: proof.sumcheck_polys.len(),
            });
        }
        if proof.q0_claim != SC::EF::ONE {
            return Err(GkrVerificationError::InvalidZeroRoundValue {
                actual: proof.q0_claim,
            });
        }
        return Ok((SC::EF::ZERO, SC::EF::ONE, vec![]));
    }

    // Verify proof shape
    if proof.claims_per_layer.len() != total_rounds {
        return Err(GkrVerificationError::IncorrectLayerCount {
            expected: total_rounds,
            actual: proof.claims_per_layer.len(),
        });
    }

    // Sumcheck polys: round j has j sub-rounds, so total = 0+1+2+...+(total_rounds-1)
    let expected_sumcheck_entries = total_rounds.saturating_sub(1);
    if proof.sumcheck_polys.len() != expected_sumcheck_entries {
        return Err(GkrVerificationError::IncorrectSumcheckPolyCount {
            expected: expected_sumcheck_entries,
            actual: proof.sumcheck_polys.len(),
        });
    }

    transcript.observe_ext(proof.q0_claim);

    // Handle round 0 (no sumcheck, direct tree evaluation)
    let layer_claims = &proof.claims_per_layer[0];
    observe_layer_claims::<SC, TS>(transcript, layer_claims);

    // Compute recursive relations for layer 1 → 0
    let (p_cross_term, q_cross_term) = compute_recursive_relations::<SC>(layer_claims);

    // Zero-check: p̂₀ must be zero
    if p_cross_term != SC::EF::ZERO {
        return Err(GkrVerificationError::ZeroCheckFailed {
            actual: p_cross_term,
        });
    }

    // Verify q0 consistency
    if q_cross_term != proof.q0_claim {
        return Err(GkrVerificationError::RootConsistencyCheckFailed {
            expected: proof.q0_claim,
            actual: q_cross_term,
        });
    }

    // Sample μ₁ and reduce to single evaluation
    let mu = transcript.sample_ext();
    debug!(gkr_round = 0, %mu);
    let (mut numer_claim, mut denom_claim) = reduce_to_single_evaluation::<SC>(layer_claims, mu);
    debug!(%numer_claim, %denom_claim);
    let mut gkr_r = vec![mu];

    // Handle rounds 1..total_rounds with sumcheck
    for round in 1..total_rounds {
        // Sample batching challenge λⱼ
        let lambda = transcript.sample_ext();
        debug!(gkr_round = round, %lambda);
        let claim = numer_claim + lambda * denom_claim;

        // Run sumcheck protocol for this round (round j has j sub-rounds)
        let (new_claim, round_r, eq_at_r_prime) =
            verify_gkr_sumcheck::<SC, TS>(proof, transcript, round, claim, &gkr_r)?;
        debug_assert_eq!(eq_at_r_prime, eval_eq_mle(&gkr_r, &round_r));

        // Observe layer evaluation claims
        let layer_claims = &proof.claims_per_layer[round];
        observe_layer_claims::<SC, TS>(transcript, layer_claims);

        // Compute recursive relations
        let (p_cross_term, q_cross_term) = compute_recursive_relations::<SC>(layer_claims);

        // Verify consistency
        let expected_claim = (p_cross_term + lambda * q_cross_term) * eq_at_r_prime;
        if expected_claim != new_claim {
            return Err(GkrVerificationError::LayerConsistencyCheckFailed {
                round,
                expected: expected_claim,
                actual: new_claim,
            });
        }

        // Sample μⱼ and reduce to single evaluation
        let mu = transcript.sample_ext();
        debug!(gkr_round = round, %mu);
        (numer_claim, denom_claim) = reduce_to_single_evaluation::<SC>(layer_claims, mu);
        // Update evaluation point: ξ^{(j)} = (μⱼ, ρ^{(j-1)})
        gkr_r = std::iter::once(mu).chain(round_r).collect();
    }

    Ok((numer_claim, denom_claim, gkr_r))
}

/// Verify sumcheck for a single GKR round.
///
/// Reduces evaluation of (p̂ⱼ₋₁ + λⱼ·q̂ⱼ₋₁)(ξ^{(j-1)}) to evaluations at the next layer.
///
/// # Returns
/// `(claim, ρ^{(j-1)}, eq(ξ^{(j-1)}, ρ^{(j-1)}))` where ρ^{(j-1)} is randomly sampled from the
/// sumcheck protocol.
fn verify_gkr_sumcheck<SC: StarkProtocolConfig, TS: FiatShamirTranscript<SC>>(
    proof: &GkrProof<SC>,
    transcript: &mut TS,
    round: usize,
    mut claim: SC::EF,
    gkr_r: &[SC::EF],
) -> Result<(SC::EF, Vec<SC::EF>, SC::EF), GkrVerificationError<SC::EF>> {
    debug_assert!(
        round > 0,
        "verify_gkr_sumcheck should not be called for round 0"
    );
    debug_assert_eq!(
        gkr_r.len(),
        round,
        "gkr_r should have exactly round elements"
    );

    // For round j, there are j sumcheck sub-rounds
    let expected_subrounds = round;
    let polys = &proof.sumcheck_polys[round - 1];
    if polys.len() != expected_subrounds {
        return Err(GkrVerificationError::IncorrectSubroundCount {
            round,
            expected: expected_subrounds,
            actual: polys.len(),
        });
    }
    let mut gkr_r_prime = Vec::with_capacity(round);
    let mut eq = SC::EF::ONE; // eq(ξ^{(j-1)}, ρ^{(j-1)}) computed incrementally

    for (sumcheck_round, poly_evals) in polys.iter().enumerate() {
        debug!(gkr_round = round, %sumcheck_round, sum_claim = %claim);
        // Observe s(1), s(2), s(3) where s is the sumcheck polynomial
        for &eval in poly_evals {
            transcript.observe_ext(eval);
        }

        let ri = transcript.sample_ext();
        gkr_r_prime.push(ri);
        debug!(gkr_round = round, %sumcheck_round, r_round = %ri);

        let ev0 = claim - poly_evals[0]; // s(0) = claim - s(1)
        let evals = [ev0, poly_evals[0], poly_evals[1], poly_evals[2]];
        claim = interpolate_cubic_at_0123(&evals, ri);

        // Update eq incrementally: eq *= ξᵢ·rᵢ + (1-ξᵢ)·(1-rᵢ)
        let xi = gkr_r[sumcheck_round];
        eq *= xi * ri + (SC::EF::ONE - xi) * (SC::EF::ONE - ri);
    }

    Ok((claim, gkr_r_prime, eq))
}

/// Observes layer claims in the transcript.
fn observe_layer_claims<SC: StarkProtocolConfig, TS: FiatShamirTranscript<SC>>(
    transcript: &mut TS,
    claims: &GkrLayerClaims<SC>,
) {
    transcript.observe_ext(claims.p_xi_0);
    transcript.observe_ext(claims.q_xi_0);
    transcript.observe_ext(claims.p_xi_1);
    transcript.observe_ext(claims.q_xi_1);
}

/// Computes recursive relations from layer claims.
fn compute_recursive_relations<SC: StarkProtocolConfig>(claims: &GkrLayerClaims<SC>) -> (SC::EF, SC::EF) {
    let p_cross_term = claims.p_xi_0 * claims.q_xi_1 + claims.p_xi_1 * claims.q_xi_0;
    let q_cross_term = claims.q_xi_0 * claims.q_xi_1;
    (p_cross_term, q_cross_term)
}

/// Reduces claims to a single evaluation point using linear interpolation.
fn reduce_to_single_evaluation<SC: StarkProtocolConfig>(claims: &GkrLayerClaims<SC>, mu: SC::EF) -> (SC::EF, SC::EF) {
    let numer = interpolate_linear_at_01(&[claims.p_xi_0, claims.p_xi_1], mu);
    let denom = interpolate_linear_at_01(&[claims.q_xi_0, claims.q_xi_1], mu);
    (numer, denom)
}

#[cfg(test)]
mod tests {
    use openvm_stark_sdk::config::setup_tracing;

    use super::*;
    use crate::{
        poseidon2::sponge::DuplexSponge,
        proof::{GkrLayerClaims, GkrProof},
        prover::fractional_sumcheck_gkr::{fractional_sumcheck, Frac},
        baby_bear_poseidon2::{BabyBearPoseidon2ConfigV2, EF, F},
    };

    type SCV2 = BabyBearPoseidon2ConfigV2;

    #[test]
    fn test_multiple_rounds_shape() {
        setup_tracing();
        let proof = GkrProof::<SCV2> {
            logup_pow_witness: F::ZERO,
            q0_claim: EF::ONE,
            claims_per_layer: vec![],
            sumcheck_polys: vec![],
        };

        let mut transcript = DuplexSponge::default();

        let result = verify_gkr::<SCV2, _>(&proof, &mut transcript, 2);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GkrVerificationError::IncorrectLayerCount { .. }
        ));

        let layer_claims = GkrLayerClaims::<SCV2> {
            p_xi_0: EF::ZERO,
            p_xi_1: EF::ZERO,
            q_xi_0: EF::ONE,
            q_xi_1: EF::ONE,
        };

        let proof2 = GkrProof::<SCV2> {
            logup_pow_witness: F::ZERO,
            q0_claim: EF::ONE,
            claims_per_layer: vec![layer_claims.clone(), layer_claims],
            sumcheck_polys: vec![],
        };

        let mut transcript = DuplexSponge::default();
        let result = verify_gkr::<SCV2, _>(&proof2, &mut transcript, 2);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GkrVerificationError::IncorrectSumcheckPolyCount { .. }
        ));
    }

    #[test]
    fn test_gkr_base_layer_numerator_zero() {
        setup_tracing();
        let layer1_claims = GkrLayerClaims::<SCV2> {
            p_xi_0: EF::from_u64(1), // Non-zero
            p_xi_1: EF::from_u64(2),
            q_xi_0: EF::from_u64(3),
            q_xi_1: EF::from_u64(4),
        };

        // p0 = 1*4 + 2*3 = 10 (non-zero)
        let proof = GkrProof::<SCV2> {
            logup_pow_witness: F::ZERO,
            q0_claim: EF::from_u64(12), // q0 = 3*4 = 12
            claims_per_layer: vec![layer1_claims],
            sumcheck_polys: vec![],
        };

        let mut transcript = DuplexSponge::default();
        let result = verify_gkr::<SCV2, _>(&proof, &mut transcript, 1);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GkrVerificationError::ZeroCheckFailed { .. }
        ));
    }

    #[test]
    fn test_gkr_1_round_integration() {
        setup_tracing();
        let fractions = vec![
            Frac {
                p: EF::ZERO,
                q: EF::ONE,
            },
            Frac {
                p: EF::ZERO,
                q: EF::ONE,
            },
        ];

        let mut prover_transcript = DuplexSponge::default();
        let (frac_proof, _xi) = fractional_sumcheck::<SCV2, _>(&mut prover_transcript, &fractions, true);

        let gkr_proof = GkrProof::<SCV2> {
            logup_pow_witness: F::ZERO,
            q0_claim: frac_proof.fractional_sum.1,
            claims_per_layer: frac_proof.claims_per_layer,
            sumcheck_polys: frac_proof.sumcheck_polys,
        };

        let mut verifier_transcript = DuplexSponge::default();
        let total_rounds = p3_util::log2_strict_usize(fractions.len());
        let result = verify_gkr::<SCV2, _>(&gkr_proof, &mut verifier_transcript, total_rounds);

        assert!(
            result.is_ok(),
            "1-round verification failed: {:?}",
            result.err()
        );
        let (numer_claim, denom_claim, _) = result.unwrap();
        assert_eq!(numer_claim, EF::ZERO);
        assert_ne!(denom_claim, EF::ZERO);
    }

    #[test]
    fn test_gkr_2_round_integration() {
        setup_tracing();
        let fractions = vec![
            Frac {
                p: EF::ZERO,
                q: EF::ONE,
            },
            Frac {
                p: EF::ZERO,
                q: EF::ONE,
            },
            Frac {
                p: EF::ZERO,
                q: EF::ONE,
            },
            Frac {
                p: EF::ZERO,
                q: EF::ONE,
            },
        ];

        let mut prover_transcript = DuplexSponge::default();
        let (frac_proof, _xi) = fractional_sumcheck::<SCV2, _>(&mut prover_transcript, &fractions, true);

        let gkr_proof = GkrProof::<SCV2> {
            logup_pow_witness: F::ZERO,
            q0_claim: frac_proof.fractional_sum.1,
            claims_per_layer: frac_proof.claims_per_layer,
            sumcheck_polys: frac_proof.sumcheck_polys,
        };

        let mut verifier_transcript = DuplexSponge::default();
        let total_rounds = p3_util::log2_strict_usize(fractions.len());
        let result = verify_gkr::<SCV2, _>(&gkr_proof, &mut verifier_transcript, total_rounds);

        assert!(
            result.is_ok(),
            "2-round verification failed: {:?}",
            result.err()
        );
        let (numer_claim, denom_claim, _) = result.unwrap();
        assert_eq!(numer_claim, EF::ZERO);
        assert_ne!(denom_claim, EF::ZERO);
    }

    #[test]
    fn test_gkr_3_round_integration() {
        setup_tracing();
        let fractions = vec![
            Frac {
                p: EF::ZERO,
                q: EF::ONE,
            },
            Frac {
                p: EF::ZERO,
                q: EF::ONE,
            },
            Frac {
                p: EF::ZERO,
                q: EF::ONE,
            },
            Frac {
                p: EF::ZERO,
                q: EF::ONE,
            },
            Frac {
                p: EF::ZERO,
                q: EF::ONE,
            },
            Frac {
                p: EF::ZERO,
                q: EF::ONE,
            },
            Frac {
                p: EF::ZERO,
                q: EF::ONE,
            },
            Frac {
                p: EF::ZERO,
                q: EF::ONE,
            },
        ];

        let mut prover_transcript = DuplexSponge::default();
        let (frac_proof, _xi) = fractional_sumcheck::<SCV2, _>(&mut prover_transcript, &fractions, true);

        let gkr_proof = GkrProof::<SCV2> {
            logup_pow_witness: F::ZERO,
            q0_claim: frac_proof.fractional_sum.1,
            claims_per_layer: frac_proof.claims_per_layer,
            sumcheck_polys: frac_proof.sumcheck_polys,
        };

        let mut verifier_transcript = DuplexSponge::default();
        let total_rounds = p3_util::log2_strict_usize(fractions.len());
        let result = verify_gkr::<SCV2, _>(&gkr_proof, &mut verifier_transcript, total_rounds);

        assert!(
            result.is_ok(),
            "3-round verification failed: {:?}",
            result.err()
        );
        let (numer_claim, denom_claim, _) = result.unwrap();
        assert_eq!(numer_claim, EF::ZERO);
        assert_ne!(denom_claim, EF::ZERO);
    }

    #[test]
    fn test_gkr_mixed_fractions() {
        setup_tracing();
        let fractions = vec![
            Frac {
                p: EF::from_u64(5),
                q: EF::ONE,
            },
            Frac {
                p: -EF::from_u64(5),
                q: EF::ONE,
            },
        ];

        let mut prover_transcript = DuplexSponge::default();
        let (frac_proof, _xi) = fractional_sumcheck::<SCV2, _>(&mut prover_transcript, &fractions, true);

        let gkr_proof = GkrProof::<SCV2> {
            logup_pow_witness: F::ZERO,
            q0_claim: frac_proof.fractional_sum.1,
            claims_per_layer: frac_proof.claims_per_layer,
            sumcheck_polys: frac_proof.sumcheck_polys,
        };

        let mut verifier_transcript = DuplexSponge::default();
        let total_rounds = p3_util::log2_strict_usize(fractions.len());
        let result = verify_gkr::<SCV2, _>(&gkr_proof, &mut verifier_transcript, total_rounds);

        assert!(
            result.is_ok(),
            "Mixed fractions verification failed: {:?}",
            result.err()
        );
        let (_numer_claim, denom_claim, _) = result.unwrap();
        assert_ne!(denom_claim, EF::ZERO);
    }

    #[test]
    fn test_gkr_empty_case() {
        setup_tracing();
        let fractions = vec![];

        let mut prover_transcript = DuplexSponge::default();
        let (frac_proof, _xi) = fractional_sumcheck::<SCV2, _>(&mut prover_transcript, &fractions, true);

        let gkr_proof = GkrProof::<SCV2> {
            logup_pow_witness: F::ZERO,
            q0_claim: frac_proof.fractional_sum.1,
            claims_per_layer: frac_proof.claims_per_layer,
            sumcheck_polys: frac_proof.sumcheck_polys,
        };

        let mut verifier_transcript = DuplexSponge::default();
        let result = verify_gkr::<SCV2, _>(&gkr_proof, &mut verifier_transcript, 0);

        assert!(
            result.is_ok(),
            "Empty case verification failed: {:?}",
            result.err()
        );
        let (numer_claim, denom_claim, gkr_r) = result.unwrap();
        assert_eq!(numer_claim, EF::ZERO);
        assert_eq!(denom_claim, EF::ONE);
        assert_eq!(gkr_r, vec![]);
    }
}
