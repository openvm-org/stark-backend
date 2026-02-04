use p3_field::PrimeCharacteristicRing;
use thiserror::Error;
use tracing::debug;

use crate::{
    block_sumcheck_sizes, gkr_block_len,
    poly_common::{eval_eq_mle, interpolate_linear_at_01, interpolate_multivariate_at_0123_grid},
    poseidon2::sponge::FiatShamirTranscript,
    proof::{GkrLayerClaims, GkrProof},
    EF,
};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GkrVerificationError {
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
        "Zero-round proof should have empty layers and block sumcheck, got {claims_len} and {block_sumcheck_len}"
    )]
    InvalidZeroRoundShape {
        claims_len: usize,
        block_sumcheck_len: usize,
    },
    #[error("Expected {expected} layers, got {actual}")]
    IncorrectLayerCount { expected: usize, actual: usize },
    #[error("Expected {expected} block sumcheck entries, got {actual}")]
    IncorrectBlockSumcheckPolyCount { expected: usize, actual: usize },
    #[error("Round {round} block {block} has invalid length {len}")]
    InvalidBlockSumcheckPolyLen {
        round: usize,
        block: usize,
        len: usize,
    },
    #[error("Round {round} expected {expected} sumcheck variables, got {actual}")]
    IncorrectBlockVarCount {
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
pub fn verify_gkr<TS: FiatShamirTranscript>(
    proof: &GkrProof,
    transcript: &mut TS,
    total_rounds: usize,
) -> Result<(EF, EF, Vec<EF>), GkrVerificationError> {
    if total_rounds == 0 {
        // Check proof shape
        if !proof.claims_per_layer.is_empty() || !proof.block_sumcheck_polys.is_empty() {
            return Err(GkrVerificationError::InvalidZeroRoundShape {
                claims_len: proof.claims_per_layer.len(),
                block_sumcheck_len: proof.block_sumcheck_polys.len(),
            });
        }
        if proof.q0_claim != EF::ONE {
            return Err(GkrVerificationError::InvalidZeroRoundValue {
                actual: proof.q0_claim,
            });
        }
        return Ok((EF::ZERO, EF::ONE, vec![]));
    }

    // Verify proof shape
    if proof.claims_per_layer.len() != total_rounds {
        return Err(GkrVerificationError::IncorrectLayerCount {
            expected: total_rounds,
            actual: proof.claims_per_layer.len(),
        });
    }

    // Block sumcheck polys: one entry per round j=1..total_rounds-1
    let expected_sumcheck_entries = total_rounds.saturating_sub(1);
    if proof.block_sumcheck_polys.len() != expected_sumcheck_entries {
        return Err(GkrVerificationError::IncorrectBlockSumcheckPolyCount {
            expected: expected_sumcheck_entries,
            actual: proof.block_sumcheck_polys.len(),
        });
    }

    transcript.observe_ext(proof.q0_claim);

    // Handle round 0 (no sumcheck, direct tree evaluation)
    let layer_claims = &proof.claims_per_layer[0];
    observe_layer_claims(transcript, layer_claims);

    // Compute recursive relations for layer 1 → 0
    let (p_cross_term, q_cross_term) = compute_recursive_relations(layer_claims);

    // Zero-check: p̂₀ must be zero
    if p_cross_term != EF::ZERO {
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
    let (mut numer_claim, mut denom_claim) = reduce_to_single_evaluation(layer_claims, mu);
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
            verify_gkr_block_sumcheck(proof, transcript, round, claim, &gkr_r)?;
        debug_assert_eq!(eq_at_r_prime, eval_eq_mle(&gkr_r, &round_r));

        // Observe layer evaluation claims
        let layer_claims = &proof.claims_per_layer[round];
        observe_layer_claims(transcript, layer_claims);

        // Compute recursive relations
        let (p_cross_term, q_cross_term) = compute_recursive_relations(layer_claims);

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
        (numer_claim, denom_claim) = reduce_to_single_evaluation(layer_claims, mu);
        // Update evaluation point: ξ^{(j)} = (μⱼ, ρ^{(j-1)})
        gkr_r = std::iter::once(mu).chain(round_r).collect();
    }

    Ok((numer_claim, denom_claim, gkr_r))
}

/// Verify block sumcheck for a single GKR round.
///
/// Reduces evaluation of (p̂ⱼ₋₁ + λⱼ·q̂ⱼ₋₁)(ξ^{(j-1)}) to evaluations at the next layer.
///
/// # Returns
/// `(claim, ρ^{(j-1)}, eq(ξ^{(j-1)}, ρ^{(j-1)}))` where ρ^{(j-1)} is randomly sampled from the
/// sumcheck protocol.
fn verify_gkr_block_sumcheck<TS: FiatShamirTranscript>(
    proof: &GkrProof,
    transcript: &mut TS,
    round: usize,
    mut claim: EF,
    gkr_r: &[EF],
) -> Result<(EF, Vec<EF>, EF), GkrVerificationError> {
    debug_assert!(
        round > 0,
        "verify_gkr_block_sumcheck should not be called for round 0"
    );
    debug_assert_eq!(
        gkr_r.len(),
        round,
        "gkr_r should have exactly round elements"
    );

    let blocks = &proof.block_sumcheck_polys[round - 1];
    let mut gkr_r_prime = Vec::with_capacity(round);
    let mut eq = EF::ONE; // eq(ξ^{(j-1)}, ρ^{(j-1)}) computed incrementally
    let mut sizes = block_sumcheck_sizes(round);
    let mut consumed = 0usize;

    for (block_idx, block_evals) in blocks.iter().enumerate() {
        debug!(gkr_round = round, block = block_idx, sum_claim = %claim);

        let Some(k) = sizes.next() else {
            return Err(GkrVerificationError::IncorrectBlockVarCount {
                round,
                expected: round,
                actual: round + 1,
            });
        };

        let expected_len = gkr_block_len(k);
        if block_evals.len() != expected_len {
            return Err(GkrVerificationError::InvalidBlockSumcheckPolyLen {
                round,
                block: block_idx,
                len: block_evals.len(),
            });
        }
        let start_idx = consumed;
        consumed += k;
        if consumed > round {
            return Err(GkrVerificationError::IncorrectBlockVarCount {
                round,
                expected: round,
                actual: consumed,
            });
        }

        // Observe all block evaluations
        for &eval in block_evals {
            transcript.observe_ext(eval);
        }

        // Sample k challenges
        let mut challenges = Vec::with_capacity(k);
        for _ in 0..k {
            let ri = transcript.sample_ext();
            challenges.push(ri);
            gkr_r_prime.push(ri);
        }

        // Verify sumcheck relation: sum over {0,1}^k == claim
        let boolean_sum = sum_at_boolean_points(block_evals, k);
        if boolean_sum != claim {
            return Err(GkrVerificationError::LayerConsistencyCheckFailed {
                round,
                expected: claim,
                actual: boolean_sum,
            });
        }

        // Interpolate at the sampled challenges
        let mut grid_evals = block_evals.to_vec();
        interpolate_multivariate_at_0123_grid(&mut grid_evals, &challenges);
        claim = grid_evals[0];

        // Update eq incrementally: eq *= ξᵢ·rᵢ + (1-ξᵢ)·(1-rᵢ)
        for i in 0..k {
            let xi = gkr_r[start_idx + i];
            let ri = challenges[i];
            eq *= xi * ri + (EF::ONE - xi) * (EF::ONE - ri);
        }
    }

    if sizes.next().is_some() || consumed != round {
        return Err(GkrVerificationError::IncorrectBlockVarCount {
            round,
            expected: round,
            actual: consumed,
        });
    }

    Ok((claim, gkr_r_prime, eq))
}

/// Observes layer claims in the transcript.
fn observe_layer_claims<TS: FiatShamirTranscript>(transcript: &mut TS, claims: &GkrLayerClaims) {
    transcript.observe_ext(claims.p_xi_0);
    transcript.observe_ext(claims.q_xi_0);
    transcript.observe_ext(claims.p_xi_1);
    transcript.observe_ext(claims.q_xi_1);
}

/// Computes recursive relations from layer claims.
fn compute_recursive_relations(claims: &GkrLayerClaims) -> (EF, EF) {
    let p_cross_term = claims.p_xi_0 * claims.q_xi_1 + claims.p_xi_1 * claims.q_xi_0;
    let q_cross_term = claims.q_xi_0 * claims.q_xi_1;
    (p_cross_term, q_cross_term)
}

/// Reduces claims to a single evaluation point using linear interpolation.
fn reduce_to_single_evaluation(claims: &GkrLayerClaims, mu: EF) -> (EF, EF) {
    let numer = interpolate_linear_at_01(&[claims.p_xi_0, claims.p_xi_1], mu);
    let denom = interpolate_linear_at_01(&[claims.q_xi_0, claims.q_xi_1], mu);
    (numer, denom)
}

/// Sums evaluations over all of {0,1}^k (including the origin).
///
/// `block_evals` contains values on the grid {0,1,2,3}^k (including origin at index 0),
/// ordered lexicographically with coord_0 varying fastest, i.e. `index = sum_i coord_i * 4^i`.
///
/// The algorithm iterates boolean masks and maps each {0,1}^k point to its index via base-4
/// positional weights 4^i.
fn sum_at_boolean_points(block_evals: &[EF], k: usize) -> EF {
    let mut sum = EF::ZERO;
    for mask in 0..(1usize << k) {
        let mut idx = 0usize;
        let mut bit = mask;
        let mut i = 0usize;
        while bit != 0 {
            if bit & 1 == 1 {
                idx += 1 << (2 * i);
            }
            bit >>= 1;
            i += 1;
        }
        sum += block_evals[idx];
    }
    sum
}

#[cfg(test)]
mod tests {
    use openvm_stark_sdk::config::setup_tracing;

    use super::*;
    use crate::{
        poseidon2::sponge::DuplexSponge,
        proof::{GkrLayerClaims, GkrProof},
        prover::fractional_sumcheck_gkr::{fractional_sumcheck, Frac},
        F,
    };

    #[test]
    fn test_multiple_rounds_shape() {
        setup_tracing();
        let proof = GkrProof {
            logup_pow_witness: F::ZERO,
            q0_claim: EF::ONE,
            claims_per_layer: vec![],
            block_sumcheck_polys: vec![],
        };

        let mut transcript = DuplexSponge::default();

        let result = verify_gkr(&proof, &mut transcript, 2);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GkrVerificationError::IncorrectLayerCount { .. }
        ));

        let layer_claims = GkrLayerClaims {
            p_xi_0: EF::ZERO,
            p_xi_1: EF::ZERO,
            q_xi_0: EF::ONE,
            q_xi_1: EF::ONE,
        };

        let proof2 = GkrProof {
            logup_pow_witness: F::ZERO,
            q0_claim: EF::ONE,
            claims_per_layer: vec![layer_claims.clone(), layer_claims],
            block_sumcheck_polys: vec![],
        };

        let mut transcript = DuplexSponge::default();
        let result = verify_gkr(&proof2, &mut transcript, 2);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GkrVerificationError::IncorrectBlockSumcheckPolyCount { .. }
        ));
    }

    #[test]
    fn test_sum_at_boolean_points() {
        let k = 3usize;
        let grid_len = 4usize.pow(k as u32);
        let mut grid = vec![EF::ZERO; grid_len];
        for idx in 0..grid_len {
            let mut tmp = idx;
            let mut acc = 0u64;
            for i in 0..k {
                let coord = (tmp % 4) as u64;
                acc += (i as u64 + 1) * (coord + 1);
                tmp /= 4;
            }
            grid[idx] = EF::from_u64(acc);
        }

        let mut expected = EF::ZERO;
        for mask in 0..(1usize << k) {
            let mut idx = 0usize;
            for i in 0..k {
                if (mask >> i) & 1 == 1 {
                    idx += 1 << (2 * i);
                }
            }
            expected += grid[idx];
        }

        let actual = sum_at_boolean_points(&grid, k);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_gkr_base_layer_numerator_zero() {
        setup_tracing();
        let layer1_claims = GkrLayerClaims {
            p_xi_0: EF::from_u64(1), // Non-zero
            p_xi_1: EF::from_u64(2),
            q_xi_0: EF::from_u64(3),
            q_xi_1: EF::from_u64(4),
        };

        // p0 = 1*4 + 2*3 = 10 (non-zero)
        let proof = GkrProof {
            logup_pow_witness: F::ZERO,
            q0_claim: EF::from_u64(12), // q0 = 3*4 = 12
            claims_per_layer: vec![layer1_claims],
            block_sumcheck_polys: vec![],
        };

        let mut transcript = DuplexSponge::default();
        let result = verify_gkr(&proof, &mut transcript, 1);
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
        let (frac_proof, _xi) = fractional_sumcheck(&mut prover_transcript, &fractions, true);

        let gkr_proof = GkrProof {
            logup_pow_witness: F::ZERO,
            q0_claim: frac_proof.fractional_sum.1,
            claims_per_layer: frac_proof.claims_per_layer,
            block_sumcheck_polys: frac_proof.block_sumcheck_polys,
        };

        let mut verifier_transcript = DuplexSponge::default();
        let total_rounds = p3_util::log2_strict_usize(fractions.len());
        let result = verify_gkr(&gkr_proof, &mut verifier_transcript, total_rounds);

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
        let (frac_proof, _xi) = fractional_sumcheck(&mut prover_transcript, &fractions, true);

        let gkr_proof = GkrProof {
            logup_pow_witness: F::ZERO,
            q0_claim: frac_proof.fractional_sum.1,
            claims_per_layer: frac_proof.claims_per_layer,
            block_sumcheck_polys: frac_proof.block_sumcheck_polys,
        };

        let mut verifier_transcript = DuplexSponge::default();
        let total_rounds = p3_util::log2_strict_usize(fractions.len());
        let result = verify_gkr(&gkr_proof, &mut verifier_transcript, total_rounds);

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
        let (frac_proof, _xi) = fractional_sumcheck(&mut prover_transcript, &fractions, true);

        let gkr_proof = GkrProof {
            logup_pow_witness: F::ZERO,
            q0_claim: frac_proof.fractional_sum.1,
            claims_per_layer: frac_proof.claims_per_layer,
            block_sumcheck_polys: frac_proof.block_sumcheck_polys,
        };

        let mut verifier_transcript = DuplexSponge::default();
        let total_rounds = p3_util::log2_strict_usize(fractions.len());
        let result = verify_gkr(&gkr_proof, &mut verifier_transcript, total_rounds);

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
        let (frac_proof, _xi) = fractional_sumcheck(&mut prover_transcript, &fractions, true);

        let gkr_proof = GkrProof {
            logup_pow_witness: F::ZERO,
            q0_claim: frac_proof.fractional_sum.1,
            claims_per_layer: frac_proof.claims_per_layer,
            block_sumcheck_polys: frac_proof.block_sumcheck_polys,
        };

        let mut verifier_transcript = DuplexSponge::default();
        let total_rounds = p3_util::log2_strict_usize(fractions.len());
        let result = verify_gkr(&gkr_proof, &mut verifier_transcript, total_rounds);

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
        let (frac_proof, _xi) = fractional_sumcheck(&mut prover_transcript, &fractions, true);

        let gkr_proof = GkrProof {
            logup_pow_witness: F::ZERO,
            q0_claim: frac_proof.fractional_sum.1,
            claims_per_layer: frac_proof.claims_per_layer,
            block_sumcheck_polys: frac_proof.block_sumcheck_polys,
        };

        let mut verifier_transcript = DuplexSponge::default();
        let result = verify_gkr(&gkr_proof, &mut verifier_transcript, 0);

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
