use p3_field::PrimeCharacteristicRing;
use thiserror::Error;
use tracing::debug;

use crate::{
    poly_common::{eval_eq_mle, interpolate_cubic_at_0123, interpolate_linear_at_01},
    poseidon2::sponge::FiatShamirTranscript,
    prover::poly::evals_eq_hypercube_serial,
    proof::{GkrLayerClaims, GkrProof},
    EF,
};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GkrVerificationError {
    #[error("Grid claim count mismatch: expected {expected}, got {actual}")]
    GridClaimCountMismatch { expected: usize, actual: usize },
    #[error("Grid denominator is zero at index {index}")]
    GridDenominatorZero { index: usize },
    #[error("Grid sum check failed: numerator should be zero, got {actual}")]
    GridSumCheckFailed { actual: EF },
    #[error("Zero-check failed: numerator at root should be zero, got {actual}")]
    ZeroCheckFailed { actual: EF },
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
    #[error("Layer {layer} has {actual} grid claims, expected {expected}")]
    IncorrectLayerGridSize {
        layer: usize,
        expected: usize,
        actual: usize,
    },
}

/// Verifies the data-parallel GKR protocol for fractional sumcheck with grid check.
///
/// Reduces the fractional sum to evaluation claims on the input layer polynomials
/// p_hat(xi) and q_hat(xi) at a random point xi.
///
/// The argument `total_rounds` is l_skip + n_logup_block (NOT the full n_logup).
/// The argument `n_grid` is the effective grid dimension n'_grid.
///
/// # Returns
/// `(p_hat(xi), q_hat(xi), xi)` where xi = concat(gkr_r, xi_grid).
/// Block dimensions come first (gkr_r), grid dimensions last (xi_grid).
pub fn verify_gkr<TS: FiatShamirTranscript>(
    proof: &GkrProof,
    transcript: &mut TS,
    total_rounds: usize,
    n_grid: usize,
) -> Result<(EF, EF, Vec<EF>), GkrVerificationError> {
    let n_grid_size = 1usize << n_grid;

    // Step 2: Verify grid claims
    if proof.grid_claims.len() != n_grid_size {
        return Err(GkrVerificationError::GridClaimCountMismatch {
            expected: n_grid_size,
            actual: proof.grid_claims.len(),
        });
    }

    // Check all q_g != 0 and compute sum of p_g/q_g = 0 via fraction addition
    // Frac addition: (p1/q1) + (p2/q2) = (p1*q2 + p2*q1) / (q1*q2)
    let mut sum_p = EF::ZERO;
    let mut sum_q = EF::ONE;
    for (g, &(p_g, q_g)) in proof.grid_claims.iter().enumerate() {
        if q_g == EF::ZERO {
            return Err(GkrVerificationError::GridDenominatorZero { index: g });
        }
        // Add p_g/q_g to sum_p/sum_q
        sum_p = sum_p * q_g + p_g * sum_q;
        sum_q *= q_g;
    }
    if sum_p != EF::ZERO {
        return Err(GkrVerificationError::GridSumCheckFailed { actual: sum_p });
    }

    // Observe grid claims in transcript.
    // When there is only one grid point (n_grid=0), p=0 is implicit and not observed.
    // When there are multiple grid points, individual p_g values are non-zero
    // and must be observed to bind them before xi_grid is sampled.
    for &(p_g, q_g) in &proof.grid_claims {
        if n_grid_size > 1 {
            transcript.observe_ext(p_g);
        }
        transcript.observe_ext(q_g);
    }

    // Step 3: If total_rounds == 0, GKR is skipped; evaluation claims are grid claims themselves
    if total_rounds == 0 {
        if !proof.claims_per_layer.is_empty() || !proof.sumcheck_polys.is_empty() {
            return Err(GkrVerificationError::InvalidZeroRoundShape {
                claims_len: proof.claims_per_layer.len(),
                sumcheck_len: proof.sumcheck_polys.len(),
            });
        }
        // Go directly to grid check with grid claims as evaluation claims
        let (p_xi, q_xi, xi) =
            grid_check(transcript, &proof.grid_claims, n_grid, vec![]);
        return Ok((p_xi, q_xi, xi));
    }

    // Verify proof shape
    if proof.claims_per_layer.len() != total_rounds {
        return Err(GkrVerificationError::IncorrectLayerCount {
            expected: total_rounds,
            actual: proof.claims_per_layer.len(),
        });
    }
    let expected_sumcheck_entries = total_rounds.saturating_sub(1);
    if proof.sumcheck_polys.len() != expected_sumcheck_entries {
        return Err(GkrVerificationError::IncorrectSumcheckPolyCount {
            expected: expected_sumcheck_entries,
            actual: proof.sumcheck_polys.len(),
        });
    }

    // Step 4: Round j=1 (direct check, no sumcheck)
    // Verify shape: claims_per_layer[0] must have n_grid_size entries
    let layer_claims_vec = &proof.claims_per_layer[0];
    if layer_claims_vec.len() != n_grid_size {
        return Err(GkrVerificationError::IncorrectLayerGridSize {
            layer: 0,
            expected: n_grid_size,
            actual: layer_claims_vec.len(),
        });
    }

    // Observe all per-grid layer claims
    for claims in layer_claims_vec {
        observe_layer_claims(transcript, claims);
    }

    // Verify per-grid-point root consistency:
    // For each g: p_cross_g should equal grid_claims[g].0
    //             q_cross_g should equal grid_claims[g].1
    for (g, claims) in layer_claims_vec.iter().enumerate() {
        let (p_cross_g, q_cross_g) = compute_recursive_relations(claims);
        let (expected_p, expected_q) = proof.grid_claims[g];
        if p_cross_g != expected_p || q_cross_g != expected_q {
            return Err(GkrVerificationError::ZeroCheckFailed {
                actual: p_cross_g - expected_p,
            });
        }
    }

    // Sample mu_1, reduce per-grid claims to single eval point
    let mu_1 = transcript.sample_ext();
    debug!(gkr_round = 0, %mu_1);

    // Per-grid numer/denom claims via linear interpolation
    let mut numer_claims: Vec<EF> = Vec::with_capacity(n_grid_size);
    let mut denom_claims: Vec<EF> = Vec::with_capacity(n_grid_size);
    for claims in layer_claims_vec {
        let (n, d) = reduce_to_single_evaluation(claims, mu_1);
        numer_claims.push(n);
        denom_claims.push(d);
    }
    let mut gkr_r = vec![mu_1];

    // Step 5: Rounds j=2,...,total_rounds (with sumcheck)
    for round in 1..total_rounds {
        // Sample gamma_j for batching
        let gamma_j = transcript.sample_ext();
        debug!(gkr_round = round, %gamma_j);
        let gamma_pows = gamma_powers(gamma_j, 2 * n_grid_size);

        // Compute batched claim
        let claim: EF = (0..n_grid_size)
            .map(|g| {
                gamma_pows[2 * g] * numer_claims[g] + gamma_pows[2 * g + 1] * denom_claims[g]
            })
            .sum();

        // Run sumcheck protocol for this round
        let (new_claim, round_r, eq_at_r_prime) =
            verify_gkr_sumcheck(proof, transcript, round, claim, &gkr_r)?;
        debug_assert_eq!(eq_at_r_prime, eval_eq_mle(&gkr_r, &round_r));

        // Read layer claims for all grid points
        let layer_claims_vec = &proof.claims_per_layer[round];
        if layer_claims_vec.len() != n_grid_size {
            return Err(GkrVerificationError::IncorrectLayerGridSize {
                layer: round,
                expected: n_grid_size,
                actual: layer_claims_vec.len(),
            });
        }
        for claims in layer_claims_vec {
            observe_layer_claims(transcript, claims);
        }

        // Compute batched recursive from layer claims
        let batched_recursive: EF = layer_claims_vec
            .iter()
            .enumerate()
            .map(|(g, claims)| {
                let (p_cross_g, q_cross_g) = compute_recursive_relations(claims);
                gamma_pows[2 * g] * p_cross_g + gamma_pows[2 * g + 1] * q_cross_g
            })
            .sum();

        // Check: batched_recursive * eq_at_r_prime == new_claim
        let expected_claim = batched_recursive * eq_at_r_prime;
        if expected_claim != new_claim {
            return Err(GkrVerificationError::LayerConsistencyCheckFailed {
                round,
                expected: expected_claim,
                actual: new_claim,
            });
        }

        // Sample mu, reduce claims
        let mu = transcript.sample_ext();
        debug!(gkr_round = round, %mu);
        numer_claims.clear();
        denom_claims.clear();
        for claims in layer_claims_vec {
            let (n, d) = reduce_to_single_evaluation(claims, mu);
            numer_claims.push(n);
            denom_claims.push(d);
        }
        // Update evaluation point: gkr_r = (mu, round_r)
        gkr_r = std::iter::once(mu).chain(round_r).collect();
    }

    // Step 6: Grid check
    // Per-grid evaluation claims are (numer_claims[g], denom_claims[g]) at xi_block = gkr_r
    let eval_claims: Vec<(EF, EF)> = numer_claims
        .into_iter()
        .zip(denom_claims)
        .collect();

    let (p_xi, q_xi, xi) = grid_check(transcript, &eval_claims, n_grid, gkr_r);
    Ok((p_xi, q_xi, xi))
}

/// Performs the grid check: combines per-grid-point evaluation claims into a single
/// evaluation claim using eq(xi_grid, g) weights.
///
/// Returns (p_xi, q_xi, xi) where xi = concat(gkr_r, xi_grid).
/// The block dimensions (gkr_r) come first, matching the LSB-first MLE convention
/// used throughout the codebase.
fn grid_check<TS: FiatShamirTranscript>(
    transcript: &mut TS,
    eval_claims: &[(EF, EF)],
    n_grid: usize,
    gkr_r: Vec<EF>,
) -> (EF, EF, Vec<EF>) {
    let n_grid_size = 1usize << n_grid;
    debug_assert_eq!(eval_claims.len(), n_grid_size);

    // Sample xi_grid (n_grid random elements)
    let xi_grid: Vec<EF> = (0..n_grid).map(|_| transcript.sample_ext()).collect();

    // Compute eq(xi_grid, g) for all g in {0,1}^n_grid
    let eq_evals = evals_eq_hypercube_serial(&xi_grid);

    // p_xi = sum_g v_{p,g} * eq(xi_grid, g)
    // q_xi = sum_g v_{q,g} * eq(xi_grid, g)
    let mut p_xi = EF::ZERO;
    let mut q_xi = EF::ZERO;
    for (g, &(v_p_g, v_q_g)) in eval_claims.iter().enumerate() {
        p_xi += v_p_g * eq_evals[g];
        q_xi += v_q_g * eq_evals[g];
    }

    // xi = concat(gkr_r, xi_grid)  [block dimensions first, then grid dimensions]
    let xi: Vec<EF> = gkr_r.into_iter().chain(xi_grid).collect();

    (p_xi, q_xi, xi)
}

/// Compute powers of gamma: gamma^0, gamma^1, ..., gamma^{count-1}
fn gamma_powers(gamma: EF, count: usize) -> Vec<EF> {
    let mut pows = Vec::with_capacity(count);
    let mut cur = EF::ONE;
    for _ in 0..count {
        pows.push(cur);
        cur *= gamma;
    }
    pows
}

/// Verify sumcheck for a single GKR round.
///
/// Reduces evaluation of (p̂ⱼ₋₁ + λⱼ·q̂ⱼ₋₁)(ξ^{(j-1)}) to evaluations at the next layer.
///
/// # Returns
/// `(claim, ρ^{(j-1)}, eq(ξ^{(j-1)}, ρ^{(j-1)}))` where ρ^{(j-1)} is randomly sampled from the
/// sumcheck protocol.
fn verify_gkr_sumcheck<TS: FiatShamirTranscript>(
    proof: &GkrProof,
    transcript: &mut TS,
    round: usize,
    mut claim: EF,
    gkr_r: &[EF],
) -> Result<(EF, Vec<EF>, EF), GkrVerificationError> {
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
    let mut eq = EF::ONE; // eq(ξ^{(j-1)}, ρ^{(j-1)}) computed incrementally

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
        eq *= xi * ri + (EF::ONE - xi) * (EF::ONE - ri);
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

#[cfg(test)]
mod tests {
    use openvm_stark_sdk::config::setup_tracing;

    use super::*;
    use crate::{
        poseidon2::sponge::DuplexSponge,
        proof::{GkrLayerClaims, GkrProof},
        prover::fractional_sumcheck_gkr::{fractional_sumcheck, Frac, FracSumcheckProof},
        F,
    };

    /// Helper to build a GkrProof from a FracSumcheckProof for testing (single grid point, n_grid=0).
    fn gkr_proof_from_frac(frac_proof: &FracSumcheckProof<EF>) -> GkrProof {
        GkrProof {
            logup_pow_witness: F::ZERO,
            grid_claims: frac_proof.grid_claims.clone(),
            claims_per_layer: frac_proof.claims_per_layer.clone(),
            sumcheck_polys: frac_proof.sumcheck_polys.clone(),
        }
    }

    #[test]
    fn test_multiple_rounds_shape() {
        setup_tracing();
        let proof = GkrProof {
            logup_pow_witness: F::ZERO,
            grid_claims: vec![(EF::ZERO, EF::ONE)],
            claims_per_layer: vec![],
            sumcheck_polys: vec![],
        };

        let mut transcript = DuplexSponge::default();

        let result = verify_gkr(&proof, &mut transcript, 2, 0);
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
            grid_claims: vec![(EF::ZERO, EF::ONE)],
            claims_per_layer: vec![vec![layer_claims.clone()], vec![layer_claims]],
            sumcheck_polys: vec![],
        };

        let mut transcript = DuplexSponge::default();
        let result = verify_gkr(&proof2, &mut transcript, 2, 0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GkrVerificationError::IncorrectSumcheckPolyCount { .. }
        ));
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

        // p_cross = 1*4 + 2*3 = 10 (non-zero)
        // q_cross = 3*4 = 12
        // grid_claims: single grid point with (p=0, q=12)
        // batched_claim = gamma^0 * 0 + gamma^1 * 12 = 12*gamma
        // batched_recursive = gamma^0 * 10 + gamma^1 * 12 = 10 + 12*gamma
        // These won't match because 10 != 0
        let proof = GkrProof {
            logup_pow_witness: F::ZERO,
            grid_claims: vec![(EF::ZERO, EF::from_u64(12))],
            claims_per_layer: vec![vec![layer1_claims]],
            sumcheck_polys: vec![],
        };

        let mut transcript = DuplexSponge::default();
        let result = verify_gkr(&proof, &mut transcript, 1, 0);
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
        let (frac_proof, _xi) = fractional_sumcheck(&mut prover_transcript, &fractions, true, 0);

        let gkr_proof = gkr_proof_from_frac(&frac_proof);

        let mut verifier_transcript = DuplexSponge::default();
        // Observe grid claims (matching prover which observes q0_claim)

        let total_rounds = p3_util::log2_strict_usize(fractions.len());
        let result = verify_gkr(&gkr_proof, &mut verifier_transcript, total_rounds, 0);

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
        let (frac_proof, _xi) = fractional_sumcheck(&mut prover_transcript, &fractions, true, 0);

        let gkr_proof = gkr_proof_from_frac(&frac_proof);

        let mut verifier_transcript = DuplexSponge::default();

        let total_rounds = p3_util::log2_strict_usize(fractions.len());
        let result = verify_gkr(&gkr_proof, &mut verifier_transcript, total_rounds, 0);

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
        let (frac_proof, _xi) = fractional_sumcheck(&mut prover_transcript, &fractions, true, 0);

        let gkr_proof = gkr_proof_from_frac(&frac_proof);

        let mut verifier_transcript = DuplexSponge::default();

        let total_rounds = p3_util::log2_strict_usize(fractions.len());
        let result = verify_gkr(&gkr_proof, &mut verifier_transcript, total_rounds, 0);

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
        let (frac_proof, _xi) = fractional_sumcheck(&mut prover_transcript, &fractions, true, 0);

        let gkr_proof = gkr_proof_from_frac(&frac_proof);

        let mut verifier_transcript = DuplexSponge::default();

        let total_rounds = p3_util::log2_strict_usize(fractions.len());
        let result = verify_gkr(&gkr_proof, &mut verifier_transcript, total_rounds, 0);

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
        // When fractions is empty, the prover returns empty grid_claims.
        // In the full protocol, verify_gkr is NOT called when total_interactions == 0.
        // This test verifies that the prover produces the expected empty proof.
        let fractions = vec![];

        let mut prover_transcript = DuplexSponge::default();
        let (frac_proof, _xi) = fractional_sumcheck(&mut prover_transcript, &fractions, true, 0);

        assert!(frac_proof.grid_claims.is_empty());
        assert!(frac_proof.claims_per_layer.is_empty());
        assert!(frac_proof.sumcheck_polys.is_empty());
    }

    #[test]
    fn test_grid_claim_count_mismatch() {
        setup_tracing();
        let proof = GkrProof {
            logup_pow_witness: F::ZERO,
            grid_claims: vec![(EF::ZERO, EF::ONE), (EF::ZERO, EF::ONE)],
            claims_per_layer: vec![],
            sumcheck_polys: vec![],
        };

        let mut transcript = DuplexSponge::default();
        // n_grid=0 => expects 1 grid claim, but we have 2
        let result = verify_gkr(&proof, &mut transcript, 0, 0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GkrVerificationError::GridClaimCountMismatch {
                expected: 1,
                actual: 2
            }
        ));
    }

    #[test]
    fn test_grid_denominator_zero() {
        setup_tracing();
        let proof = GkrProof {
            logup_pow_witness: F::ZERO,
            grid_claims: vec![(EF::ZERO, EF::ZERO)],
            claims_per_layer: vec![],
            sumcheck_polys: vec![],
        };

        let mut transcript = DuplexSponge::default();
        let result = verify_gkr(&proof, &mut transcript, 0, 0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GkrVerificationError::GridDenominatorZero { index: 0 }
        ));
    }

    #[test]
    fn test_grid_sum_nonzero() {
        setup_tracing();
        // Two grid points: 1/1 and 1/1 => sum = 2 != 0
        let proof = GkrProof {
            logup_pow_witness: F::ZERO,
            grid_claims: vec![(EF::ONE, EF::ONE), (EF::ONE, EF::ONE)],
            claims_per_layer: vec![],
            sumcheck_polys: vec![],
        };

        let mut transcript = DuplexSponge::default();
        let result = verify_gkr(&proof, &mut transcript, 0, 1);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GkrVerificationError::GridSumCheckFailed { .. }
        ));
    }

    #[test]
    fn test_grid_check_trivial_n_grid_0() {
        setup_tracing();
        // n_grid=0, single grid point, total_rounds=0
        // Grid check should be trivial: eq evaluation = 1, xi_grid = empty
        let proof = GkrProof {
            logup_pow_witness: F::ZERO,
            grid_claims: vec![(EF::ZERO, EF::ONE)],
            claims_per_layer: vec![],
            sumcheck_polys: vec![],
        };

        let mut transcript = DuplexSponge::default();
        let result = verify_gkr(&proof, &mut transcript, 0, 0);
        assert!(result.is_ok());
        let (p_xi, q_xi, xi) = result.unwrap();
        assert_eq!(p_xi, EF::ZERO);
        assert_eq!(q_xi, EF::ONE);
        assert_eq!(xi, vec![]);
    }

    #[test]
    fn test_grid_check_multiple_grid_points_zero_rounds() {
        setup_tracing();
        // n_grid=1, 2 grid points, total_rounds=0
        // Grid claims sum to zero: 3/2 + (-3)/2 = 0
        let proof = GkrProof {
            logup_pow_witness: F::ZERO,
            grid_claims: vec![
                (EF::from_u64(3), EF::from_u64(2)),
                (-EF::from_u64(3), EF::from_u64(2)),
            ],
            claims_per_layer: vec![],
            sumcheck_polys: vec![],
        };

        let mut transcript = DuplexSponge::default();
        let result = verify_gkr(&proof, &mut transcript, 0, 1);
        assert!(result.is_ok());
        let (_p_xi, _q_xi, xi) = result.unwrap();
        // xi should have 1 element (xi_grid only, no gkr_r)
        assert_eq!(xi.len(), 1);
    }

    /// Integration test: prover + verifier with n_grid=1 and 4 fractions (2 block rounds, 2 grid points).
    #[test]
    fn test_gkr_grid_1_integration() {
        setup_tracing();
        let fractions = vec![
            Frac {
                p: EF::from_u64(3),
                q: EF::ONE,
            },
            Frac {
                p: -EF::from_u64(3),
                q: EF::ONE,
            },
            Frac {
                p: EF::from_u64(7),
                q: EF::from_u64(2),
            },
            Frac {
                p: -EF::from_u64(7),
                q: EF::from_u64(2),
            },
        ];

        let n_grid = 1; // 2 grid points, each with block_size=2
        let total_rounds = p3_util::log2_strict_usize(fractions.len());

        let mut prover_transcript = DuplexSponge::default();
        let (frac_proof, xi) =
            fractional_sumcheck(&mut prover_transcript, &fractions, true, n_grid);

        assert_eq!(xi.len(), total_rounds);
        assert_eq!(frac_proof.grid_claims.len(), 2);

        let gkr_proof = gkr_proof_from_frac(&frac_proof);
        let mut verifier_transcript = DuplexSponge::default();
        let total_rounds_block = total_rounds - n_grid;
        let result =
            verify_gkr(&gkr_proof, &mut verifier_transcript, total_rounds_block, n_grid);
        assert!(
            result.is_ok(),
            "Grid integration (n_grid=1) failed: {:?}",
            result.err()
        );
        let (_numer_claim, denom_claim, v_xi) = result.unwrap();
        // With n_grid > 0, numer_claim is sum_g p_g * eq(xi_grid, g),
        // which is generally non-zero (individual p_g can be non-zero).
        assert_ne!(denom_claim, EF::ZERO);
        assert_eq!(v_xi, xi);
    }

    /// Integration test: prover + verifier with n_grid=2 and 8 fractions (1 block round, 4 grid points).
    #[test]
    fn test_gkr_grid_2_integration() {
        setup_tracing();
        // 8 fractions => total_rounds_full=3. With n_grid=2 => 4 grid points, block_size=2.
        let fractions = vec![
            Frac {
                p: EF::from_u64(1),
                q: EF::ONE,
            },
            Frac {
                p: -EF::from_u64(1),
                q: EF::ONE,
            },
            Frac {
                p: EF::from_u64(2),
                q: EF::from_u64(3),
            },
            Frac {
                p: -EF::from_u64(2),
                q: EF::from_u64(3),
            },
            Frac {
                p: EF::from_u64(5),
                q: EF::from_u64(7),
            },
            Frac {
                p: -EF::from_u64(5),
                q: EF::from_u64(7),
            },
            Frac {
                p: EF::from_u64(4),
                q: EF::from_u64(2),
            },
            Frac {
                p: -EF::from_u64(4),
                q: EF::from_u64(2),
            },
        ];

        let n_grid = 2;
        let total_rounds_full = p3_util::log2_strict_usize(fractions.len());

        let mut prover_transcript = DuplexSponge::default();
        let (frac_proof, xi) =
            fractional_sumcheck(&mut prover_transcript, &fractions, true, n_grid);

        assert_eq!(xi.len(), total_rounds_full);
        assert_eq!(frac_proof.grid_claims.len(), 4);

        let gkr_proof = gkr_proof_from_frac(&frac_proof);
        let mut verifier_transcript = DuplexSponge::default();
        let total_rounds_block = total_rounds_full - n_grid;
        let result =
            verify_gkr(&gkr_proof, &mut verifier_transcript, total_rounds_block, n_grid);
        assert!(
            result.is_ok(),
            "Grid integration (n_grid=2) failed: {:?}",
            result.err()
        );
        let (_numer_claim, denom_claim, v_xi) = result.unwrap();
        assert_ne!(denom_claim, EF::ZERO);
        assert_eq!(v_xi, xi);
    }

    /// Integration test: n_grid equals total_rounds_full (all rounds are grid, block has 1 element).
    #[test]
    fn test_gkr_grid_all_grid_rounds() {
        setup_tracing();
        // 4 fractions, n_grid=2 => 4 grid points, block_size=1
        // Each grid point is a single fraction => no GKR rounds needed
        let fractions = vec![
            Frac {
                p: EF::from_u64(10),
                q: EF::from_u64(3),
            },
            Frac {
                p: -EF::from_u64(10),
                q: EF::from_u64(3),
            },
            Frac {
                p: EF::from_u64(1),
                q: EF::ONE,
            },
            Frac {
                p: -EF::from_u64(1),
                q: EF::ONE,
            },
        ];

        let n_grid = 2;
        let total_rounds_full = p3_util::log2_strict_usize(fractions.len());
        assert_eq!(total_rounds_full, n_grid);

        let mut prover_transcript = DuplexSponge::default();
        let (frac_proof, xi) =
            fractional_sumcheck(&mut prover_transcript, &fractions, true, n_grid);

        assert_eq!(xi.len(), total_rounds_full);
        assert_eq!(frac_proof.grid_claims.len(), 4);
        assert!(frac_proof.claims_per_layer.is_empty());
        assert!(frac_proof.sumcheck_polys.is_empty());

        let gkr_proof = gkr_proof_from_frac(&frac_proof);
        let mut verifier_transcript = DuplexSponge::default();
        let total_rounds_block = total_rounds_full - n_grid; // == 0
        let result =
            verify_gkr(&gkr_proof, &mut verifier_transcript, total_rounds_block, n_grid);
        assert!(
            result.is_ok(),
            "Grid all-grid-rounds verification failed: {:?}",
            result.err()
        );
        let (_numer_claim, denom_claim, v_xi) = result.unwrap();
        assert_ne!(denom_claim, EF::ZERO);
        assert_eq!(v_xi, xi);
    }
}
