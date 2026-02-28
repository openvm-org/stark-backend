//! Verifier-only unit tests for the GKR fractional sumcheck.
//!
//! These test `verify_gkr` with hand-constructed `GkrProof` structs, covering
//! error paths (wrong layer count, wrong poly count, zero-check failure) and
//! integration paths (1/2/3 rounds, mixed fractions, empty case). No engine or
//! prover backend is involved — all verifier-side math — so they are not in the
//! shared backend test suite.

use openvm_stark_backend::{
    proof::{GkrLayerClaims, GkrProof},
    prover::fractional_sumcheck_gkr::{fractional_sumcheck, Frac},
    verifier::fractional_sumcheck_gkr::{verify_gkr, GkrVerificationError},
};
use openvm_stark_sdk::{config::baby_bear_poseidon2::*, utils::setup_tracing};
use p3_field::PrimeCharacteristicRing;

type SC = BabyBearPoseidon2Config;

#[test]
fn test_multiple_rounds_shape() {
    setup_tracing();
    let proof = GkrProof::<SC> {
        logup_pow_witness: F::ZERO,
        q0_claim: EF::ONE,
        claims_per_layer: vec![],
        sumcheck_polys: vec![],
    };

    let mut transcript = default_duplex_sponge();

    let result = verify_gkr::<SC, _>(&proof, &mut transcript, 2);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        GkrVerificationError::IncorrectLayerCount { .. }
    ));

    let layer_claims = GkrLayerClaims::<SC> {
        p_xi_0: EF::ZERO,
        p_xi_1: EF::ZERO,
        q_xi_0: EF::ONE,
        q_xi_1: EF::ONE,
    };

    let proof2 = GkrProof::<SC> {
        logup_pow_witness: F::ZERO,
        q0_claim: EF::ONE,
        claims_per_layer: vec![layer_claims.clone(), layer_claims],
        sumcheck_polys: vec![],
    };

    let mut transcript = default_duplex_sponge();
    let result = verify_gkr::<SC, _>(&proof2, &mut transcript, 2);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        GkrVerificationError::IncorrectSumcheckPolyCount { .. }
    ));
}

#[test]
fn test_gkr_base_layer_numerator_zero() {
    setup_tracing();
    let layer1_claims = GkrLayerClaims::<SC> {
        p_xi_0: EF::from_u64(1), // Non-zero
        p_xi_1: EF::from_u64(2),
        q_xi_0: EF::from_u64(3),
        q_xi_1: EF::from_u64(4),
    };

    // p0 = 1*4 + 2*3 = 10 (non-zero)
    let proof = GkrProof::<SC> {
        logup_pow_witness: F::ZERO,
        q0_claim: EF::from_u64(12), // q0 = 3*4 = 12
        claims_per_layer: vec![layer1_claims],
        sumcheck_polys: vec![],
    };

    let mut transcript = default_duplex_sponge();
    let result = verify_gkr::<SC, _>(&proof, &mut transcript, 1);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        GkrVerificationError::ZeroCheckFailed { .. }
    ));
}

#[test]
fn test_gkr_1_round_integration() -> eyre::Result<()> {
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

    let mut prover_transcript = default_duplex_sponge();
    let (frac_proof, _xi) = fractional_sumcheck::<SC, _>(&mut prover_transcript, &fractions, true)?;

    let gkr_proof = GkrProof::<SC> {
        logup_pow_witness: F::ZERO,
        q0_claim: frac_proof.fractional_sum.1,
        claims_per_layer: frac_proof.claims_per_layer,
        sumcheck_polys: frac_proof.sumcheck_polys,
    };

    let mut verifier_transcript = default_duplex_sponge();
    let total_rounds = p3_util::log2_strict_usize(fractions.len());
    let (numer_claim, denom_claim, _) =
        verify_gkr::<SC, _>(&gkr_proof, &mut verifier_transcript, total_rounds)?;
    assert_eq!(numer_claim, EF::ZERO);
    assert_ne!(denom_claim, EF::ZERO);
    Ok(())
}

#[test]
fn test_gkr_2_round_integration() -> eyre::Result<()> {
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

    let mut prover_transcript = default_duplex_sponge();
    let (frac_proof, _xi) = fractional_sumcheck::<SC, _>(&mut prover_transcript, &fractions, true)?;

    let gkr_proof = GkrProof::<SC> {
        logup_pow_witness: F::ZERO,
        q0_claim: frac_proof.fractional_sum.1,
        claims_per_layer: frac_proof.claims_per_layer,
        sumcheck_polys: frac_proof.sumcheck_polys,
    };

    let mut verifier_transcript = default_duplex_sponge();
    let total_rounds = p3_util::log2_strict_usize(fractions.len());
    let (numer_claim, denom_claim, _) =
        verify_gkr::<SC, _>(&gkr_proof, &mut verifier_transcript, total_rounds)?;
    assert_eq!(numer_claim, EF::ZERO);
    assert_ne!(denom_claim, EF::ZERO);
    Ok(())
}

#[test]
fn test_gkr_3_round_integration() -> eyre::Result<()> {
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

    let mut prover_transcript = default_duplex_sponge();
    let (frac_proof, _xi) = fractional_sumcheck::<SC, _>(&mut prover_transcript, &fractions, true)?;

    let gkr_proof = GkrProof::<SC> {
        logup_pow_witness: F::ZERO,
        q0_claim: frac_proof.fractional_sum.1,
        claims_per_layer: frac_proof.claims_per_layer,
        sumcheck_polys: frac_proof.sumcheck_polys,
    };

    let mut verifier_transcript = default_duplex_sponge();
    let total_rounds = p3_util::log2_strict_usize(fractions.len());
    let (numer_claim, denom_claim, _) =
        verify_gkr::<SC, _>(&gkr_proof, &mut verifier_transcript, total_rounds)?;
    assert_eq!(numer_claim, EF::ZERO);
    assert_ne!(denom_claim, EF::ZERO);
    Ok(())
}

#[test]
fn test_gkr_mixed_fractions() -> eyre::Result<()> {
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

    let mut prover_transcript = default_duplex_sponge();
    let (frac_proof, _xi) = fractional_sumcheck::<SC, _>(&mut prover_transcript, &fractions, true)?;

    let gkr_proof = GkrProof::<SC> {
        logup_pow_witness: F::ZERO,
        q0_claim: frac_proof.fractional_sum.1,
        claims_per_layer: frac_proof.claims_per_layer,
        sumcheck_polys: frac_proof.sumcheck_polys,
    };

    let mut verifier_transcript = default_duplex_sponge();
    let total_rounds = p3_util::log2_strict_usize(fractions.len());
    let (_numer_claim, denom_claim, _) =
        verify_gkr::<SC, _>(&gkr_proof, &mut verifier_transcript, total_rounds)?;
    assert_ne!(denom_claim, EF::ZERO);
    Ok(())
}

#[test]
fn test_gkr_empty_case() -> eyre::Result<()> {
    setup_tracing();
    let fractions = vec![];

    let mut prover_transcript = default_duplex_sponge();
    let (frac_proof, _xi) = fractional_sumcheck::<SC, _>(&mut prover_transcript, &fractions, true)?;

    let gkr_proof = GkrProof::<SC> {
        logup_pow_witness: F::ZERO,
        q0_claim: frac_proof.fractional_sum.1,
        claims_per_layer: frac_proof.claims_per_layer,
        sumcheck_polys: frac_proof.sumcheck_polys,
    };

    let mut verifier_transcript = default_duplex_sponge();
    let (numer_claim, denom_claim, gkr_r) =
        verify_gkr::<SC, _>(&gkr_proof, &mut verifier_transcript, 0)?;
    assert_eq!(numer_claim, EF::ZERO);
    assert_eq!(denom_claim, EF::ONE);
    assert_eq!(gkr_r, vec![]);
    Ok(())
}
