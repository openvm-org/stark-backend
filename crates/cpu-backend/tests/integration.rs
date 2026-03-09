use openvm_stark_backend::{
    prover::sumcheck::{sumcheck_multilinear, sumcheck_prismalinear},
    verifier::sumcheck::{verify_sumcheck_multilinear, verify_sumcheck_prismalinear},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::*;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use rand::{rngs::StdRng, Rng, SeedableRng};

type SC = BabyBearPoseidon2Config;

// ============================================================================
// Shared backend test suite via macro
// ============================================================================

type Engine =
    openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine<DuplexSponge>;
openvm_backend_tests::backend_test_suite!(Engine);

// ============================================================================
// Tests unique to cpu-backend (not in the shared backend-tests suite)
// ============================================================================

#[test]
fn test_plain_multilinear_sumcheck() -> Result<(), String> {
    let n = 15;
    let mut rng = StdRng::from_seed([228; 32]);

    let num_pts = 1 << n;
    assert!((F::ORDER_U32 - 1) % num_pts == 0);

    let evals = (0..num_pts)
        .map(|_| F::from_u32(rng.random_range(0..F::ORDER_U32)))
        .collect::<Vec<_>>();
    let mut prover_sponge = default_duplex_sponge();
    let mut verifier_sponge = default_duplex_sponge();

    let (proof, _) = sumcheck_multilinear::<SC, _, _>(&mut prover_sponge, &evals).unwrap();
    verify_sumcheck_multilinear::<SC, _>(&mut verifier_sponge, &proof)
}

#[test]
fn test_plain_prismalinear_sumcheck() -> Result<(), String> {
    openvm_stark_sdk::utils::setup_tracing();
    let n = 5;
    let l_skip = 10;
    let mut rng = StdRng::from_seed([228; 32]);

    let dim = n + l_skip;
    let num_pts = 1 << dim;
    assert!((F::ORDER_U32 - 1) % num_pts == 0);

    let evals = (0..num_pts)
        .map(|_| F::from_u32(rng.random_range(0..F::ORDER_U32)))
        .collect::<Vec<_>>();
    let mut prover_sponge = default_duplex_sponge();
    let mut verifier_sponge = default_duplex_sponge();

    let (proof, _) = sumcheck_prismalinear::<SC, _, _>(&mut prover_sponge, l_skip, &evals).unwrap();
    verify_sumcheck_prismalinear::<SC, _>(&mut verifier_sponge, l_skip, &proof)
}
