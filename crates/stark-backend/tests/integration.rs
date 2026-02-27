//! CPU-specific integration tests.
//!
//! These tests call CPU-only prover APIs directly (`sumcheck_multilinear`,
//! `sumcheck_prismalinear`, `StackedReductionCpu`) and are therefore not
//! engine-generic. The cuda-backend has its own GPU variants of these tests
//! with different prover APIs.
//!
//! Engine-generic tests live in the shared `openvm-backend-tests` crate and
//! are included via the `backend_test_suite!` macro above.

use itertools::Itertools;
use openvm_stark_backend::{
    prover::{
        stacked_pcs::stacked_commit,
        stacked_reduction::{prove_stacked_opening_reduction, StackedReductionCpu},
        sumcheck::{sumcheck_multilinear, sumcheck_prismalinear},
        DeviceDataTransporter, MultiRapProver,
    },
    test_utils::{default_test_params_small, FibFixture, TestFixture},
    verifier::{
        stacked_reduction::{verify_stacked_reduction, StackedReductionError},
        sumcheck::{verify_sumcheck_multilinear, verify_sumcheck_prismalinear},
    },
    StarkEngine, StarkProtocolConfig,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::*,
    utils::{setup_tracing, setup_tracing_with_log_level},
};
use p3_field::{PrimeCharacteristicRing, PrimeField32, TwoAdicField};
use rand::{rngs::StdRng, Rng, SeedableRng};
use test_case::test_case;
use tracing::{debug, Level};

type SC = BabyBearPoseidon2Config;
type Engine = BabyBearPoseidon2CpuEngine<DuplexSponge>;

// ===========================================================================
// Shared test suite (engine-generic + WHIR)
// ===========================================================================

openvm_backend_tests::backend_test_suite!(Engine);

// ===========================================================================
// CPU-specific tests (not in shared suite)
// ===========================================================================

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

    let (proof, _) = sumcheck_multilinear::<SC, _, _>(&mut prover_sponge, &evals);
    verify_sumcheck_multilinear::<SC, _>(&mut verifier_sponge, &proof)
}

#[test]
fn test_plain_prismalinear_sumcheck() -> Result<(), String> {
    setup_tracing();
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

    let (proof, _) = sumcheck_prismalinear::<SC, _, _>(&mut prover_sponge, l_skip, &evals);
    verify_sumcheck_prismalinear::<SC, _>(&mut verifier_sponge, l_skip, &proof)
}

#[test_case(9)]
#[test_case(2 ; "when log_height equals l_skip")]
#[test_case(1 ; "when log_height less than l_skip")]
#[test_case(0 ; "when log_height is zero")]
fn test_stacked_opening_reduction(
    log_trace_degree: usize,
) -> Result<(), StackedReductionError<EF>> {
    setup_tracing_with_log_level(Level::DEBUG);

    let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(default_test_params_small());
    let config = engine.config();
    let params = config.params();
    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);
    let (pk, _vk) = fib.keygen(&engine);
    let pk = engine.device().transport_pk_to_device(&pk);
    let mut ctx = fib.generate_proving_ctx();

    ctx.sort_for_stacking();

    let (_, common_main_pcs_data) = {
        stacked_commit(
            config.hasher(),
            params.l_skip,
            params.n_stack,
            params.log_blowup,
            params.k_whir(),
            &ctx.common_main_traces()
                .map(|(_, trace)| trace)
                .collect_vec(),
        )
    };

    let omega_skip = F::two_adic_generator(params.l_skip);
    let omega_skip_pows = omega_skip.powers().take(1 << params.l_skip).collect_vec();

    let device = engine.device();
    let ((_, batch_proof), r) = device
        .prove_rap_constraints(
            &mut default_duplex_sponge(),
            &pk,
            &ctx,
            &common_main_pcs_data,
        )
        .unwrap();

    let need_rot = pk.per_air[ctx.per_trace[0].0].vk.params.need_rot;
    let need_rot_per_commit = vec![vec![need_rot]];
    let (stacking_proof, _) = prove_stacked_opening_reduction::<SC, _, _, _, StackedReductionCpu<SC>>(
        device,
        &mut default_duplex_sponge(),
        params.n_stack,
        vec![&common_main_pcs_data],
        need_rot_per_commit.clone(),
        &r,
    );

    debug!(?batch_proof.column_openings);

    let u_prism = verify_stacked_reduction::<SC, _>(
        &mut default_duplex_sponge(),
        &stacking_proof,
        &[common_main_pcs_data.layout],
        &need_rot_per_commit,
        params.l_skip,
        params.n_stack,
        &batch_proof.column_openings,
        &r,
        &omega_skip_pows,
    )?;
    assert_eq!(u_prism.len(), params.n_stack + 1);
    Ok(())
}
