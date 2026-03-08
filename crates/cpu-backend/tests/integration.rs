use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

use itertools::Itertools;
use openvm_stark_backend::{
    any_air_arc_vec,
    keygen::types::LinearConstraint,
    prover::{
        sumcheck::{sumcheck_multilinear, sumcheck_prismalinear},
        AirProvingContext, DeviceDataTransporter, ProvingContext,
    },
    test_utils::{
        default_test_params_small,
        dummy_airs::{
            fib_air::air::FibonacciAir,
            fib_selector_air::air::FibonacciSelectorAir,
            interaction::dummy_interaction_air::{
                DummyInteractionAir, DummyInteractionChip, DummyInteractionData,
            },
        },
        test_system_params_small, CachedFixture11, FibFixture, InteractionsFixture11,
        MixtureFixture, PreprocessedFibFixture, SelfInteractionFixture, TestFixture,
    },
    utils::disable_debug_builder,
    verifier::{
        proof_shape::{verify_proof_shape, ProofShapeError},
        sumcheck::{verify_sumcheck_multilinear, verify_sumcheck_prismalinear},
        verify, VerifierError,
    },
    AirRef, StarkEngine, SystemParams, TranscriptHistory, WhirConfig, WhirParams,
    WhirRoundConfig,
};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::*, log_up_params::log_up_security_params_baby_bear_100_bits},
    utils::{setup_tracing, setup_tracing_with_log_level},
};
// Use our engine instead of the SDK's reference engine.
use openvm_cpu_backend::{engine::BabyBearPoseidon2CpuEngine, RowMajorMatrixWrapper};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use rand::{rngs::StdRng, Rng, SeedableRng};
use test_case::test_case;
use tracing::Level;

type SC = BabyBearPoseidon2Config;

// Type alias for our backend's AirProvingContext
type RmAirProvingContext = AirProvingContext<openvm_cpu_backend::CpuBackend<SC>>;

/// Helper: convert a RowMajorMatrix to an AirProvingContext for our CpuBackend.
fn rm_ctx(rm: &RowMajorMatrix<F>) -> RmAirProvingContext {
    AirProvingContext::simple_no_pis(RowMajorMatrixWrapper::new(rm.clone()))
}


pub fn test_engine_small() -> BabyBearPoseidon2CpuEngine<DuplexSponge> {
    setup_tracing();
    BabyBearPoseidon2CpuEngine::new(default_test_params_small())
}

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

#[test]
fn test_proof_shape_verifier() -> Result<(), ProofShapeError> {
    setup_tracing();
    let log_trace_degree = 3;

    // without interactions
    let engine = test_engine_small();
    let (vk, proof) = FibFixture::new(0, 1, 1 << log_trace_degree).keygen_and_prove(&engine);
    verify_proof_shape(&vk.inner, &proof)?;

    // with interactions
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    verify_proof_shape(&vk.inner, &proof)?;

    // with cached trace
    let params = engine.config().clone();
    let (vk, proof) = CachedFixture11::new(params).keygen_and_prove(&engine);
    verify_proof_shape(&vk.inner, &proof)?;

    // with preprocessed trace
    let height = 1 << log_trace_degree;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
    let (vk, proof) = PreprocessedFibFixture::new(0, 1, sels).keygen_and_prove(&engine);
    verify_proof_shape(&vk.inner, &proof)?;

    Ok(())
}

#[test]
fn test_proof_shape_verifier_rng_system_params() -> Result<(), ProofShapeError> {
    setup_tracing();
    let mut rng = StdRng::from_seed([228; 32]);
    for _ in 0..10 {
        let l_skip = rng.random_range(1usize..=2);
        let n_stack = rng.random_range(8usize..=9);
        let k_whir = rng.random_range(1usize..=4);
        let log_blowup = rng.random_range(1usize..=3);
        let num_whir_rounds = rng.random_range(1..=2);
        let mut rounds = Vec::with_capacity(num_whir_rounds);
        for _ in 0..num_whir_rounds {
            rounds.push(WhirRoundConfig {
                num_queries: rng.random_range(1..=10),
            });
        }
        let whir = WhirConfig {
            k: k_whir,
            rounds,
            mu_pow_bits: 1,
            query_phase_pow_bits: 1,
            folding_pow_bits: 1,
        };
        let params = SystemParams {
            l_skip,
            n_stack,
            log_blowup,
            whir,
            logup: log_up_security_params_baby_bear_100_bits(),
            max_constraint_degree: 3,
        };
        let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
        let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
        verify_proof_shape(&vk.inner, &proof)?;
    }
    Ok(())
}

#[test_case(2, 10)]
#[test_case(2, 1; "where log_trace_degree=1 less than l_skip=2")]
#[test_case(2, 0; "where log_trace_degree=0 less than l_skip=2")]
#[test_case(3, 2; "where log_trace_degree=2 less than l_skip=3")]
fn test_fib_air_roundtrip(l_skip: usize, log_trace_degree: usize) -> Result<(), VerifierError<EF>> {
    setup_tracing_with_log_level(Level::DEBUG);

    let n_stack = 8;
    let k_whir = 4;
    let whir_params = WhirParams {
        k: k_whir,
        log_final_poly_len: k_whir,
        query_phase_pow_bits: 1,
    };
    let log_blowup = 1;
    let whir = WhirConfig::new(log_blowup, l_skip + n_stack, whir_params, 80);
    let params = SystemParams {
        l_skip,
        n_stack,
        log_blowup,
        whir,
        logup: log_up_security_params_baby_bear_100_bits(),
        max_constraint_degree: 3,
    };
    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);

    let engine = BabyBearPoseidon2CpuEngine::new(params);
    let (pk, vk) = fib.keygen(&engine);
    let mut recorder = default_duplex_sponge_recorder();
    let proof = fib.prove_from_transcript(&engine, &pk, &mut recorder);

    let mut validator_sponge =
        DuplexSpongeValidator::new(poseidon2_perm().clone(), recorder.into_log());
    verify(engine.config(), &vk, &proof, &mut validator_sponge)
}

#[test_case(2, 8, 3)]
#[test_case(5, 5, 4)]
fn test_dummy_interactions_roundtrip(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
) -> Result<(), VerifierError<EF>> {
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let engine = BabyBearPoseidon2CpuEngine::new(params);
    let fx = InteractionsFixture11;
    let (pk, vk) = fx.keygen(&engine);

    let mut recorder = default_duplex_sponge_recorder();
    let proof = fx.prove_from_transcript(&engine, &pk, &mut recorder);

    let mut validator_sponge = default_duplex_sponge_validator(recorder.into_log());
    verify(engine.config(), &vk, &proof, &mut validator_sponge)
}

#[test_case(2, 8, 3)]
#[test_case(5, 5, 4)]
#[test_case(5, 8, 3)]
fn test_cached_trace_roundtrip(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
) -> Result<(), VerifierError<EF>> {
    setup_tracing_with_log_level(Level::DEBUG);
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let engine = BabyBearPoseidon2CpuEngine::new(params);
    let fx = CachedFixture11::new(engine.config().clone());
    let (pk, vk) = fx.keygen(&engine);

    let mut recorder = default_duplex_sponge_recorder();
    let proof = fx.prove_from_transcript(&engine, &pk, &mut recorder);

    let mut validator_sponge = default_duplex_sponge_validator(recorder.into_log());
    verify(engine.config(), &vk, &proof, &mut validator_sponge)
}

#[test_case(2, 8, 3)]
#[test_case(5, 5, 4)]
fn test_preprocessed_trace_roundtrip(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
) -> Result<(), VerifierError<EF>> {
    use itertools::Itertools;
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let engine = BabyBearPoseidon2CpuEngine::new(params);
    let log_trace_degree = 8;
    let height = 1 << log_trace_degree;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
    let fx = PreprocessedFibFixture::new(0, 1, sels);
    let (pk, vk) = fx.keygen(&engine);

    let mut recorder = default_duplex_sponge_recorder();
    let proof = fx.prove_from_transcript(&engine, &pk, &mut recorder);

    let mut validator_sponge = default_duplex_sponge_validator(recorder.into_log());
    verify(engine.config(), &vk, &proof, &mut validator_sponge)
}

#[test]
fn test_interactions_single_sender_receiver_happy() {
    setup_tracing();

    let engine = test_engine_small();
    let fx = InteractionsFixture11;
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof).unwrap();
}

#[test]
fn test_single_cached_trace_stark() {
    setup_tracing();
    let engine = test_engine_small();
    let fx = CachedFixture11::new(engine.config().clone());
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof).unwrap();
}

#[test_case(10 ; "when log_height equals n_stack l_skip")]
#[test_case(3 ; "when log_height greater than l_skip")]
#[test_case(2 ; "when log_height equals l_skip")]
#[test_case(1 ; "when log_height less than l_skip")]
#[test_case(0 ; "when log_height is zero")]
fn test_single_preprocessed_trace_stark(log_trace_degree: usize) {
    setup_tracing();
    let engine = test_engine_small();
    let height = 1 << log_trace_degree;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
    let fx = PreprocessedFibFixture::new(0, 1, sels);
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof).unwrap();
}

#[test_case(10 ; "when log_height equals n_stack l_skip")]
#[test_case(3 ; "when log_height greater than l_skip")]
#[test_case(2 ; "when log_height equals l_skip")]
#[test_case(1 ; "when log_height less than l_skip")]
#[test_case(0 ; "when log_height is zero")]
fn test_multi_interaction_traces_stark(log_trace_degree: usize) {
    setup_tracing();
    let engine = test_engine_small();
    let fx = SelfInteractionFixture {
        widths: vec![4, 7, 8, 8, 10, 100],
        log_height: log_trace_degree,
        bus_index: 4,
    };
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof).unwrap();
}

#[test_case(10 ; "when log_height equals n_stack l_skip")]
#[test_case(3 ; "when log_height greater than l_skip")]
#[test_case(2 ; "when log_height equals l_skip")]
#[test_case(1 ; "when log_height less than l_skip")]
#[test_case(0 ; "when log_height is zero")]
fn test_mixture_traces_stark(log_trace_degree: usize) {
    setup_tracing();
    let engine = test_engine_small();
    let fx = MixtureFixture::standard(log_trace_degree, engine.config().clone());
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof).unwrap();
}

#[test]
fn test_matrix_stacking_overflow() {
    setup_tracing();
    let params = test_system_params_small(3, 5, 3);
    let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
    let fx = SelfInteractionFixture {
        widths: vec![4, 7, 8, 8, 10],
        log_height: 1,
        bus_index: 4,
    };
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof).unwrap();
}

#[test]
fn test_vkey_methods() {
    setup_tracing();

    let engine = test_engine_small();
    let fib_air = FibonacciAir;
    let send_air = DummyInteractionAir::new(1, true, 0);
    let recv_air = DummyInteractionAir::new(1, false, 0);

    let airs = any_air_arc_vec![fib_air, send_air, recv_air];
    let (_pk, vk) = engine.keygen(&airs);

    assert_eq!(vk.inner.per_air.len(), 3);
    assert_eq!(vk.inner.per_air[0].params.width.main_width(), 2);
    assert_eq!(vk.inner.per_air[1].params.width.main_width(), 2);
    assert_eq!(vk.inner.per_air[2].params.width.main_width(), 2);
    assert_eq!(vk.inner.per_air[0].num_interactions(), 0);
    assert_eq!(vk.inner.per_air[1].num_interactions(), 1);
    assert_eq!(vk.inner.per_air[2].num_interactions(), 1);
}

#[test]
fn test_interaction_trace_height_constraints() {
    let log_trace_degree = 3;
    let n = 1usize << log_trace_degree;
    let sels: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let fib_air = FibonacciSelectorAir::new(sels, true);
    let mut sender_air = DummyInteractionAir::new(1, true, 0);
    sender_air.count_weight = 3;
    let mut sender_air_2 = DummyInteractionAir::new(1, true, 0);
    sender_air_2.count_weight = 1;
    let mut sender_air_3 = DummyInteractionAir::new(1, true, 1);
    sender_air_3.count_weight = 7;

    let engine = test_engine_small();
    let airs: Vec<AirRef<SC>> = vec![
        Arc::new(fib_air),
        Arc::new(sender_air),
        Arc::new(sender_air_2),
        Arc::new(sender_air_3),
    ];
    let (_pk, vk) = engine.keygen(&airs);

    assert_eq!(vk.inner.trace_height_constraints.len(), 3);
    assert_eq!(
        vk.inner.trace_height_constraints[0],
        LinearConstraint {
            coefficients: vec![0, 3, 1, 0],
            threshold: F::ORDER_U32,
        }
    );
    assert_eq!(
        vk.inner.trace_height_constraints[1],
        LinearConstraint {
            coefficients: vec![0, 0, 0, 7],
            threshold: F::ORDER_U32,
        }
    );
    assert_eq!(
        vk.inner.trace_height_constraints[2],
        LinearConstraint {
            coefficients: vec![1, 1, 1, 1],
            threshold: engine.params().logup.max_interaction_count,
        }
    );
}

#[test]
fn test_trace_height_constraints_implied_removal() {
    let log_trace_degree = 3;
    let n = 1usize << log_trace_degree;
    let sels: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let fib_air = FibonacciSelectorAir::new(sels, true);
    let sender_air = DummyInteractionAir::new(1, true, 0);
    let sender_air_2 = DummyInteractionAir::new(1, true, 1);

    let engine = test_engine_small();
    let airs: Vec<AirRef<SC>> = vec![
        Arc::new(fib_air),
        Arc::new(sender_air),
        Arc::new(sender_air_2),
    ];
    let (_pk, vk) = engine.keygen(&airs);

    assert_eq!(vk.inner.trace_height_constraints.len(), 1);
    assert_eq!(
        vk.inner.trace_height_constraints[0],
        LinearConstraint {
            coefficients: vec![1, 1, 1],
            threshold: engine.params().logup.max_interaction_count,
        }
    );
}

#[test]
fn test_interaction_multi_rows_neg() {
    setup_tracing();

    let sender_trace = RowMajorMatrix::new(
        [0, 1, 3, 5, 7, 4, 546, 0]
            .into_iter()
            .map(F::from_usize)
            .collect(),
        2,
    );
    let sender_air = DummyInteractionAir::new(1, true, 0);

    let receiver_trace = RowMajorMatrix::new(
        [1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 0, 0, 0, 0, 456]
            .into_iter()
            .map(F::from_usize)
            .collect(),
        2,
    );
    let receiver_air = DummyInteractionAir::new(1, false, 0);

    disable_debug_builder();
    let engine = test_engine_small();
    let result = catch_unwind(AssertUnwindSafe(|| {
        engine.run_test(
            any_air_arc_vec![sender_air, receiver_air],
            vec![rm_ctx(&sender_trace), rm_ctx(&receiver_trace)],
        )
    }));
    assert!(result.is_err() || result.unwrap().is_err());
}

#[test]
fn test_interaction_all_zero_sender() {
    setup_tracing();

    let sender_trace = RowMajorMatrix::new(
        [0, 1, 0, 5, 0, 4, 0, 889]
            .into_iter()
            .map(F::from_usize)
            .collect(),
        2,
    );
    let sender_air = DummyInteractionAir::new(1, true, 0);

    let engine = test_engine_small();
    engine
        .run_test(
            any_air_arc_vec![sender_air],
            vec![rm_ctx(&sender_trace)],
        )
        .expect("Verification failed");
}

#[test]
fn test_interaction_multi_senders() {
    setup_tracing();

    let sender_trace1 = RowMajorMatrix::new(
        [0, 1, 3, 5, 6, 4, 333, 889]
            .into_iter()
            .map(F::from_usize)
            .collect(),
        2,
    );
    let sender_trace2 =
        RowMajorMatrix::new([1, 4, 213, 889].into_iter().map(F::from_usize).collect(), 2);
    let sender_air = DummyInteractionAir::new(1, true, 0);

    let receiver_trace = RowMajorMatrix::new(
        [1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 889, 1, 889, 0, 456]
            .into_iter()
            .map(F::from_usize)
            .collect(),
        2,
    );
    let receiver_air = DummyInteractionAir::new(1, false, 0);

    let engine = test_engine_small();
    engine
        .run_test(
            any_air_arc_vec![sender_air, sender_air, receiver_air],
            vec![
                rm_ctx(&sender_trace1),
                rm_ctx(&sender_trace2),
                rm_ctx(&receiver_trace),
            ],
        )
        .expect("Verification failed");
}

#[test]
fn test_interaction_multi_senders_neg() {
    setup_tracing();

    let sender_trace1 = RowMajorMatrix::new(
        [0, 1, 3, 5, 5, 4, 333, 889]
            .into_iter()
            .map(F::from_usize)
            .collect(),
        2,
    );
    let sender_trace2 =
        RowMajorMatrix::new([1, 4, 213, 889].into_iter().map(F::from_usize).collect(), 2);
    let sender_air = DummyInteractionAir::new(1, true, 0);

    let receiver_trace = RowMajorMatrix::new(
        [1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 889, 1, 889, 0, 456]
            .into_iter()
            .map(F::from_usize)
            .collect(),
        2,
    );
    let receiver_air = DummyInteractionAir::new(1, false, 0);

    disable_debug_builder();
    let engine = test_engine_small();
    let result = catch_unwind(AssertUnwindSafe(|| {
        engine.run_test(
            any_air_arc_vec![sender_air, sender_air, receiver_air],
            vec![
                rm_ctx(&sender_trace1),
                rm_ctx(&sender_trace2),
                rm_ctx(&receiver_trace),
            ],
        )
    }));
    assert!(result.is_err() || result.unwrap().is_err());
}

#[test]
fn test_interaction_multi_sender_receiver() {
    setup_tracing();

    let sender_trace1 = RowMajorMatrix::new(
        [0, 1, 3, 5, 6, 4, 333, 889]
            .into_iter()
            .map(F::from_usize)
            .collect(),
        2,
    );
    let sender_trace2 =
        RowMajorMatrix::new([1, 4, 213, 889].into_iter().map(F::from_usize).collect(), 2);
    let sender_air = DummyInteractionAir::new(1, true, 0);

    let receiver_trace1 = RowMajorMatrix::new(
        [1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 889, 0, 289, 0, 456]
            .into_iter()
            .map(F::from_usize)
            .collect(),
        2,
    );
    let receiver_trace2 = RowMajorMatrix::new([1, 889].into_iter().map(F::from_usize).collect(), 2);
    let receiver_air = DummyInteractionAir::new(1, false, 0);

    let engine = test_engine_small();
    engine
        .run_test(
            any_air_arc_vec![sender_air, sender_air, receiver_air, receiver_air],
            vec![
                rm_ctx(&sender_trace1),
                rm_ctx(&sender_trace2),
                rm_ctx(&receiver_trace1),
                rm_ctx(&receiver_trace2),
            ],
        )
        .expect("Verification failed");
}

#[test]
fn test_interaction_cached_trace_neg() {
    setup_tracing();

    let engine = test_engine_small();
    let config = engine.config().clone();

    let mut sender_chip: DummyInteractionChip<SC> =
        DummyInteractionChip::new_without_partition(2, true, 0);
    sender_chip.load_data(DummyInteractionData {
        count: vec![0, 7, 3, 546],
        fields: vec![vec![1, 1], vec![4, 2], vec![5, 1], vec![889, 4]],
    });

    let mut receiver_chip: DummyInteractionChip<SC> =
        DummyInteractionChip::new_with_partition(config, 2, false, 0);
    receiver_chip.load_data(DummyInteractionData {
        count: vec![1, 3, 4, 2, 0, 545, 1, 0],
        fields: vec![
            vec![5, 1],
            vec![4, 2],
            vec![4, 2],
            vec![5, 1],
            vec![123, 3],
            vec![889, 4],
            vec![889, 10],
            vec![456, 5],
        ],
    });

    // These chips generate AirProvingContext<ReferenceBackend> via generate_proving_ctx().
    // The TestFixture machinery (keygen_and_prove) handles conversion via transport_proving_ctx_to_device.
    // So we use the TestFixture approach: build airs + proving ctx, then keygen_and_prove.
    let airs = vec![receiver_chip.air(), sender_chip.air()];

    // Build a ProvingContext<ReferenceBackend> and use engine.prove() via TestFixture path
    let (pk, _vk) = engine.keygen(&airs);
    let d_pk = engine.device().transport_pk_to_device(&pk);
    let recv_ctx = receiver_chip.generate_proving_ctx();
    let send_ctx = sender_chip.generate_proving_ctx();
    let ctx = ProvingContext::new(vec![(0, recv_ctx), (1, send_ctx)]);
    let d_ctx = engine.device().transport_proving_ctx_to_device(&ctx);

    disable_debug_builder();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let proof = engine.prove(&d_pk, d_ctx);
        engine.verify(&pk.get_vk(), &proof)
    }));
    assert!(result.is_err() || result.unwrap().is_err());
}

#[test]
fn test_optional_air() {
    setup_tracing();

    let engine = test_engine_small();
    let config = engine.config().clone();

    let fib_air = Arc::new(FibonacciAir) as AirRef<SC>;
    let send_chip1: DummyInteractionChip<SC> =
        DummyInteractionChip::new_without_partition(1, true, 0);
    let send_chip2: DummyInteractionChip<SC> =
        DummyInteractionChip::new_with_partition(config.clone(), 1, true, 0);
    let recv_chip1: DummyInteractionChip<SC> =
        DummyInteractionChip::new_without_partition(1, false, 0);

    let airs = vec![
        fib_air,
        send_chip1.air(),
        send_chip2.air(),
        recv_chip1.air(),
    ];
    let (pk, _vk) = engine.keygen(&airs);
    let d_pk = engine.device().transport_pk_to_device(&pk);

    // Case 1: All AIRs are present.
    {
        let fib = FibFixture::new(0, 1, 8);
        let fib_ctx = fib.generate_proving_ctx();

        let mut s1: DummyInteractionChip<SC> =
            DummyInteractionChip::new_without_partition(1, true, 0);
        s1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        let mut s2: DummyInteractionChip<SC> =
            DummyInteractionChip::new_with_partition(config.clone(), 1, true, 0);
        s2.load_data(DummyInteractionData {
            count: vec![1, 2, 8],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        let mut r1: DummyInteractionChip<SC> =
            DummyInteractionChip::new_without_partition(1, false, 0);
        r1.load_data(DummyInteractionData {
            count: vec![2, 4, 12],
            fields: vec![vec![1], vec![2], vec![3]],
        });

        // Build ProvingContext<ReferenceBackend> then transport
        let fib_air_ctx = fib_ctx.per_trace.into_iter().next().unwrap().1;
        let ref_ctx = ProvingContext::new(vec![
            (0, fib_air_ctx),
            (1, s1.generate_proving_ctx()),
            (2, s2.generate_proving_ctx()),
            (3, r1.generate_proving_ctx()),
        ]);
        let d_ctx = engine.device().transport_proving_ctx_to_device(&ref_ctx);
        let proof = engine.prove(&d_pk, d_ctx);
        engine.verify(&pk.get_vk(), &proof).unwrap();
    }

    // Case 2: Only send_chip1 and recv_chip1 present (fib and send_chip2 omitted).
    {
        let mut s1: DummyInteractionChip<SC> =
            DummyInteractionChip::new_without_partition(1, true, 0);
        s1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        let mut r1: DummyInteractionChip<SC> =
            DummyInteractionChip::new_without_partition(1, false, 0);
        r1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });

        let ref_ctx = ProvingContext::new(vec![
            (1, s1.generate_proving_ctx()),
            (3, r1.generate_proving_ctx()),
        ]);
        let d_ctx = engine.device().transport_proving_ctx_to_device(&ref_ctx);
        let proof = engine.prove(&d_pk, d_ctx);
        engine.verify(&pk.get_vk(), &proof).unwrap();
    }

    // Case 3: Negative - unbalanced interactions.
    {
        disable_debug_builder();
        let mut r1: DummyInteractionChip<SC> =
            DummyInteractionChip::new_without_partition(1, false, 0);
        r1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });

        let d_pk = &d_pk;
        let pk = &pk;
        let engine = &engine;
        let result = catch_unwind(AssertUnwindSafe(|| {
            let ref_ctx = ProvingContext::new(vec![(3, r1.generate_proving_ctx())]);
            let d_ctx = engine.device().transport_proving_ctx_to_device(&ref_ctx);
            let proof = engine.prove(d_pk, d_ctx);
            engine.verify(&pk.get_vk(), &proof)
        }));
        assert!(result.is_err() || result.unwrap().is_err());
    }
}
