//! Shared test suite for STARK backend implementations.
//!
//! This crate provides generic test functions that can be used to verify any
//! [`StarkEngine`] implementation against the `BabyBearPoseidon2` configuration.
//!
//! # Usage
//!
//! The easiest way to use this crate is via the [`backend_test_suite!`] macro,
//! which generates the complete shared test suite with a single invocation:
//!
//! ```ignore
//! openvm_backend_tests::backend_test_suite!(MyEngine);
//! ```
//!
//! Individual test functions can also be called directly for custom wrappers:
//!
//! ```ignore
//! #[test]
//! fn test_proof_shape_verifier() {
//!     openvm_backend_tests::proof_shape_verifier::<MyEngine>().unwrap();
//! }
//! ```
//!
//! # Adding New Tests
//!
//! 1. Add a `pub fn` here (with `E: StarkEngine<SC = SC>` bound for engine-generic tests).
//! 2. Add the test to the [`backend_test_suite!`] macro.
//! 3. Each backend automatically picks it up on next build.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

use itertools::Itertools;
use openvm_stark_backend::{
    any_air_arc_vec,
    duplex_sponge::DuplexSpongeValidator,
    keygen::types::LinearConstraint,
    poly_common::Squarable,
    prover::{
        poly::Ple, stacked_pcs::stacked_commit, whir::prove_whir_opening, AirProvingContext,
        ColMajorMatrix, CpuBackend, DeviceDataTransporter, DeviceMultiStarkProvingKey,
        MatrixDimensions, ProvingContext,
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
        prove_up_to_batch_constraints, test_system_params_small, test_whir_config_small,
        CachedFixture11, FibFixture, InteractionsFixture11, MixtureFixture,
        PreprocessedAndCachedFixture, PreprocessedFibFixture, SelfInteractionFixture, TestFixture,
    },
    utils::disable_debug_builder,
    verifier::{
        batch_constraints::verify_zerocheck_and_logup,
        fractional_sumcheck_gkr::verify_gkr,
        proof_shape::verify_proof_shape,
        verify,
        whir::{binary_k_fold, verify_whir, VerifyWhirError},
    },
    AirRef, FiatShamirTranscript, StarkEngine, StarkProtocolConfig, SystemParams,
    TranscriptHistory, WhirConfig, WhirParams, WhirRoundConfig,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{
            default_duplex_sponge, default_duplex_sponge_recorder, poseidon2_perm,
            BabyBearPoseidon2Config, BabyBearPoseidon2CpuEngine, DuplexSponge, EF, F,
        },
        log_up_params::log_up_security_params_baby_bear_100_bits,
    },
    utils::{setup_tracing, setup_tracing_with_log_level},
};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField32, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tracing::{debug, Level};

pub type SC = BabyBearPoseidon2Config;

// ---------------------------------------------------------------------------
// Helper: run_test equivalent that handles device transport generically
// ---------------------------------------------------------------------------

/// Generic version of `StarkEngine::run_test` that accepts CPU-side contexts
/// and handles device transport automatically.
pub fn run_test_on_cpu_ctx<E: StarkEngine<SC = SC>>(
    engine: &E,
    airs: Vec<AirRef<SC>>,
    ctxs: Vec<AirProvingContext<CpuBackend<SC>>>,
) -> eyre::Result<()> {
    let cpu_ctx = ProvingContext::new(ctxs.into_iter().enumerate().collect());
    let d_ctx = engine.device().transport_proving_ctx_to_device(&cpu_ctx);
    engine.run_test(airs, d_ctx.per_trace.into_iter().map(|(_, c)| c).collect())?;
    Ok(())
}

// ===========================================================================
// 1. Proof shape verification
// ===========================================================================

pub fn proof_shape_verifier<E: StarkEngine<SC = SC>>() -> eyre::Result<()> {
    setup_tracing();
    let log_trace_degree = 3;

    let engine = E::new(default_test_params_small());

    // without interactions
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

pub fn proof_shape_verifier_rng_system_params<E: StarkEngine<SC = SC>>() -> eyre::Result<()> {
    setup_tracing();
    let mut rng = StdRng::from_seed([228; 32]);
    let w_stack = 16;
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
            w_stack,
            log_blowup,
            whir,
            logup: log_up_security_params_baby_bear_100_bits(),
            max_constraint_degree: 3,
        };
        let engine = E::new(params);
        let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
        verify_proof_shape(&vk.inner, &proof)?;
    }
    Ok(())
}

// ===========================================================================
// 2. Simple end-to-end prove + verify
// ===========================================================================

pub fn interactions_single_sender_receiver_happy<E: StarkEngine<SC = SC>>() -> eyre::Result<()> {
    setup_tracing();
    let engine = E::new(default_test_params_small());
    let fx = InteractionsFixture11;
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof)?;
    Ok(())
}

pub fn single_cached_trace_stark<E: StarkEngine<SC = SC>>() -> eyre::Result<()> {
    setup_tracing();
    let engine = E::new(default_test_params_small());
    let fx = CachedFixture11::new(engine.config().clone());
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof)?;
    Ok(())
}

pub fn single_preprocessed_trace_stark<E: StarkEngine<SC = SC>>(
    log_trace_degree: usize,
) -> eyre::Result<()> {
    setup_tracing();
    let engine = E::new(default_test_params_small());
    let height = 1 << log_trace_degree;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
    let fx = PreprocessedFibFixture::new(0, 1, sels);
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof)?;
    Ok(())
}

pub fn multi_interaction_traces_stark<E: StarkEngine<SC = SC>>(
    log_trace_degree: usize,
) -> eyre::Result<()> {
    setup_tracing();
    let engine = E::new(default_test_params_small());
    let fx = SelfInteractionFixture {
        widths: vec![4, 7, 8, 8, 10, 100],
        log_height: log_trace_degree,
        bus_index: 4,
    };
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof)?;
    Ok(())
}

pub fn mixture_traces_stark<E: StarkEngine<SC = SC>>(log_trace_degree: usize) -> eyre::Result<()> {
    setup_tracing();
    let engine = E::new(default_test_params_small());
    let fx = MixtureFixture::standard(log_trace_degree, engine.config().clone());
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof)?;
    Ok(())
}

pub fn matrix_stacking_overflow<E: StarkEngine<SC = SC>>() -> eyre::Result<()> {
    setup_tracing();
    let params = test_system_params_small(3, 5, 3);
    let engine = E::new(params);
    let fx = SelfInteractionFixture {
        widths: vec![4, 7, 8, 8, 10],
        log_height: 1,
        bus_index: 4,
    };
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof)?;
    Ok(())
}

// ===========================================================================
// 3. Roundtrip tests (custom system params)
// ===========================================================================

pub fn fib_air_roundtrip<E: StarkEngine<SC = SC>>(
    l_skip: usize,
    log_trace_degree: usize,
) -> eyre::Result<()> {
    setup_tracing_with_log_level(Level::DEBUG);

    let n_stack = 8;
    let w_stack = 8;
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
        w_stack,
        log_blowup,
        whir,
        logup: log_up_security_params_baby_bear_100_bits(),
        max_constraint_degree: 3,
    };
    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);

    let engine = E::new(params);
    let (pk, vk) = fib.keygen(&engine);
    let mut prover_transcript = engine.initial_transcript();
    let proof = fib.prove_from_transcript(&engine, &pk, &mut prover_transcript);

    let mut verifier_sponge = default_duplex_sponge();
    verify(engine.config(), &vk, &proof, &mut verifier_sponge)?;
    Ok(())
}

pub fn dummy_interactions_roundtrip<E: StarkEngine<SC = SC>>(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
) -> eyre::Result<()> {
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let engine = E::new(params);
    let fx = InteractionsFixture11;
    let (pk, vk) = fx.keygen(&engine);

    let mut prover_transcript = engine.initial_transcript();
    let proof = fx.prove_from_transcript(&engine, &pk, &mut prover_transcript);

    let mut verifier_sponge = default_duplex_sponge();
    verify(engine.config(), &vk, &proof, &mut verifier_sponge)?;
    Ok(())
}

pub fn cached_trace_roundtrip<E: StarkEngine<SC = SC>>(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
) -> eyre::Result<()> {
    setup_tracing_with_log_level(Level::DEBUG);
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let engine = E::new(params);
    let fx = CachedFixture11::new(engine.config().clone());
    let (pk, vk) = fx.keygen(&engine);

    let mut prover_transcript = engine.initial_transcript();
    let proof = fx.prove_from_transcript(&engine, &pk, &mut prover_transcript);

    let mut verifier_sponge = default_duplex_sponge();
    verify(engine.config(), &vk, &proof, &mut verifier_sponge)?;
    Ok(())
}

pub fn preprocessed_trace_roundtrip<E: StarkEngine<SC = SC>>(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
) -> eyre::Result<()> {
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let engine = E::new(params);
    let log_trace_degree = 8;
    let height = 1 << log_trace_degree;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
    let fx = PreprocessedFibFixture::new(0, 1, sels);
    let (pk, vk) = fx.keygen(&engine);

    let mut prover_transcript = engine.initial_transcript();
    let proof = fx.prove_from_transcript(&engine, &pk, &mut prover_transcript);

    let mut verifier_sponge = default_duplex_sponge();
    verify(engine.config(), &vk, &proof, &mut verifier_sponge)?;
    Ok(())
}

pub fn preprocessed_and_cached_trace_roundtrip<E: StarkEngine<SC = SC>>(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
    num_cached_parts: usize,
) -> eyre::Result<()> {
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let engine = E::new(params);
    let log_trace_degree = 8;
    let height = 1 << log_trace_degree;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
    let fx = PreprocessedAndCachedFixture::new(sels, engine.config().clone(), num_cached_parts);
    let (pk, vk) = fx.keygen(&engine);

    let mut prover_transcript = engine.initial_transcript();
    let proof = fx.prove_from_transcript(&engine, &pk, &mut prover_transcript);

    let mut verifier_sponge = default_duplex_sponge();
    verify(engine.config(), &vk, &proof, &mut verifier_sponge)?;
    Ok(())
}

// ===========================================================================
// 4. Pipeline decomposition tests
// ===========================================================================

pub fn batch_sumcheck_zero_interactions<E: StarkEngine<SC = SC>>(
    log_trace_degree: usize,
) -> eyre::Result<()> {
    setup_tracing_with_log_level(Level::DEBUG);

    let engine = E::new(default_test_params_small());
    let params = engine.params();
    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);
    let (pk, vk) = fib.keygen(&engine);
    let device = engine.device();
    let d_pk = device.transport_pk_to_device(&pk);
    let cpu_ctx = fib.generate_proving_ctx();
    let ctx = device.transport_proving_ctx_to_device(&cpu_ctx);

    let mut n_per_air: Vec<isize> = Vec::with_capacity(ctx.per_trace.len());
    for (_, trace) in ctx.common_main_traces() {
        let trace_height = trace.height();
        let log_height = log2_strict_usize(trace_height);
        let n = log_height as isize - params.l_skip as isize;
        n_per_air.push(n);
    }

    let mut prover_transcript = engine.initial_transcript();
    let mut verifier_sponge = default_duplex_sponge();

    let omega_skip = F::two_adic_generator(params.l_skip);
    let omega_skip_pows = omega_skip.powers().take(1 << params.l_skip).collect_vec();

    let pvs = vec![ctx.per_trace[0].1.public_values.clone()];
    let (partial_proof, _) =
        prove_up_to_batch_constraints(&engine, &mut prover_transcript, &d_pk, ctx);
    let (gkr_proof, batch_proof) = partial_proof.into();

    let r = verify_zerocheck_and_logup::<SC, _>(
        &mut verifier_sponge,
        &vk.inner,
        &pvs,
        &gkr_proof,
        &batch_proof,
        &[0],
        &n_per_air,
        &omega_skip_pows,
    )?;
    assert_eq!(r.len(), log_trace_degree.saturating_sub(params.l_skip) + 1);
    Ok(())
}

pub fn gkr_verify_zero_interactions<E: StarkEngine<SC = SC>>() -> eyre::Result<()> {
    setup_tracing_with_log_level(Level::DEBUG);

    let engine = E::new(default_test_params_small());
    let params = engine.params();
    let fx = InteractionsFixture11;
    let (pk, _vk) = fx.keygen(&engine);
    let device = engine.device();
    let d_pk = device.transport_pk_to_device(&pk);
    let cpu_ctx = fx.generate_proving_ctx();
    let ctx = device
        .transport_proving_ctx_to_device(&cpu_ctx)
        .into_sorted();

    let mut prover_transcript = engine.initial_transcript();
    let (partial_proof, _) =
        prove_up_to_batch_constraints(&engine, &mut prover_transcript, &d_pk, ctx);
    let (gkr_proof, _) = partial_proof.into();

    let mut transcript = default_duplex_sponge();
    assert!(FiatShamirTranscript::<SC>::check_witness(
        &mut transcript,
        params.logup.pow_bits,
        gkr_proof.logup_pow_witness
    ));
    let _alpha = FiatShamirTranscript::<SC>::sample_ext(&mut transcript);
    let _beta = FiatShamirTranscript::<SC>::sample_ext(&mut transcript);
    let total_rounds = gkr_proof.claims_per_layer.len();
    let _ = verify_gkr::<SC, _>(&gkr_proof, &mut transcript, total_rounds)?;

    Ok(())
}

pub fn batch_constraints_with_interactions<E: StarkEngine<SC = SC>>() -> eyre::Result<()> {
    setup_tracing_with_log_level(Level::DEBUG);

    let engine = E::new(default_test_params_small());
    let fx = InteractionsFixture11;
    let (pk, vk) = fx.keygen(&engine);
    let device = engine.device();
    let d_pk = device.transport_pk_to_device(&pk);
    let cpu_ctx = fx.generate_proving_ctx();
    let ctx = device
        .transport_proving_ctx_to_device(&cpu_ctx)
        .into_sorted();
    let l_skip = engine.params().l_skip;

    let mut pvs = vec![vec![]; vk.inner.per_air.len()];
    let (trace_id_to_air_ids, ns): (Vec<_>, Vec<_>) = ctx
        .per_trace
        .iter()
        .map(|(air_idx, trace_ctx)| {
            pvs[*air_idx] = trace_ctx.public_values.clone();
            (
                *air_idx,
                log2_strict_usize(trace_ctx.common_main.height()) as isize - l_skip as isize,
            )
        })
        .multiunzip();
    debug!(?trace_id_to_air_ids);
    debug!(n_per_trace = ?ns);
    let omega_pows = F::two_adic_generator(l_skip)
        .powers()
        .take(1 << l_skip)
        .collect_vec();

    let mut prover_transcript = engine.initial_transcript();
    let (partial_proof, _) =
        prove_up_to_batch_constraints(&engine, &mut prover_transcript, &d_pk, ctx);
    let (gkr_proof, batch_proof) = partial_proof.into();

    let mut verifier_transcript = default_duplex_sponge();
    verify_zerocheck_and_logup::<SC, _>(
        &mut verifier_transcript,
        &vk.inner,
        &pvs,
        &gkr_proof,
        &batch_proof,
        &trace_id_to_air_ids,
        &ns,
        &omega_pows,
    )?;
    Ok(())
}

// ===========================================================================
// 5. Custom context construction
// ===========================================================================

pub fn single_fib_and_dummy_trace_stark<E: StarkEngine<SC = SC>>(
    log_trace_degree: usize,
) -> eyre::Result<()> {
    setup_tracing();

    let engine = E::new(default_test_params_small());

    // Create interactions fixture with larger trace
    let sender_height = 2 * (1 << 3);
    let sender_trace = RowMajorMatrix::new(
        [0, 1, 3, 5, 7, 4, 546, 889]
            .into_iter()
            .cycle()
            .take(2 * sender_height)
            .map(F::from_usize)
            .collect(),
        2,
    );
    let receiver_trace = RowMajorMatrix::new(
        [1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 889, 1, 889, 0, 456]
            .into_iter()
            .cycle()
            .take(4 * sender_height)
            .map(F::from_usize)
            .collect(),
        2,
    );

    // Create fibonacci fixture with small trace
    let height = 2 * (1 << log_trace_degree);
    let fib = FibFixture::new(0, 1, height);

    // Generate AIRs and proving keys
    let fx_fixture = InteractionsFixture11;
    let fx_airs = fx_fixture.airs();
    let fib_airs = fib.airs();
    let mut combined_airs = fx_airs;
    combined_airs.extend(fib_airs);
    let (combined_pk, _combined_vk) = engine.keygen(&combined_airs);
    let device = engine.device();
    let combined_pk = device.transport_pk_to_device(&combined_pk);

    // Generate custom contexts for interactions with modified traces
    let mut per_trace: Vec<_> = [sender_trace, receiver_trace]
        .into_iter()
        .enumerate()
        .map(|(air_idx, trace)| {
            (
                air_idx,
                AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&trace)),
            )
        })
        .collect();
    let fib_ctx = fib.generate_proving_ctx().per_trace.pop().unwrap().1;

    // Combine contexts
    per_trace.push((per_trace.len(), fib_ctx));
    let cpu_ctx = ProvingContext::new(per_trace);
    let combined_ctx = device
        .transport_proving_ctx_to_device(&cpu_ctx)
        .into_sorted();

    let proof = engine.prove(&combined_pk, combined_ctx);
    engine.verify(&combined_pk.get_vk(), &proof)?;
    Ok(())
}

// ===========================================================================
// 6. Interaction tests (positive and negative)
// ===========================================================================

pub fn optional_air<E: StarkEngine<SC = SC>>() -> eyre::Result<()> {
    setup_tracing();

    let engine = E::new(default_test_params_small());
    let config = engine.config().clone();
    let device = engine.device();

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
    let d_pk = device.transport_pk_to_device(&pk);

    // Case 1: All AIRs are present.
    {
        let fib = FibFixture::new(0, 1, 8);
        let fib_air_ctx = fib
            .generate_proving_ctx()
            .per_trace
            .into_iter()
            .next()
            .unwrap()
            .1;

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

        let cpu_ctx = ProvingContext::new(vec![
            (0, fib_air_ctx),
            (1, s1.generate_proving_ctx()),
            (2, s2.generate_proving_ctx()),
            (3, r1.generate_proving_ctx()),
        ]);
        let d_ctx = device.transport_proving_ctx_to_device(&cpu_ctx);
        let proof = engine.prove(&d_pk, d_ctx);
        engine.verify(&pk.get_vk(), &proof)?;
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

        let cpu_ctx = ProvingContext::new(vec![
            (1, s1.generate_proving_ctx()),
            (3, r1.generate_proving_ctx()),
        ]);
        let d_ctx = device.transport_proving_ctx_to_device(&cpu_ctx);
        let proof = engine.prove(&d_pk, d_ctx);
        engine.verify(&pk.get_vk(), &proof)?;
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
        let device = engine.device();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let cpu_ctx = ProvingContext::new(vec![(3, r1.generate_proving_ctx())]);
            let d_ctx = device.transport_proving_ctx_to_device(&cpu_ctx);
            let proof = engine.prove(d_pk, d_ctx);
            engine.verify(&pk.get_vk(), &proof)
        }));
        assert!(result.is_err() || result.unwrap().is_err());
    }
    Ok(())
}

pub fn vkey_methods<E: StarkEngine<SC = SC>>() {
    setup_tracing();

    let engine = E::new(default_test_params_small());
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

pub fn interaction_trace_height_constraints<E: StarkEngine<SC = SC>>() {
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

    let engine = E::new(default_test_params_small());
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

pub fn trace_height_constraints_implied_removal<E: StarkEngine<SC = SC>>() {
    let log_trace_degree = 3;
    let n = 1usize << log_trace_degree;
    let sels: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let fib_air = FibonacciSelectorAir::new(sels, true);
    let sender_air = DummyInteractionAir::new(1, true, 0);
    let sender_air_2 = DummyInteractionAir::new(1, true, 1);

    let engine = E::new(default_test_params_small());
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

pub fn interaction_multi_rows_neg<E: StarkEngine<SC = SC>>() {
    setup_tracing();

    let sender_trace = RowMajorMatrix::new(
        [0, 1, 3, 5, 7, 4, 546, 0]
            .into_iter()
            .map(F::from_usize)
            .collect(),
        2,
    );
    let sender_air = DummyInteractionAir::new(1, true, 0);

    // count of 0 is 545 != 546 in sender
    let receiver_trace = RowMajorMatrix::new(
        [1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 0, 0, 0, 0, 456]
            .into_iter()
            .map(F::from_usize)
            .collect(),
        2,
    );
    let receiver_air = DummyInteractionAir::new(1, false, 0);

    disable_debug_builder();
    let engine = E::new(default_test_params_small());
    let result = catch_unwind(AssertUnwindSafe(|| {
        run_test_on_cpu_ctx(
            &engine,
            any_air_arc_vec![sender_air, receiver_air],
            vec![
                AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&sender_trace)),
                AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&receiver_trace)),
            ],
        )
    }));
    assert!(result.is_err() || result.unwrap().is_err());
}

pub fn interaction_all_zero_sender<E: StarkEngine<SC = SC>>() -> eyre::Result<()> {
    setup_tracing();

    let sender_trace = RowMajorMatrix::new(
        [0, 1, 0, 5, 0, 4, 0, 889]
            .into_iter()
            .map(F::from_usize)
            .collect(),
        2,
    );
    let sender_air = DummyInteractionAir::new(1, true, 0);

    let engine = E::new(default_test_params_small());
    run_test_on_cpu_ctx(
        &engine,
        any_air_arc_vec![sender_air],
        vec![AirProvingContext::simple_no_pis(
            ColMajorMatrix::from_row_major(&sender_trace),
        )],
    )?;
    Ok(())
}

pub fn interaction_multi_senders<E: StarkEngine<SC = SC>>() -> eyre::Result<()> {
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

    let engine = E::new(default_test_params_small());
    run_test_on_cpu_ctx(
        &engine,
        any_air_arc_vec![sender_air, sender_air, receiver_air],
        vec![
            AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&sender_trace1)),
            AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&sender_trace2)),
            AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&receiver_trace)),
        ],
    )?;
    Ok(())
}

pub fn interaction_multi_senders_neg<E: StarkEngine<SC = SC>>() {
    setup_tracing();

    // Changed 6→5 for sender1 so sums don't balance
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
    let engine = E::new(default_test_params_small());
    let result = catch_unwind(AssertUnwindSafe(|| {
        run_test_on_cpu_ctx(
            &engine,
            any_air_arc_vec![sender_air, sender_air, receiver_air],
            vec![
                AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&sender_trace1)),
                AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&sender_trace2)),
                AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&receiver_trace)),
            ],
        )
    }));
    assert!(result.is_err() || result.unwrap().is_err());
}

pub fn interaction_multi_sender_receiver<E: StarkEngine<SC = SC>>() -> eyre::Result<()> {
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

    let engine = E::new(default_test_params_small());
    run_test_on_cpu_ctx(
        &engine,
        any_air_arc_vec![sender_air, sender_air, receiver_air, receiver_air],
        vec![
            AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&sender_trace1)),
            AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&sender_trace2)),
            AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&receiver_trace1)),
            AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&receiver_trace2)),
        ],
    )?;
    Ok(())
}

pub fn interaction_cached_trace_neg<E: StarkEngine<SC = SC>>() {
    setup_tracing();

    let engine = E::new(default_test_params_small());
    let config = engine.config().clone();

    let mut sender_chip: DummyInteractionChip<SC> =
        DummyInteractionChip::new_without_partition(2, true, 0);
    sender_chip.load_data(DummyInteractionData {
        count: vec![0, 7, 3, 546],
        fields: vec![vec![1, 1], vec![4, 2], vec![5, 1], vec![889, 4]],
    });

    // field [889, 4] has count 545 != 546 in sender
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
            vec![889, 10], // changed from [889, 4] to cause mismatch
            vec![456, 5],
        ],
    });

    let airs = vec![receiver_chip.air(), sender_chip.air()];
    let ctxs = vec![
        receiver_chip.generate_proving_ctx(),
        sender_chip.generate_proving_ctx(),
    ];

    disable_debug_builder();
    let device = engine.device();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let (pk, vk) = engine.keygen(&airs);
        let d_pk = device.transport_pk_to_device(&pk);
        let cpu_ctx = ProvingContext::new(ctxs.into_iter().enumerate().collect());
        let d_ctx = device.transport_proving_ctx_to_device(&cpu_ctx);
        let proof = engine.prove(&d_pk, d_ctx);
        engine.verify(&vk, &proof)
    }));
    assert!(result.is_err() || result.unwrap().is_err());
}

// ===========================================================================
// 7. WHIR tests (PCS-level, not engine-generic)
// ===========================================================================

/// Generate random evaluation points for WHIR tests.
pub fn generate_random_z(params: &SystemParams, rng: &mut StdRng) -> (Vec<EF>, Vec<EF>) {
    let z_prism: Vec<_> = (0..params.n_stack + 1)
        .map(|_| EF::from_u64(rng.random()))
        .collect();

    let z_cube = {
        let z_cube = z_prism[0]
            .exp_powers_of_2()
            .take(params.l_skip)
            .chain(z_prism[1..].iter().copied())
            .collect_vec();
        debug_assert_eq!(z_cube.len(), params.n_stack + params.l_skip);
        z_cube
    };

    (z_prism, z_cube)
}

/// Compute stacking openings for a single matrix at the given evaluation point.
pub fn stacking_openings_for_matrix(
    params: &SystemParams,
    z_prism: &[EF],
    matrix: &ColMajorMatrix<F>,
) -> Vec<EF> {
    matrix
        .columns()
        .map(|col| {
            Ple::from_evaluations(params.l_skip, col).eval_at_point(
                params.l_skip,
                z_prism[0],
                &z_prism[1..],
            )
        })
        .collect()
}

/// Run a CPU WHIR prove-then-verify cycle for the given proving key and context.
fn run_whir_test(
    config: &SC,
    pk: DeviceMultiStarkProvingKey<CpuBackend<SC>>,
    ctx: &ProvingContext<CpuBackend<SC>>,
) -> eyre::Result<()> {
    let params = config.params();
    let (common_main_commit, common_main_pcs_data) = {
        let traces = ctx
            .common_main_traces()
            .map(|(_, trace)| trace)
            .collect_vec();
        stacked_commit(
            config.hasher(),
            params.l_skip,
            params.n_stack,
            params.log_blowup,
            params.k_whir(),
            &traces,
        )?
    };

    let mut commits = vec![common_main_commit];
    let mut committed_mats = vec![(&common_main_pcs_data.matrix, &common_main_pcs_data.tree)];
    for (air_id, trace_ctx) in &ctx.per_trace {
        let pcs_datas = pk.per_air[*air_id]
            .preprocessed_data
            .iter()
            .chain(&trace_ctx.cached_mains);
        for cd in pcs_datas {
            let data = &cd.data;
            committed_mats.push((&data.matrix, &data.tree));
            commits.push(data.commit()?);
        }
    }

    let mut rng = StdRng::seed_from_u64(0);
    let (z_prism, z_cube) = generate_random_z(params, &mut rng);

    let mut prover_sponge = default_duplex_sponge_recorder();

    let proof = prove_whir_opening::<SC, _>(
        &mut prover_sponge,
        config.hasher(),
        params.l_skip,
        params.log_blowup,
        params.whir(),
        &committed_mats,
        &z_cube,
    )?;

    let stacking_openings = committed_mats
        .iter()
        .map(|(matrix, _)| stacking_openings_for_matrix(params, &z_prism, matrix))
        .collect_vec();

    let mut verifier_sponge =
        DuplexSpongeValidator::new(poseidon2_perm().clone(), prover_sponge.into_log());
    verify_whir::<SC, _>(
        &mut verifier_sponge,
        config,
        &proof,
        &stacking_openings,
        &commits,
        &z_cube,
    )?;
    Ok(())
}

fn run_whir_fib_test(params: SystemParams) -> eyre::Result<()> {
    let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params.clone());
    let fib = FibFixture::new(0, 1, 1 << params.log_stacked_height());
    let (pk, _vk) = fib.keygen(&engine);
    let pk = engine.device().transport_pk_to_device(&pk);
    let ctx = fib.generate_proving_ctx();
    run_whir_test(engine.config(), pk, &ctx)
}

/// Test binary k-fold operation with k=1.
pub fn fold_single() {
    let mut rng = StdRng::seed_from_u64(0);

    let a0 = EF::from_u32(rng.random());
    let a1 = EF::from_u32(rng.random());
    let alpha = EF::from_u32(rng.random());
    let x = F::from_u32(rng.random());

    let result = binary_k_fold(vec![a0, a1], &[alpha], x);
    assert_eq!(result, a0 + (alpha - x) * (a0 - a1) * x.double().inverse());
}

/// Test binary k-fold operation with k=2.
pub fn fold_double() {
    let mut rng = StdRng::seed_from_u64(0);

    let a0 = EF::from_u32(rng.random());
    let a1 = EF::from_u32(rng.random());
    let a2 = EF::from_u32(rng.random());
    let a3 = EF::from_u32(rng.random());
    let alpha0 = EF::from_u32(rng.random());
    let alpha1 = EF::from_u32(rng.random());

    let x = F::from_u32(rng.random());

    let result = binary_k_fold(vec![a0, a1, a2, a3], &[alpha0, alpha1], x);
    let tw = F::two_adic_generator(2);

    let b0 = a0 + (alpha0 - x) * (a0 - a2) * x.double().inverse();
    let b1 = a1 + (alpha0 - (tw * x)) * (a1 - a3) * (tw * x).double().inverse();
    let x2 = x.square();
    let expected = b0 + (alpha1 - x2) * (b0 - b1) * x2.double().inverse();

    assert_eq!(result, expected);
}

/// Test CPU WHIR prover + verifier with a single Fibonacci commitment under
/// varying stacking and folding parameters.
pub fn whir_single_fib(
    n_stack: usize,
    log_blowup: usize,
    k_whir: usize,
    log_final_poly_len: usize,
) -> eyre::Result<()> {
    setup_tracing_with_log_level(Level::DEBUG);
    let l_skip = 2;
    let w_stack = 8;
    let whir = test_whir_config_small(log_blowup, l_skip + n_stack, k_whir, log_final_poly_len);

    let params = SystemParams {
        l_skip,
        n_stack,
        w_stack,
        log_blowup,
        whir,
        logup: log_up_security_params_baby_bear_100_bits(),
        max_constraint_degree: 3,
    };
    run_whir_fib_test(params)
}

/// Build a [`WhirConfig`] with 2 rounds for multi-commitment WHIR tests.
pub fn whir_test_config(k_whir: usize) -> WhirConfig {
    WhirConfig {
        k: k_whir,
        rounds: vec![
            WhirRoundConfig { num_queries: 6 },
            WhirRoundConfig { num_queries: 5 },
        ],
        mu_pow_bits: 1,
        query_phase_pow_bits: 1,
        folding_pow_bits: 1,
    }
}

/// Test CPU WHIR prover + verifier with 5 randomly generated commitments.
pub fn whir_multiple_commitments() -> eyre::Result<()> {
    setup_tracing_with_log_level(Level::DEBUG);

    let mut rng = StdRng::seed_from_u64(42);

    let params = SystemParams {
        l_skip: 3,
        n_stack: 3,
        w_stack: 64,
        log_blowup: 1,
        whir: whir_test_config(2),
        logup: log_up_security_params_baby_bear_100_bits(),
        max_constraint_degree: 3,
    };
    let config = BabyBearPoseidon2Config::default_from_params(params);
    let params = config.params();

    let n_rows = 1 << (params.n_stack + params.l_skip);

    let mut matrices = vec![];
    let mut commits = vec![];
    let mut trees = vec![];

    let num_commitments = 5;
    for _ in 0..num_commitments {
        let n_cols = (rng.random::<u64>() % 10 + 3) as usize;
        let data = (0..n_rows * n_cols)
            .map(|_| F::from_u64(rng.random()))
            .collect_vec();
        let mat = ColMajorMatrix::new(data, n_cols);

        let (commit, pcs_data) = stacked_commit(
            config.hasher(),
            params.l_skip,
            params.n_stack,
            params.log_blowup,
            params.k_whir(),
            &[&mat],
        )?;

        matrices.push(mat);
        commits.push(commit);
        trees.push(pcs_data.tree);
    }

    debug_assert_eq!(matrices[0].height(), 1 << (params.n_stack + params.l_skip));

    let (z_prism, z_cube) = generate_random_z(params, &mut rng);

    let mut prover_sponge = default_duplex_sponge_recorder();

    let committed_mats = matrices.iter().zip(trees.iter()).collect_vec();
    let proof = prove_whir_opening::<SC, _>(
        &mut prover_sponge,
        config.hasher(),
        params.l_skip,
        params.log_blowup,
        params.whir(),
        &committed_mats,
        &z_cube,
    )?;

    let stacking_openings: Vec<Vec<EF>> = matrices
        .iter()
        .map(|mat| stacking_openings_for_matrix(params, &z_prism, mat))
        .collect();

    let mut verifier_sponge =
        DuplexSpongeValidator::new(poseidon2_perm().clone(), prover_sponge.into_log());
    verify_whir::<SC, _>(
        &mut verifier_sponge,
        &config,
        &proof,
        &stacking_openings,
        &commits,
        &z_cube,
    )?;
    Ok(())
}

/// Soundness test: verify that WHIR correctly rejects tampered openings.
pub fn whir_multiple_commitments_negative() {
    setup_tracing_with_log_level(Level::DEBUG);

    let mut rng = StdRng::seed_from_u64(42);

    let params = SystemParams {
        l_skip: 3,
        n_stack: 3,
        w_stack: 64,
        log_blowup: 1,
        whir: whir_test_config(2),
        logup: log_up_security_params_baby_bear_100_bits(),
        max_constraint_degree: 3,
    };
    let config = BabyBearPoseidon2Config::default_from_params(params);
    let params = config.params();

    let n_rows = 1 << (params.n_stack + params.l_skip);

    let mut matrices = vec![];
    let mut commits = vec![];
    let mut trees = vec![];

    let num_commitments = 5;
    for _ in 0..num_commitments {
        let n_cols = (rng.random::<u64>() % 10 + 3) as usize;
        let data = (0..n_rows * n_cols)
            .map(|_| F::from_u64(rng.random()))
            .collect_vec();
        let mat = ColMajorMatrix::new(data, n_cols);

        let (commit, pcs_data) = stacked_commit(
            config.hasher(),
            params.l_skip,
            params.n_stack,
            params.log_blowup,
            params.k_whir(),
            &[&mat],
        )
        .unwrap();

        matrices.push(mat);
        commits.push(commit);
        trees.push(pcs_data.tree);
    }

    debug_assert_eq!(matrices[0].height(), 1 << (params.n_stack + params.l_skip));

    let (z_prism, z_cube) = generate_random_z(params, &mut rng);

    let mut prover_sponge = default_duplex_sponge();
    let mut verifier_sponge = default_duplex_sponge();

    let committed_mats = matrices.iter().zip(trees.iter()).collect_vec();
    let proof = prove_whir_opening::<SC, _>(
        &mut prover_sponge,
        config.hasher(),
        params.l_skip,
        params.log_blowup,
        params.whir(),
        &committed_mats,
        &z_cube,
    )
    .unwrap();

    let mut stacking_openings: Vec<Vec<EF>> = matrices
        .iter()
        .map(|mat| stacking_openings_for_matrix(params, &z_prism, mat))
        .collect();

    // change an opening to test soundness
    stacking_openings[1][2] = EF::ONE;

    assert!(matches!(
        verify_whir::<SC, _>(
            &mut verifier_sponge,
            &config,
            &proof,
            &stacking_openings,
            &commits,
            &z_cube,
        ),
        Err(VerifyWhirError::FinalPolyConstraint)
    ));
}

// ===========================================================================
// Test suite macros
// ===========================================================================

/// Helper: generate `#[test]` functions for parameterized engine-generic tests.
///
/// Proc macro attributes like `#[test_case]` cannot be used inside
/// `macro_rules!` expansions — Rust's macro hygiene assigns different syntax
/// contexts to the generated tokens, causing the proc macro to emit wrapper
/// functions that fail to connect arguments to parameters. This helper
/// side-steps the issue by expanding each parameter set into its own `#[test]`
/// function via pure `macro_rules!`.
///
/// # Variants
///
/// ```ignore
/// // Function returning Result (calls .unwrap()):
/// __test_cases!($engine, fib_air_roundtrip, unwrap, {
///     test_fib_air_roundtrip(2, 10),
///     test_fib_air_roundtrip_small(2, 1),
/// });
///
/// // Function returning ():
/// __test_cases!($engine, single_preprocessed_trace_stark, {
///     test_single_preprocessed_trace_stark(10),
///     test_single_preprocessed_trace_stark_zero(0),
/// });
/// ```
#[doc(hidden)]
// Must export macro for $crate::__test_cases hygiene inside backend_test_suite!
#[macro_export]
macro_rules! __test_cases {
    // Engine-generic, returns Result (call .unwrap())
    ($engine:ty, $func:ident, unwrap, { $( $test_name:ident( $($arg:expr),* ) ),+ $(,)? }) => {
        $(
            #[test]
            fn $test_name() {
                $crate::$func::<$engine>($($arg),*).unwrap();
            }
        )+
    };
    // Engine-generic, returns ()
    ($engine:ty, $func:ident, { $( $test_name:ident( $($arg:expr),* ) ),+ $(,)? }) => {
        $(
            #[test]
            fn $test_name() {
                $crate::$func::<$engine>($($arg),*);
            }
        )+
    };
}

/// Generate the complete shared backend test suite for the given engine type.
///
/// This macro expands to `#[test]` functions covering all engine-generic tests
/// plus WHIR PCS tests. Parameterized tests use the [`__test_cases!`] helper.
///
/// # Example
///
/// ```ignore
/// use openvm_stark_sdk::config::baby_bear_poseidon2::*;
///
/// type Engine = BabyBearPoseidon2CpuEngine<DuplexSponge>;
/// openvm_backend_tests::backend_test_suite!(Engine);
/// ```
#[macro_export]
macro_rules! backend_test_suite {
    ($engine:ty) => {
        // === 1. Proof shape verification ===

        $crate::__test_cases!($engine, proof_shape_verifier, unwrap, {
            test_proof_shape_verifier(),
        });

        $crate::__test_cases!($engine, proof_shape_verifier_rng_system_params, unwrap, {
            test_proof_shape_verifier_rng_system_params(),
        });

        // === 2. Simple end-to-end ===

        $crate::__test_cases!($engine, interactions_single_sender_receiver_happy, unwrap, {
            test_interactions_single_sender_receiver_happy(),
        });

        $crate::__test_cases!($engine, single_cached_trace_stark, unwrap, {
            test_single_cached_trace_stark(),
        });

        $crate::__test_cases!($engine, single_preprocessed_trace_stark, unwrap, {
            test_single_preprocessed_trace_stark(10),
            test_single_preprocessed_trace_stark_log_height_gt_l_skip(3),
            test_single_preprocessed_trace_stark_log_height_eq_l_skip(2),
            test_single_preprocessed_trace_stark_log_height_lt_l_skip(1),
            test_single_preprocessed_trace_stark_log_height_zero(0),
        });

        $crate::__test_cases!($engine, multi_interaction_traces_stark, unwrap, {
            test_multi_interaction_traces_stark(10),
            test_multi_interaction_traces_stark_log_height_gt_l_skip(3),
            test_multi_interaction_traces_stark_log_height_eq_l_skip(2),
            test_multi_interaction_traces_stark_log_height_lt_l_skip(1),
            test_multi_interaction_traces_stark_log_height_zero(0),
        });

        $crate::__test_cases!($engine, mixture_traces_stark, unwrap, {
            test_mixture_traces_stark(10),
            test_mixture_traces_stark_log_height_gt_l_skip(3),
            test_mixture_traces_stark_log_height_eq_l_skip(2),
            test_mixture_traces_stark_log_height_lt_l_skip(1),
            test_mixture_traces_stark_log_height_zero(0),
        });

        $crate::__test_cases!($engine, matrix_stacking_overflow, unwrap, {
            test_matrix_stacking_overflow(),
        });

        // === 3. Roundtrip tests ===

        $crate::__test_cases!($engine, fib_air_roundtrip, unwrap, {
            test_fib_air_roundtrip(2, 10),
            test_fib_air_roundtrip_log_trace_degree_1_lt_l_skip_2(2, 1),
            test_fib_air_roundtrip_log_trace_degree_0_lt_l_skip_2(2, 0),
            test_fib_air_roundtrip_log_trace_degree_2_lt_l_skip_3(3, 2),
            test_fib_air_roundtrip_large_l_skip(6, 10),
        });

        $crate::__test_cases!($engine, dummy_interactions_roundtrip, unwrap, {
            test_dummy_interactions_roundtrip_2_8_3(2, 8, 3),
            test_dummy_interactions_roundtrip_5_5_4(5, 5, 4),
        });

        $crate::__test_cases!($engine, cached_trace_roundtrip, unwrap, {
            test_cached_trace_roundtrip_2_8_3(2, 8, 3),
            test_cached_trace_roundtrip_5_5_4(5, 5, 4),
            test_cached_trace_roundtrip_5_8_3(5, 8, 3),
            test_cached_trace_roundtrip_6_7_3(6, 7, 3),
        });

        $crate::__test_cases!($engine, preprocessed_trace_roundtrip, unwrap, {
            test_preprocessed_trace_roundtrip_2_8_3(2, 8, 3),
            test_preprocessed_trace_roundtrip_5_5_4(5, 5, 4),
        });

        $crate::__test_cases!($engine, preprocessed_and_cached_trace_roundtrip, unwrap, {
            test_preprocessed_and_cached_trace_roundtrip_2_8_3_1(2, 8, 3, 1),
            test_preprocessed_and_cached_trace_roundtrip_2_8_3_2(2, 8, 3, 2),
            test_preprocessed_and_cached_trace_roundtrip_2_8_3_3(2, 8, 3, 3),
            test_preprocessed_and_cached_trace_roundtrip_5_5_4_1(5, 5, 4, 1),
            test_preprocessed_and_cached_trace_roundtrip_5_5_4_2(5, 5, 4, 2),
            test_preprocessed_and_cached_trace_roundtrip_5_5_4_3(5, 5, 4, 3),
        });

        // === 4. Pipeline decomposition ===

        $crate::__test_cases!($engine, batch_sumcheck_zero_interactions, unwrap, {
            test_batch_sumcheck_zero_interactions(4),
            test_batch_sumcheck_zero_interactions_log_height_eq_l_skip(2),
            test_batch_sumcheck_zero_interactions_log_height_lt_l_skip(1),
            test_batch_sumcheck_zero_interactions_log_height_zero(0),
        });

        $crate::__test_cases!($engine, gkr_verify_zero_interactions, unwrap, {
            test_gkr_verify_zero_interactions(),
        });

        $crate::__test_cases!($engine, batch_constraints_with_interactions, unwrap, {
            test_batch_constraints_with_interactions(),
        });

        // === 5. Custom context construction ===

        $crate::__test_cases!($engine, single_fib_and_dummy_trace_stark, unwrap, {
            test_single_fib_and_dummy_trace_stark(3),
            test_single_fib_and_dummy_trace_stark_log_height_eq_l_skip(2),
            test_single_fib_and_dummy_trace_stark_log_height_lt_l_skip(1),
            test_single_fib_and_dummy_trace_stark_log_height_zero(0),
        });

        // === 6. Interaction tests ===

        $crate::__test_cases!($engine, optional_air, unwrap, {
            test_optional_air(),
        });

        $crate::__test_cases!($engine, vkey_methods, {
            test_vkey_methods(),
        });

        $crate::__test_cases!($engine, interaction_trace_height_constraints, {
            test_interaction_trace_height_constraints(),
        });

        $crate::__test_cases!($engine, trace_height_constraints_implied_removal, {
            test_trace_height_constraints_implied_removal(),
        });

        $crate::__test_cases!($engine, interaction_multi_rows_neg, {
            test_interaction_multi_rows_neg(),
        });

        $crate::__test_cases!($engine, interaction_all_zero_sender, unwrap, {
            test_interaction_all_zero_sender(),
        });

        $crate::__test_cases!($engine, interaction_multi_senders, unwrap, {
            test_interaction_multi_senders(),
        });

        $crate::__test_cases!($engine, interaction_multi_senders_neg, {
            test_interaction_multi_senders_neg(),
        });

        $crate::__test_cases!($engine, interaction_multi_sender_receiver, unwrap, {
            test_interaction_multi_sender_receiver(),
        });

        $crate::__test_cases!($engine, interaction_cached_trace_neg, {
            test_interaction_cached_trace_neg(),
        });

        // === 7. WHIR PCS tests (not engine-generic, plain wrappers) ===

        #[test]
        fn test_fold_single() {
            $crate::fold_single();
        }

        #[test]
        fn test_fold_double() {
            $crate::fold_double();
        }

        #[test]
        fn test_whir_single_fib_0_1_1_0() {
            $crate::whir_single_fib(0, 1, 1, 0).unwrap();
        }

        #[test]
        fn test_whir_single_fib_2_1_1_2() {
            $crate::whir_single_fib(2, 1, 1, 2).unwrap();
        }

        #[test]
        fn test_whir_single_fib_2_1_2_0() {
            $crate::whir_single_fib(2, 1, 2, 0).unwrap();
        }

        #[test]
        fn test_whir_single_fib_2_1_3_1() {
            $crate::whir_single_fib(2, 1, 3, 1).unwrap();
        }

        #[test]
        fn test_whir_single_fib_2_1_4_0() {
            $crate::whir_single_fib(2, 1, 4, 0).unwrap();
        }

        #[test]
        fn test_whir_single_fib_2_2_4_0() {
            $crate::whir_single_fib(2, 2, 4, 0).unwrap();
        }

        #[test]
        fn test_whir_multiple_commitments() {
            $crate::whir_multiple_commitments().unwrap();
        }

        #[test]
        fn test_whir_multiple_commitments_negative() {
            $crate::whir_multiple_commitments_negative();
        }
    };
}
