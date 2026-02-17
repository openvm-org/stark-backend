//! Tests copied from openvm-stark-backend crate and then modified to use Metal backend.
use itertools::Itertools;
use openvm_stark_backend::{
    p3_matrix::dense::RowMajorMatrix,
    p3_util::log2_strict_usize,
    prover::{
        stacked_pcs::stacked_commit,
        stacked_reduction::{prove_stacked_opening_reduction, StackedReductionCpu},
        AirProvingContext, ColMajorMatrix, DeviceDataTransporter, MatrixDimensions, MultiRapProver,
        ProvingContext,
    },
    test_utils::{
        default_test_params_small, prove_up_to_batch_constraints, test_system_params_small,
        CachedFixture11, FibFixture, InteractionsFixture11, MixtureFixture, PreprocessedFibFixture,
        SelfInteractionFixture, TestFixture,
    },
    verifier::{
        batch_constraints::{verify_zerocheck_and_logup, BatchConstraintError},
        fractional_sumcheck_gkr::verify_gkr,
        proof_shape::{verify_proof_shape, ProofShapeError},
        stacked_reduction::{verify_stacked_reduction, StackedReductionError},
        verify, VerifierError,
    },
    FiatShamirTranscript, StarkEngine, StarkProtocolConfig, SystemParams, TranscriptHistory,
    WhirConfig, WhirParams, WhirRoundConfig,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{
            default_duplex_sponge, default_duplex_sponge_recorder, default_duplex_sponge_validator,
            BabyBearPoseidon2CpuEngine, DuplexSponge, DuplexSpongeRecorder,
        },
        log_up_params::log_up_security_params_baby_bear_100_bits,
    },
    utils::{setup_tracing, setup_tracing_with_log_level},
};
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use rand::{rngs::StdRng, Rng, SeedableRng};
use test_case::test_case;
use tracing::{debug, Level};

use crate::{
    prelude::{EF, F, SC},
    sponge::DuplexSpongeMetal,
    BabyBearPoseidon2MetalEngine,
};

pub fn test_metal_engine_small() -> BabyBearPoseidon2MetalEngine {
    setup_tracing();
    BabyBearPoseidon2MetalEngine::new(default_test_params_small())
}

#[test]
fn test_proof_shape_verifier() -> Result<(), ProofShapeError> {
    setup_tracing();
    let log_trace_degree = 3;

    // without interactions
    let engine = test_metal_engine_small();
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
            mu_pow_bits: 2,
            query_phase_pow_bits: 2,
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

#[test_case(4)]
#[test_case(2 ; "when log_height equals l_skip")]
#[test_case(1 ; "when log_height less than l_skip")]
#[test_case(0 ; "when log_height is zero")]
fn test_batch_sumcheck_zero_interactions(
    log_trace_degree: usize,
) -> Result<(), BatchConstraintError<EF>> {
    setup_tracing_with_log_level(Level::DEBUG);

    let engine = test_metal_engine_small();
    let params = engine.config().params();
    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);
    let (pk, vk) = fib.keygen(&engine);
    let device = engine.device();
    let pk = device.transport_pk_to_device(&pk);
    let ctx = device.transport_proving_ctx_to_device(&fib.generate_proving_ctx());

    let mut n_per_air: Vec<isize> = Vec::with_capacity(ctx.per_trace.len());
    for (_, trace) in ctx.common_main_traces() {
        let trace_height = trace.height();
        let log_height = log2_strict_usize(trace_height);
        let n = log_height as isize - params.l_skip as isize;
        n_per_air.push(n);
    }

    let mut prover_sponge = DuplexSpongeMetal::default();
    let mut verifier_sponge = default_duplex_sponge();

    let omega_skip = F::two_adic_generator(params.l_skip);
    let omega_skip_pows = omega_skip.powers().take(1 << params.l_skip).collect_vec();

    let pvs = vec![ctx.per_trace[0].1.public_values.clone()];
    let ((gkr_proof, batch_proof), _) =
        prove_up_to_batch_constraints(&engine, &mut prover_sponge, &pk, ctx);
    let r = verify_zerocheck_and_logup(
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

#[test_case(9)]
#[test_case(2 ; "when log_height equals l_skip")]
#[test_case(1 ; "when log_height less than l_skip")]
#[test_case(0 ; "when log_height is zero")]
fn test_stacked_opening_reduction(
    log_trace_degree: usize,
) -> Result<(), StackedReductionError<EF>> {
    setup_tracing_with_log_level(Level::DEBUG);

    let metal_engine = test_metal_engine_small();
    let params = metal_engine.config().params();

    let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params.clone());
    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);
    let (pk, _vk) = fib.keygen(&engine);
    let pk = engine.device().transport_pk_to_device(&pk);
    let mut ctx = fib.generate_proving_ctx();

    ctx.sort_for_stacking();

    let (_, common_main_pcs_data) = {
        stacked_commit(
            engine.config().hasher(),
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
    // We need batch_proof to obtain the column openings
    let ((_, batch_proof), r) = device.prove_rap_constraints(
        &mut DuplexSpongeMetal::default(),
        &pk,
        &ctx,
        &common_main_pcs_data,
    );

    let need_rot = pk.per_air[ctx.per_trace[0].0].vk.params.need_rot;
    let need_rot_per_commit = vec![vec![need_rot]];
    let (stacking_proof, _) = prove_stacked_opening_reduction::<SC, _, _, _, StackedReductionCpu<SC>>(
        device,
        &mut DuplexSpongeMetal::default(),
        params.n_stack,
        vec![&common_main_pcs_data],
        need_rot_per_commit.clone(),
        &r,
    );

    debug!(?batch_proof.column_openings);

    let u_prism = verify_stacked_reduction(
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

#[test_case(3)]
#[test_case(2 ; "when fib log_height equals l_skip")]
#[test_case(1 ; "when fib log_height less than l_skip")]
#[test_case(0 ; "when fib log_height is zero")]
fn test_single_fib_and_dummy_trace_stark(log_trace_degree: usize) {
    setup_tracing();

    // 1. Create parameters
    let engine = test_metal_engine_small();

    // 2. Create interactions fixture with larger trace - generate custom traces
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

    // 3. Create fibonacci fixture with small trace
    let height = 2 * (1 << log_trace_degree);
    let fib = FibFixture::new(0, 1, height);

    // 4. Generate AIRs and proving keys
    let fx_fixture = InteractionsFixture11;
    let fx_airs = fx_fixture.airs();
    let fib_airs = fib.airs();
    let mut combined_airs = fx_airs;
    combined_airs.extend(fib_airs);
    let (combined_pk, _combined_vk) = engine.keygen(&combined_airs);
    let combined_pk = engine.device().transport_pk_to_device(&combined_pk);

    // 5. Generate custom contexts for interactions with modified traces
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

    // 6. Update air_ids in fib context and combine contexts
    per_trace.push((per_trace.len(), fib_ctx));
    let combined_ctx = engine
        .device()
        .transport_proving_ctx_to_device(&ProvingContext::new(per_trace))
        .into_sorted();

    let proof = engine.prove(&combined_pk, combined_ctx);
    engine.verify(&combined_pk.get_vk(), &proof).unwrap();
}

#[test_case(2, 10)]
#[test_case(2, 1; "where log_trace_degree=1 less than l_skip=2")]
#[test_case(2, 0; "where log_trace_degree=0 less than l_skip=2")]
#[test_case(3, 2; "where log_trace_degree=2 less than l_skip=3")]
#[test_case(6, 10; "where l_skip exceeds log_warp_size")]
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

    let engine = BabyBearPoseidon2MetalEngine::new(params);
    let (pk, vk) = fib.keygen(&engine);
    let mut prover_sponge = DuplexSpongeMetal::default();
    let proof = fib.prove_from_transcript(&engine, &pk, &mut prover_sponge);

    let mut validator_sponge = default_duplex_sponge();
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
    let engine = BabyBearPoseidon2MetalEngine::new(params);
    let fx = InteractionsFixture11;
    let (pk, vk) = fx.keygen(&engine);

    let mut prover_sponge = DuplexSpongeMetal::default();
    let proof = fx.prove_from_transcript(&engine, &pk, &mut prover_sponge);

    let mut validator_sponge = default_duplex_sponge();
    verify(engine.config(), &vk, &proof, &mut validator_sponge)
}

#[test_case(2, 8, 3)]
#[test_case(5, 5, 4)]
#[test_case(5, 8, 3)]
#[test_case(6, 7, 3; "where l_skip exceeds log_warp_size")]
fn test_cached_trace_roundtrip(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
) -> Result<(), VerifierError<EF>> {
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let engine = BabyBearPoseidon2MetalEngine::new(params);
    let fx = CachedFixture11::new(engine.config().clone());
    let (pk, vk) = fx.keygen(&engine);

    let mut prover_sponge = DuplexSpongeMetal::default();
    let proof = fx.prove_from_transcript(&engine, &pk, &mut prover_sponge);

    let mut validator_sponge = default_duplex_sponge();
    verify(engine.config(), &vk, &proof, &mut validator_sponge)
}

#[test_case(2, 8, 3)]
#[test_case(5, 5, 4)]
fn test_preprocessed_trace_roundtrip(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
) -> Result<(), VerifierError<EF>> {
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let engine = BabyBearPoseidon2CpuEngine::<DuplexSpongeRecorder>::new(params);
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

    let engine = test_metal_engine_small();
    let fx = InteractionsFixture11;
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof).unwrap();
}

#[test]
fn test_single_cached_trace_stark() {
    setup_tracing();
    let engine = test_metal_engine_small();
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
    let engine = test_metal_engine_small();
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
    let engine = test_metal_engine_small();
    let fx = SelfInteractionFixture {
        widths: vec![4, 7, 8, 8, 10, 100],
        log_height: log_trace_degree,
        bus_index: 4,
    };
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof).unwrap();
}

#[test_case(10 ; "when log_height equals n_stack plus l_skip")]
#[test_case(3 ; "when log_height greater than l_skip")]
#[test_case(2 ; "when log_height equals l_skip")]
#[test_case(1 ; "when log_height less than l_skip")]
#[test_case(0 ; "when log_height is zero")]
fn test_mixture_traces_stark(log_trace_degree: usize) {
    setup_tracing();
    let engine = test_metal_engine_small();
    let fx = MixtureFixture::standard(log_trace_degree, engine.config().clone());
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof).unwrap();
}

#[test]
fn test_gkr_verify_zero_interactions() -> eyre::Result<()> {
    setup_tracing_with_log_level(Level::DEBUG);

    let engine = test_metal_engine_small();
    let device = engine.device();
    let params = engine.config().params();
    let fx = InteractionsFixture11;
    let (pk, _vk) = fx.keygen(&engine);
    let pk = device.transport_pk_to_device(&pk);
    let ctx = device
        .transport_proving_ctx_to_device(&fx.generate_proving_ctx())
        .into_sorted();
    let mut transcript = DuplexSpongeMetal::default();
    let ((gkr_proof, _), _) = prove_up_to_batch_constraints(&engine, &mut transcript, &pk, ctx);

    let mut transcript = default_duplex_sponge_recorder();
    assert!(FiatShamirTranscript::<SC>::check_witness(
        &mut transcript,
        params.logup.pow_bits,
        gkr_proof.logup_pow_witness
    ));
    let _alpha = FiatShamirTranscript::<SC>::sample_ext(&mut transcript);
    let _beta = FiatShamirTranscript::<SC>::sample_ext(&mut transcript);
    let total_rounds = gkr_proof.claims_per_layer.len();
    let _ = verify_gkr(&gkr_proof, &mut transcript, total_rounds)?;

    Ok(())
}

#[test]
fn test_batch_constraints_with_interactions() -> eyre::Result<()> {
    setup_tracing_with_log_level(Level::DEBUG);

    let engine = test_metal_engine_small();
    let device = engine.device();
    let fx = InteractionsFixture11;
    let (pk, vk) = fx.keygen(&engine);
    let pk = device.transport_pk_to_device(&pk);
    let ctx = device
        .transport_proving_ctx_to_device(&fx.generate_proving_ctx())
        .into_sorted();
    let l_skip = device.config().l_skip;
    let mut pvs = vec![vec![]; vk.inner.per_air.len()];
    let (trace_id_to_air_ids, ns): (Vec<_>, Vec<_>) = ctx
        .per_trace
        .iter()
        .map(|(air_idx, air_ctx)| {
            pvs[*air_idx] = air_ctx.public_values.clone();
            (
                *air_idx,
                log2_strict_usize(air_ctx.common_main.height()) as isize - l_skip as isize,
            )
        })
        .multiunzip();
    debug!(?trace_id_to_air_ids);
    debug!(n_per_trace = ?ns);
    let omega_pows = F::two_adic_generator(l_skip)
        .powers()
        .take(1 << l_skip)
        .collect_vec();

    let mut transcript = DuplexSpongeMetal::default();
    let ((gkr_proof, batch_proof), _) =
        prove_up_to_batch_constraints(&engine, &mut transcript, &pk, ctx);
    let mut transcript = default_duplex_sponge_recorder();
    verify_zerocheck_and_logup(
        &mut transcript,
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

#[test]
fn test_matrix_stacking_overflow() {
    setup_tracing();
    let params = test_system_params_small(3, 5, 3);
    let engine = BabyBearPoseidon2MetalEngine::new(params);
    let fx = SelfInteractionFixture {
        widths: vec![4, 7, 8, 8, 10],
        log_height: 1,
        bus_index: 4,
    };
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof).unwrap();
}
