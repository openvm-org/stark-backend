//! Tests copied from stark-backend-v2 crate and then modified to use GPU backend.
use itertools::Itertools;
use openvm_cuda_backend::prelude::F;
use openvm_stark_backend::{
    p3_matrix::dense::RowMajorMatrix, p3_util::log2_strict_usize, prover::MatrixDimensions,
};
use openvm_stark_sdk::config::{
    log_up_params::log_up_security_params_baby_bear_100_bits, setup_tracing,
    setup_tracing_with_log_level,
};
use p3_field::{PrimeCharacteristicRing, PrimeField32, TwoAdicField};
use rand::{Rng, SeedableRng, rngs::StdRng};
use stark_backend_v2::{
    BabyBearPoseidon2CpuEngineV2, StarkEngineV2, SystemParams, WhirConfig, WhirParams,
    WhirRoundConfig,
    poseidon2::sponge::{
        DuplexSponge, DuplexSpongeRecorder, FiatShamirTranscript, TranscriptHistory,
    },
    prover::{
        AirProvingContextV2, ColMajorMatrix, DeviceDataTransporterV2, MultiRapProver,
        ProvingContextV2,
        stacked_pcs::stacked_commit,
        stacked_reduction::{StackedReductionCpu, prove_stacked_opening_reduction},
    },
    test_utils::{
        CachedFixture11, DuplexSpongeValidator, FibFixture, InteractionsFixture11, MixtureFixture,
        PreprocessedFibFixture, SelfInteractionFixture, TestFixture, default_test_params_small,
        prove_up_to_batch_constraints, test_system_params_small,
    },
    verifier::{
        VerifierError,
        batch_constraints::{BatchConstraintError, verify_zerocheck_and_logup},
        fractional_sumcheck_gkr::verify_gkr,
        proof_shape::{ProofShapeError, verify_proof_shape},
        stacked_reduction::{StackedReductionError, verify_stacked_reduction},
        sumcheck::{verify_sumcheck_multilinear, verify_sumcheck_prismalinear},
        verify,
    },
};
use test_case::test_case;
use tracing::{Level, debug};

use crate::{
    BabyBearPoseidon2GpuEngineV2,
    sponge::DuplexSpongeGpu,
    sumcheck::{sumcheck_multilinear_gpu, sumcheck_prismalinear_gpu},
};

pub fn test_gpu_engine_small() -> BabyBearPoseidon2GpuEngineV2 {
    setup_tracing();
    BabyBearPoseidon2GpuEngineV2::new(default_test_params_small())
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
    let mut prover_sponge_gpu = DuplexSpongeGpu::default();
    let mut verifier_sponge = DuplexSponge::default();

    let (proof_gpu, _) = sumcheck_multilinear_gpu(&mut prover_sponge_gpu, &evals);

    verify_sumcheck_multilinear::<F, _>(&mut verifier_sponge, &proof_gpu)
}

#[test]
fn test_plain_prismalinear_sumcheck() -> Result<(), String> {
    let n = 5;
    let l_skip = 10;
    let mut rng = StdRng::from_seed([228; 32]);

    let dim = n + l_skip;
    let num_pts = 1 << dim;
    assert!((F::ORDER_U32 - 1) % num_pts == 0);

    let evals = (0..num_pts)
        .map(|_| F::from_u32(rng.random_range(0..F::ORDER_U32)))
        .collect::<Vec<_>>();

    let mut prover_sponge = DuplexSpongeGpu::default();
    let mut verifier_sponge = DuplexSponge::default();

    let (proof, _) = sumcheck_prismalinear_gpu(&mut prover_sponge, l_skip, &evals);
    verify_sumcheck_prismalinear::<F, _>(&mut verifier_sponge, l_skip, &proof)
}

#[test]
fn test_proof_shape_verifier() -> Result<(), ProofShapeError> {
    setup_tracing();
    let log_trace_degree = 3;

    // without interactions
    let engine = test_gpu_engine_small();
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
        let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);
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
) -> Result<(), BatchConstraintError> {
    setup_tracing_with_log_level(Level::DEBUG);

    let engine = test_gpu_engine_small();
    let params = engine.config();
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

    let mut prover_sponge = DuplexSpongeGpu::default();
    let mut verifier_sponge = DuplexSponge::default();

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
fn test_stacked_opening_reduction(log_trace_degree: usize) -> Result<(), StackedReductionError> {
    setup_tracing_with_log_level(Level::DEBUG);

    let engine = test_gpu_engine_small();
    let params = engine.config();

    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params.clone());
    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);
    let (pk, _vk) = fib.keygen(&engine);
    let pk = engine.device().transport_pk_to_device(&pk);
    let mut ctx = fib.generate_proving_ctx();

    ctx.sort_for_stacking();

    let (_, common_main_pcs_data) = {
        stacked_commit(
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
        &mut DuplexSpongeGpu::default(),
        &pk,
        &ctx,
        &common_main_pcs_data,
    );

    let need_rot = pk.per_air[ctx.per_trace[0].0].vk.params.need_rot;
    let need_rot_per_commit = vec![vec![need_rot]];
    let (stacking_proof, _) = prove_stacked_opening_reduction::<_, _, _, StackedReductionCpu>(
        device,
        &mut DuplexSpongeGpu::default(),
        params.n_stack,
        vec![&common_main_pcs_data],
        need_rot_per_commit.clone(),
        &r,
    );

    debug!(?batch_proof.column_openings);

    let u_prism = verify_stacked_reduction(
        &mut DuplexSponge::default(),
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
    let engine = test_gpu_engine_small();

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
                AirProvingContextV2::simple_no_pis(ColMajorMatrix::from_row_major(&trace)),
            )
        })
        .collect();
    let fib_ctx = fib.generate_proving_ctx().per_trace.pop().unwrap().1;

    // 6. Update air_ids in fib context and combine contexts
    per_trace.push((per_trace.len(), fib_ctx));
    let combined_ctx = engine
        .device()
        .transport_proving_ctx_to_device(&ProvingContextV2::new(per_trace))
        .into_sorted();

    let proof = engine.prove(&combined_pk, combined_ctx);
    engine.verify(&combined_pk.get_vk(), &proof).unwrap();
}

#[test_case(2, 10)]
#[test_case(2, 1; "where log_trace_degree=1 less than l_skip=2")]
#[test_case(2, 0; "where log_trace_degree=0 less than l_skip=2")]
#[test_case(3, 2; "where log_trace_degree=2 less than l_skip=3")]
#[test_case(6, 10; "where l_skip exceeds log_warp_size")]
fn test_fib_air_roundtrip(l_skip: usize, log_trace_degree: usize) -> Result<(), VerifierError> {
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

    let engine = BabyBearPoseidon2GpuEngineV2::new(params);
    let (pk, vk) = fib.keygen(&engine);
    let mut prover_sponge = DuplexSpongeGpu::default();
    let proof = fib.prove_from_transcript(&engine, &pk, &mut prover_sponge);

    let mut validator_sponge = DuplexSponge::default();
    verify(&vk, &proof, &mut validator_sponge)
}

#[test_case(2, 8, 3)]
#[test_case(5, 5, 4)]
fn test_dummy_interactions_roundtrip(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
) -> Result<(), VerifierError> {
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let engine = BabyBearPoseidon2GpuEngineV2::new(params);
    let fx = InteractionsFixture11;
    let (pk, vk) = fx.keygen(&engine);

    let mut prover_sponge = DuplexSpongeGpu::default();
    let proof = fx.prove_from_transcript(&engine, &pk, &mut prover_sponge);

    let mut validator_sponge = DuplexSponge::default();
    verify(&vk, &proof, &mut validator_sponge)
}

#[test_case(2, 8, 3)]
#[test_case(5, 5, 4)]
#[test_case(5, 8, 3)]
#[test_case(6, 7, 3; "where l_skip exceeds log_warp_size")]
fn test_cached_trace_roundtrip(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
) -> Result<(), VerifierError> {
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let engine = BabyBearPoseidon2GpuEngineV2::new(params.clone());
    let fx = CachedFixture11::new(params);
    let (pk, vk) = fx.keygen(&engine);

    let mut prover_sponge = DuplexSpongeGpu::default();
    let proof = fx.prove_from_transcript(&engine, &pk, &mut prover_sponge);

    let mut validator_sponge = DuplexSponge::default();
    verify(&vk, &proof, &mut validator_sponge)
}

#[test_case(2, 8, 3)]
#[test_case(5, 5, 4)]
fn test_preprocessed_trace_roundtrip(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
) -> Result<(), VerifierError> {
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let engine = BabyBearPoseidon2CpuEngineV2::new(params);
    let log_trace_degree = 8;
    let height = 1 << log_trace_degree;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
    let fx = PreprocessedFibFixture::new(0, 1, sels);
    let (pk, vk) = fx.keygen(&engine);

    let mut recorder = DuplexSpongeRecorder::default();
    let proof = fx.prove_from_transcript(&engine, &pk, &mut recorder);

    let mut validator_sponge = DuplexSpongeValidator::new(recorder.into_log());
    verify(&vk, &proof, &mut validator_sponge)
}

#[test]
fn test_interactions_single_sender_receiver_happy() {
    setup_tracing();

    let engine = test_gpu_engine_small();
    let fx = InteractionsFixture11;
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof).unwrap();
}

#[test]
fn test_single_cached_trace_stark() {
    setup_tracing();
    let engine = test_gpu_engine_small();
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
    let engine = test_gpu_engine_small();
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
    let engine = test_gpu_engine_small();
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
    let engine = test_gpu_engine_small();
    let fx = MixtureFixture::standard(log_trace_degree, engine.config().clone());
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof).unwrap();
}

#[test]
fn test_gkr_verify_zero_interactions() -> eyre::Result<()> {
    setup_tracing_with_log_level(Level::DEBUG);

    let engine = test_gpu_engine_small();
    let device = engine.device();
    let params = engine.config();
    let fx = InteractionsFixture11;
    let (pk, _vk) = fx.keygen(&engine);
    let pk = device.transport_pk_to_device(&pk);
    let ctx = device
        .transport_proving_ctx_to_device(&fx.generate_proving_ctx())
        .into_sorted();
    let mut transcript = DuplexSpongeGpu::default();
    let ((gkr_proof, _), _) = prove_up_to_batch_constraints(&engine, &mut transcript, &pk, ctx);

    let mut transcript = DuplexSpongeRecorder::default();
    assert!(transcript.check_witness(params.logup.pow_bits, gkr_proof.logup_pow_witness));
    let _alpha = transcript.sample_ext();
    let _beta = transcript.sample_ext();
    let total_rounds = gkr_proof.claims_per_layer.len();
    verify_gkr(&gkr_proof, &mut transcript, total_rounds)?;

    Ok(())
}

#[test]
fn test_batch_constraints_with_interactions() -> eyre::Result<()> {
    setup_tracing_with_log_level(Level::DEBUG);

    let engine = test_gpu_engine_small();
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

    let mut transcript = DuplexSpongeGpu::default();
    let ((gkr_proof, batch_proof), _) =
        prove_up_to_batch_constraints(&engine, &mut transcript, &pk, ctx);
    let mut transcript = DuplexSpongeRecorder::default();
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
    let engine = BabyBearPoseidon2GpuEngineV2::new(params);
    let fx = SelfInteractionFixture {
        widths: vec![4, 7, 8, 8, 10],
        log_height: 1,
        bus_index: 4,
    };
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof).unwrap();
}

/// Tests that monomial-based and DAG-based zerocheck evaluation paths produce identical results.
///
/// This test verifies that `ZerocheckMonomialBatchBuilder` and `ZerocheckMleBatchBuilder`
/// compute the same output for traces where both paths are applicable.
#[test]
fn test_monomial_vs_dag_equivalence() {
    use openvm_cuda_common::copy::{MemCopyD2H, MemCopyH2D};
    use p3_util::log2_strict_usize;
    use stark_backend_v2::{
        poly_common::eval_eq_uni_at_one, test_utils::prove_up_to_batch_constraints,
    };

    use crate::{
        EF,
        cuda::logup_zerocheck::{MainMatrixPtrs, fold_selectors_round0, interpolate_columns_gpu},
        logup_zerocheck::{
            batch_mle::{TraceCtx, ZerocheckMleBatchBuilder},
            batch_mle_monomial::{ZerocheckMonomialBatch, compute_lambda_combinations},
            fold_ple::fold_ple_evals_rotate,
        },
        poly::EqEvalSegments,
    };

    setup_tracing_with_log_level(Level::DEBUG);

    // Use FibFixture for a simple AIR with only constraints (no interactions)
    let log_trace_degree = 5; // 32 rows - small enough for monomial path

    // Threshold for monomial path - test traces with num_y <= this value
    let threshold = 32u32;

    let engine = test_gpu_engine_small();
    let device = engine.device();
    let params = engine.config();
    let l_skip = params.l_skip;

    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);
    let (pk, _vk) = fib.keygen(&engine);
    let pk = device.transport_pk_to_device(&pk);

    // Run GKR/batch constraints to get random challenges
    let ctx_for_challenges = device
        .transport_proving_ctx_to_device(&fib.generate_proving_ctx())
        .into_sorted();
    let mut prover_sponge = DuplexSpongeGpu::default();
    let ((_, _), r) =
        prove_up_to_batch_constraints(&engine, &mut prover_sponge, &pk, ctx_for_challenges);

    // Regenerate context for the actual test
    let ctx = device
        .transport_proving_ctx_to_device(&fib.generate_proving_ctx())
        .into_sorted();

    // Build xi vector with proper length: need l_skip + n_lift elements for EqEvalSegments
    // r contains the sumcheck round challenges, but we need to prefix with l_skip elements
    let height = ctx.per_trace[0].1.common_main.height();
    let n_calc = log2_strict_usize(height).saturating_sub(l_skip);
    let xi_len = l_skip + n_calc + 1;

    // Generate deterministic xi values (using r and extending with sponge samples)
    let mut xi: Vec<EF> = Vec::with_capacity(xi_len);
    // Add l_skip initial elements (these would come from GKR in real prover)
    for _ in 0..l_skip {
        xi.push(prover_sponge.sample_ext());
    }
    // Add the sumcheck round challenges
    xi.extend_from_slice(&r);
    // Ensure we have enough elements
    while xi.len() < xi_len {
        xi.push(prover_sponge.sample_ext());
    }
    assert!(xi.len() > l_skip, "xi vector must have enough elements");

    // Setup omega skip powers
    let omega_skip = F::two_adic_generator(l_skip);
    let omega_skip_pows: Vec<F> = omega_skip.powers().take(1 << l_skip).collect();
    let d_omega_skip_pows = omega_skip_pows.to_device().unwrap();

    // Get trace info
    let (air_idx, air_ctx) = &ctx.per_trace[0];
    let height = air_ctx.common_main.height();
    let n = log2_strict_usize(height) as isize - l_skip as isize;
    let n_lift = n.max(0) as usize;

    // Setup eq_xis
    let eq_xis = EqEvalSegments::new(&xi[l_skip..]).expect("failed to compute eq_xis");

    // Setup selectors (is_first, is_transition, is_last)
    let sel_height = 1 << n_lift;
    let mut sel_cols = F::zero_vec(3 * sel_height);
    sel_cols[sel_height..2 * sel_height - 1].fill(F::ONE); // is_transition
    sel_cols[0] = F::ONE; // is_first
    sel_cols[2 * sel_height + sel_height - 1] = F::ONE; // is_last
    let d_sels_base = sel_cols.to_device().unwrap();

    // Fold selectors
    let (l, r_fold) = if n.is_negative() {
        (
            l_skip.wrapping_add_signed(n),
            r[0].exp_power_of_2(-n as usize),
        )
    } else {
        (l_skip, r[0])
    };
    let omega = F::two_adic_generator(l);
    let is_first = eval_eq_uni_at_one(l, r_fold);
    let is_last = eval_eq_uni_at_one(l, r_fold * omega);
    let d_sels_folded =
        openvm_cuda_common::d_buffer::DeviceBuffer::<EF>::with_capacity(sel_height * 3);
    unsafe {
        fold_selectors_round0(
            d_sels_folded.as_mut_ptr(),
            d_sels_base.as_ptr(),
            is_first,
            is_last,
            sel_height,
        )
        .unwrap();
    }

    // Setup inv_lagrange_denoms for PLE fold
    let inv_lagrange_denoms_r0 =
        crate::utils::compute_barycentric_inv_lagrange_denoms(l_skip, &omega_skip_pows, r[0]);
    let d_inv_lagrange_denoms_r0 = inv_lagrange_denoms_r0.to_device().unwrap();

    // Fold common_main trace
    let mat_folded = fold_ple_evals_rotate(
        l_skip,
        &d_omega_skip_pows,
        &air_ctx.common_main,
        &d_inv_lagrange_denoms_r0,
        true,
    )
    .unwrap();

    // Setup lambda powers
    let lambda = prover_sponge.sample_ext();
    let air_pk = &pk.per_air[*air_idx];
    let max_num_constraints = air_pk
        .vk
        .symbolic_constraints
        .constraints
        .constraint_idx
        .len();
    let h_lambda_pows: Vec<EF> = lambda.powers().take(max_num_constraints).collect();
    let d_lambda_pows = h_lambda_pows.to_device().unwrap();

    // Public values
    let d_public_values = if air_ctx.public_values.is_empty() {
        openvm_cuda_common::d_buffer::DeviceBuffer::new()
    } else {
        air_ctx.public_values.to_device().unwrap()
    };

    // Verify the AIR has constraints and monomials available
    let dag = &air_pk.vk.symbolic_constraints;
    let has_constraints = dag.constraints.num_constraints() > 0;
    assert!(has_constraints, "FibFixture should have constraints");

    let has_monomials = air_pk
        .other_data
        .zerocheck_monomials
        .as_ref()
        .map(|m| m.num_monomials > 0)
        .unwrap_or(false);
    assert!(
        has_monomials,
        "Proving key should have expanded monomials for monomial path"
    );

    // Test at multiple num_y values to ensure equivalence across different sizes.
    // The threshold was set on the engine above - use it to filter test cases.
    let s_deg = params.max_constraint_degree as usize + 1;

    for test_round in 1..=n_lift.min(3) {
        let n_round = n_lift.saturating_sub(test_round - 1);
        let test_height = 1 << n_round;
        let num_y = (test_height / 2) as u32;

        if num_y == 0 || num_y > threshold {
            continue;
        }

        debug!(test_round, num_y, %threshold, "testing monomial vs DAG equivalence");

        // For this test, we need to interpolate columns first (like in sumcheck_polys_batch_eval)
        let has_interactions = false;
        let mut columns: Vec<*const EF> = Vec::new();
        columns.push(eq_xis.get_ptr(n_round));
        // Add selector columns
        for col in 0..3 {
            columns.push(d_sels_folded.as_ptr().wrapping_add(col * sel_height));
        }
        // Add main trace columns
        for col in 0..mat_folded.width() {
            columns.push(
                mat_folded
                    .buffer()
                    .as_ptr()
                    .wrapping_add(col * mat_folded.height()),
            );
        }

        let interpolated = openvm_cuda_backend::base::DeviceMatrix::<EF>::with_capacity(
            s_deg * num_y as usize,
            columns.len(),
        );
        let d_columns = columns.to_device().unwrap();
        unsafe {
            interpolate_columns_gpu(interpolated.buffer(), &d_columns, s_deg, num_y as usize)
                .expect("failed to interpolate columns");
        }

        let interpolated_height = interpolated.height();
        // eq_xi_ptr should point to non-interpolated eq_xi (num_y elements)
        // The kernel accesses eq_xi[y_int], not eq_xi[row]
        let eq_xi_ptr = eq_xis.get_ptr(n_round);
        let sels_ptr = interpolated
            .buffer()
            .as_ptr()
            .wrapping_add(interpolated_height);

        let main_ptrs = [MainMatrixPtrs {
            data: interpolated
                .buffer()
                .as_ptr()
                .wrapping_add(4 * interpolated_height), // after eq_xi + 3 sels
            air_width: mat_folded.width() as u32 / 2,
        }];
        let main_ptrs_dev = main_ptrs.to_device().unwrap();

        // Build TraceCtx
        let trace_ctx = TraceCtx {
            trace_idx: 0,
            air_idx: *air_idx,
            n_lift,
            num_y,
            has_constraints: true,
            has_interactions,
            norm_factor: F::ONE,
            eq_xi_ptr,
            sels_ptr,
            prep_ptr: MainMatrixPtrs {
                data: std::ptr::null(),
                air_width: 0,
            },
            main_ptrs_dev,
            public_ptr: d_public_values.as_ptr(),
            eq_3bs_ptr: std::ptr::null(),
        };

        // Run DAG-based evaluation
        let dag_builder =
            ZerocheckMleBatchBuilder::new(std::iter::once(&trace_ctx), &pk, s_deg as u32);
        let dag_output = dag_builder.evaluate(&d_lambda_pows, s_deg as u32);
        let dag_results: Vec<EF> = dag_output.to_host().expect("copy DAG output");

        // Run monomial-based evaluation
        let lambda_comb = compute_lambda_combinations(&pk, 0, &d_lambda_pows).unwrap();
        let mono_batch =
            ZerocheckMonomialBatch::new(std::iter::once(&trace_ctx), &pk, &[&lambda_comb]);
        let mono_output = mono_batch.evaluate(s_deg as u32);
        let mono_results: Vec<EF> = mono_output.to_host().expect("copy monomial output");

        // Compare results
        assert_eq!(
            dag_results.len(),
            mono_results.len(),
            "Output lengths should match"
        );
        for (i, (dag_val, mono_val)) in dag_results.iter().zip(mono_results.iter()).enumerate() {
            assert_eq!(
                dag_val, mono_val,
                "Mismatch at index {i} for num_y={num_y}: DAG={dag_val:?}, monomial={mono_val:?}"
            );
        }
        debug!(
            num_y,
            num_results = dag_results.len(),
            "monomial vs DAG equivalence verified"
        );
    }
}
