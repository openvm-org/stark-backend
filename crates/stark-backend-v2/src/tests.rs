use itertools::Itertools;
use openvm_stark_backend::prover::MatrixDimensions;
use openvm_stark_sdk::config::{setup_tracing, setup_tracing_with_log_level};
use p3_field::{FieldAlgebra, PrimeField32, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use rand::{Rng, SeedableRng, rngs::StdRng};
use tracing::{Level, debug};

use crate::{
    BabyBearPoseidon2CpuEngineV2, F, StarkEngineV2,
    keygen::types::SystemParams,
    poseidon2::sponge::{DuplexSponge, FiatShamirTranscript},
    prover::{
        AirProvingContextV2, ColMajorMatrix, DeviceDataTransporterV2, ProvingContextV2,
        batch_constraints::prove_zerocheck_and_logup,
        stacked_pcs::stacked_commit,
        stacked_reduction::stacked_opening_reduction,
        sumcheck::{sumcheck_multilinear, sumcheck_prismalinear},
    },
    test_utils::{
        CachedFixture11, FibFixture, InteractionsFixture11, PreprocessedFibFixture, TestFixture,
        test_engine_small,
    },
    verifier::{
        batch_constraints::{BatchConstraintError, verify_zerocheck_and_logup},
        fractional_sumcheck_gkr::verify_gkr,
        proof_shape::{ProofShapeError, verify_proof_shape},
        stacked_reduction::{StackedReductionError, verify_stacked_reduction},
        sumcheck::{verify_sumcheck_multilinear, verify_sumcheck_prismalinear},
    },
};

#[test]
fn test_plain_multilinear_sumcheck() -> Result<(), String> {
    let n = 15;
    let mut rng = StdRng::from_seed([228; 32]);

    let num_pts = 1 << n;
    assert!((F::ORDER_U32 - 1) % num_pts == 0);

    let evals = (0..num_pts)
        .map(|_| F::from_canonical_u32(rng.random_range(0..F::ORDER_U32)))
        .collect::<Vec<_>>();
    let mut prover_sponge = DuplexSponge::default();
    let mut verifier_sponge = DuplexSponge::default();

    let (proof, _) = sumcheck_multilinear(&mut prover_sponge, &evals);
    verify_sumcheck_multilinear::<F, _>(&mut verifier_sponge, &proof)
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
        .map(|_| F::from_canonical_u32(rng.random_range(0..F::ORDER_U32)))
        .collect::<Vec<_>>();
    let mut prover_sponge = DuplexSponge::default();
    let mut verifier_sponge = DuplexSponge::default();

    let (proof, _) = sumcheck_prismalinear(&mut prover_sponge, l_skip, &evals);
    verify_sumcheck_prismalinear::<F, _>(&mut verifier_sponge, l_skip, &proof)
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
    let params = engine.config();
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
        let num_whir_queries = rng.random_range(1..=10);
        let num_whir_rounds = rng.random_range(1..=2);
        let log_final_poly_len = n_stack + l_skip - num_whir_rounds * k_whir;
        let params = SystemParams {
            l_skip,
            n_stack,
            log_blowup,
            k_whir,
            num_whir_queries,
            log_final_poly_len,
            logup_pow_bits: 1,
            whir_pow_bits: 1,
        };
        let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);
        let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
        verify_proof_shape(&vk.inner, &proof)?;
    }
    Ok(())
}

#[test]
fn test_batch_sumcheck_zero_interactions() -> Result<(), BatchConstraintError> {
    setup_tracing_with_log_level(Level::DEBUG);

    let engine = test_engine_small();
    let params = engine.config();
    let log_trace_degree = 4;
    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);
    let (pk, vk) = fib.keygen(&engine);
    let pk = engine.device().transport_pk_to_device(&pk);
    let ctx = fib.generate_proving_ctx();

    let mut n_per_air: Vec<usize> = Vec::with_capacity(ctx.per_trace.len());
    for (_, trace) in ctx.common_main_traces() {
        let trace_height = trace.height();
        let prism_dim = log2_strict_usize(trace_height);
        assert!(prism_dim >= params.l_skip);
        let n = prism_dim - params.l_skip;
        n_per_air.push(n);
    }

    let mut prover_sponge = DuplexSponge::default();
    let mut verifier_sponge = DuplexSponge::default();

    let omega_skip = F::two_adic_generator(params.l_skip);
    let omega_skip_pows = omega_skip.powers().take(1 << params.l_skip).collect_vec();

    let (gkr_proof, batch_proof, _) = prove_zerocheck_and_logup(&mut prover_sponge, &pk, &ctx);
    let pvs = vec![ctx.per_trace[0].1.public_values.clone()];
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
    assert_eq!(r.len(), log_trace_degree - params.l_skip + 1);
    Ok(())
}

#[test]
fn test_stacked_opening_reduction() -> Result<(), StackedReductionError> {
    setup_tracing_with_log_level(Level::DEBUG);

    let engine = test_engine_small();
    let params = engine.config();

    let log_trace_degree = 9;
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);
    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);
    let (pk, _vk) = fib.keygen(&engine);
    let pk = engine.device().transport_pk_to_device(&pk);
    let mut ctx = fib.generate_proving_ctx();

    ctx.per_trace
        .sort_by(|a, b| b.1.common_main.height().cmp(&a.1.common_main.height()));

    let (_, common_main_pcs_data) = {
        stacked_commit(
            params.l_skip,
            params.n_stack,
            params.log_blowup,
            params.k_whir,
            &ctx.common_main_traces()
                .map(|(_, trace)| trace)
                .collect_vec(),
        )
    };

    let mut ns: Vec<usize> = Vec::with_capacity(ctx.per_trace.len());
    for (_, trace) in ctx.common_main_traces() {
        let trace_height = trace.height();
        let prism_dim = log2_strict_usize(trace_height);
        assert!(prism_dim >= params.l_skip);
        let n = prism_dim - params.l_skip;
        ns.push(n);
    }

    let omega_skip = F::two_adic_generator(params.l_skip);
    let omega_skip_pows = omega_skip.powers().take(1 << params.l_skip).collect_vec();

    // We need batch_proof to obtain the column openings
    let (_, batch_proof, r) = prove_zerocheck_and_logup(&mut DuplexSponge::default(), &pk, &ctx);

    let (stacking_proof, _) = stacked_opening_reduction(
        &mut DuplexSponge::default(),
        params.l_skip,
        params.n_stack,
        &[(&common_main_pcs_data.matrix, &common_main_pcs_data.layout)],
        &r,
    );

    debug!(?batch_proof.column_openings);

    let u_prism = verify_stacked_reduction(
        &mut DuplexSponge::default(),
        &stacking_proof,
        &[common_main_pcs_data.layout],
        params.l_skip,
        params.n_stack,
        &batch_proof.column_openings,
        &r,
        &omega_skip_pows,
    )?;
    assert_eq!(u_prism.len(), params.n_stack + 1);
    Ok(())
}

#[test]
fn test_single_fib_and_dummy_trace_stark() {
    setup_tracing();

    // 1. Create parameters
    let log_trace_degree = 3;
    let engine = test_engine_small();

    // 2. Create interactions fixture with larger trace - generate custom traces
    let height = 2 * (1 << log_trace_degree);
    let sender_trace = RowMajorMatrix::new(
        [0, 1, 3, 5, 7, 4, 546, 889]
            .into_iter()
            .cycle()
            .take(2 * height)
            .map(F::from_canonical_usize)
            .collect(),
        2,
    );
    let receiver_trace = RowMajorMatrix::new(
        [1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 889, 1, 889, 0, 456]
            .into_iter()
            .cycle()
            .take(4 * height)
            .map(F::from_canonical_usize)
            .collect(),
        2,
    );

    // 3. Create fibonacci fixture with small trace
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
    per_trace.sort_by(|a, b| b.1.common_main.height().cmp(&a.1.common_main.height()));
    let combined_ctx = ProvingContextV2::new(per_trace);

    let l_skip = engine.config().l_skip;
    let mut pvs = vec![vec![]; 3];
    let (trace_id_to_air_ids, ns): (Vec<_>, Vec<_>) = combined_ctx
        .per_trace
        .iter()
        .map(|(air_idx, air_ctx)| {
            pvs[*air_idx] = air_ctx.public_values.clone();
            (
                *air_idx,
                log2_strict_usize(air_ctx.common_main.height()) - l_skip,
            )
        })
        .multiunzip();
    let omega_pows = F::two_adic_generator(l_skip)
        .powers()
        .take(1 << l_skip)
        .collect_vec();

    let mut transcript = DuplexSponge::default();
    let (gkr_proof, batch_proof, _) =
        prove_zerocheck_and_logup(&mut transcript, &combined_pk, &combined_ctx);
    let mut transcript = DuplexSponge::default();
    verify_zerocheck_and_logup(
        &mut transcript,
        &combined_pk.get_vk().inner,
        &pvs,
        &gkr_proof,
        &batch_proof,
        &trace_id_to_air_ids,
        &ns,
        &omega_pows,
    )
    .unwrap();
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
    let fx = CachedFixture11::new(engine.config());
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof).unwrap();
}

#[test]
fn test_single_preprocessed_trace_stark() {
    setup_tracing();
    let engine = test_engine_small();
    let log_trace_degree = 3;
    let height = 1 << log_trace_degree;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
    let fx = PreprocessedFibFixture::new(0, 1, sels);
    let (vk, proof) = fx.keygen_and_prove(&engine);
    engine.verify(&vk, &proof).unwrap();
}

#[test]
fn test_gkr_verify_zero_interactions() -> eyre::Result<()> {
    setup_tracing_with_log_level(Level::DEBUG);

    let engine = test_engine_small();
    let params = engine.config();
    let fx = InteractionsFixture11;
    let (pk, _vk) = fx.keygen(&engine);
    let pk = engine.device().transport_pk_to_device(&pk);
    let mut ctx = fx.generate_proving_ctx();
    ctx.per_trace
        .sort_by(|a, b| b.1.common_main.height().cmp(&a.1.common_main.height()));
    let mut transcript = DuplexSponge::default();
    let (gkr_proof, _, _) = prove_zerocheck_and_logup(&mut transcript, &pk, &ctx);

    let mut transcript = DuplexSponge::default();
    assert!(transcript.check_witness(params.logup_pow_bits, gkr_proof.logup_pow_witness));
    let _alpha = transcript.sample_ext();
    let _beta = transcript.sample_ext();
    let total_rounds = gkr_proof.claims_per_layer.len();
    verify_gkr(&gkr_proof, &mut transcript, total_rounds)?;

    Ok(())
}

#[test]
fn test_batch_constraints_with_interactions() -> eyre::Result<()> {
    setup_tracing_with_log_level(Level::DEBUG);

    let engine = test_engine_small();
    let fx = InteractionsFixture11;
    let (pk, vk) = fx.keygen(&engine);
    let pk = engine.device().transport_pk_to_device(&pk);
    let mut ctx = fx.generate_proving_ctx();
    ctx.per_trace
        .sort_by(|a, b| b.1.common_main.height().cmp(&a.1.common_main.height()));
    let l_skip = engine.config().l_skip;
    let mut pvs = vec![vec![]; vk.inner.per_air.len()];
    let (trace_id_to_air_ids, ns): (Vec<_>, Vec<_>) = ctx
        .per_trace
        .iter()
        .map(|(air_idx, air_ctx)| {
            pvs[*air_idx] = air_ctx.public_values.clone();
            (
                *air_idx,
                log2_strict_usize(air_ctx.common_main.height()) - l_skip,
            )
        })
        .multiunzip();
    debug!(?trace_id_to_air_ids);
    debug!(n_per_trace = ?ns);
    let omega_pows = F::two_adic_generator(l_skip)
        .powers()
        .take(1 << l_skip)
        .collect_vec();

    let mut transcript = DuplexSponge::default();
    let (gkr_proof, batch_proof, _) = prove_zerocheck_and_logup(&mut transcript, &pk, &ctx);
    let mut transcript = DuplexSponge::default();
    verify_zerocheck_and_logup(
        &mut transcript,
        &vk.inner,
        &pvs,
        &gkr_proof,
        &batch_proof,
        &trace_id_to_air_ids,
        &ns,
        &omega_pows,
    )
    .unwrap();
    Ok(())
}
