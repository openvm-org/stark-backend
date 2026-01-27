use itertools::Itertools;
use openvm_stark_backend::{
    duplex_sponge::DuplexSpongeValidator,
    poly_common::Squarable,
    prover::{
        poly::Ple, stacked_pcs::stacked_commit, whir::prove_whir_opening, ColMajorMatrix,
        CpuBackend, DeviceDataTransporter, DeviceMultiStarkProvingKey, MatrixDimensions,
        ProvingContext,
    },
    test_utils::{test_whir_config_small, FibFixture, TestFixture},
    verifier::whir::{binary_k_fold, verify_whir, VerifyWhirError},
    StarkEngine, StarkProtocolConfig, SystemParams, TranscriptHistory, WhirConfig, WhirRoundConfig,
};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::*, log_up_params::log_up_security_params_baby_bear_100_bits},
    utils::setup_tracing_with_log_level,
};
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use rand::{rngs::StdRng, Rng, SeedableRng};
use test_case::test_case;
use tracing::Level;

type SC = BabyBearPoseidon2Config;

fn generate_random_z(params: &SystemParams, rng: &mut StdRng) -> (Vec<EF>, Vec<EF>) {
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

fn stacking_openings_for_matrix(
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

fn run_whir_test(
    config: &SC,
    pk: DeviceMultiStarkProvingKey<CpuBackend<SC>>,
    ctx: &ProvingContext<CpuBackend<SC>>,
) -> Result<(), VerifyWhirError> {
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
        )
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
            commits.push(data.commit());
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
    );

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
    )
}

fn run_whir_fib_test(params: SystemParams) -> Result<(), VerifyWhirError> {
    let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params.clone());
    let fib = FibFixture::new(0, 1, 1 << params.log_stacked_height());
    let (pk, _vk) = fib.keygen(&engine);
    let pk = engine.device().transport_pk_to_device(&pk);
    let ctx = fib.generate_proving_ctx();
    run_whir_test(engine.config(), pk, &ctx)
}

#[test_case(0, 1, 1, 0)]
#[test_case(2, 1, 1, 2)]
#[test_case(2, 1, 2, 0)]
#[test_case(2, 1, 3, 1)]
#[test_case(2, 1, 4, 0)]
#[test_case(2, 2, 4, 0)]
fn test_whir_single_fib(
    n_stack: usize,
    log_blowup: usize,
    k_whir: usize,
    log_final_poly_len: usize,
) -> Result<(), VerifyWhirError> {
    setup_tracing_with_log_level(Level::DEBUG);
    let l_skip = 2;
    let whir = test_whir_config_small(log_blowup, l_skip + n_stack, k_whir, log_final_poly_len);

    let params = SystemParams {
        l_skip,
        n_stack,
        log_blowup,
        whir,
        logup: log_up_security_params_baby_bear_100_bits(),
        max_constraint_degree: 3,
    };
    run_whir_fib_test(params)
}

#[test]
fn test_fold_single() {
    let mut rng = StdRng::seed_from_u64(0);

    let a0 = EF::from_u32(rng.random());
    let a1 = EF::from_u32(rng.random());
    let alpha = EF::from_u32(rng.random());
    let x = F::from_u32(rng.random());

    let result = binary_k_fold(vec![a0, a1], &[alpha], x);
    assert_eq!(result, a0 + (alpha - x) * (a0 - a1) * x.double().inverse());
}

#[test]
fn test_fold_double() {
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

fn whir_test_config(k_whir: usize) -> WhirConfig {
    WhirConfig {
        k: k_whir,
        rounds: vec![
            WhirRoundConfig {
                num_queries: 6,
                num_ood_samples: 1,
            },
            WhirRoundConfig {
                num_queries: 5,
                num_ood_samples: 1,
            },
        ],
        mu_pow_bits: 1,
        query_phase_pow_bits: 1,
        folding_pow_bits: 1,
    }
}

#[test]
fn test_whir_multiple_commitments() -> Result<(), VerifyWhirError> {
    setup_tracing_with_log_level(Level::DEBUG);

    let mut rng = StdRng::seed_from_u64(42);

    let params = SystemParams {
        l_skip: 3,
        n_stack: 3,
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
        );

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
    );

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
    )
}

#[test]
fn test_whir_multiple_commitments_negative() {
    setup_tracing_with_log_level(Level::DEBUG);

    let mut rng = StdRng::seed_from_u64(42);

    let params = SystemParams {
        l_skip: 3,
        n_stack: 3,
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
        );

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
    );

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
