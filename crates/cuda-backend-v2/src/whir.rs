use std::sync::Arc;

use stark_backend_v2::{
    poseidon2::sponge::FiatShamirTranscript,
    proof::WhirProof,
    prover::{CpuDeviceV2, stacked_pcs::StackedPcsData, whir::WhirProver},
};

use crate::{
    Digest, EF, F, GpuBackendV2, GpuDeviceV2, gpu_backend::transport_stacked_pcs_data_to_host,
    stacked_pcs::StackedPcsDataGpu,
};

impl<TS: FiatShamirTranscript> WhirProver<GpuBackendV2, GpuDeviceV2, TS> for GpuDeviceV2 {
    fn prove_whir(
        &self,
        transcript: &mut TS,
        common_main_pcs_data: StackedPcsDataGpu<F, Digest>,
        pre_cached_pcs_data_per_commit: Vec<Arc<StackedPcsDataGpu<F, Digest>>>,
        u_cube: &[EF],
    ) -> WhirProof {
        let host_common = transport_stacked_pcs_data_to_host(&common_main_pcs_data);
        let host_pre_cached: Vec<Arc<StackedPcsData<F, Digest>>> = pre_cached_pcs_data_per_commit
            .into_iter()
            .map(|data| Arc::new(transport_stacked_pcs_data_to_host(data.as_ref())))
            .collect();
        CpuDeviceV2::new(self.config).prove_whir(transcript, host_common, host_pre_cached, u_cube)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use itertools::Itertools;
    use openvm_stark_sdk::config::setup_tracing_with_log_level;
    use p3_field::FieldAlgebra;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use stark_backend_v2::{
        BabyBearPoseidon2CpuEngineV2, EF, F,
        keygen::types::{MultiStarkProvingKeyV2, SystemParams},
        poly_common::Squarable,
        poseidon2::sponge::{DuplexSponge, DuplexSpongeRecorder, TranscriptHistory},
        prover::{
            ColMajorMatrix, CpuBackendV2, DeviceDataTransporterV2, ProvingContextV2, poly::Ple,
            stacked_pcs::stacked_commit, whir::WhirProver,
        },
        test_utils::{DuplexSpongeValidator, FibFixture, TestFixture},
        verifier::whir::{VerifyWhirError, verify_whir},
    };
    use tracing::Level;

    use crate::GpuDeviceV2;

    fn generate_random_z(params: &SystemParams, rng: &mut StdRng) -> (Vec<EF>, Vec<EF>) {
        let z_prism: Vec<_> = (0..params.n_stack + 1)
            .map(|_| EF::from_wrapped_u64(rng.random()))
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
        params: SystemParams,
        pk: MultiStarkProvingKeyV2,
        ctx: ProvingContextV2<CpuBackendV2>,
    ) -> Result<(), VerifyWhirError> {
        let device = GpuDeviceV2::new(params);
        let mut rng = StdRng::seed_from_u64(0);
        let (z_prism, z_cube) = generate_random_z(&params, &mut rng);

        let (common_main_commit, common_main_pcs_data) = {
            let traces = ctx
                .common_main_traces()
                .map(|(_, trace)| trace)
                .collect_vec();
            stacked_commit(
                params.l_skip,
                params.n_stack,
                params.log_blowup,
                params.k_whir,
                &traces,
            )
        };
        let d_common_main_pcs_data = device.transport_pcs_data_to_device(&common_main_pcs_data);

        let mut stacking_openings = vec![stacking_openings_for_matrix(
            &params,
            &z_prism,
            &common_main_pcs_data.matrix,
        )];
        let mut commits = vec![common_main_commit];
        let mut pre_cached_pcs_data_per_commit = Vec::new();
        for (air_id, air_ctx) in ctx.per_trace {
            let pcs_datas = pk.per_air[air_id]
                .preprocessed_data
                .iter()
                .chain(air_ctx.cached_mains.iter().map(|(_, d)| d));
            for data in pcs_datas {
                commits.push(data.commit());
                stacking_openings.push(stacking_openings_for_matrix(
                    &params,
                    &z_prism,
                    &data.matrix,
                ));
                let d_data = device.transport_pcs_data_to_device(data);
                pre_cached_pcs_data_per_commit.push(Arc::new(d_data));
            }
        }

        let mut prover_sponge = DuplexSpongeRecorder::default();

        let proof = device.prove_whir(
            &mut prover_sponge,
            d_common_main_pcs_data,
            pre_cached_pcs_data_per_commit,
            &z_cube,
        );

        let mut verifier_sponge = DuplexSpongeValidator::new(prover_sponge.into_log());
        verify_whir(
            &mut verifier_sponge,
            &params,
            &proof,
            &stacking_openings,
            &commits,
            &z_cube,
        )
    }

    fn run_whir_fib_test(params: SystemParams) -> Result<(), VerifyWhirError> {
        let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);
        let fib = FibFixture::new(0, 1, 1 << (params.n_stack + params.l_skip));
        let (pk, _vk) = fib.keygen(&engine);
        let ctx = fib.generate_proving_ctx();
        run_whir_test(params, pk, ctx)
    }

    #[test]
    fn test_whir_single_fib_nstack_0() -> Result<(), VerifyWhirError> {
        setup_tracing_with_log_level(Level::DEBUG);

        let params = SystemParams {
            l_skip: 2,
            n_stack: 0,
            log_blowup: 1,
            k_whir: 1,
            num_whir_queries: 5,
            log_final_poly_len: 0,
            logup_pow_bits: 1,
            whir_pow_bits: 0,
        };
        run_whir_fib_test(params)
    }

    #[test]
    fn test_whir_single_fib_nstack_2() -> Result<(), VerifyWhirError> {
        setup_tracing_with_log_level(Level::DEBUG);

        let params = SystemParams {
            l_skip: 2,
            n_stack: 2,
            log_blowup: 1,
            k_whir: 1,
            num_whir_queries: 5,
            log_final_poly_len: 2,
            logup_pow_bits: 1,
            whir_pow_bits: 0,
        };
        run_whir_fib_test(params)
    }

    #[test]
    fn test_whir_single_fib_kwhir_2() -> Result<(), VerifyWhirError> {
        setup_tracing_with_log_level(Level::DEBUG);

        let params = SystemParams {
            l_skip: 2,
            n_stack: 2,
            log_blowup: 1,
            k_whir: 2,
            num_whir_queries: 5,
            log_final_poly_len: 0,
            logup_pow_bits: 1,
            whir_pow_bits: 0,
        };
        run_whir_fib_test(params)
    }

    #[test]
    fn test_whir_single_fib_kwhir_3() -> Result<(), VerifyWhirError> {
        setup_tracing_with_log_level(Level::DEBUG);

        let params = SystemParams {
            l_skip: 2,
            n_stack: 2,
            log_blowup: 1,
            k_whir: 3,
            num_whir_queries: 5,
            log_final_poly_len: 1,
            logup_pow_bits: 1,
            whir_pow_bits: 0,
        };
        run_whir_fib_test(params)
    }

    #[test]
    fn test_whir_single_fib_kwhir_4() -> Result<(), VerifyWhirError> {
        setup_tracing_with_log_level(Level::DEBUG);

        let params = SystemParams {
            l_skip: 2,
            n_stack: 2,
            log_blowup: 1,
            k_whir: 4,
            num_whir_queries: 5,
            log_final_poly_len: 0,
            logup_pow_bits: 1,
            whir_pow_bits: 0,
        };
        run_whir_fib_test(params)
    }

    #[test]
    fn test_whir_single_fib_log_blowup_2() -> Result<(), VerifyWhirError> {
        setup_tracing_with_log_level(Level::DEBUG);

        let params = SystemParams {
            l_skip: 2,
            n_stack: 2,
            log_blowup: 1,
            k_whir: 4,
            num_whir_queries: 5,
            log_final_poly_len: 0,
            logup_pow_bits: 2,
            whir_pow_bits: 0,
        };
        run_whir_fib_test(params)
    }

    // TODO: test multiple commitments
}
