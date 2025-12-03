use std::{cmp::max, fmt::Debug, sync::Arc};

use itertools::Itertools;
use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D, cuda_memcpy},
    d_buffer::DeviceBuffer,
    error::MemCopyError,
    memory_manager::MemTracker,
    stream::current_stream_sync,
};
use openvm_stark_backend::prover::hal::MatrixDimensions;
use stark_backend_v2::{
    SystemParams,
    keygen::types::MultiStarkProvingKeyV2,
    poly_common::Squarable,
    poseidon2::sponge::FiatShamirTranscript,
    proof::*,
    prover::{
        AirProvingContextV2, ColMajorMatrix, CommittedTraceDataV2, CpuBackendV2,
        DeviceDataTransporterV2, DeviceMultiStarkProvingKeyV2, DeviceStarkProvingKeyV2, MatrixView,
        MultiRapProver, OpeningProverV2, ProverBackendV2, ProverDeviceV2, ProvingContextV2,
        TraceCommitterV2, prove_zerocheck_and_logup,
        stacked_pcs::{MerkleTree, StackedPcsData},
    },
};

use crate::{
    D_EF, Digest, EF, F, GpuDeviceV2, GpuProverConfig, ProverError,
    cuda::matrix::collapse_strided_matrix,
    logup_zerocheck::LogupZerocheckGpu,
    merkle_tree::MerkleTreeGpu,
    poly::PleMatrix,
    stacked_pcs::{StackedPcsDataGpu, stacked_commit},
    stacked_reduction::prove_stacked_opening_reduction_gpu,
    whir::prove_whir_opening_gpu,
};

#[derive(Clone, Copy)]
pub struct GpuBackendV2;

impl ProverBackendV2 for GpuBackendV2 {
    const CHALLENGE_EXT_DEGREE: u8 = D_EF as u8;

    type Val = F;
    type Challenge = EF;
    type Commitment = Digest;
    type Matrix = DeviceMatrix<F>;
    type PcsData = StackedPcsDataGpu<F, Digest>;
}

impl<TS: FiatShamirTranscript> ProverDeviceV2<GpuBackendV2, TS> for GpuDeviceV2 {
    fn config(&self) -> SystemParams {
        self.config()
    }
}

impl TraceCommitterV2<GpuBackendV2> for GpuDeviceV2 {
    fn commit(&self, traces: &[&DeviceMatrix<F>]) -> (Digest, StackedPcsDataGpu<F, Digest>) {
        stacked_commit(
            self.config.l_skip,
            self.config.n_stack,
            self.config.log_blowup,
            self.config.k_whir,
            traces,
            self.prover_config.cache_stacked_matrix,
        )
        .unwrap()
    }
}

impl<TS: FiatShamirTranscript> MultiRapProver<GpuBackendV2, TS> for GpuDeviceV2 {
    type PartialProof = (GkrProof, BatchConstraintProof);
    /// The random opening point `r` where the batch constraint sumcheck reduces to evaluation
    /// claims of trace matrices `T, T_{rot}` at `r_{n_T}`.
    type Artifacts = Vec<EF>;

    fn prove_rap_constraints(
        &self,
        transcript: &mut TS,
        mpk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
        ctx: &ProvingContextV2<GpuBackendV2>,
        common_main_pcs_data: &StackedPcsDataGpu<F, Digest>,
    ) -> ((GkrProof, BatchConstraintProof), Vec<EF>) {
        let mem = MemTracker::start_and_reset_peak("prover.rap_constraints");
        let (gkr_proof, batch_constraint_proof, r) =
            prove_zerocheck_and_logup::<_, _, TS, LogupZerocheckGpu>(
                self,
                transcript,
                mpk,
                ctx,
                common_main_pcs_data,
            );
        mem.emit_metrics();
        ((gkr_proof, batch_constraint_proof), r)
    }
}

impl<TS: FiatShamirTranscript> OpeningProverV2<GpuBackendV2, TS> for GpuDeviceV2 {
    type OpeningProof = (StackingProof, WhirProof);
    /// The shared vector `r` where each trace matrix `T, T_{rot}` is opened at `r_{n_T}`.
    type OpeningPoints = Vec<EF>;

    fn prove_openings(
        &self,
        transcript: &mut TS,
        mpk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
        ctx: ProvingContextV2<GpuBackendV2>,
        common_main_pcs_data: StackedPcsDataGpu<F, Digest>,
        r: Vec<EF>,
    ) -> (StackingProof, WhirProof) {
        let mut mem = MemTracker::start_and_reset_peak("prover.openings");
        let params = self.config;
        let (stacking_proof, u_prisma, stacked_per_commit) = prove_stacked_opening_reduction_gpu(
            self,
            transcript,
            mpk,
            ctx,
            common_main_pcs_data,
            &r,
        )
        .unwrap();

        let (&u0, u_rest) = u_prisma.split_first().unwrap();
        let u_cube = u0
            .exp_powers_of_2()
            .take(params.l_skip)
            .chain(u_rest.iter().copied())
            .collect_vec();

        let whir_proof =
            prove_whir_opening_gpu(&params, transcript, stacked_per_commit, &u_cube).unwrap();
        mem.emit_metrics();
        mem.reset_peak();
        (stacking_proof, whir_proof)
    }
}

impl DeviceDataTransporterV2<GpuBackendV2> for GpuDeviceV2 {
    fn transport_pk_to_device(
        &self,
        mpk: &MultiStarkProvingKeyV2,
    ) -> DeviceMultiStarkProvingKeyV2<GpuBackendV2> {
        let per_air = mpk
            .per_air
            .iter()
            .map(|pk| {
                let preprocessed_data = pk.preprocessed_data.as_ref().map(|d| {
                    transport_and_unstack_single_data_h2d(d.as_ref(), &self.prover_config).unwrap()
                });

                DeviceStarkProvingKeyV2 {
                    air_name: pk.air_name.clone(),
                    vk: pk.vk.clone(),
                    preprocessed_data,
                }
            })
            .collect();
        // Synchronize in case the proving key is shared between threads/streams
        current_stream_sync().unwrap();

        DeviceMultiStarkProvingKeyV2::new(
            per_air,
            mpk.trace_height_constraints.clone(),
            mpk.max_constraint_degree,
            mpk.params,
            mpk.vk_pre_hash,
        )
    }

    fn transport_matrix_to_device(&self, matrix: &ColMajorMatrix<F>) -> DeviceMatrix<F> {
        transport_matrix_h2d_col_major(matrix).unwrap()
    }

    fn transport_pcs_data_to_device(
        &self,
        pcs_data: &StackedPcsData<F, Digest>,
    ) -> StackedPcsDataGpu<F, Digest> {
        transport_pcs_data_h2d(pcs_data, &self.prover_config).unwrap()
    }

    fn transport_matrix_from_device_to_host(&self, matrix: &DeviceMatrix<F>) -> ColMajorMatrix<F> {
        transport_matrix_d2h_col_major(matrix).unwrap()
    }
}

pub fn transport_matrix_h2d_col_major<T>(
    matrix: &ColMajorMatrix<T>,
) -> Result<DeviceMatrix<T>, MemCopyError> {
    // matrix is already col-major, so this is just H2D buffer transfer
    let buffer = matrix.values.to_device()?;
    Ok(DeviceMatrix::new(
        Arc::new(buffer),
        matrix.height(),
        matrix.width(),
    ))
}

/// `d` must be the stacked pcs data of a single trace matrix.
/// This function will transport `d` to device and then unstack it (allocating device memory) to
/// return `CommittedTraceDataV2<F, Digest>`.
pub fn transport_and_unstack_single_data_h2d(
    d: &StackedPcsData<F, Digest>,
    prover_config: &GpuProverConfig,
) -> Result<CommittedTraceDataV2<GpuBackendV2>, ProverError> {
    debug_assert!(
        d.layout
            .sorted_cols
            .iter()
            .all(|(mat_idx, _, _)| *mat_idx == 0)
    );
    let l_skip = d.layout.l_skip();
    let trace_view = d.mat_view(0);
    let height = trace_view.height();
    let width = trace_view.width();
    let stride = trace_view.stride();
    let lifted_height = height * stride;
    debug_assert_eq!(lifted_height, max(height, 1 << l_skip));
    debug_assert_eq!(lifted_height * width, trace_view.values().len());
    debug_assert!(d.matrix.values.len() >= lifted_height * width);
    let stacked_width = d.matrix.width();
    let stacked_height = d.matrix.height();
    let d_matrix_evals = d.matrix.values.to_device()?;
    let strided_trace = DeviceBuffer::<F>::with_capacity(lifted_height * width);
    // SAFETY: D2D copy
    // - `d_matrix_evals` is the stacked matrix, guaranteed to have length `>= lifted_height *
    //   width` by definition of stacking
    // - `d_matrix_evals` stacks a single trace matrix
    unsafe {
        cuda_memcpy::<true, true>(
            strided_trace.as_mut_raw_ptr(),
            d_matrix_evals.as_raw_ptr(),
            lifted_height * width * size_of::<F>(),
        )?;
    }
    let trace_buffer = if stride == 1 {
        strided_trace
    } else {
        let buf = DeviceBuffer::<F>::with_capacity(height * width);
        unsafe {
            collapse_strided_matrix(
                buf.as_mut_ptr(),
                strided_trace.as_ptr(),
                width as u32,
                height as u32,
                stride as u32,
            )?;
        }
        // Wait for kernel to finish so we can safely drop strided_trace
        current_stream_sync()?;
        drop(strided_trace);
        buf
    };
    let d_matrix = prover_config
        .cache_stacked_matrix
        .then(|| PleMatrix::from_evals(l_skip, d_matrix_evals, stacked_height, stacked_width));
    let d_tree = transport_merkle_tree_h2d(&d.tree)?;
    let d_data = StackedPcsDataGpu {
        layout: d.layout.clone(),
        matrix: d_matrix,
        tree: d_tree,
    };
    // Sanity check. Not a strong assert because we transport the merkle tree
    // instead of recomputing it above.
    assert_eq!(d_data.tree.root(), d.commit());
    Ok(CommittedTraceDataV2 {
        commitment: d.commit(),
        trace: DeviceMatrix::new(Arc::new(trace_buffer), height, width),
        data: Arc::new(d_data),
    })
}

/// Transports backing matrix and tree digest layers from host to device.
pub fn transport_merkle_tree_h2d<F, Digest>(
    tree: &MerkleTree<F, Digest>,
) -> Result<MerkleTreeGpu<F, Digest>, MemCopyError> {
    let backing_matrix = transport_matrix_h2d_col_major(tree.backing_matrix())?;
    let digest_layers = tree
        .digest_layers()
        .iter()
        .map(|layer| layer.to_device())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MerkleTreeGpu {
        backing_matrix,
        digest_layers,
        rows_per_query: tree.rows_per_query(),
    })
}

pub fn transport_pcs_data_h2d(
    pcs_data: &StackedPcsData<F, Digest>,
    prover_config: &GpuProverConfig,
) -> Result<StackedPcsDataGpu<F, Digest>, ProverError> {
    let StackedPcsData {
        layout,
        matrix,
        tree,
    } = pcs_data;
    let width = matrix.width();
    let height = matrix.height();
    let d_matrix_evals = matrix.values.to_device()?;
    let d_matrix = prover_config
        .cache_stacked_matrix
        .then(|| PleMatrix::from_evals(layout.l_skip(), d_matrix_evals, height, width));
    let d_tree = transport_merkle_tree_h2d(tree)?;

    Ok(StackedPcsDataGpu {
        layout: layout.clone(),
        matrix: d_matrix,
        tree: d_tree,
    })
}

pub fn transport_air_proving_ctx_to_device(
    cpu_ctx: AirProvingContextV2<CpuBackendV2>,
) -> AirProvingContextV2<GpuBackendV2> {
    assert!(
        cpu_ctx.cached_mains.is_empty(),
        "CPU to GPU transfer of cached traces not supported"
    );
    let trace = transport_matrix_h2d_col_major(&cpu_ctx.common_main).unwrap();
    AirProvingContextV2 {
        cached_mains: vec![],
        common_main: trace,
        public_values: cpu_ctx.public_values,
    }
}

pub fn transport_matrix_d2h_col_major<T>(
    matrix: &DeviceMatrix<T>,
) -> Result<ColMajorMatrix<T>, MemCopyError> {
    let values_host = matrix.buffer().to_host()?;
    Ok(ColMajorMatrix::new(values_host, matrix.width()))
}

pub fn transport_merkle_tree_to_host(tree: &MerkleTreeGpu<F, Digest>) -> MerkleTree<F, Digest> {
    let backing_matrix = transport_matrix_d2h_col_major(&tree.backing_matrix).unwrap();
    let digest_layers = tree
        .digest_layers
        .iter()
        .map(|layer| layer.to_host().unwrap())
        .collect_vec();
    // Safety: assuming the tree is properly constructed on device, the layers are correct after D2H
    // transfer.
    unsafe {
        MerkleTree::<F, Digest>::from_raw_parts(backing_matrix, digest_layers, tree.rows_per_query)
    }
}

pub fn assert_eq_host_and_device_matrix_col_maj<T: Clone + Send + Sync + PartialEq + Debug>(
    cpu: &ColMajorMatrix<T>,
    gpu: &DeviceMatrix<T>,
) {
    assert_eq!(gpu.width(), cpu.width());
    assert_eq!(gpu.height(), cpu.height());
    let gpu = gpu.to_host().unwrap();
    for r in 0..cpu.height() {
        for c in 0..cpu.width() {
            assert_eq!(
                gpu[c * cpu.height() + r],
                *cpu.get(r, c).unwrap(),
                "Mismatch at row {} column {}",
                r,
                c
            );
        }
    }
}

mod v1_shims {
    use std::sync::Arc;

    use openvm_cuda_backend::{base::DeviceMatrix, prover_backend::GpuBackend};
    use stark_backend_v2::{SystemParams, prover::CommittedTraceDataV2, v1_shims::V1Compat};

    use crate::{F, GpuBackendV2, stacked_pcs::stacked_commit};

    impl V1Compat for GpuBackendV2 {
        type V1 = GpuBackend;

        fn dummy_matrix() -> DeviceMatrix<F> {
            DeviceMatrix::dummy()
        }

        fn convert_trace(matrix: DeviceMatrix<F>) -> DeviceMatrix<F> {
            matrix
        }

        fn convert_committed_trace(
            params: SystemParams,
            matrix: DeviceMatrix<F>,
        ) -> CommittedTraceDataV2<GpuBackendV2> {
            let (commitment, data) = stacked_commit(
                params.l_skip,
                params.n_stack,
                params.log_blowup,
                params.k_whir,
                &[&matrix],
                false,
            )
            .unwrap();
            CommittedTraceDataV2 {
                commitment,
                data: Arc::new(data),
                trace: matrix,
            }
        }
    }
}
