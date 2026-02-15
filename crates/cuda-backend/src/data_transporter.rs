use std::{cmp::max, fmt::Debug, sync::Arc};

use itertools::Itertools;
use openvm_cuda_common::{
    copy::{cuda_memcpy, MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    error::MemCopyError,
    stream::current_stream_sync,
};
use openvm_stark_backend::{
    keygen::types::MultiStarkProvingKey,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::{
        stacked_pcs::{MerkleTree, StackedPcsData},
        AirProvingContext, ColMajorMatrix, CommittedTraceData, CpuBackend, DeviceDataTransporter,
        DeviceMultiStarkProvingKey, DeviceStarkProvingKey, MatrixDimensions, MatrixView,
        ProvingContext,
    },
};
use tracing::debug;

use crate::{
    base::DeviceMatrix,
    cuda::matrix::{collapse_strided_matrix, matrix_transpose_fp},
    merkle_tree::MerkleTreeGpu,
    poly::PleMatrix,
    prelude::{Digest, F, SC},
    stacked_pcs::StackedPcsDataGpu,
    AirDataGpu, GpuBackend, GpuDevice, GpuProverConfig, ProverError,
};

impl DeviceDataTransporter<SC, GpuBackend> for GpuDevice {
    fn transport_pk_to_device(
        &self,
        mpk: &MultiStarkProvingKey<SC>,
    ) -> DeviceMultiStarkProvingKey<GpuBackend> {
        let per_air = mpk
            .per_air
            .iter()
            .map(|pk| {
                let preprocessed_data = pk.preprocessed_data.as_ref().map(|d| {
                    transport_and_unstack_single_data_h2d(d.as_ref(), &self.prover_config).unwrap()
                });
                let other_data = AirDataGpu::new(pk).unwrap();
                let num_monomials = other_data
                    .zerocheck_monomials
                    .as_ref()
                    .map(|m| m.num_monomials)
                    .unwrap_or(0);
                debug!(air = %pk.air_name, num_monomials, "monomial expansion");

                DeviceStarkProvingKey {
                    air_name: pk.air_name.clone(),
                    vk: pk.vk.clone(),
                    preprocessed_data,
                    other_data,
                }
            })
            .collect();
        // Synchronize in case the proving key is shared between threads/streams
        current_stream_sync().unwrap();

        DeviceMultiStarkProvingKey::new(
            per_air,
            mpk.trace_height_constraints.clone(),
            mpk.max_constraint_degree,
            mpk.params.clone(),
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

pub fn transport_matrix_h2d_row(
    matrix: &RowMajorMatrix<F>,
) -> Result<DeviceMatrix<F>, MemCopyError> {
    let data = matrix.values.as_slice();
    let input_buffer = data.to_device().unwrap();
    let output = DeviceMatrix::<F>::with_capacity(matrix.height(), matrix.width());
    unsafe {
        matrix_transpose_fp(
            output.buffer(),
            &input_buffer,
            matrix.width(),
            matrix.height(),
        )?;
    }
    assert_eq!(output.strong_count(), 1);
    Ok(output)
}

/// `d` must be the stacked pcs data of a single trace matrix.
/// This function will transport `d` to device and then unstack it (allocating device memory) to
/// return `CommittedTraceData<F, Digest>`.
pub fn transport_and_unstack_single_data_h2d(
    d: &StackedPcsData<F, Digest>,
    prover_config: &GpuProverConfig,
) -> Result<CommittedTraceData<GpuBackend>, ProverError> {
    debug_assert!(d
        .layout
        .sorted_cols
        .iter()
        .all(|(mat_idx, _, _)| *mat_idx == 0));
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
            )
            .map_err(ProverError::CollapseStrided)?;
        }
        // Wait for kernel to finish so we can safely drop strided_trace
        current_stream_sync().map_err(ProverError::CurrentStreamSync)?;
        drop(strided_trace);
        buf
    };
    let d_matrix = prover_config
        .cache_stacked_matrix
        .then(|| PleMatrix::from_evals(l_skip, d_matrix_evals, stacked_height, stacked_width));
    let d_tree = transport_merkle_tree_h2d(&d.tree, prover_config.cache_rs_code_matrix)?;
    let d_data = StackedPcsDataGpu {
        layout: d.layout.clone(),
        matrix: d_matrix,
        tree: d_tree,
    };
    // Sanity check. Not a strong assert because we transport the merkle tree
    // instead of recomputing it above.
    assert_eq!(d_data.tree.root(), d.commit());
    Ok(CommittedTraceData {
        commitment: d.commit(),
        trace: DeviceMatrix::new(Arc::new(trace_buffer), height, width),
        data: Arc::new(d_data),
    })
}

/// Transports backing matrix and tree digest layers from host to device.
pub fn transport_merkle_tree_h2d<F, Digest: Clone>(
    tree: &MerkleTree<F, Digest>,
    cache_backing_matrix: bool,
) -> Result<MerkleTreeGpu<F, Digest>, MemCopyError> {
    let backing_matrix = if cache_backing_matrix {
        Some(transport_matrix_h2d_col_major(tree.backing_matrix())?)
    } else {
        None
    };
    let digest_layers = tree
        .digest_layers()
        .iter()
        .map(|layer| layer.to_device())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MerkleTreeGpu {
        backing_matrix,
        digest_layers,
        rows_per_query: tree.rows_per_query(),
        root: tree.root(),
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
    let d_tree = transport_merkle_tree_h2d(tree, prover_config.cache_rs_code_matrix)?;

    Ok(StackedPcsDataGpu {
        layout: layout.clone(),
        matrix: d_matrix,
        tree: d_tree,
    })
}

pub fn transport_air_proving_ctx_to_device(
    cpu_ctx: AirProvingContext<CpuBackend<SC>>,
) -> AirProvingContext<GpuBackend> {
    assert!(
        cpu_ctx.cached_mains.is_empty(),
        "CPU to GPU transfer of cached traces not supported"
    );
    let trace = transport_matrix_h2d_col_major(&cpu_ctx.common_main).unwrap();
    AirProvingContext {
        cached_mains: vec![],
        common_main: trace,
        public_values: cpu_ctx.public_values,
    }
}

pub fn transport_proving_ctx_to_host(
    gpu_ctx: ProvingContext<GpuBackend>,
    l_skip: usize,
) -> ProvingContext<CpuBackend<SC>> {
    let per_trace = gpu_ctx
        .per_trace
        .into_iter()
        .map(|(i, ctx)| (i, transport_air_proving_ctx_to_host(ctx, l_skip)))
        .collect_vec();
    ProvingContext { per_trace }
}

pub fn transport_air_proving_ctx_to_host(
    gpu_ctx: AirProvingContext<GpuBackend>,
    l_skip: usize,
) -> AirProvingContext<CpuBackend<SC>> {
    let trace = transport_matrix_d2h_col_major(&gpu_ctx.common_main).unwrap();
    let cached_mains = gpu_ctx
        .cached_mains
        .into_iter()
        .map(|mat| {
            // WARNING: By default this matrix isn't cached. For this to work, ensure that
            // GpuProverConfig fields cache_stacked_matrix and cache_rs_code_matrix are set
            // to be true.
            let evals_matrix = mat.data.matrix.as_ref().unwrap().to_evals(l_skip).unwrap();
            CommittedTraceData {
                commitment: mat.commitment,
                trace: transport_matrix_d2h_col_major(&mat.trace).unwrap(),
                data: Arc::new(StackedPcsData {
                    layout: mat.data.layout.clone(),
                    matrix: transport_matrix_d2h_col_major(&evals_matrix).unwrap(),
                    tree: transport_merkle_tree_to_host(&mat.data.tree),
                }),
            }
        })
        .collect_vec();
    AirProvingContext {
        cached_mains,
        common_main: trace,
        public_values: gpu_ctx.public_values,
    }
}

pub fn transport_matrix_d2h_col_major<T>(
    matrix: &DeviceMatrix<T>,
) -> Result<ColMajorMatrix<T>, MemCopyError> {
    let values_host = matrix.buffer().to_host()?;
    Ok(ColMajorMatrix::new(values_host, matrix.width()))
}

pub fn transport_matrix_d2h_row_major(
    matrix: &DeviceMatrix<F>,
) -> Result<RowMajorMatrix<F>, MemCopyError> {
    let matrix_buffer = DeviceBuffer::<F>::with_capacity(matrix.height() * matrix.width());
    unsafe {
        matrix_transpose_fp(
            &matrix_buffer,
            matrix.buffer(),
            matrix.height(),
            matrix.width(),
        )?;
    }
    Ok(RowMajorMatrix::<F>::new(
        matrix_buffer.to_host()?,
        matrix.width(),
    ))
}

/// For debugging purposes only.
///
/// # Panics
/// If `tree.backing_matrix` is `None`.
pub fn transport_merkle_tree_to_host(tree: &MerkleTreeGpu<F, Digest>) -> MerkleTree<F, Digest> {
    let backing_matrix =
        transport_matrix_d2h_col_major(tree.backing_matrix.as_ref().unwrap()).unwrap();
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

pub fn assert_eq_device_matrix<T: Clone + Send + Sync + PartialEq + Debug>(
    a: &DeviceMatrix<T>,
    b: &DeviceMatrix<T>,
) {
    assert_eq!(a.height(), b.height());
    assert_eq!(a.width(), b.width());
    assert_eq!(a.buffer().len(), b.buffer().len());
    let a_host = a.to_host().unwrap();
    let b_host = b.to_host().unwrap();
    for r in 0..a.height() {
        for c in 0..a.width() {
            assert_eq!(
                a_host[c * a.height() + r],
                b_host[c * b.height() + r],
                "Mismatch at row {} column {}",
                r,
                c
            );
        }
    }
}

pub fn assert_eq_host_and_device_matrix<T: Clone + Send + Sync + PartialEq + Debug>(
    cpu: Arc<RowMajorMatrix<T>>,
    gpu: &DeviceMatrix<T>,
) {
    assert_eq!(gpu.width(), cpu.width());
    assert_eq!(gpu.height(), cpu.height());
    let gpu = gpu.to_host().unwrap();
    for r in 0..cpu.height() {
        for c in 0..cpu.width() {
            assert_eq!(
                gpu[c * cpu.height() + r],
                cpu.get(r, c).expect("matrix index out of bounds"),
                "Mismatch at row {} column {}",
                r,
                c
            );
        }
    }
}
