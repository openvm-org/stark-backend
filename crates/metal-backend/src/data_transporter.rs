use std::{cmp::max, sync::Arc};

use openvm_metal_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::MetalBuffer,
};
use openvm_stark_backend::{
    keygen::types::MultiStarkProvingKey,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::{
        stacked_pcs::{MerkleTree, StackedPcsData},
        ColMajorMatrix, CommittedTraceData, DeviceDataTransporter, DeviceMultiStarkProvingKey,
        DeviceStarkProvingKey, MatrixDimensions,
    },
};
use tracing::debug;

use crate::{
    base::MetalMatrix,
    merkle_tree::MerkleTreeMetal,
    metal::matrix::{collapse_strided_matrix, matrix_transpose_fp},
    poly::PleMatrix,
    prelude::{Digest, F, SC},
    stacked_pcs::StackedPcsDataMetal,
    AirDataMetal, MetalBackend, MetalDevice, MetalProverConfig, ProverError,
};

impl DeviceDataTransporter<SC, MetalBackend> for MetalDevice {
    fn transport_pk_to_device(
        &self,
        mpk: &MultiStarkProvingKey<SC>,
    ) -> DeviceMultiStarkProvingKey<MetalBackend> {
        debug!(num_airs = mpk.per_air.len(), "transport_pk_to_device start");
        let per_air = mpk
            .per_air
            .iter()
            .map(|pk| {
                let preprocessed_data = pk.preprocessed_data.as_ref().map(|d| {
                    transport_and_unstack_single_data_h2d(d.as_ref(), &self.prover_config).unwrap()
                });
                let other_data = AirDataMetal::new(pk);
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

        debug!("transport_pk_to_device done");
        DeviceMultiStarkProvingKey::new(
            per_air,
            mpk.trace_height_constraints.clone(),
            mpk.max_constraint_degree,
            mpk.params.clone(),
            mpk.vk_pre_hash,
        )
    }

    fn transport_matrix_to_device(&self, matrix: &ColMajorMatrix<F>) -> MetalMatrix<F> {
        transport_matrix_h2d_col_major(matrix)
    }

    fn transport_pcs_data_to_device(
        &self,
        pcs_data: &StackedPcsData<F, Digest>,
    ) -> StackedPcsDataMetal<F, Digest> {
        transport_pcs_data_h2d(pcs_data, &self.prover_config).unwrap()
    }

    fn transport_matrix_from_device_to_host(&self, matrix: &MetalMatrix<F>) -> ColMajorMatrix<F> {
        transport_matrix_d2h_col_major(matrix)
    }
}

pub fn transport_matrix_h2d_col_major<T>(matrix: &ColMajorMatrix<T>) -> MetalMatrix<T> {
    // matrix is already col-major, so this is just H2D buffer transfer
    let buffer = matrix.values.to_device();
    MetalMatrix::new(Arc::new(buffer), matrix.height(), matrix.width())
}

pub fn transport_matrix_h2d_row(matrix: &RowMajorMatrix<F>) -> MetalMatrix<F> {
    let data = matrix.values.as_slice();
    let input_buffer = data.to_device();
    let output = MetalMatrix::<F>::with_capacity(matrix.height(), matrix.width());
    unsafe {
        matrix_transpose_fp(
            output.buffer(),
            &input_buffer,
            matrix.width(),
            matrix.height(),
        )
        .unwrap();
    }
    assert_eq!(output.strong_count(), 1);
    output
}

/// `d` must be the stacked pcs data of a single trace matrix.
/// This function will transport `d` to device and then unstack it (allocating device memory) to
/// return `CommittedTraceData<F, Digest>`.
pub fn transport_and_unstack_single_data_h2d(
    d: &StackedPcsData<F, Digest>,
    prover_config: &MetalProverConfig,
) -> Result<CommittedTraceData<MetalBackend>, ProverError> {
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
    let d_matrix_evals = d.matrix.values.to_device();
    let strided_trace = MetalBuffer::<F>::with_capacity(lifted_height * width);
    trace_view.values().copy_to(&strided_trace)?;
    let trace_buffer = if stride == 1 {
        strided_trace
    } else {
        let buf = MetalBuffer::<F>::with_capacity(height * width);
        unsafe {
            collapse_strided_matrix(
                &buf,
                0,
                &strided_trace,
                0,
                width as u32,
                height as u32,
                stride as u32,
            )
            .map_err(ProverError::CollapseStrided)?;
        }
        drop(strided_trace);
        buf
    };
    let d_matrix = prover_config
        .cache_stacked_matrix
        .then(|| PleMatrix::from_evals(l_skip, d_matrix_evals, stacked_height, stacked_width));
    let d_tree = transport_merkle_tree_h2d(&d.tree, prover_config.cache_rs_code_matrix);
    let d_data = StackedPcsDataMetal {
        layout: d.layout.clone(),
        matrix: d_matrix,
        tree: d_tree,
    };
    // Sanity check.
    assert_eq!(d_data.tree.root(), d.commit());
    Ok(CommittedTraceData {
        commitment: d.commit(),
        trace: MetalMatrix::new(Arc::new(trace_buffer), height, width),
        data: Arc::new(d_data),
    })
}

/// Transports backing matrix and tree digest layers from host to device.
pub fn transport_merkle_tree_h2d<F, Digest: Clone>(
    tree: &MerkleTree<F, Digest>,
    cache_backing_matrix: bool,
) -> MerkleTreeMetal<F, Digest> {
    let backing_matrix = if cache_backing_matrix {
        Some(transport_matrix_h2d_col_major(tree.backing_matrix()))
    } else {
        None
    };
    let digest_layers = tree
        .digest_layers()
        .iter()
        .map(|layer| layer.to_device())
        .collect::<Vec<_>>();
    MerkleTreeMetal {
        backing_matrix,
        digest_layers,
        rows_per_query: tree.rows_per_query(),
        root: tree.root(),
    }
}

pub fn transport_pcs_data_h2d(
    pcs_data: &StackedPcsData<F, Digest>,
    prover_config: &MetalProverConfig,
) -> Result<StackedPcsDataMetal<F, Digest>, ProverError> {
    let StackedPcsData {
        layout,
        matrix,
        tree,
    } = pcs_data;
    let width = matrix.width();
    let height = matrix.height();
    let d_matrix_evals = matrix.values.to_device();
    let d_matrix = prover_config
        .cache_stacked_matrix
        .then(|| PleMatrix::from_evals(layout.l_skip(), d_matrix_evals, height, width));
    let d_tree = transport_merkle_tree_h2d(tree, prover_config.cache_rs_code_matrix);

    Ok(StackedPcsDataMetal {
        layout: layout.clone(),
        matrix: d_matrix,
        tree: d_tree,
    })
}

pub fn transport_matrix_d2h_col_major<T>(matrix: &MetalMatrix<T>) -> ColMajorMatrix<T> {
    let values_host = matrix.to_host();
    ColMajorMatrix::new(values_host, matrix.width())
}

pub fn transport_matrix_d2h_row_major(matrix: &MetalMatrix<F>) -> RowMajorMatrix<F> {
    let matrix_buffer = MetalBuffer::<F>::with_capacity(matrix.height() * matrix.width());
    unsafe {
        matrix_transpose_fp(
            &matrix_buffer,
            matrix.buffer(),
            matrix.height(),
            matrix.width(),
        )
        .unwrap();
    }
    RowMajorMatrix::<F>::new(matrix_buffer.to_host(), matrix.width())
}
