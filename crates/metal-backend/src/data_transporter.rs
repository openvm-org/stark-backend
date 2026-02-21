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
    metal::matrix::matrix_transpose_fp,
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
    let width = matrix.width();
    let height = matrix.height();
    let data = matrix.values.as_slice();
    if transpose_is_layout_identity(width, height) {
        debug!(
            height,
            width,
            transport_kernel_dispatches = 0,
            transport_sync_points = 0,
            transport_allocations = 1,
            "row-major upload reused contiguous layout"
        );
        let buffer = data.to_device();
        return MetalMatrix::new(Arc::new(buffer), height, width);
    }

    let input_buffer = data.to_device();
    let output = MetalMatrix::<F>::with_capacity(height, width);
    unsafe {
        matrix_transpose_fp(
            output.buffer(),
            &input_buffer,
            width,
            height,
        )
        .unwrap();
    }
    debug!(
        height,
        width,
        transport_kernel_dispatches = 1,
        transport_sync_points = 1,
        transport_allocations = 2,
        "row-major upload transpose kernel"
    );
    assert_eq!(output.strong_count(), 1);
    output
}

#[inline]
fn transpose_is_layout_identity(width: usize, height: usize) -> bool {
    width <= 1 || height <= 1
}

fn collapse_strided_trace_values<T: Copy>(
    src: &[T],
    dst: &mut [T],
    width: usize,
    height: usize,
    lifted_height: usize,
    stride: usize,
) {
    debug_assert_eq!(src.len(), lifted_height * width);
    debug_assert_eq!(dst.len(), height * width);
    debug_assert_eq!(lifted_height, height * stride);
    debug_assert!(stride > 0);
    for (src_col, dst_col) in src
        .chunks_exact(lifted_height)
        .zip(dst.chunks_exact_mut(height))
    {
        for (row_idx, dst_val) in dst_col.iter_mut().enumerate() {
            *dst_val = src_col[row_idx * stride];
        }
    }
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
    let cache_stacked_matrix = prover_config.cache_stacked_matrix;
    debug_assert_eq!(lifted_height, max(height, 1 << l_skip));
    debug_assert_eq!(lifted_height * width, trace_view.values().len());
    debug_assert!(d.matrix.values.len() >= lifted_height * width);
    let trace_buffer = if stride == 1 {
        debug!(
            height,
            width,
            stride,
            transport_allocations = 1 + usize::from(cache_stacked_matrix),
            transport_kernel_dispatches = 0,
            transport_sync_points = 0,
            cached_matrix_transport = cache_stacked_matrix,
            "trace transport direct contiguous upload"
        );
        trace_view.values().to_device()
    } else {
        let trace_buffer = MetalBuffer::<F>::with_capacity(height * width);
        // SAFETY:
        // - `trace_buffer` was freshly allocated in this function and has no concurrent GPU access.
        // - The returned slice length is exactly `height * width`, matching `dst`.
        // - Source and destination do not overlap.
        let trace_slice = unsafe { trace_buffer.as_mut_slice() };
        collapse_strided_trace_values(
            trace_view.values(),
            trace_slice,
            width,
            height,
            lifted_height,
            stride,
        );
        debug!(
            height,
            width,
            stride,
            lifted_height,
            transport_allocations = 1 + usize::from(cache_stacked_matrix),
            transport_kernel_dispatches = 0,
            transport_sync_points = 0,
            removed_kernel_dispatches = 1,
            removed_sync_points = 1,
            cached_matrix_transport = cache_stacked_matrix,
            "trace transport collapsed to contiguous layout during upload"
        );
        trace_buffer
    };
    let d_matrix = if cache_stacked_matrix {
        let stacked_width = d.matrix.width();
        let stacked_height = d.matrix.height();
        let d_matrix_evals = d.matrix.values.to_device();
        Some(PleMatrix::from_evals(
            l_skip,
            d_matrix_evals,
            stacked_height,
            stacked_width,
        ))
    } else {
        None
    };
    debug!(
        cache_stacked_matrix,
        cached_matrix_transport_allocations = usize::from(cache_stacked_matrix),
        cached_matrix_transport_elements = if cache_stacked_matrix {
            d.matrix.values.len()
        } else {
            0
        },
        "trace transport stacked matrix cache"
    );
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
    let non_root_layer_ptrs = digest_layers
        .iter()
        .take(digest_layers.len().saturating_sub(1))
        .map(|layer| layer.as_device_ptr() as u64)
        .collect::<Vec<_>>()
        .to_device();
    MerkleTreeMetal {
        backing_matrix,
        digest_layers,
        non_root_layer_ptrs,
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
    let cache_stacked_matrix = prover_config.cache_stacked_matrix;
    let d_matrix = if cache_stacked_matrix {
        let d_matrix_evals = matrix.values.to_device();
        Some(PleMatrix::from_evals(
            layout.l_skip(),
            d_matrix_evals,
            height,
            width,
        ))
    } else {
        None
    };
    debug!(
        width,
        height,
        cache_stacked_matrix,
        cached_matrix_transport_allocations = usize::from(cache_stacked_matrix),
        cached_matrix_transport_elements = if cache_stacked_matrix {
            matrix.values.len()
        } else {
            0
        },
        "pcs transport stacked matrix cache"
    );
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

pub(crate) fn read_folded_matrix_first_row<T: Copy>(matrix: &MetalMatrix<T>) -> Vec<T> {
    let height = matrix.height();
    let width = matrix.width();
    debug_assert!(height > 0, "Folded matrix height must be non-zero");
    if height == 0 || width == 0 {
        return Vec::new();
    }

    let src = matrix.buffer().as_ptr();
    if height == 1 {
        let mut row = Vec::with_capacity(width);
        unsafe {
            std::ptr::copy_nonoverlapping(src, row.as_mut_ptr(), width);
            row.set_len(width);
        }
        row
    } else {
        (0..width).map(|col| unsafe { *src.add(col * height) }).collect()
    }
}

pub fn transport_matrix_d2h_row_major(matrix: &MetalMatrix<F>) -> RowMajorMatrix<F> {
    let height = matrix.height();
    let width = matrix.width();
    if transpose_is_layout_identity(width, height) {
        debug!(
            height,
            width,
            transport_kernel_dispatches = 0,
            transport_sync_points = 0,
            transport_allocations = 0,
            "row-major download reused contiguous layout"
        );
        return RowMajorMatrix::<F>::new(matrix.to_host(), width);
    }

    let matrix_buffer = MetalBuffer::<F>::with_capacity(height * width);
    unsafe {
        matrix_transpose_fp(
            &matrix_buffer,
            matrix.buffer(),
            height,
            width,
        )
        .unwrap();
    }
    debug!(
        height,
        width,
        transport_kernel_dispatches = 1,
        transport_sync_points = 1,
        transport_allocations = 1,
        "row-major download transpose kernel"
    );
    RowMajorMatrix::<F>::new(matrix_buffer.to_host(), width)
}

#[cfg(test)]
mod tests {
    use super::{collapse_strided_trace_values, transpose_is_layout_identity};

    #[test]
    fn collapse_strided_trace_stride_one_identity() {
        let width = 3;
        let height = 4;
        let stride = 1;
        let lifted_height = height * stride;
        let src: Vec<u32> = (0..(width * lifted_height) as u32).collect();
        let mut dst = vec![0u32; width * height];
        collapse_strided_trace_values(&src, &mut dst, width, height, lifted_height, stride);
        assert_eq!(dst, src);
    }

    #[test]
    fn collapse_strided_trace_stride_two_col_major() {
        let width = 2;
        let height = 4;
        let stride = 2;
        let lifted_height = height * stride;
        let src = vec![
            // col 0
            0u32, 1, 2, 3, 4, 5, 6, 7,
            // col 1
            10u32, 11, 12, 13, 14, 15, 16, 17,
        ];
        let mut dst = vec![0u32; width * height];
        collapse_strided_trace_values(&src, &mut dst, width, height, lifted_height, stride);
        assert_eq!(dst, vec![0, 2, 4, 6, 10, 12, 14, 16]);
    }

    #[test]
    fn transpose_layout_identity_only_for_degenerate_shapes() {
        assert!(transpose_is_layout_identity(1, 8));
        assert!(transpose_is_layout_identity(8, 1));
        assert!(transpose_is_layout_identity(0, 9));
        assert!(transpose_is_layout_identity(9, 0));
        assert!(!transpose_is_layout_identity(2, 2));
        assert!(!transpose_is_layout_identity(3, 5));
    }
}
