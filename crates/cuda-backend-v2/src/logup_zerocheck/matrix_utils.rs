use std::{ffi::c_void, mem::size_of, sync::Arc};

use itertools::Itertools;
use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::{copy::cuda_memcpy, d_buffer::DeviceBuffer};
use openvm_stark_backend::prover::MatrixDimensions;

use super::errors::{InteractionGpuError, Round0PrepError};
use crate::{
    Digest, F, cuda::matrix::collapse_and_lift_strided_matrix, stacked_pcs::StackedPcsDataGpu,
};

/// Unstack a matrix from stacked PCS data. Returns the lifted matrix and the unlifted height.
///
/// # Performance Note
/// This function performs a Device-to-Device copy. For better performance, consider
/// using `data.matrix.mixed` directly which already stores the DFT over 2^l_skip version.
pub(crate) fn unstack_matrix(
    data: &StackedPcsDataGpu<F, Digest>,
    mat_idx: usize,
) -> Result<(DeviceMatrix<F>, usize), InteractionGpuError> {
    let stacked_matrix = &data.matrix;
    let stacked_height = stacked_matrix.height();
    let col_slices = data
        .layout
        .sorted_cols
        .iter()
        .filter(|(idx, _, _)| *idx == mat_idx)
        .map(|(_, _, s)| s)
        .collect_vec();
    if col_slices.is_empty() {
        return Err(InteractionGpuError::Layout);
    }
    let width = col_slices.len();
    let l_skip = data.layout.l_skip();
    let s = col_slices[0];
    debug_assert!(
        col_slices
            .iter()
            .all(|s2| s2.log_height() == s.log_height())
    );
    let lifted_height = s.len(l_skip);
    let buffer = DeviceBuffer::<F>::with_capacity(width * lifted_height);
    let stride = s.stride(l_skip);
    let height = 1 << s.log_height();
    // SAFETY: stacked matrix is column major, and due to definition of stacking, the column slices
    // will be contiguous in memory.
    unsafe {
        let src_offset = s.col_idx * stacked_height + s.row_idx;
        let src = stacked_matrix.evals.buffer().as_ptr().add(src_offset);
        if stride == 1 {
            // No striding
            cuda_memcpy::<true, true>(
                buffer.as_mut_ptr() as *mut c_void,
                src as *const c_void,
                width * lifted_height * size_of::<F>(),
            )?;
        } else {
            // Stacked matrix is strided: we need to collapse the strides and then lift. Evaluations
            // of lift is equivalent to cyclically repeating the unlifted evaluations.
            collapse_and_lift_strided_matrix(
                buffer.as_mut_ptr(),
                src,
                width as u32,
                height as u32,
                stride as u32,
            )?;
        }
    }
    Ok((
        DeviceMatrix::new(Arc::new(buffer), lifted_height, width),
        height,
    ))
}

/// Unstack a matrix from stacked PCS data (for round0, returns Round0PrepError).
pub(crate) fn unstack_matrix_round0(
    data: &StackedPcsDataGpu<F, Digest>,
    mat_idx: usize,
) -> Result<(DeviceMatrix<F>, usize), Round0PrepError> {
    unstack_matrix(data, mat_idx).map_err(|_| Round0PrepError::Layout)
}
