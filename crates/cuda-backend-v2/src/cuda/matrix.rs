use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

use crate::prelude::{EF, F};

extern "C" {
    fn _matrix_transpose_fp(
        output: *mut F,
        input: *const F,
        col_size: usize,
        row_size: usize,
    ) -> i32;

    fn _matrix_transpose_fpext(
        output: *mut EF,
        input: *const EF,
        col_size: usize,
        row_size: usize,
    ) -> i32;

    fn _matrix_get_rows_fp(
        output: *mut F,
        input: *const F,
        row_indices: *const u32,
        matrix_width: u64,
        matrix_height: u64,
        row_indices_len: u32,
    ) -> i32;

    fn _split_ext_to_base_col_major_matrix(
        d_matrix: *mut F,
        d_poly: *const EF,
        poly_len: u64,
        matrix_height: u32,
    ) -> i32;

    fn _batch_rotate_pad(
        out: *mut F,
        input: *const F,
        width: u32,
        num_x: u32,
        domain_size: u32,
        padded_size: u32,
    ) -> i32;

    fn _lift_padded_matrix_evals(
        matrix: *mut F,
        width: u32,
        height: u32,
        lifted_height: u32,
        padded_height: u32,
    ) -> i32;

    fn _collapse_strided_matrix(
        out: *mut F,
        input: *const F,
        width: u32,
        height: u32,
        stride: u32,
    ) -> i32;

    fn _batch_expand_pad(
        output: *mut F,
        input: *const F,
        poly_count: u32,
        out_size: u32,
        in_size: u32,
    ) -> i32;

    fn _batch_expand_pad_wide(
        out: *mut F,
        input: *const F,
        width: u32,
        padded_height: u32,
        height: u32,
    ) -> i32;
}

/// Safety:
/// - `input` and `output` must not overlap.
pub unsafe fn matrix_transpose_fp(
    output: &mut DeviceBuffer<F>,
    input: &DeviceBuffer<F>,
    width: usize,
    height: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_matrix_transpose_fp(
        output.as_mut_ptr(),
        input.as_ptr(),
        width,
        height,
    ))
}

/// Safety:
/// - `input` and `output` must not overlap.
pub unsafe fn matrix_transpose_fpext(
    output: &mut DeviceBuffer<EF>,
    input: &DeviceBuffer<EF>,
    width: usize,
    height: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_matrix_transpose_fpext(
        output.as_mut_ptr(),
        input.as_ptr(),
        width,
        height,
    ))
}

pub unsafe fn matrix_get_rows_fp_kernel(
    output: &mut DeviceBuffer<F>,
    input: &DeviceBuffer<F>,
    row_indices: &DeviceBuffer<u32>,
    matrix_width: u64,
    matrix_height: u64,
    row_indices_len: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_matrix_get_rows_fp(
        output.as_mut_ptr(),
        input.as_ptr(),
        row_indices.as_ptr(),
        matrix_width,
        matrix_height,
        row_indices_len,
    ))
}

pub unsafe fn split_ext_to_base_col_major_matrix(
    d_matrix: &mut DeviceBuffer<F>,
    d_poly: &DeviceBuffer<EF>,
    poly_len: u64,
    matrix_height: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_split_ext_to_base_col_major_matrix(
        d_matrix.as_mut_ptr(),
        d_poly.as_ptr(),
        poly_len,
        matrix_height,
    ))
}

/// This kernel has the effect of rotating `input` column-wise as a `height x width` matrix, where
/// `height = domain_size * num_x` **and then** batch expanding it as `width * num_x` vectors of
/// length `domain_size` by zero padding to vectors of length `padded_size`. The output is written
/// to `output`. This kernel zeros all padding entries.
///
/// # Safety
/// - `output` must be a pointer to DeviceBuffer with capacity for `padded_size * width * num_x`
///   elements.
/// - `input` must be a pointer to DeviceBuffer with at least `domain_size * width * num_x`
///   elements.
/// - Must have `width * num_x < (u16::MAX)^2` due to grid dimension restrictions.
pub unsafe fn batch_rotate_pad(
    output: *mut F,
    input: *const F,
    width: u32,
    num_x: u32,
    domain_size: u32,
    padded_size: u32,
) -> Result<(), CudaError> {
    debug_assert!(domain_size <= padded_size);
    debug_assert!(width.checked_mul(num_x).unwrap() < u16::MAX as u32 * u16::MAX as u32);
    CudaError::from_result(_batch_rotate_pad(
        output,
        input,
        width,
        num_x,
        domain_size,
        padded_size,
    ))
}

/// This lifts `matrix` inplace, where `matrix` is already vertically padded.
/// - `matrix` is a `padded_height x width` matrix.
///
/// This kernel cyclically repeats the first `height` rows of the matrix vertically for
/// `lifted_height` rows. The rows `lifted_height..padded_height` are left untouched.
///
/// # Safety
/// - `matrix` is a pointer to a device buffer with length at least `width * padded_height`.
/// - `height <= lifted_height <= padded_height`
pub unsafe fn lift_padded_matrix_evals(
    matrix: *mut F,
    width: u32,
    height: u32,
    lifted_height: u32,
    padded_height: u32,
) -> Result<(), CudaError> {
    debug_assert!(height <= lifted_height && lifted_height <= padded_height);
    CudaError::from_result(_lift_padded_matrix_evals(
        matrix,
        width,
        height,
        lifted_height,
        padded_height,
    ))
}

/// Let `lifted_height = height * stride`. Collapses a `lifted_height × width` matrix to a `height x
/// width` by taking vertical row `stride`.
///
/// # Safety
/// - `output` must be a pointer to `DeviceBuffer<F>` with length at least `height * width`.
/// - `input` must be a pointer to `DeviceBuffer<F>` with length at least `lifted_height * width`.
pub unsafe fn collapse_strided_matrix(
    output: *mut F,
    input: *const F,
    width: u32,
    height: u32,
    stride: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_collapse_strided_matrix(
        output, input, width, height, stride,
    ))
}

pub unsafe fn batch_expand_pad(
    output: *mut F,
    input: *const F,
    poly_count: u32,
    out_size: u32,
    in_size: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_batch_expand_pad(
        output, input, poly_count, out_size, in_size,
    ))
}

/// Expands a `height × width` column-major matrix to a `padded_height × width` column-major matrix
/// by padding with zeros. This kernel is intended for use when `width` is large (~2^20), so each
/// column is handled by a different block.
///
/// # Safety
/// - `out` must be a pointer to `DeviceBuffer<F>` with length at least `padded_height * width`.
/// - `input` must be a pointer to `DeviceBuffer<F>` with length at least `height * width`.
pub unsafe fn batch_expand_pad_wide(
    out: *mut F,
    input: *const F,
    width: u32,
    padded_height: u32,
    height: u32,
) -> Result<(), CudaError> {
    debug_assert!(padded_height > height);
    CudaError::from_result(_batch_expand_pad_wide(
        out,
        input,
        width,
        padded_height,
        height,
    ))
}
