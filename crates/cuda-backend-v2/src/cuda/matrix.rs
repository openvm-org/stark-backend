use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

use crate::F;

extern "C" {
    fn _batch_rotate_pad(
        out: *mut F,
        input: *const F,
        width: u32,
        num_x: u32,
        domain_size: u32,
        padded_size: u32,
    ) -> i32;

    fn _batch_rotate_lift_and_pad(
        out: *mut F,
        input: *const F,
        width: u32,
        height: u32,
        lifted_height: u32,
        padded_height: u32,
    ) -> i32;

    fn _collapse_and_lift_strided_matrix(
        out: *mut F,
        input: *const F,
        width: u32,
        height: u32,
        stride: u32,
    ) -> i32;

    fn _batch_expand_pad_wide(
        out: *mut F,
        input: *const F,
        width: u32,
        padded_height: u32,
        height: u32,
    ) -> i32;
}

/// Copies a `domain_size × width` matrix into a `padded_size × (2 * width)` matrix,
/// duplicating each column with a one-step rotation applied to the second copy.
///
/// # Safety
/// - `output` must have capacity for `padded_size * 2 * width` elements.
/// - `input` must have at least `domain_size * width` elements.
pub unsafe fn batch_rotate_pad(
    output: &DeviceBuffer<F>,
    input: &DeviceBuffer<F>,
    width: u32,
    num_x: u32,
    domain_size: u32,
    padded_size: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_batch_rotate_pad(
        output.as_mut_ptr(),
        input.as_ptr(),
        width,
        num_x,
        domain_size,
        padded_size,
    ))
}

/// This is a weird kernel for a specific purpose and may be deleted later.
/// `input` is a `width x lifted_height` matrix that is **assumed** to be the lifting of a `width x
/// height` matrix (so it is vertically repeating every `height` rows). This kernel will rotate the
/// **unlifted** matrix, lift it back to `width x lifted_height`, and then zero-expand each column
/// to `width x padded_height`.
///
/// NOTE: unlike `batch_rotate_pad_kernel`, this kernel does not do plain expand without rotate.
///
/// # Safety
/// - `output` is a pointer to a device buffer with length at least `width * padded_height`.
/// - `input` is a pointer to a device buffer with length at least `width * lifted_height` and the
///   device buffer corresponds to a lifting of a `width x height` matrix.
/// - `height` divides `lifted_height`, `lifted_height` divides `padded_height`
pub unsafe fn batch_rotate_lift_and_pad(
    output: *mut F,
    input: *const F,
    width: u32,
    height: u32,
    lifted_height: u32,
    padded_height: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_batch_rotate_lift_and_pad(
        output,
        input,
        width,
        height,
        lifted_height,
        padded_height,
    ))
}

/// Let `lifted_height = height * stride`. Collapses a `lifted_height × width` matrix to a `height x
/// width` by taking vertical row `stride`, and then cyclically repeats vertically to "lift" again
/// to a `lifted_height x width` matrix.
///
/// # Safety
/// - `output` must be a pointer to `DeviceBuffer<F>` with length at least `lifted_height * width`.
/// - `input` must be a pointer to `DeviceBuffer<F>` with length at least `lifted_height * width`.
pub unsafe fn collapse_and_lift_strided_matrix(
    output: *mut F,
    input: *const F,
    width: u32,
    height: u32,
    stride: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_collapse_and_lift_strided_matrix(
        output, input, width, height, stride,
    ))
}

/// Expands a `height × width` column-major matrix to a `padded_height × width` column-major matrix
/// by padding with zeros. This kernel is intended for use when `width` is large (~2^20), so each
/// column is handled by a different block.
///
/// # Safety
/// - `output` must be a pointer to `DeviceBuffer<F>` with length at least `padded_height * width`.
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
