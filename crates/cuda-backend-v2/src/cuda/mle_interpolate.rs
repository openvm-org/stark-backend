use openvm_cuda_common::{
    d_buffer::DeviceBuffer,
    error::{CudaError, check},
};

use crate::{EF, F, cuda::LOG_WARP_SIZE};

extern "C" {
    fn _mle_interpolate_stage(
        buffer: *mut F,
        total_len: usize,
        step: u32,
        is_eval_to_coeff: bool,
    ) -> i32;

    fn _mle_interpolate_stage_ext(
        buffer: *mut EF,
        total_len: usize,
        step: u32,
        is_eval_to_coeff: bool,
    ) -> i32;

    fn _mle_interpolate_stage_2d(
        buffer: *mut F,
        width: u16,
        height: u32,
        padded_height: u32,
        step: u32,
        is_eval_to_coeff: bool,
    ) -> i32;

    fn _mle_interpolate_fused_2d(
        buffer: *mut F,
        width: u16,
        padded_height: u32,
        log_stride: u32,
        start_step: u32,
        num_stages: u32,
        is_eval_to_coeff: bool,
        right_pad: bool,
    ) -> i32;

    fn _mle_interpolate_shared_2d(
        buffer: *mut F,
        width: u16,
        padded_height: u32,
        log_stride: u32,
        start_log_step: u32,
        end_log_step: u32,
        is_eval_to_coeff: bool,
        right_pad: bool,
    ) -> i32;
}

/// Does in-place interpolation on `buffer` from eval to coeff form in one coordinate, assuming the
/// associated polynomial is linear in that coordinate. Effectively it performs `v -= u` for `v, u`
/// that are `step` apart in the buffer.
///
/// The boolean `is_eval_to_coeff` indicates the direction of the transformation:
/// - If `true`, it performs eval to coeff transformation.
/// - If `false`, it performs coeff to eval transformation.
///
/// # Safety
/// - The `buffer` must be allocated and initialized on device in the default stream.
/// - `step` must be a power of two and less than or equal to half the length of the buffer.
pub unsafe fn mle_interpolate_stage(
    buffer: &mut DeviceBuffer<F>,
    step: u32,
    is_eval_to_coeff: bool,
) -> Result<(), CudaError> {
    check(_mle_interpolate_stage(
        buffer.as_mut_ptr(),
        buffer.len(),
        step,
        is_eval_to_coeff,
    ))
}

/// Same as [mle_interpolate_stage] for extension field `EF`.
pub unsafe fn mle_interpolate_stage_ext(
    buffer: &mut DeviceBuffer<EF>,
    step: u32,
    is_eval_to_coeff: bool,
) -> Result<(), CudaError> {
    check(_mle_interpolate_stage_ext(
        buffer.as_mut_ptr(),
        buffer.len(),
        step,
        is_eval_to_coeff,
    ))
}

/// Same as [mle_interpolate_stage] but `buffer` is now a `padded_height x width` column-major
/// matrix, and we only perform the interpolation on the first `height` rows.
///
/// # Safety
/// - `width` must fit in a `u16` for the CUDA grid dimension.
/// - `buffer` must have length `>= padded_height * width`.
/// - `padded_height` must be a multiple of `step * 2`.
pub unsafe fn mle_interpolate_stage_2d(
    buffer: *mut F,
    width: u16,
    height: u32,
    padded_height: u32,
    step: u32,
    is_eval_to_coeff: bool,
) -> Result<(), CudaError> {
    debug_assert!(height <= padded_height);
    debug_assert_eq!(padded_height % (step * 2), 0);
    check(_mle_interpolate_stage_2d(
        buffer,
        width,
        height,
        padded_height,
        step,
        is_eval_to_coeff,
    ))
}

/// Fused MLE interpolation that processes multiple consecutive stages using warp shuffle.
/// This fuses stages from `start_step` up to `start_step << (num_stages - 1)`.
///
/// For example, with `start_step=1` and `num_stages=5`, this processes stages with
/// steps 1, 2, 4, 8, 16 in a single kernel launch using warp shuffle for data exchange.
///
/// # Parameters
/// - `log_stride`: meaningful_count = padded_height >> log_stride
/// - `bit_reversed`: if true, physical index = tidx << log_stride (strided access) if false,
///   physical index = tidx (contiguous access)
///
/// # Safety
/// - `width` must fit in a `u16` for the CUDA grid dimension.
/// - `buffer` must have length `>= padded_height * width`.
/// - `start_step` must be a power of 2.
/// - `start_step << (num_stages - 1)` must be <= 16 (warp shuffle constraint).
/// - `num_stages` must be in range `1..=5`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn mle_interpolate_fused_2d(
    buffer: *mut F,
    width: u16,
    padded_height: u32,
    log_stride: u32,
    start_step: u32,
    num_stages: u32,
    is_eval_to_coeff: bool,
    right_pad: bool,
) -> Result<(), CudaError> {
    debug_assert!((1..=LOG_WARP_SIZE as u32).contains(&num_stages));
    debug_assert!((start_step << (num_stages - 1)) <= 16);
    check(_mle_interpolate_fused_2d(
        buffer,
        width,
        padded_height,
        log_stride,
        start_step,
        num_stages,
        is_eval_to_coeff,
        right_pad,
    ))
}

/// Tile log size for shared memory kernel (must match CUDA kernel's MLE_SHARED_TILE_LOG_SIZE).
pub const MLE_SHARED_TILE_LOG_SIZE: u32 = 12;

/// Shared memory MLE interpolation that processes multiple stages within tiles.
/// Processes stages from `start_log_step` to `end_log_step` (both inclusive).
///
/// # Parameters
/// - `log_stride`: meaningful_count = padded_height >> log_stride
/// - `bit_reversed`: if true, physical index = tidx << log_stride (strided access) if false,
///   physical index = tidx (contiguous access)
///
/// # Safety
/// - `width` must fit in a `u16` for the CUDA grid dimension.
/// - `buffer` must have length `>= padded_height * width`.
/// - `end_log_step` must be < MLE_SHARED_TILE_LOG_SIZE (12).
#[allow(clippy::too_many_arguments)]
pub unsafe fn mle_interpolate_shared_2d(
    buffer: *mut F,
    width: u16,
    padded_height: u32,
    log_stride: u32,
    start_log_step: u32,
    end_log_step: u32,
    is_eval_to_coeff: bool,
    right_pad: bool,
) -> Result<(), CudaError> {
    debug_assert!(end_log_step < MLE_SHARED_TILE_LOG_SIZE);
    check(_mle_interpolate_shared_2d(
        buffer,
        width,
        padded_height,
        log_stride,
        start_log_step,
        end_log_step,
        is_eval_to_coeff,
        right_pad,
    ))
}
