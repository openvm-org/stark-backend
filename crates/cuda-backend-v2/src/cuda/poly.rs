use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

extern "C" {
    fn _ple_interpolate_stage(buffer: *mut std::ffi::c_void, total_len: usize, step: u32) -> i32;
}

/// Does in-place interpolation on `buffer` from eval to coeff form in one coordinate, assuming the
/// associated polynomial is linear in that coordinate. Effectively it performs `v -= u` for `v, u`
/// that are `step` apart in the buffer.
///
/// # Safety
/// - The `buffer` must be allocated and initialized on device in the default stream.
/// - `step` must be a power of two and less than or equal to half the length of the buffer.
pub unsafe fn ple_interpolate_stage<F>(
    buffer: &mut DeviceBuffer<F>,
    step: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_ple_interpolate_stage(
        buffer.as_mut_raw_ptr(),
        buffer.len(),
        step,
    ))
}
