use std::{ffi::c_void, sync::Once};

use openvm_cuda_common::{
    d_buffer::DeviceBuffer,
    error::{CudaError, check},
};

use crate::F;

/// Size of the device NTT twiddle table (2^11 - 2 = 2046 elements for MAX_NTT_LEVEL=10)
pub const DEVICE_NTT_TWIDDLES_SIZE: usize = (1 << 11) - 2;

extern "C" {
    pub fn _batch_ntt_small(
        d_buffer: *mut F,
        l_skip: usize,
        cnt_blocks: usize,
        is_intt: bool,
    ) -> i32;

    fn _generate_device_ntt_twiddles(d_twiddles: *mut c_void) -> i32;
}

static INIT_DEVICE_NTT_TWIDDLES: Once = Once::new();

/// Ensure device NTT twiddles are initialized in constant memory.
/// Safe to call multiple times - initialization happens only once.
pub fn ensure_device_ntt_twiddles_initialized() {
    INIT_DEVICE_NTT_TWIDDLES.call_once(|| {
        let twiddles = DeviceBuffer::<F>::with_capacity(DEVICE_NTT_TWIDDLES_SIZE);
        unsafe {
            generate_device_ntt_twiddles(&twiddles)
                .expect("failed to initialize device NTT twiddles");
        }
    });
}

/// Generate device NTT twiddles and copy to constant memory.
/// `d_twiddles` must have capacity for `DEVICE_NTT_TWIDDLES_SIZE` elements.
unsafe fn generate_device_ntt_twiddles(d_twiddles: &DeviceBuffer<F>) -> Result<(), CudaError> {
    CudaError::from_result(_generate_device_ntt_twiddles(d_twiddles.as_mut_raw_ptr()))
}

pub unsafe fn batch_ntt_small(
    buffer: &mut DeviceBuffer<F>,
    l_skip: usize,
    cnt_blocks: usize,
    is_intt: bool,
) -> Result<(), CudaError> {
    ensure_device_ntt_twiddles_initialized();
    check(_batch_ntt_small(
        buffer.as_mut_ptr(),
        l_skip,
        cnt_blocks,
        is_intt,
    ))
}
