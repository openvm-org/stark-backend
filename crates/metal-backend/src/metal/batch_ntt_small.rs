//! Batch NTT small kernel dispatch wrappers.
//! Ported from cuda-backend/src/cuda/batch_ntt_small.rs

use std::ffi::c_void;
use std::sync::Once;

use openvm_metal_common::{d_buffer::MetalBuffer, error::MetalError};

use crate::prelude::F;

use super::{dispatch_sync, get_kernels, grid_size_1d, DEFAULT_THREADS_PER_GROUP};

/// Size of the device NTT twiddle table (2^11 - 2 = 2046 elements for MAX_NTT_LEVEL=10)
pub const DEVICE_NTT_TWIDDLES_SIZE: usize = (1 << 11) - 2;

static INIT_DEVICE_NTT_TWIDDLES: Once = Once::new();

/// Ensure device NTT twiddles are initialized.
/// Safe to call multiple times - initialization happens only once.
pub fn ensure_device_ntt_twiddles_initialized() {
    INIT_DEVICE_NTT_TWIDDLES.call_once(|| {
        let twiddles = MetalBuffer::<F>::with_capacity(DEVICE_NTT_TWIDDLES_SIZE);
        unsafe {
            generate_device_ntt_twiddles(&twiddles)
                .expect("failed to initialize device NTT twiddles");
        }
    });
}

unsafe fn generate_device_ntt_twiddles(d_twiddles: &MetalBuffer<F>) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("generate_device_ntt_twiddles")?;
    let (grid, group) = grid_size_1d(d_twiddles.len(), DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(d_twiddles.gpu_buffer()), 0);
    })
}

pub unsafe fn batch_ntt_small(
    buffer: &mut MetalBuffer<F>,
    l_skip: usize,
    cnt_blocks: usize,
    is_intt: bool,
) -> Result<(), MetalError> {
    ensure_device_ntt_twiddles_initialized();
    let pipeline = get_kernels().get_pipeline("batch_ntt_small")?;
    let (grid, group) = grid_size_1d(cnt_blocks * 32, 32);
    let l_skip_u32 = l_skip as u32;
    let cnt_blocks_u32 = cnt_blocks as u32;
    let is_intt_u32: u32 = if is_intt { 1 } else { 0 };
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(buffer.gpu_buffer()), 0);
        encoder.set_bytes(1, 4, &l_skip_u32 as *const u32 as *const c_void);
        encoder.set_bytes(2, 4, &cnt_blocks_u32 as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &is_intt_u32 as *const u32 as *const c_void);
    })
}
