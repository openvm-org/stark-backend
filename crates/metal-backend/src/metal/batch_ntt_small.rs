//! Batch NTT small kernel dispatch wrappers.
//! Ported from cuda-backend/src/cuda/batch_ntt_small.rs

use std::{ffi::c_void, mem::size_of, sync::OnceLock};

use openvm_metal_common::{d_buffer::MetalBuffer, error::MetalError};

use super::{
    dispatch_sync, get_kernels, grid_size_1d, grid_size_2d, DEFAULT_THREADS_PER_GROUP,
    LOG_SIMD_SIZE,
};
use crate::prelude::F;

/// Size of the device NTT twiddle table (2^11 - 2 = 2046 elements for MAX_NTT_LEVEL=10)
pub const DEVICE_NTT_TWIDDLES_SIZE: usize = (1 << 11) - 2;

static DEVICE_NTT_TWIDDLES: OnceLock<MetalBuffer<F>> = OnceLock::new();

fn device_ntt_twiddles() -> &'static MetalBuffer<F> {
    DEVICE_NTT_TWIDDLES.get_or_init(|| {
        let twiddles = MetalBuffer::<F>::with_capacity(DEVICE_NTT_TWIDDLES_SIZE);
        unsafe {
            generate_device_ntt_twiddles(&twiddles)
                .expect("failed to initialize device NTT twiddles");
        }
        twiddles
    })
}

/// Ensure device NTT twiddles are initialized.
/// Safe to call multiple times - initialization happens only once.
pub fn ensure_device_ntt_twiddles_initialized() {
    let _ = device_ntt_twiddles();
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
    if cnt_blocks == 0 {
        return Ok(());
    }

    let d_twiddles = device_ntt_twiddles();
    let pipeline = get_kernels().get_pipeline("batch_ntt_small")?;
    let threads_x = 1usize << l_skip;
    debug_assert!(threads_x <= 1024);
    let threads_y = 1024usize / threads_x;
    let num_groups_x = cnt_blocks.div_ceil(threads_y);
    let (grid, group) = grid_size_2d(num_groups_x * threads_x, threads_y, threads_x, threads_y);
    // Metal validation requires a threadgroup binding whenever the kernel declares one.
    // For non-shared-memory paths we bind one element and keep it unused.
    let smem_bytes = if l_skip > LOG_SIMD_SIZE {
        size_of::<F>() * 1024
    } else {
        16
    };
    let l_skip_u32 = l_skip as u32;
    let cnt_blocks_u32 = cnt_blocks as u32;
    let is_intt_u32: u32 = if is_intt { 1 } else { 0 };
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(buffer.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(d_twiddles.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &l_skip_u32 as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &cnt_blocks_u32 as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &is_intt_u32 as *const u32 as *const c_void);
        encoder.set_threadgroup_memory_length(0, smem_bytes as u64);
    })
}
