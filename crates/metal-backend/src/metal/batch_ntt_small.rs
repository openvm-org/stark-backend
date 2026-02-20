//! Batch NTT small kernel dispatch wrappers.
//! Ported from cuda-backend/src/cuda/batch_ntt_small.rs

use std::ffi::c_void;
use std::mem::size_of;
use std::sync::OnceLock;

use openvm_metal_common::{d_buffer::MetalBuffer, device::get_context, error::MetalError};

use crate::prelude::F;

use super::{
    dispatch_sync, get_kernels, grid_size_1d, grid_size_2d, DEFAULT_THREADS_PER_GROUP,
    LOG_SIMD_SIZE,
};

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
    let ctx = get_context();
    let device_threads = ctx.device.max_threads_per_threadgroup();
    let threads_x = 1usize << l_skip;
    let max_threads_total = pipeline.max_total_threads_per_threadgroup() as usize;
    let max_threads_x = device_threads.width as usize;
    let max_threads_y = device_threads.height as usize;
    if threads_x > max_threads_total || threads_x > max_threads_x {
        return Err(MetalError::ExecutionFailed(format!(
            "batch_ntt_small requires threads_x={threads_x}, but max_total={max_threads_total}, max_x={max_threads_x}",
        )));
    }

    let max_threads_y_from_total = max_threads_total / threads_x;
    let mut threads_y = max_threads_y.min(max_threads_y_from_total);
    if threads_y == 0 {
        return Err(MetalError::ExecutionFailed(format!(
            "batch_ntt_small cannot launch with threads_x={threads_x}: max_total={max_threads_total}, max_y={max_threads_y}",
        )));
    }

    let needs_tg_mem = l_skip > LOG_SIMD_SIZE;
    let max_dynamic_tg_mem = (ctx.device.max_threadgroup_memory_length() as usize)
        .saturating_sub(pipeline.static_threadgroup_memory_length() as usize);
    // Metal validation requires a threadgroup binding whenever the kernel declares one.
    // For non-shared-memory paths we bind one element and keep it unused.
    let smem_bytes = if needs_tg_mem {
        let smem_bytes_per_ntt = size_of::<F>() * threads_x;
        let max_threads_y_from_mem = max_dynamic_tg_mem / smem_bytes_per_ntt;
        threads_y = threads_y.min(max_threads_y_from_mem);
        if threads_y == 0 {
            return Err(MetalError::ExecutionFailed(format!(
                "batch_ntt_small shared memory requires {smem_bytes_per_ntt} bytes per packed NTT, but max dynamic threadgroup memory is {max_dynamic_tg_mem}",
            )));
        }
        smem_bytes_per_ntt * threads_y
    } else {
        const MIN_THREADGROUP_BINDING_BYTES: usize = 16;
        if max_dynamic_tg_mem < MIN_THREADGROUP_BINDING_BYTES {
            return Err(MetalError::ExecutionFailed(format!(
                "batch_ntt_small requires at least {MIN_THREADGROUP_BINDING_BYTES} bytes of dynamic threadgroup memory binding, but max is {max_dynamic_tg_mem}",
            )));
        }
        MIN_THREADGROUP_BINDING_BYTES
    };

    let num_groups_x = cnt_blocks.div_ceil(threads_y);
    let (grid, group) = grid_size_2d(num_groups_x * threads_x, threads_y, threads_x, threads_y);
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
