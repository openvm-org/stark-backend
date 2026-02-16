//! Device info queries for Metal.
//! Ported from cuda-backend/src/cuda/device_info.rs
//!
//! Metal does not have an exact equivalent of CUDA SM count.
//! Instead, we query the device's max threadgroup memory and other properties.

use openvm_metal_common::{device::get_context, error::MetalError};

/// Returns the number of GPU cores (approximate equivalent of SM count).
///
/// Metal does not directly expose SM count. On Apple Silicon, we use the
/// max threads per threadgroup as a proxy for compute capability.
/// For dispatch sizing, this returns a reasonable default.
pub fn get_gpu_core_count(_device_id: u32) -> Result<u32, MetalError> {
    let ctx = get_context();
    // Apple Silicon GPUs don't expose core counts directly.
    // Use max threads per threadgroup as a heuristic proxy.
    // M1: ~128 EUs, M2: ~160 EUs, M3: ~192 EUs, etc.
    // A reasonable default is 128 which works for dispatch sizing.
    let max_threads = ctx.device.max_threads_per_threadgroup();
    // Use max_threads.width as a rough proxy; default to 128 if unavailable.
    let core_count = (max_threads.width as u32).max(128);
    Ok(core_count)
}
