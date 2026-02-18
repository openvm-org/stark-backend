use openvm_metal_common::{device::get_context, error::MetalError};

/// Returns a heuristic thread parallelism count for the Metal GPU.
///
/// Apple's Metal API does not expose the actual GPU core count.
/// We use `max_threads_per_threadgroup` as a rough proxy for scheduling
/// granularity, with a minimum of 128. This is NOT the physical core count
/// but serves as a comparable parameter to CUDA's SM count for workload sizing.
pub fn get_gpu_core_count(_device_id: u32) -> Result<u32, MetalError> {
    let ctx = get_context();
    let max_threads = ctx.device.max_threads_per_threadgroup();
    let core_count = (max_threads.width as u32).max(128);
    Ok(core_count)
}
