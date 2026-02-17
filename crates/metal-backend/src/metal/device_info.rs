use openvm_metal_common::{device::get_context, error::MetalError};

/// Returns the number of GPU cores (approximate equivalent of SM count).
pub fn get_gpu_core_count(_device_id: u32) -> Result<u32, MetalError> {
    let ctx = get_context();
    let max_threads = ctx.device.max_threads_per_threadgroup();
    let core_count = (max_threads.width as u32).max(128);
    Ok(core_count)
}
