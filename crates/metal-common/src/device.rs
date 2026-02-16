use std::sync::OnceLock;

use metal::{CommandQueue, Device};

use crate::error::MetalError;

static METAL_CONTEXT: OnceLock<MetalContext> = OnceLock::new();

pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
}

// Safety: Metal Device and CommandQueue are thread-safe on Apple platforms.
// The Metal framework guarantees that command queue submission is synchronized.
unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

/// Returns a reference to the global Metal context, initializing it on first call.
pub fn get_context() -> &'static MetalContext {
    METAL_CONTEXT.get_or_init(init_context)
}

/// Tries to get the global Metal context, returning an error if no device is available.
pub fn try_get_context() -> Result<&'static MetalContext, MetalError> {
    Ok(METAL_CONTEXT.get_or_init(|| {
        try_init_context().expect("Metal initialization failed")
    }))
}

fn init_context() -> MetalContext {
    try_init_context().expect("Metal initialization failed")
}

fn try_init_context() -> Result<MetalContext, MetalError> {
    let device = Device::system_default().ok_or(MetalError::DeviceNotFound)?;
    let queue = device.new_command_queue();
    tracing::info!(
        "Metal device: {}, max buffer length: {} bytes",
        device.name(),
        device.max_buffer_length()
    );
    Ok(MetalContext { device, queue })
}
