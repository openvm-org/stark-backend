use metal::{CommandBufferRef, CommandQueue, MTLCommandBufferStatus};

use crate::error::MetalError;

/// Creates a new command buffer from the given command queue.
pub fn new_command_buffer(queue: &CommandQueue) -> &CommandBufferRef {
    queue.new_command_buffer()
}

/// Commits the command buffer, waits until completed, and checks status.
/// Returns an error if the command buffer execution failed.
pub fn sync_and_check(cmd_buffer: &CommandBufferRef) -> Result<(), MetalError> {
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

    let status = cmd_buffer.status();
    if status == MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionFailed(
            "command buffer execution failed".to_string(),
        ));
    }
    Ok(())
}

/// Executes a blit (block transfer) operation using a blit command encoder.
/// The closure receives the blit encoder to schedule copy/fill operations.
/// The command buffer is committed and waited on before returning.
pub fn blit_operation<F>(queue: &CommandQueue, f: F) -> Result<(), MetalError>
where
    F: FnOnce(&metal::BlitCommandEncoderRef),
{
    let cmd_buffer = queue.new_command_buffer();
    let blit_encoder = cmd_buffer.new_blit_command_encoder();
    f(blit_encoder);
    blit_encoder.end_encoding();
    sync_and_check(cmd_buffer)
}
