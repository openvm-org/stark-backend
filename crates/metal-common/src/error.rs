use thiserror::Error;

#[derive(Error, Debug)]
pub enum MetalError {
    #[error("Metal device not found")]
    DeviceNotFound,
    #[error("Failed to create command queue")]
    CommandQueueCreation,
    #[error("Kernel not found: {0}")]
    KernelNotFound(String),
    #[error("Pipeline creation failed: {0}")]
    PipelineCreation(String),
    #[error("Command buffer execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Library creation failed: {0}")]
    LibraryCreation(String),
}

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Buffer allocation failed: requested {requested} bytes")]
    AllocationFailed { requested: u64 },
    #[error("Out of memory: requested {requested}, available {available}")]
    OutOfMemory { requested: u64, available: u64 },
}

#[derive(Error, Debug)]
pub enum MemCopyError {
    #[error("Size mismatch: source {src_len} != destination {dst_len}")]
    SizeMismatch { src_len: usize, dst_len: usize },
}
