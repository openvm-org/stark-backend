use openvm_cuda_common::error::{CudaError, MemCopyError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProverError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),
    #[error("Memory copy error: {0}")]
    MemCopy(#[from] MemCopyError),
}
