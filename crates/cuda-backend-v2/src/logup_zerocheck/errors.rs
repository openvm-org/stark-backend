use openvm_cuda_common::error::{CudaError, MemCopyError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Round0PrepError {
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),
    #[error("memory copy failed: {0}")]
    Mem(#[from] MemCopyError),
    #[error("invalid stacked layout for round-0 staging")]
    Layout,
}

#[derive(Debug, Error)]
pub enum Round0EvalError {
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),
    #[error("memcpy error: {0}")]
    Copy(#[from] MemCopyError),
    #[error("round-0 layout mismatch")]
    Layout,
}

impl From<Round0PrepError> for Round0EvalError {
    fn from(err: Round0PrepError) -> Self {
        match err {
            Round0PrepError::Cuda(e) => Round0EvalError::Cuda(e),
            Round0PrepError::Mem(e) => Round0EvalError::Copy(e),
            Round0PrepError::Layout => Round0EvalError::Layout,
        }
    }
}

#[derive(Debug, Error)]
pub enum FractionalSegmentTreeError {
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),
    #[error("memcpy error: {0}")]
    Copy(#[from] MemCopyError),
}

#[derive(Debug, Error)]
pub enum FractionalSumcheckError {
    #[error("frac_prepare_round cuda error: {0}")]
    PrepareRound(CudaError),
    #[error("frac_compute_round cuda error: {0}")]
    ComputeRound(CudaError),
    #[error("frac_fold_columns cuda error: {0}")]
    FoldColumns(CudaError),
    #[error("frac_extract_claims cuda error: {0}")]
    ExtractClaims(CudaError),
    #[error("memcpy error: {0}")]
    Copy(#[from] MemCopyError),
}

#[derive(Debug, Error)]
pub enum InteractionGpuError {
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),
    #[error("memcpy error: {0}")]
    Copy(#[from] MemCopyError),
    #[error("interaction layout mismatch")]
    Layout,
}

#[derive(Debug, Error)]
pub enum FoldPleError {
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),
    #[error("memcpy error: {0}")]
    Copy(#[from] MemCopyError),
    #[error("invalid matrix dimensions")]
    InvalidDimensions,
}

#[derive(Debug, Error)]
pub enum UnstackMatrixError {
    #[error("matrix idx={mat_idx} not found")]
    MatrixNotFound { mat_idx: usize },
}

