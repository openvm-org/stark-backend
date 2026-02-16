use openvm_metal_common::error::MetalError;
use thiserror::Error;

use crate::{prelude::EF, sponge::GrindError, KernelError};

#[derive(Debug, Error)]
pub enum Round0PrepError {
    #[error("metal error: {0}")]
    Metal(#[from] MetalError),
    #[error("invalid stacked layout for round-0 staging")]
    Layout,
}

#[derive(Debug, Error)]
pub enum Round0EvalError {
    #[error("metal error: {0}")]
    Metal(#[from] MetalError),
    #[error("round-0 layout mismatch")]
    Layout,
}

impl From<Round0PrepError> for Round0EvalError {
    fn from(err: Round0PrepError) -> Self {
        match err {
            Round0PrepError::Metal(e) => Round0EvalError::Metal(e),
            Round0PrepError::Layout => Round0EvalError::Layout,
        }
    }
}

#[derive(Debug, Error)]
pub enum FractionalSumcheckError {
    #[error("nonzero root sum: p={p}, q={q}")]
    NonzeroRootSum { p: EF, q: EF },
    #[error("bit reversal: {0}")]
    BitReversal(MetalError),
    #[error("segment tree: {0}")]
    SegmentTree(MetalError),
    #[error("frac_compute_round: {0}")]
    ComputeRound(MetalError),
    #[error("frac_fold_columns: {0}")]
    FoldColumns(MetalError),
    #[error("evals_eq_hypercube: {0}")]
    EvalEqHypercube(KernelError),
    #[error("grind error: {0}")]
    Grind(#[from] GrindError),
}

#[derive(Debug, Error)]
pub enum InteractionMetalError {
    #[error("metal error: {0}")]
    Metal(#[from] MetalError),
    #[error("interaction layout mismatch")]
    Layout,
}

#[derive(Debug, Error)]
pub enum FoldPleError {
    #[error("metal error: {0}")]
    Metal(#[from] MetalError),
    #[error("invalid matrix dimensions")]
    InvalidDimensions,
}

#[derive(Debug, Error)]
pub enum UnstackMatrixError {
    #[error("matrix idx={mat_idx} not found")]
    MatrixNotFound { mat_idx: usize },
}
