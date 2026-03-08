use openvm_stark_backend::prover::error::{
    LogupZerocheckError, StackedPcsError, StackedReductionError, WhirProverError,
};
use thiserror::Error;

/// Top-level error type for the CPU backend prover.
#[derive(Error, Debug)]
pub enum CpuBackendError {
    #[error("Stacked PCS: {0}")]
    StackedPcs(#[from] StackedPcsError),
    #[error("LogupZerocheck: {0}")]
    LogupZerocheck(#[from] LogupZerocheckError),
    #[error("Stacked reduction: {0}")]
    StackedReduction(#[from] StackedReductionError),
    #[error("WHIR: {0}")]
    Whir(#[from] WhirProverError),
}
