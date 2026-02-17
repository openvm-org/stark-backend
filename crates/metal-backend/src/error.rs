use openvm_metal_common::error::MetalError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProverError {
    #[error("Metal execution: {0}")]
    MetalExecution(MetalError),
    #[error("Not implemented: {0}")]
    NotImplemented(String),
}
