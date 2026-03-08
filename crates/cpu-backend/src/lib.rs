//! Row-major CPU prover backend for the SWIRL proof system.
//!
//! This crate provides [`CpuBackend`] and [`CpuDevice`], a prover backend that uses
//! row-major matrix layout for the constraint evaluation hot path (logup_zerocheck).
//! This gives significantly better cache locality and enables SIMD vectorization
//! via plonky3's `PackedField`.
//!
//! At PCS boundaries (trace commitment, WHIR opening), the row-major matrices are
//! converted to column-major format, which is a one-time cost.

mod backend;
mod device;
pub mod engine;
pub mod logup_zerocheck;
mod stacked_reduction;
pub mod transcript;
mod whir;

pub use backend::*;
pub use device::*;
pub use transcript::CpuTranscript;
