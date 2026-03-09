#![allow(
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::useless_conversion,
    clippy::unnecessary_cast,
    clippy::wrong_self_convention
)]
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
pub mod error;
pub mod logup_zerocheck;
mod stacked_reduction;
mod whir;

pub use backend::*;
pub use device::*;
pub use error::CpuProverError;
