//! Abstraction layer for prover implementations of multi-matrix circuits on a single machine.
//!
//! Provides a coordinator that implements a full prover given a set of device kernels.
//!
//! Currently includes full prover implementations for:
//! - CPU

/// Host prover implementation that uses custom device kernels
pub mod coordinator;
pub mod hal;
pub mod helper;
/// Metrics about trace and other statistics related to prover performance
pub mod metrics;
/// Polynomial opening proofs
pub mod opener;
/// Computation of DEEP quotient polynomial and commitment
pub mod quotient;
/// Trace commitment computation
mod trace;
/// Types used by the prover
pub mod types;

/// Trait for STARK/SNARK proving at the highest abstraction level.
pub trait Prover {
    type ProvingKeyRef<'a>;
    type ProvingContext<'a>;
    type Proof;

    fn prove<'a>(&self, pk: Self::ProvingKeyRef<'a>, ctx: Self::ProvingContext<'a>) -> Self::Proof;
}
