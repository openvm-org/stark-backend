//! Abstraction layer for prover implementations of multi-matrix circuits on a single machine.
//!
//! Provides a coordinator that implements a full prover by coordinating between host and device, where
//! the host implementation is common and the device implementation relies on custom-specified device kernels.
//!
//! Currently includes full prover implementations for:
//! - CPU

/// Host prover implementation that uses custom device kernels
pub mod coordinator;
/// CPU implementation of proving backend
pub mod cpu;
pub mod hal;
/// Types used by the prover
pub mod types;

/// Testing helper
pub mod helper; // [jpw]: maybe this should be moved to sdk
/// Metrics about trace and other statistics related to prover performance
pub mod metrics;

/// Trait for STARK/SNARK proving at the highest abstraction level.
pub trait Prover {
    type ProvingKeyView<'a>;
    type ProvingContext;
    type Proof;

    /// The prover should own the challenger, whose state mutates during proving.
    fn prove<'a>(&mut self, pk: Self::ProvingKeyView<'a>, ctx: Self::ProvingContext)
        -> Self::Proof;
}
