//! Metal-native logup/zerocheck prover.
//!
//! Implements the GKR + batch sumcheck protocol for zerocheck and logup constraints.
//! Uses CPU computation on Metal unified memory data for correctness.
//!
//! # Strategy
//!
//! The GKR + batch sumcheck algorithm is very complex (~1600 lines in the CUDA backend).
//! Rather than reimplementing from scratch, we leverage Metal's unified memory model:
//!
//! 1. Read MetalMatrix data from unified memory (MetalBuffer) -- this is a direct memory
//!    read, no GPU sync needed.
//! 2. Convert Metal types (MetalMatrix, StackedPcsDataMetal) to CPU equivalents
//!    (ColMajorMatrix, StackedPcsData) via the `convert` module.
//! 3. Call the stark-backend CPU `prove_zerocheck_and_logup` function.
//! 4. Return the protocol-agnostic proof types unchanged.
//!
//! The `to_cpu()` method on `StackedPcsDataMetal` is used internally by the `convert`
//! module for type conversion. It is NOT called from `metal_backend.rs`.

use openvm_stark_backend::{
    proof::{BatchConstraintProof, GkrProof},
    prover::{prove_zerocheck_and_logup, DeviceMultiStarkProvingKey, ProvingContext},
};
use tracing::instrument;

use crate::{
    prelude::{EF, SC},
    sponge::DuplexSpongeMetal,
    MetalBackend, MetalDevice,
};

/// Prove zerocheck and logup constraints for the Metal backend.
///
/// Converts Metal proving key and context to CPU types, then delegates to the
/// stark-backend CPU algorithm. This is correct because Metal's unified memory
/// allows direct reads from MetalBuffer without GPU synchronization.
///
/// # Arguments
///
/// * `transcript` - Fiat-Shamir transcript (Metal sponge, delegates to CPU challenger)
/// * `mpk` - Multi-STARK proving key with MetalMatrix traces
/// * `ctx` - Proving context with MetalMatrix traces and public values
/// * `_device` - Metal device handle (unused, reserved for future GPU acceleration)
///
/// # Returns
///
/// A tuple of:
/// * `(GkrProof, BatchConstraintProof)` - the GKR and batch constraint proofs
/// * `Vec<EF>` - the random evaluation point `r` for the opening phase
#[instrument(name = "metal.logup_zerocheck", skip_all)]
pub fn prove_constraints_metal(
    transcript: &mut DuplexSpongeMetal,
    mpk: &DeviceMultiStarkProvingKey<MetalBackend>,
    ctx: &ProvingContext<MetalBackend>,
    _device: &MetalDevice,
) -> ((GkrProof<SC>, BatchConstraintProof<SC>), Vec<EF>) {
    // Convert Metal types to CPU types for algorithm execution.
    // This reads from unified memory (direct memcpy, no GPU sync).
    let cpu_mpk = crate::convert::mpk_to_cpu(mpk);
    let cpu_ctx = crate::convert::ctx_to_cpu(ctx);

    // Run the CPU algorithm on the converted data.
    let (gkr_proof, batch_constraint_proof, r) =
        prove_zerocheck_and_logup::<SC, _>(transcript, &cpu_mpk, &cpu_ctx);

    ((gkr_proof, batch_constraint_proof), r)
}
