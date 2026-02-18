//! Metal-native WHIR opening proof.
//!
//! Implements the WHIR opening proof protocol, mirroring
//! the CUDA `whir.rs` module structure.

use openvm_stark_backend::{
    proof::WhirProof,
    prover::whir::prove_whir_opening,
    StarkProtocolConfig, SystemParams,
};
use tracing::instrument;

use crate::{
    prelude::{EF, SC},
    sponge::DuplexSpongeMetal,
    StackedPcsDataMetal,
};

/// Metal-native WHIR opening proof.
///
/// Converts Metal PCS data to CPU types internally and delegates to the CPU algorithm.
/// This is correct because Metal uses unified memory (data is directly accessible).
#[instrument(name = "metal.whir", skip_all)]
pub fn prove_whir_opening_metal(
    params: &SystemParams,
    transcript: &mut DuplexSpongeMetal,
    stacked_per_commit: Vec<&StackedPcsDataMetal>,
    u: &[EF],
) -> WhirProof<SC> {
    let sc = SC::default_from_params(params.clone());
    let hasher = sc.hasher();

    // Access the CPU data from StackedPcsDataMetal
    let committed_mats: Vec<_> = stacked_per_commit
        .iter()
        .map(|d| (&d.inner().matrix, &d.inner().tree))
        .collect();

    prove_whir_opening::<SC, _>(
        transcript,
        hasher,
        params.l_skip,
        params.log_blowup,
        &params.whir,
        &committed_mats,
        u,
    )
}
