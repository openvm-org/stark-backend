//! Metal backend marker type and ProverBackend implementation.
//!
//! This module defines `MetalBackend` and implements the prover traits on `MetalDevice`.
//! All proving operations delegate to Metal-native modules in this crate.

use openvm_stark_backend::{
    proof::*,
    prover::{
        DeviceMultiStarkProvingKey, MultiRapProver, OpeningProver, ProverBackend, ProverDevice,
        ProvingContext, TraceCommitter,
    },
};
use tracing::instrument;

use crate::{
    base::MetalMatrix,
    prelude::{Digest, D_EF, EF, F, SC},
    sponge::DuplexSpongeMetal,
    stacked_pcs::StackedPcsDataMetal,
    AirDataMetal, MetalDevice,
};

#[derive(Clone, Copy)]
pub struct MetalBackend;

impl ProverBackend for MetalBackend {
    const CHALLENGE_EXT_DEGREE: u8 = D_EF as u8;

    type Val = F;
    type Challenge = EF;
    type Commitment = Digest;
    type Matrix = MetalMatrix<F>;
    type PcsData = StackedPcsDataMetal;
    type OtherAirData = AirDataMetal;
}

impl ProverDevice<MetalBackend, DuplexSpongeMetal> for MetalDevice {}

impl TraceCommitter<MetalBackend> for MetalDevice {
    #[instrument(name = "metal.trace_commit", skip_all, fields(phase = "prover"))]
    fn commit(&self, traces: &[&MetalMatrix<F>]) -> (Digest, StackedPcsDataMetal) {
        crate::stacked_pcs::commit_traces_metal(self.config(), traces, self.prover_config())
    }
}

impl MultiRapProver<MetalBackend, DuplexSpongeMetal> for MetalDevice {
    type PartialProof = (GkrProof<SC>, BatchConstraintProof<SC>);
    type Artifacts = Vec<EF>;

    #[instrument(name = "metal.rap_constraints", skip_all, fields(phase = "prover"))]
    fn prove_rap_constraints(
        &self,
        transcript: &mut DuplexSpongeMetal,
        mpk: &DeviceMultiStarkProvingKey<MetalBackend>,
        ctx: &ProvingContext<MetalBackend>,
        _common_main_pcs_data: &StackedPcsDataMetal,
    ) -> ((GkrProof<SC>, BatchConstraintProof<SC>), Vec<EF>) {
        crate::logup_zerocheck::prove_constraints_metal(transcript, mpk, ctx, self)
    }
}

impl OpeningProver<MetalBackend, DuplexSpongeMetal> for MetalDevice {
    type OpeningProof = (StackingProof<SC>, WhirProof<SC>);
    type OpeningPoints = Vec<EF>;

    #[instrument(name = "metal.openings", skip_all, fields(phase = "prover"))]
    fn prove_openings(
        &self,
        transcript: &mut DuplexSpongeMetal,
        mpk: &DeviceMultiStarkProvingKey<MetalBackend>,
        ctx: ProvingContext<MetalBackend>,
        common_main_pcs_data: StackedPcsDataMetal,
        r: Vec<EF>,
    ) -> (StackingProof<SC>, WhirProof<SC>) {
        crate::openings::prove_openings_metal(
            self,
            transcript,
            mpk,
            ctx,
            common_main_pcs_data,
            r,
        )
    }
}
