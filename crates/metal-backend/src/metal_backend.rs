use itertools::Itertools;
use openvm_stark_backend::{
    poly_common::Squarable,
    proof::*,
    prover::{
        DeviceMultiStarkProvingKey, MultiRapProver, OpeningProver, ProverBackend, ProverDevice,
        ProvingContext, TraceCommitter,
    },
};
use tracing::instrument;

use crate::{
    base::MetalMatrix,
    logup_zerocheck::prove_zerocheck_and_logup_metal,
    prelude::{Digest, D_EF, EF, F, SC},
    sponge::DuplexSpongeMetal,
    stacked_pcs::{stacked_commit, StackedPcsDataMetal},
    stacked_reduction::prove_stacked_opening_reduction_metal,
    whir::prove_whir_opening_metal,
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
    type PcsData = StackedPcsDataMetal<F, Digest>;
    type OtherAirData = AirDataMetal;
}

impl ProverDevice<MetalBackend, DuplexSpongeMetal> for MetalDevice {}

impl TraceCommitter<MetalBackend> for MetalDevice {
    fn commit(&self, traces: &[&MetalMatrix<F>]) -> (Digest, StackedPcsDataMetal<F, Digest>) {
        stacked_commit(
            self.config.l_skip,
            self.config.n_stack,
            self.config.log_blowup,
            self.config.k_whir(),
            traces,
            self.prover_config,
        )
        .unwrap()
    }
}

impl MultiRapProver<MetalBackend, DuplexSpongeMetal> for MetalDevice {
    type PartialProof = (GkrProof<SC>, BatchConstraintProof<SC>);
    type Artifacts = Vec<EF>;

    #[instrument(name = "prover.rap_constraints", skip_all, fields(phase = "prover"))]
    fn prove_rap_constraints(
        &self,
        transcript: &mut DuplexSpongeMetal,
        mpk: &DeviceMultiStarkProvingKey<MetalBackend>,
        ctx: &ProvingContext<MetalBackend>,
        _common_main_pcs_data: &StackedPcsDataMetal<F, Digest>,
    ) -> ((GkrProof<SC>, BatchConstraintProof<SC>), Vec<EF>) {
        let save_memory = self.prover_config.zerocheck_save_memory;
        // Threshold for monomial evaluation path based on proof type:
        // - App proofs (log_blowup=1): higher threshold (512)
        // - Recursion proofs: lower threshold (64)
        let monomial_num_y_threshold = if self.config.log_blowup == 1 { 512 } else { 64 };
        let (gkr_proof, batch_constraint_proof, r) = prove_zerocheck_and_logup_metal(
            transcript,
            mpk,
            ctx,
            save_memory,
            monomial_num_y_threshold,
            self.sm_count,
        );
        ((gkr_proof, batch_constraint_proof), r)
    }
}

impl OpeningProver<MetalBackend, DuplexSpongeMetal> for MetalDevice {
    type OpeningProof = (StackingProof<SC>, WhirProof<SC>);
    type OpeningPoints = Vec<EF>;

    #[instrument(name = "prover.openings", skip_all, fields(phase = "prover"))]
    fn prove_openings(
        &self,
        transcript: &mut DuplexSpongeMetal,
        mpk: &DeviceMultiStarkProvingKey<MetalBackend>,
        ctx: ProvingContext<MetalBackend>,
        common_main_pcs_data: StackedPcsDataMetal<F, Digest>,
        r: Vec<EF>,
    ) -> (StackingProof<SC>, WhirProof<SC>) {
        let params = self.config();
        let (stacking_proof, u_prisma, stacked_per_commit) = prove_stacked_opening_reduction_metal(
            self,
            transcript,
            mpk,
            ctx,
            common_main_pcs_data,
            &r,
        )
        .unwrap();

        let (&u0, u_rest) = u_prisma.split_first().unwrap();
        let u_cube = u0
            .exp_powers_of_2()
            .take(params.l_skip)
            .chain(u_rest.iter().copied())
            .collect_vec();

        let whir_proof =
            prove_whir_opening_metal(params, transcript, stacked_per_commit, &u_cube).unwrap();
        (stacking_proof, whir_proof)
    }
}
