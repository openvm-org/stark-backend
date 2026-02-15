use itertools::Itertools;
use openvm_cuda_common::memory_manager::MemTracker;
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
    base::DeviceMatrix,
    logup_zerocheck::prove_zerocheck_and_logup_gpu,
    prelude::{Digest, D_EF, EF, F, SC},
    sponge::DuplexSpongeGpu,
    stacked_pcs::{stacked_commit, StackedPcsDataGpu},
    stacked_reduction::prove_stacked_opening_reduction_gpu,
    whir::prove_whir_opening_gpu,
    AirDataGpu, GpuDevice,
};

#[derive(Clone, Copy)]
pub struct GpuBackend;

impl ProverBackend for GpuBackend {
    const CHALLENGE_EXT_DEGREE: u8 = D_EF as u8;

    type Val = F;
    type Challenge = EF;
    type Commitment = Digest;
    type Matrix = DeviceMatrix<F>;
    type PcsData = StackedPcsDataGpu<F, Digest>;
    type OtherAirData = AirDataGpu;
}

impl ProverDevice<GpuBackend, DuplexSpongeGpu> for GpuDevice {}

impl TraceCommitter<GpuBackend> for GpuDevice {
    fn commit(&self, traces: &[&DeviceMatrix<F>]) -> (Digest, StackedPcsDataGpu<F, Digest>) {
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

impl MultiRapProver<GpuBackend, DuplexSpongeGpu> for GpuDevice {
    type PartialProof = (GkrProof<SC>, BatchConstraintProof<SC>);
    /// The random opening point `r` where the batch constraint sumcheck reduces to evaluation
    /// claims of trace matrices `T, T_{rot}` at `r_{n_T}`.
    type Artifacts = Vec<EF>;

    #[instrument(name = "prover.rap_constraints", skip_all, fields(phase = "prover"))]
    fn prove_rap_constraints(
        &self,
        transcript: &mut DuplexSpongeGpu,
        mpk: &DeviceMultiStarkProvingKey<GpuBackend>,
        ctx: &ProvingContext<GpuBackend>,
        _common_main_pcs_data: &StackedPcsDataGpu<F, Digest>,
    ) -> ((GkrProof<SC>, BatchConstraintProof<SC>), Vec<EF>) {
        let mem = MemTracker::start_and_reset_peak("prover.rap_constraints");
        let save_memory = self.prover_config.zerocheck_save_memory;
        // Threshold for monomial evaluation path based on proof type:
        // - App proofs (log_blowup=1): higher threshold (512)
        // - Recursion proofs: lower threshold (64)
        let monomial_num_y_threshold = if self.config.log_blowup == 1 { 512 } else { 64 };
        let (gkr_proof, batch_constraint_proof, r) = prove_zerocheck_and_logup_gpu(
            transcript,
            mpk,
            ctx,
            save_memory,
            monomial_num_y_threshold,
            self.sm_count,
        );
        mem.emit_metrics();
        ((gkr_proof, batch_constraint_proof), r)
    }
}

impl OpeningProver<GpuBackend, DuplexSpongeGpu> for GpuDevice {
    type OpeningProof = (StackingProof<SC>, WhirProof<SC>);
    /// The shared vector `r` where each trace matrix `T, T_{rot}` is opened at `r_{n_T}`.
    type OpeningPoints = Vec<EF>;

    #[instrument(name = "prover.openings", skip_all, fields(phase = "prover"))]
    fn prove_openings(
        &self,
        transcript: &mut DuplexSpongeGpu,
        mpk: &DeviceMultiStarkProvingKey<GpuBackend>,
        ctx: ProvingContext<GpuBackend>,
        common_main_pcs_data: StackedPcsDataGpu<F, Digest>,
        r: Vec<EF>,
    ) -> (StackingProof<SC>, WhirProof<SC>) {
        let mut mem = MemTracker::start_and_reset_peak("prover.openings");
        let params = self.config();
        let (stacking_proof, u_prisma, stacked_per_commit) = prove_stacked_opening_reduction_gpu(
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
            prove_whir_opening_gpu(params, transcript, stacked_per_commit, &u_cube).unwrap();
        mem.emit_metrics();
        mem.reset_peak();
        (stacking_proof, whir_proof)
    }
}
