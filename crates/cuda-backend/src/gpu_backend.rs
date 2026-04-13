use std::marker::PhantomData;

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
    hash_scheme::{DefaultHashScheme, GpuHashScheme},
    logup_zerocheck::prove_zerocheck_and_logup_gpu,
    merkle_tree::{MerkleProofQueryDigest, MerkleTreeConstructor},
    prelude::{D_EF, EF, F},
    sponge::GpuFiatShamirTranscript,
    stacked_pcs::{stacked_commit, StackedPcsDataGpu},
    stacked_reduction::prove_stacked_opening_reduction_gpu,
    whir::prove_whir_opening_gpu,
    AirDataGpu, GpuDevice, ProverError,
};

/// Generic GPU prover backend parameterised by a hash scheme `HS`.
///
/// Use the [`GpuBackend`] type alias to refer to the concrete BabyBear-Poseidon2
/// backend without spelling out the generic parameter.
#[derive(Clone, Copy)]
pub struct GenericGpuBackend<HS: GpuHashScheme>(PhantomData<HS>);

impl<HS: GpuHashScheme> Default for GenericGpuBackend<HS> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Concrete GPU backend using the default BabyBear-Poseidon2 hash scheme.
pub type GpuBackend = GenericGpuBackend<DefaultHashScheme>;

impl<HS: GpuHashScheme> ProverBackend for GenericGpuBackend<HS> {
    const CHALLENGE_EXT_DEGREE: u8 = D_EF as u8;

    type Val = F;
    type Challenge = EF;
    type Commitment = HS::Digest;
    type Matrix = DeviceMatrix<F>;
    type PcsData = StackedPcsDataGpu<F, HS::Digest>;
    type OtherAirData = AirDataGpu;
}

impl<HS: GpuHashScheme> TraceCommitter<GenericGpuBackend<HS>> for GpuDevice
where
    HS::MerkleHash: MerkleTreeConstructor,
{
    type Error = ProverError;

    fn commit(
        &self,
        traces: &[&DeviceMatrix<F>],
    ) -> Result<(HS::Digest, StackedPcsDataGpu<F, HS::Digest>), Self::Error> {
        let cfg = self.params();
        stacked_commit::<HS::MerkleHash>(
            cfg.l_skip,
            cfg.n_stack,
            cfg.log_blowup,
            cfg.k_whir(),
            traces,
            *self.prover_config(),
            &self.device_ctx,
        )
    }
}

impl<HS: GpuHashScheme, TS: GpuFiatShamirTranscript<HS::SC>> ProverDevice<GenericGpuBackend<HS>, TS>
    for GpuDevice
where
    HS::MerkleHash: MerkleTreeConstructor,
    HS::Digest: MerkleProofQueryDigest,
{
    type Error = ProverError;
    type DeviceCtx = openvm_cuda_common::stream::GpuDeviceCtx;

    fn device_ctx(&self) -> &openvm_cuda_common::stream::GpuDeviceCtx {
        &self.device_ctx
    }
}

impl<HS: GpuHashScheme, TS: GpuFiatShamirTranscript<HS::SC>>
    MultiRapProver<GenericGpuBackend<HS>, TS> for GpuDevice
{
    type PartialProof = (GkrProof<HS::SC>, BatchConstraintProof<HS::SC>);
    /// The random opening point `r` where the batch constraint sumcheck reduces to evaluation
    /// claims of trace matrices `T, T_{rot}` at `r_{n_T}`.
    type Artifacts = Vec<EF>;
    type Error = ProverError;

    #[allow(clippy::type_complexity)]
    #[instrument(name = "prover.rap_constraints", skip_all, fields(phase = "prover"))]
    fn prove_rap_constraints(
        &self,
        transcript: &mut TS,
        mpk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
        ctx: &ProvingContext<GenericGpuBackend<HS>>,
        _common_main_pcs_data: &StackedPcsDataGpu<F, HS::Digest>,
    ) -> Result<((GkrProof<HS::SC>, BatchConstraintProof<HS::SC>), Vec<EF>), Self::Error> {
        let mem = MemTracker::start_and_reset_peak("prover.rap_constraints");
        let save_memory = self.prover_config().zerocheck_save_memory;
        // Threshold for monomial evaluation path based on proof type:
        // - App proofs (log_blowup=1): higher threshold (512)
        // - Recursion proofs: lower threshold (64)
        let monomial_num_y_threshold = if self.params().log_blowup == 1 {
            512
        } else {
            64
        };
        let (gkr_proof, batch_constraint_proof, r) = prove_zerocheck_and_logup_gpu::<HS, TS>(
            transcript,
            mpk,
            ctx,
            save_memory,
            monomial_num_y_threshold,
            self.sm_count(),
            &self.device_ctx,
        )?;
        mem.emit_metrics();
        Ok(((gkr_proof, batch_constraint_proof), r))
    }
}

impl<HS: GpuHashScheme, TS: GpuFiatShamirTranscript<HS::SC>>
    OpeningProver<GenericGpuBackend<HS>, TS> for GpuDevice
where
    HS::MerkleHash: MerkleTreeConstructor,
    HS::Digest: MerkleProofQueryDigest,
{
    type OpeningProof = (StackingProof<HS::SC>, WhirProof<HS::SC>);
    /// The shared vector `r` where each trace matrix `T, T_{rot}` is opened at `r_{n_T}`.
    type OpeningPoints = Vec<EF>;
    type Error = ProverError;

    #[instrument(name = "prover.openings", skip_all, fields(phase = "prover"))]
    fn prove_openings(
        &self,
        transcript: &mut TS,
        mpk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
        ctx: ProvingContext<GenericGpuBackend<HS>>,
        common_main_pcs_data: StackedPcsDataGpu<F, HS::Digest>,
        r: Vec<EF>,
    ) -> Result<Self::OpeningProof, Self::Error> {
        let mut mem = MemTracker::start_and_reset_peak("prover.openings");
        let params = self.params();
        #[cfg(debug_assertions)]
        {
            let total_stacked_width: usize = std::iter::once(common_main_pcs_data.layout().width())
                .chain(ctx.per_trace.iter().flat_map(|(air_idx, air_ctx)| {
                    mpk.per_air[*air_idx]
                        .preprocessed_data
                        .iter()
                        .map(|committed| committed.data.layout().width())
                        .chain(
                            air_ctx
                                .cached_mains
                                .iter()
                                .map(|committed| committed.data.layout().width()),
                        )
                }))
                .sum();
            debug_assert!(
                total_stacked_width <= mpk.params.w_stack,
                "total stacked width across commits ({total_stacked_width}) exceeds w_stack ({})",
                mpk.params.w_stack
            );
        }
        let (stacking_proof, u_prisma, stacked_per_commit) =
            prove_stacked_opening_reduction_gpu::<HS, TS>(
                self,
                transcript,
                mpk,
                ctx,
                common_main_pcs_data,
                &r,
            )?;

        let (&u0, u_rest) = u_prisma.split_first().unwrap();
        let u_cube = u0
            .exp_powers_of_2()
            .take(params.l_skip)
            .chain(u_rest.iter().copied())
            .collect_vec();

        let whir_proof = prove_whir_opening_gpu::<HS, TS>(
            params,
            transcript,
            stacked_per_commit,
            &u_cube,
            &self.device_ctx,
        )?;
        mem.emit_metrics();
        mem.reset_peak();
        Ok((stacking_proof, whir_proof))
    }
}
