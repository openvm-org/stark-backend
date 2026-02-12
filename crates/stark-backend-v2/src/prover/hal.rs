// TODO[jpw]: replace v1 hal.rs file
// Keep from v1:
// - MatrixDimensions
//
// Changed ProverBackendV2 to remove Challenger(=Transcript) and non-essential types. Only keep the
// types you really need for interfaces. Protocol specific types moved to ProverDeviceV2 (possibly
// could be renamed ProtocolProver)

use std::sync::Arc;

use serde::{de::DeserializeOwned, Serialize};

use crate::{
    keygen::types::MultiStarkProvingKeyV2,
    prover::{
        stacked_pcs::StackedPcsData, AirProvingContextV2, ColMajorMatrix, CommittedTraceDataV2,
        CpuBackendV2, DeviceMultiStarkProvingKeyV2, ProvingContextV2,
    },
    StarkProtocolConfig, SystemParams,
};

pub trait MatrixDimensions {
    fn height(&self) -> usize;
    fn width(&self) -> usize;
}

/// Associated types needed by the prover, in the form of buffers and views,
/// specific to a specific hardware backend.
///
/// Memory allocation and copying is not handled by this trait.
pub trait ProverBackendV2 {
    /// Extension field degree for the challenge field `Self::Challenge` over base field
    /// `Self::Val`.
    const CHALLENGE_EXT_DEGREE: u8;
    // ==== Host Types ====
    /// Base field type, on host.
    type Val: Copy + Send + Sync + Serialize + DeserializeOwned;
    /// Challenge field (extension field of base field), on host.
    type Challenge: Copy + Send + Sync + Serialize + DeserializeOwned;
    /// Single commitment on host.
    // Commitments are small in size and need to be transferred back to host to be included in
    // proof.
    type Commitment: Clone + Send + Sync + Serialize + DeserializeOwned;

    // ==== Device Types ====
    /// Single matrix buffer on device together with dimension metadata. Owning this means nothing
    /// else has a shared reference to the buffer.
    type Matrix: MatrixDimensions + Send + Sync;
    /// Backend specific type for any pre-computed data associated with a single AIR. For example,
    /// it may contain prover-specific precomputations based on the AIR constraints (but
    /// independent from any trace data).
    type OtherAirData: Send + Sync;
    /// Owned buffer for the preimage of a PCS commitment on device, together with any metadata
    /// necessary for computing opening proofs.
    ///
    /// For example, multiple buffers for LDE matrices, their trace domain sizes, and pointer to
    /// mixed merkle tree.
    type PcsData: Send + Sync;
}

pub trait ProverDeviceV2<PB: ProverBackendV2, TS>:
    TraceCommitterV2<PB> + MultiRapProver<PB, TS> + OpeningProverV2<PB, TS>
{
    fn config(&self) -> &SystemParams;
}

/// Provides functionality for committing to a batch of trace matrices, possibly of different
/// heights.
pub trait TraceCommitterV2<PB: ProverBackendV2> {
    fn commit(&self, traces: &[&PB::Matrix]) -> (PB::Commitment, PB::PcsData);
}

/// This trait is responsible for all proving steps to prove a collection of trace matrices
/// satisfies all constraints of a Randomized AIR with Preprocessing. Such constraints include AIR
/// constraints as well as bus balancing constraints for interactions between AIRs. These
/// constraints may be grouped into challenge phases, where new randomness is sampled between phases
/// via Fiat-Shamir (which would involve committing to more data).
///
/// This trait is _not_ responsible for committing to the trace matrices or for proving polynomial
/// openings with respect to the committed trace matrices.
pub trait MultiRapProver<PB: ProverBackendV2, TS> {
    /// The partial proof is the proof that the trace matrices satisfy all constraints assuming that
    /// certain polynomial opening claims are validated. In other words, it is a proof that reduces
    /// the constraint satisfaction claim to certain polynomial opening claims.
    type PartialProof: Clone + Send + Sync + Serialize + DeserializeOwned;
    /// Other artifacts of the proof (e.g., sampled randomness) that may be passed to later stages
    /// of the protocol.
    type Artifacts;

    fn prove_rap_constraints(
        &self,
        transcript: &mut TS,
        mpk: &DeviceMultiStarkProvingKeyV2<PB>,
        ctx: &ProvingContextV2<PB>,
        common_main_pcs_data: &PB::PcsData,
    ) -> (Self::PartialProof, Self::Artifacts);
}

/// This trait is responsible for proving the evaluation claims of a collection of polynomials at a
/// collection of points. The opening point may be the same across polynomials. The polynomials may
/// be defined over different domains and are hence of "mixed" nature. The polynomials are already
/// committed and provided in their committed form.
pub trait OpeningProverV2<PB: ProverBackendV2, TS> {
    /// PCS opening proof on host. This should not be a reference.
    type OpeningProof: Clone + Send + Sync + Serialize + DeserializeOwned;
    type OpeningPoints;

    /// Computes the opening proof.
    /// The `common_main_pcs_data` is the `PcsData` for the collection of common main trace
    /// matrices. It is owned by the function and may be mutated.
    /// The `pre_cached_pcs_data_per_commit` is the `PcsData` for the preprocessed and cached trace
    /// matrices. These are specified by their `PcsData` per commitment.
    fn prove_openings(
        &self,
        transcript: &mut TS,
        mpk: &DeviceMultiStarkProvingKeyV2<PB>,
        ctx: ProvingContextV2<PB>,
        common_main_pcs_data: PB::PcsData,
        points: Self::OpeningPoints,
    ) -> Self::OpeningProof;
}

/// Trait to manage data transport of prover types from host to device.
pub trait DeviceDataTransporterV2<SC, PB>
where
    SC: StarkProtocolConfig,
    PB: ProverBackendV2<Val = SC::F, Challenge = SC::EF, Commitment = SC::Digest>,
{
    /// Transport the proving key to the device, filtering for only the provided `air_ids`.
    fn transport_pk_to_device(
        &self,
        mpk: &MultiStarkProvingKeyV2<SC>,
    ) -> DeviceMultiStarkProvingKeyV2<PB>;

    fn transport_matrix_to_device(&self, matrix: &ColMajorMatrix<SC::F>) -> PB::Matrix;

    /// The `commitment` and `prover_data` are assumed to have been previously computed from the
    /// `trace`.
    fn transport_pcs_data_to_device(
        &self,
        pcs_data: &StackedPcsData<SC::F, SC::Digest>,
    ) -> PB::PcsData;

    fn transport_committed_trace_data_to_device(
        &self,
        committed_trace: &CommittedTraceDataV2<CpuBackendV2<SC>>,
    ) -> CommittedTraceDataV2<PB> {
        let trace = self.transport_matrix_to_device(&committed_trace.trace);
        let data = self.transport_pcs_data_to_device(committed_trace.data.as_ref());

        CommittedTraceDataV2 {
            commitment: committed_trace.commitment,
            trace,
            data: Arc::new(data),
        }
    }

    fn transport_proving_ctx_to_device(
        &self,
        ctx: &ProvingContextV2<CpuBackendV2<SC>>,
    ) -> ProvingContextV2<PB> {
        let per_trace = ctx
            .per_trace
            .iter()
            .map(|(air_idx, air_ctx)| {
                let common_main = self.transport_matrix_to_device(&air_ctx.common_main);
                let cached_mains = air_ctx
                    .cached_mains
                    .iter()
                    .map(|cd| self.transport_committed_trace_data_to_device(cd))
                    .collect();
                let air_ctx_gpu = AirProvingContextV2::new(
                    cached_mains,
                    common_main,
                    air_ctx.public_values.clone(),
                );
                (*air_idx, air_ctx_gpu)
            })
            .collect();
        ProvingContextV2::new(per_trace)
    }

    // ==================================================================================
    // Device-to-Host methods below should only be used for testing / debugging purposes.
    // ==================================================================================

    /// Transport a device matrix to host. This should only be used for testing / debugging
    /// purposes.
    fn transport_matrix_from_device_to_host(&self, matrix: &PB::Matrix) -> ColMajorMatrix<SC::F>;
}
