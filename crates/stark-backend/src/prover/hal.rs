//! # Hardware Abstraction Layer
//!
//! Not all hardware implementations need to implement this.
//! A pure external device implementation can just implement the [Prover](super::Prover) trait directly.

use std::sync::Arc;

use p3_challenger::CanObserve;
use p3_matrix::dense::RowMajorMatrix;
use serde::{de::DeserializeOwned, Serialize};

use super::types::{
    MultiStarkProvingKeyView, PairView, ProverViewAfterRapPhases, RapView, StarkProvingKeyView,
};
use crate::{
    air_builders::symbolic::SymbolicExpressionDag,
    config::{Com, RapPhaseSeqProvingKey, StarkGenericConfig, Val},
    keygen::types::MultiStarkProvingKey,
};

/// Associated types needed by the prover, in the form of buffers and views,
/// specific to a specific hardware backend.
///
/// Memory allocation and copying is not handled by this trait.
pub trait ProverBackend {
    /// Extension field degree for the challenge field `Self::Challenge` over base field `Self::Val`.
    const CHALLENGE_EXT_DEGREE: u8;
    // ==== Host Types ====
    /// Base field type, on host.
    type Val: Copy + Send + Sync + Serialize + DeserializeOwned;
    /// Challenge field (extension field of base field), on host.
    type Challenge: Copy + Send + Sync + Serialize + DeserializeOwned;
    /// PCS opening proof on host (see [OpeningProver]). This should not be a reference.
    type OpeningProof: Clone + Send + Sync + Serialize + DeserializeOwned;
    /// Single commitment on host.
    // Commitments are small in size and need to be transferred back to host to be included in proof.
    type Commitment: Clone + Send + Sync + Serialize + DeserializeOwned;
    /// Challenger to observe commitments. Sampling is left to other trait implementations.
    /// We anticipate that the challenger largely operates on the host.
    type Challenger: CanObserve<Self::Val> + CanObserve<Self::Commitment>;

    // ==== Device Types ====
    /// Single matrix buffer on device together with dimension metadata.
    type MatrixView: MatrixView + Clone + Send + Sync;
    /// Reference for the preimage of a PCS commitment on device, together with any metadata
    /// necessary for computing opening proofs.
    ///
    /// For example, multiple buffers for LDE matrices, their trace domain sizes, and pointer to mixed merkle tree.
    type PcsDataView: Clone + Send + Sync;
}

pub trait MatrixView {
    fn height(&self) -> usize;
    fn width(&self) -> usize;
}

pub trait ProverDevice<PB: ProverBackend>:
    TraceCommitter<PB> + RapPartialProver<PB> + QuotientCommitter<PB> + OpeningProver<PB>
{
}

/// Provides functionality for committing to a batch of trace matrices, possibly of different heights.
pub trait TraceCommitter<PB: ProverBackend> {
    fn commit(&self, traces: &[PB::MatrixView]) -> (PB::Commitment, PB::PcsDataView);
}

/// This trait is responsible for all partial proving of after challenge rounds (a.k.a layers) in a
/// RAP after the main trace has been committed.
///
/// The partial prover *may*:
/// - observe and/or sample challenges
/// - commit to additional trace data
/// - generate other partial proof data
pub trait RapPartialProver<PB: ProverBackend> {
    // ==== Host Types ====
    /// Partial proof for multiple RAPs
    type RapPartialProof: Clone + Send + Sync;
    /// Part of proving key for a single RAP specific for the RAP challenge phases
    type RapPartialProvingKeyView<'a>: Clone + Send + Sync
    where
        Self: 'a;

    /// The `trace_views` are the views of the respective trace matrices, evaluated on the trace domain.
    /// Currently this function does not provide a view of any already committed data associated
    /// with the trace views, although that data is available.
    fn partially_prove<'a>(
        &self,
        challenger: &mut PB::Challenger,
        pk_views: &[StarkProvingKeyView<'a, PB, Self::RapPartialProvingKeyView<'a>>],
        trace_views: Vec<PairView<PB::MatrixView, PB::Val>>,
    ) -> (Self::RapPartialProof, ProverViewAfterRapPhases<PB>);
}

/// Only needed in proof systems that use quotient polynomials.
pub trait QuotientCommitter<PB: ProverBackend> {
    /// Given PCS prover data for a commitment and an index of a matrix in
    /// the commitment, return a view of the matrix evaluated
    /// on the quotient domain. In practice this will be the LDE matrix
    /// evaluated on the LDE domain which contains the quotient domain as a
    /// subgroup.
    fn get_extended_matrix(
        &self,
        view: &PB::PcsDataView,
        matrix_idx: usize,
        quotient_degree: u8,
    ) -> Option<PB::MatrixView>;

    /// Evaluate the quotient polynomial on the quotient domain and then commit to it.
    /// The `extended_views` are extensions of the respective trace matrices
    /// to evaluations on the quotient domain (or an even larger domain).
    ///
    /// The lengths of `quotient_degrees`, `constraints`, `ldes`, and `public_values` must be equal
    /// and zip together to correspond to a list of RAPs.
    ///
    /// Currently we assume that the quotient domain is a subgroup of the LDE domain so that the
    /// quotient polynomial evaluation can be done on the LDE domain. This avoids a separate
    /// cosetDFT step.
    ///
    /// For each RAP, `quotient_degree` is the number of quotient chunks that were committed.
    /// So `quotient_degree = quotient_domain_size / trace_domain_size`.
    ///
    /// The `constraints` contains the serializable symbolic constraints of each RAP across all challenge phases.
    ///
    /// Quotient polynomials for multiple RAP matrices are committed together into a single commitment.
    /// The quotient polynomials can be committed together even if the corresponding trace matrices
    /// are committed separately.
    fn eval_and_commit_quotient(
        &self,
        challenger: &mut PB::Challenger,
        constraints: &[&SymbolicExpressionDag<PB::Val>],
        extended_views: Vec<RapView<PB::MatrixView, PB::Val, PB::Challenge>>,
        quotient_degrees: &[u8],
    ) -> (PB::Commitment, PB::PcsDataView);
}

/// Polynomial commitment scheme (PCS) opening proof generator.
pub trait OpeningProver<PB: ProverBackend> {
    /// Opening proof for multiple RAP matrices, where
    /// - (for now) each preprocessed trace matrix has a separate commitment
    /// - main trace matrices can have multiple commitments
    /// - for each after_challenge phase, all matrices in the phase share a commitment
    /// - quotient poly chunks are all committed together
    fn open(
        &self,
        challenger: &mut PB::Challenger,
        // For each preprocessed trace commitment, the prover data and
        // the log height of the matrix, in order
        preprocessed: Vec<PB::PcsDataView>,
        // For each main trace commitment, the prover data and
        // the log height of each matrix, in order
        // Note: this is all one challenge phase.
        main: Vec<PB::PcsDataView>,
        // `after_phase[i]` has shared commitment prover data for all matrices in phase `i + 1`.
        after_phase: Vec<PB::PcsDataView>,
        // Quotient poly commitment prover data
        quotient_data: PB::PcsDataView,
        // Quotient degree for each RAP committed in quotient_data, in order
        quotient_degrees: &[u8],
    ) -> PB::OpeningProof;
}

/// Trait to manage data transport of prover types from host to device.
pub trait DeviceDataTransporterpter<SC, PB>
where
    SC: StarkGenericConfig,
    PB: ProverBackend<Val = Val<SC>, Challenge = SC::Challenge, Commitment = Com<SC>>,
{
    /// Transport the proving key to the device, filtering for only the provided `air_ids`.
    fn transport_pk_to_device<'a>(
        &'a self,
        mpk: &'a MultiStarkProvingKey<SC>,
        air_ids: Vec<usize>,
    ) -> MultiStarkProvingKeyView<'a, PB, &'a RapPhaseSeqProvingKey<SC>>
    where
        SC: 'a;

    fn transport_matrix_to_device(&self, matrix: &Arc<RowMajorMatrix<Val<SC>>>) -> PB::MatrixView;

    fn transport_pcs_data_to_device(&self, data: &super::cpu::PcsDataView<SC>) -> PB::PcsDataView;
}
