//! # Hardware Abstraction Layer
//!
//! Not all hardware implementations need to implement this.
//! A pure external device implementation can just implement the [Prover](super::Prover) trait directly.

use serde::{de::DeserializeOwned, Serialize};

use super::types::{CommittedDataRef, SingleRapView};
use crate::air_builders::symbolic::dag::SymbolicExpressionDag;

/// Associated types needed by the prover, in the form of buffers, specific to a specific hardware backend.
pub trait ProverBackend {
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

    // ==== Device Types ====
    /// Buffer of base field elements on device.
    type ValBuffer<'a>: SizedArray + Copy + Send + Sync + 'a;
    /// Buffer of challenge field (an extension field of base field) elements on device.
    type ChallengeBuffer<'a>: SizedArray + Copy + Send + Sync + 'a;
    /// Single matrix buffer on device.
    type MatrixBuffer<'a>: SizedMatrix + Copy + Send + Sync + 'a;
    /// Reference for the preimage of a PCS commitment on device.
    /// For example, buffer of the LDE matrix and pointer to its merkle tree.
    type PcsDataRef<'a>: Copy + Send + Sync + 'a;
}

pub trait SizedArray {
    /// Length in number of elements, where element type depends on the implementation.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait SizedMatrix {
    fn height(&self) -> usize;
    fn width(&self) -> usize;
}

/// Provides functionality for committing to a batch of trace matrices, possibly of different heights.
pub trait TraceCommitter<PB: ProverBackend> {
    fn commit<'tr, 'com>(&self, traces: &[PB::MatrixBuffer<'tr>]) -> CommittedDataRef<'com, PB>;
}

/// Only needed in proof systems that use quotient polynomials.
pub trait QuotientCommitter<PB: ProverBackend> {
    fn load_challenges(&mut self, challenges: PB::ChallengeBuffer<'_>, alpha: PB::Challenge);

    /// Evaluate the quotient polynomial on the quotient domain and then commit to it.
    /// The `ldes` are the Low Degree Extensions (LDE) matrices of the respective trace matrices.
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
    fn eval_and_commit<'a, 'b>(
        &self,
        quotient_degrees: &[u8],
        constraints: &[&'a SymbolicExpressionDag<PB::Val>],
        ldes: &[SingleRapView<PB::MatrixBuffer<'a>, PB::ChallengeBuffer<'a>>],
        public_values: &'a [PB::ValBuffer<'a>],
    ) -> CommittedDataRef<'a, PB>;
}

/// Polynomial commitment scheme (PCS) opening proof generator.
pub trait OpeningProver<PB: ProverBackend> {
    /// Opening proof for multiple RAP matrices, where
    /// - (for now) each preprocessed trace matrix has a separate commitment
    /// - main trace matrices can have multiple commitments
    /// - for each after_challenge phase, all matrices in the phase share a commitment
    /// - quotient poly chunks are all committed together
    fn open<'a>(
        &self,
        // For each preprocessed trace commitment, the prover data and
        // the log height of the matrix, in order
        preprocessed: Vec<(PB::PcsDataRef<'a>, u8)>,
        // For each main trace commitment, the prover data and
        // the log height of each matrix, in order
        main: Vec<(PB::PcsDataRef<'a>, Vec<u8>)>,
        // after_challenge[i] has shared commitment prover data for all matrices in that phase, and log height of those matrices, in order
        after_challenge: Vec<(PB::PcsDataRef<'a>, Vec<u8>)>,
        // Quotient poly commitment prover data
        quotient_data: PB::PcsDataRef<'a>,
        // Quotient degree for each RAP committed in quotient_data, in order
        quotient_degrees: &[u8],
    ) -> PB::OpeningProof;
}
