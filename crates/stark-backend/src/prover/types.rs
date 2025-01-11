use std::sync::Arc;

use derivative::Derivative;
use itertools::Itertools;
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use serde::{Deserialize, Serialize};

use super::hal::ProverBackend;
use crate::{
    config::{Com, StarkGenericConfig, Val},
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey, StarkVerifyingKey},
    proof::{AirProofData, Commitments},
    rap::AnyRap,
};

pub struct MultiStarkProvingKeyView<'a, PB: ProverBackend> {
    pub per_air: Vec<StarkProvingKeyView<'a, PB>>,
}

pub struct StarkProvingKeyView<'a, PB: ProverBackend> {
    /// Type name of the AIR, for display purposes only
    pub air_name: &'a str,
    pub vk: &'a StarkVerifyingKey<PB::Val, PB::Commitment>,
    /// Prover only data for preprocessed trace
    pub preprocessed_data: Option<ProverOnlySinglePreprocessedView<'a, PB>>,
    pub rap_phase_seq_pk: PB::RapPhaseSeqProvingKeyView<'a>,
}

pub struct RapPhaseSeqProvingKeyView<'a, PB: ProverBackend> {
    pub rap_phase_seq_pk: &'a (),
}

pub struct ProverOnlySinglePreprocessedView<'a, PB: ProverBackend> {
    pub trace: PB::MatrixView<'a>,
    pub data: PB::PcsDataRef<'a>,
}

#[derive(Clone, derive_new::new)]
pub struct ProvingContext<'a, PB: ProverBackend> {
    /// (AIR id, AIR input)
    pub per_air: Vec<(usize, AirProvingContext<'a, PB>)>,
}

impl<'a, PB: ProverBackend> ProvingContext<'a, PB> {
    pub fn into_air_proof_input_vec(self) -> Vec<AirProvingContext<'a, PB>> {
        self.per_air.into_iter().map(|(_, x)| x).collect()
    }
}

impl<'a, PB: ProverBackend> IntoIterator for ProvingContext<'a, PB> {
    type Item = (usize, AirProvingContext<'a, PB>);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.per_air.into_iter()
    }
}

/// Necessary context for proving a single AIR.
#[derive(Clone)]
pub struct AirProvingContext<'a, PB: ProverBackend> {
    /// Prover data for cached main traces
    pub cached_mains_com_data: Vec<CommittedDataRef<'a, PB>>,
    pub raw: RawAirProvingContext<'a, PB>,
}

/// Reference to the prover data corresponding to a single shared commitment,
/// as well as the owned commitment.
#[derive(Clone)]
pub struct CommittedDataRef<'com, PB: ProverBackend> {
    /// Commitment to the trace matrices.
    pub commit: PB::Commitment,
    /// Preimage of commitment. Memory layout may vary.
    pub data: PB::PcsDataRef<'com>,
}

/// Raw context for proving a single AIR.
/// Consists of the trace and public values (witness and instance data).
#[derive(Clone, Debug)]
pub struct RawAirProvingContext<'a, PB: ProverBackend> {
    /// Cached main trace matrices
    pub cached_mains: Vec<PB::MatrixView<'a>>,
    /// Common main trace matrix
    pub common_main: Option<PB::MatrixView<'a>>,
    /// Public values
    pub public_values: PB::ValBuffer<'a>,
}

impl<SC: StarkGenericConfig> MultiStarkVerifyingKey<SC> {
    pub fn validate<PB: ProverBackend>(&self, ctx: &ProvingContext<PB>) -> bool {
        if !ctx
            .per_air
            .iter()
            .all(|(air_id, _)| *air_id < self.per_air.len())
        {
            return false;
        }
        if !ctx.per_air.iter().tuple_windows().all(|(a, b)| a.0 < b.0) {
            return false;
        }
        true
    }
}

impl<SC: StarkGenericConfig> MultiStarkProvingKey<SC> {
    pub fn validate<PB: ProverBackend>(&self, ctx: &ProvingContext<PB>) -> bool {
        self.get_vk().validate(ctx)
    }
}

/// The full RAP trace consists of horizontal concatenation of multiple matrices of the same height:
/// - preprocessed trace matrix
/// - the main trace matrix is horizontally partitioned into multiple matrices,
///   where each matrix can belong to a separate matrix commitment.
/// - after each round of challenges, a trace matrix for trace allowed to use those challenges
///
/// Each of these matrices is allowed to be in a separate commitment.
///
/// Only the main trace matrix is allowed to be partitioned, so that different parts may belong to
/// different commitments. We do not see any use cases where the `preprocessed` or `after_challenge`
/// matrices need to be partitioned.
#[derive(Clone)]
pub struct SingleRapView<T, ChallengeBuffer> {
    /// Log_2 of the domain size (i.e., height of matrices)
    pub log_domain_size: u8,
    // Maybe public values should be included in this struct
    /// Preprocessed trace data, if any
    pub preprocessed: Option<T>,
    /// Main trace data, horizontally partitioned into multiple matrices
    pub partitioned_main: Vec<T>,
    /// `after_challenge[i] = (matrix, exposed_values)`
    /// where `matrix` is the trace matrix which uses challenges drawn
    /// after observing commitments to `preprocessed`, `partitioned_main`, and `after_challenge[..i]`,
    /// and `exposed_values` are certain values in this phase that are exposed to the verifier.
    pub after_challenge: Vec<(T, ChallengeBuffer)>,
}

/// The full proof for multiple RAPs where trace matrices are committed into
/// multiple commitments, where each commitment is multi-matrix.
///
/// Includes the quotient commitments and FRI opening proofs for the constraints as well.
#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound = "")]
#[derivative(Clone(bound = ""))]
pub struct HalProof<PB: ProverBackend> {
    /// The PCS commitments
    pub commitments: Commitments<PB::Commitment>,
    /// Opening proofs separated by partition, but this may change
    pub opening: PB::OpeningProof,
    /// Proof data for each AIR
    pub per_air: Vec<AirProofData<PB::Val, PB::Challenge>>,
    /// Partial proof for rap phase if it exists
    pub rap_phase_seq_proof: (), // todo
}

// ============= Below are common types independent of hardware ============
// These are legacy types. They should be removed but affect many testing codepaths.

// Legacy type
/// Necessary input for proving a single AIR.
#[derive(Derivative)]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct AirProofInput<SC: StarkGenericConfig> {
    pub air: Arc<dyn AnyRap<SC>>,
    /// Prover data for cached main traces
    pub cached_mains_pdata: Vec<super::cpu::trace::ProverTraceData<SC>>,
    pub raw: AirProofRawInput<Val<SC>>,
}

/// Raw input for proving a single AIR.
#[derive(Clone, Debug)]
pub struct AirProofRawInput<F: Field> {
    /// Cached main trace matrices
    pub cached_mains: Vec<Arc<RowMajorMatrix<F>>>,
    /// Common main trace matrix
    pub common_main: Option<RowMajorMatrix<F>>,
    /// Public values
    pub public_values: Vec<F>,
}

impl<F: Field> AirProofRawInput<F> {
    pub fn height(&self) -> usize {
        let mut height = None;
        for m in self.cached_mains.iter() {
            if let Some(h) = height {
                assert_eq!(h, m.height());
            } else {
                height = Some(m.height());
            }
        }
        let common_h = self.common_main.as_ref().map(|trace| trace.height());
        if let Some(h) = height {
            if let Some(common_h) = common_h {
                assert_eq!(h, common_h);
            }
            h
        } else {
            common_h.unwrap_or(0)
        }
    }
}
