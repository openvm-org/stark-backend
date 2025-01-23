use std::fmt::Debug;

use p3_air::AirBuilder;
use p3_commit::PolynomialSpace;
use p3_matrix::dense::RowMajorMatrix;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{
    air_builders::symbolic::{symbolic_expression::SymbolicExpression, SymbolicConstraints},
    interaction::stark_log_up::{STARK_LU_NUM_CHALLENGES, STARK_LU_NUM_EXPOSED_VALUES},
    prover::PairTraceView,
};

/// Interaction debugging tools
pub mod debug;
pub mod gkr_log_up;
pub mod rap;
pub mod stark_log_up;
pub mod trace;
mod utils;

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum InteractionType {
    Send,
    Receive,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Interaction<Expr> {
    pub fields: Vec<Expr>,
    pub count: Expr,
    pub bus_index: usize,
    pub interaction_type: InteractionType,
}

pub type SymbolicInteraction<F> = Interaction<SymbolicExpression<F>>;

/// An [AirBuilder] with additional functionality to build special logUp arguments for
/// communication between AIRs across buses. These arguments use randomness to
/// add additional trace columns (in the extension field) and constraints to the AIR.
///
/// An interactive AIR is a AIR that can specify buses for sending and receiving data
/// to other AIRs. The original AIR is augmented by virtual columns determined by
/// the interactions to define a [RAP](crate::rap::Rap).
pub trait InteractionBuilder: AirBuilder {
    /// Stores a new send interaction in the builder.
    fn push_send<E: Into<Self::Expr>>(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Self::Expr>,
    ) {
        self.push_interaction(bus_index, fields, count, InteractionType::Send);
    }

    /// Stores a new receive interaction in the builder.
    fn push_receive<E: Into<Self::Expr>>(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Self::Expr>,
    ) {
        self.push_interaction(bus_index, fields, count, InteractionType::Receive);
    }

    /// Stores a new interaction in the builder.
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Self::Expr>,
        interaction_type: InteractionType,
    );

    /// Returns the current number of interactions.
    fn num_interactions(&self) -> usize;

    /// Returns all interactions stored.
    fn all_interactions(&self) -> &[Interaction<Self::Expr>];
}

pub struct RapPhaseProverData<Challenge> {
    /// Challenges from the challenger that determine RAP constraints and exposed values.
    pub challenges_per_phase: Vec<Vec<Challenge>>,

    /// After challenge trace per air computed as a function of `challenges`.
    pub after_challenge_trace_per_air: Vec<Option<RowMajorMatrix<Challenge>>>,

    /// Public values of the phase that are functions of `challenges`.
    pub exposed_values_per_air: Vec<Option<Vec<Challenge>>>,
}

pub struct RapPhaseVerifierData<Challenge> {
    /// Challenges from the challenger in this phase that determine RAP constraints and exposed values.
    pub challenges_per_phase: Vec<Vec<Challenge>>,
}

#[derive(Debug)]
pub struct RapPhaseShape {
    // The number of challenges serialized in the vkey (i.e., used in the RAP constraints).
    pub num_challenges: usize,
    pub num_exposed_values: usize,
}

/// Supported challenge phases in a RAP.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum RapPhaseSeqKind {
    GkrLogUp,
    /// Up to one phase with prover/verifier given by [[stark_log_up::StarkLogUpPhase]] and
    /// constraints given by [[stark_log_up::eval_stark_log_up_phase]].
    StarkLogUp,
}

impl RapPhaseSeqKind {
    pub fn shape(&self) -> Vec<RapPhaseShape> {
        match self {
            RapPhaseSeqKind::StarkLogUp => vec![RapPhaseShape {
                num_challenges: STARK_LU_NUM_CHALLENGES,
                num_exposed_values: STARK_LU_NUM_EXPOSED_VALUES,
            }],
            RapPhaseSeqKind::GkrLogUp => vec![RapPhaseShape {
                num_challenges: 2,
                num_exposed_values: 1,
            }],
        }
    }
}

pub trait HasInteractionChunkSize {
    fn interaction_chunk_size(&self) -> Option<usize>;
}

/// Defines a particular protocol for the "after challenge" phases in a RAP.
///
/// A [RapPhaseSeq] is defined by the proving and verifying methods implemented in this trait,
/// as well as via some "eval" method that is determined by `RapPhaseSeqKind`.
pub trait RapPhaseSeq<F, Challenge, Challenger> {
    type PartialProof: Clone + Serialize + DeserializeOwned;
    type ProvingKey: Clone + Serialize + DeserializeOwned + HasInteractionChunkSize;
    type Error: Debug;

    const KIND: RapPhaseSeqKind;

    /// The protocol parameters for the challenge phases may depend on the AIR constraints.
    fn generate_pk_per_air(
        &self,
        symbolic_constraints_per_air: Vec<SymbolicConstraints<F>>,
    ) -> Vec<Self::ProvingKey>;

    /// Partially prove the challenge phases,
    ///
    /// Samples challenges, generates after challenge traces and exposed values, and proves any
    /// extra-STARK part of the protocol.
    ///
    /// "Partial" refers to the fact that some STARK parts of the protocol---namely, the constraints
    /// on the after challenge traces returned in `RapPhaseProverData`---are handled external to
    /// this function.
    fn partially_prove(
        &self,
        challenger: &mut Challenger,
        params_per_air: &[&Self::ProvingKey],
        constraints_per_air: &[&SymbolicConstraints<F>],
        trace_view_per_air: &[PairTraceView<'_, F>],
    ) -> Option<(Self::PartialProof, RapPhaseProverData<Challenge>)>;

    /// Returns the additional openings for the challenge phases.
    ///
    /// Returns a sequence of extension field elements per phase per air.
    fn extra_opening_points<Domain>(
        &self,
        zeta: Challenge,
        domains_per_air: &[Domain],
    ) -> Vec<Vec<Vec<Challenge>>>
    where
        Domain: PolynomialSpace<Val = F>;

    /// Partially verifies the challenge phases.
    ///
    /// Assumes the shape of `exposed_values_per_air_per_phase` is verified externally.
    ///
    /// An implementation of this function must sample challenges for the challenge phases and then
    /// observe the exposed values and commitment.
    fn partially_verify(
        &self,
        challenger: &mut Challenger,
        partial_proof: Option<&Self::PartialProof>,
        constraints_per_air: &[&SymbolicConstraints<F>],
        exposed_values_per_air_per_phase: &[Vec<Vec<Challenge>>],
    ) -> Result<RapPhaseVerifierData<Challenge>, Self::Error>;
}
