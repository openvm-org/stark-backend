// NOTE[jpw]: copied from stark-backend but renamed for V2, now generic in SC

// Keygen API for STARK backend
// Changes:
// - All AIRs can be optional
use std::sync::Arc;

use derivative::Derivative;
use openvm_stark_backend::{
    air_builders::symbolic::{symbolic_variable::SymbolicVariable, SymbolicConstraintsDag},
    keygen::types::{LinearConstraint, TraceWidth},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{prover::stacked_pcs::StackedPcsData, StarkProtocolConfig, SystemParams};

#[derive(Error, Debug)]
pub enum KeygenError {
    #[error("Max constraint degree exceeded for AIR {name}: {degree} > {max_degree}")]
    MaxConstraintDegreeExceeded {
        name: String,
        degree: usize,
        max_degree: usize,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct StarkVerifyingParamsV2 {
    /// Trace sub-matrix widths
    pub width: TraceWidth,
    /// Number of public values for this STARK only
    pub num_public_values: usize,
    /// A flag indicating whether at least one rotated variable is used in any
    /// of the constraints and/or interactions across all trace parts (common,
    /// preprocessed if there is one, all cached).
    pub need_rot: bool,
}

/// Verifier data for preprocessed trace for a single AIR.
///
/// Currently assumes each AIR has it's own preprocessed commitment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifierSinglePreprocessedData<Digest> {
    /// Commitment to the preprocessed trace.
    pub commit: Digest,
    /// The hypercube dimension of the preprocessed data _before stacking_ (log_height -
    /// vk.l_skip).
    pub hypercube_dim: isize,
    /// The width of the data after stacking.
    pub stacking_width: usize,
}

/// Verifying key for a single STARK (corresponding to single AIR matrix)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct StarkVerifyingKeyV2<F, Digest> {
    /// Preprocessed trace data, if any
    pub preprocessed_data: Option<VerifierSinglePreprocessedData<Digest>>,
    /// Parameters of the STARK
    pub params: StarkVerifyingParamsV2,
    /// Symbolic constraints of the AIR in all challenge phases. This is
    /// a serialization of the constraints in the AIR.
    pub symbolic_constraints: SymbolicConstraintsDag<F>,
    /// The maximum degree of any polynomial (constraint or interaction) for this AIR.
    pub max_constraint_degree: u8,
    /// True means this AIR must have non-empty trace.
    pub is_required: bool,
    /// Symbolic variables referenced unreferenced by the AIR.
    pub unused_variables: Vec<SymbolicVariable<F>>,
}

/// Common verifying key for multiple AIRs.
///
/// This struct contains the necessary data for the verifier to verify proofs generated for
/// multiple AIRs using a single verifying key.
#[derive(Derivative, Serialize, Deserialize)]
#[derivative(Clone(bound = ""), Debug(bound = ""))]
#[serde(bound = "")]
pub struct MultiStarkVerifyingKeyV2<SC: StarkProtocolConfig> {
    /// All parts of the verifying key needed by the verifier, except
    /// the `pre_hash` used to initialize the Fiat-Shamir transcript.
    pub inner: MultiStarkVerifyingKey0V2<SC>,
    /// The hash of all other parts of the verifying key. The Fiat-Shamir hasher will
    /// initialize by observing this hash.
    pub pre_hash: SC::Digest,
}

/// Everything in [MultiStarkVerifyingKey] except the `pre_hash` used to initialize the Fiat-Shamir
/// transcript.
#[derive(Derivative, Serialize, Deserialize)]
#[derivative(Clone(bound = ""), Debug(bound = ""))]
#[serde(bound = "")]
pub struct MultiStarkVerifyingKey0V2<SC: StarkProtocolConfig> {
    pub params: SystemParams,
    pub per_air: Vec<StarkVerifyingKeyV2<SC::F, SC::Digest>>,
    pub trace_height_constraints: Vec<LinearConstraint>,
}

/// Proving key for a single STARK (corresponding to single AIR matrix)
#[derive(Derivative, Serialize, Deserialize)]
#[derivative(Clone(bound = ""))]
#[serde(bound = "")]
pub struct StarkProvingKeyV2<SC: StarkProtocolConfig> {
    /// Type name of the AIR, for display purposes only
    pub air_name: String,
    /// Verifying key
    pub vk: StarkVerifyingKeyV2<SC::F, SC::Digest>,
    /// Prover only data for preprocessed trace
    pub preprocessed_data: Option<Arc<StackedPcsData<SC::F, SC::Digest>>>,
}

/// Common proving key for multiple AIRs.
///
/// This struct contains the necessary data for the prover to generate proofs for multiple AIRs
/// using a single proving key.
#[derive(Derivative, Serialize, Deserialize)]
#[derivative(Clone(bound = ""))]
#[serde(bound = "")]
pub struct MultiStarkProvingKeyV2<SC: StarkProtocolConfig> {
    pub per_air: Vec<StarkProvingKeyV2<SC>>,
    pub trace_height_constraints: Vec<LinearConstraint>,
    /// Maximum degree of constraints across all AIRs
    pub max_constraint_degree: usize,
    pub params: SystemParams,
    /// See [MultiStarkVerifyingKey]
    pub vk_pre_hash: SC::Digest,
}

impl<Val, Com> StarkVerifyingKeyV2<Val, Com> {
    pub fn num_cached_mains(&self) -> usize {
        self.params.width.cached_mains.len()
    }

    pub fn num_parts(&self) -> usize {
        1 + self.num_cached_mains() + (self.preprocessed_data.is_some() as usize)
    }

    pub fn has_interaction(&self) -> bool {
        !self.symbolic_constraints.interactions.is_empty()
    }

    pub fn num_interactions(&self) -> usize {
        self.symbolic_constraints.interactions.len()
    }

    /// Converts from a main part index (as used by the constraint DAG) to the
    /// commitment part indexing scheme that includes preprocessed trace.
    pub fn dag_main_part_index_to_commit_index(&self, index: usize) -> usize {
        // In the dag, common main is the final part index.
        if index == self.num_cached_mains() {
            0
        } else {
            index + 1 + self.preprocessed_data.is_some() as usize
        }
    }
}

impl<SC: StarkProtocolConfig> MultiStarkProvingKeyV2<SC> {
    pub fn get_vk(&self) -> MultiStarkVerifyingKeyV2<SC> {
        MultiStarkVerifyingKeyV2 {
            inner: self.get_vk0(),
            pre_hash: self.vk_pre_hash,
        }
    }

    fn get_vk0(&self) -> MultiStarkVerifyingKey0V2<SC> {
        MultiStarkVerifyingKey0V2 {
            params: self.params.clone(),
            per_air: self.per_air.iter().map(|pk| pk.vk.clone()).collect(),
            trace_height_constraints: self.trace_height_constraints.clone(),
        }
    }
}

impl<SC: StarkProtocolConfig> MultiStarkVerifyingKeyV2<SC> {
    /// Global maximum constraint degree across all AIRs and Interactions.
    pub fn max_constraint_degree(&self) -> usize {
        self.inner.max_constraint_degree()
    }
}

impl<SC: StarkProtocolConfig> MultiStarkVerifyingKey0V2<SC> {
    pub fn max_constraint_degree(&self) -> usize {
        self.params.max_constraint_degree
    }
}
