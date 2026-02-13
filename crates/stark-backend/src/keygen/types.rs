// NOTE[jpw]: copied from stark-backend but renamed for , now generic in SC

// Keygen API for STARK backend
// Changes:
// - All AIRs can be optional
use std::sync::Arc;

use derivative::Derivative;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    air_builders::symbolic::{symbolic_variable::SymbolicVariable, SymbolicConstraintsDag},
    prover::stacked_pcs::StackedPcsData,
    StarkProtocolConfig, SystemParams,
};

/// Widths of different parts of trace matrix
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceWidth {
    pub preprocessed: Option<usize>,
    pub cached_mains: Vec<usize>,
    pub common_main: usize,
    /// Width counted by extension field elements, _not_ base field elements
    pub after_challenge: Vec<usize>,
}

impl TraceWidth {
    /// Returns the widths of all main traces, including the common main trace if it exists.
    pub fn main_widths(&self) -> Vec<usize> {
        let mut ret = self.cached_mains.clone();
        if self.common_main != 0 {
            ret.push(self.common_main);
        }
        ret
    }

    /// Returns the width of the main trace, i.e., the sum of all cached main widths and the common
    /// main width.
    pub fn main_width(&self) -> usize {
        self.cached_mains.iter().sum::<usize>() + self.common_main
    }

    /// Total width of the trace matrix, including the preprocessed width, main width, and
    /// after-challenge widths.
    pub fn total_width(&self, ext_degree: usize) -> usize {
        self.preprocessed.unwrap_or(0)
            + self.main_width()
            + self.after_challenge.iter().sum::<usize>() * ext_degree
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct LinearConstraint {
    pub coefficients: Vec<u32>,
    pub threshold: u32,
}

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
pub struct StarkVerifyingParams {
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
pub struct StarkVerifyingKey<F, Digest> {
    /// Preprocessed trace data, if any
    pub preprocessed_data: Option<VerifierSinglePreprocessedData<Digest>>,
    /// Parameters of the STARK
    pub params: StarkVerifyingParams,
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
pub struct MultiStarkVerifyingKey<SC: StarkProtocolConfig> {
    /// All parts of the verifying key needed by the verifier, except
    /// the `pre_hash` used to initialize the Fiat-Shamir transcript.
    pub inner: MultiStarkVerifyingKey0<SC>,
    /// The hash of all other parts of the verifying key. The Fiat-Shamir hasher will
    /// initialize by observing this hash.
    pub pre_hash: SC::Digest,
}

/// Everything in [MultiStarkVerifyingKey] except the `pre_hash` used to initialize the Fiat-Shamir
/// transcript.
#[derive(Derivative, Serialize, Deserialize)]
#[derivative(Clone(bound = ""), Debug(bound = ""))]
#[serde(bound = "")]
pub struct MultiStarkVerifyingKey0<SC: StarkProtocolConfig> {
    pub params: SystemParams,
    pub per_air: Vec<StarkVerifyingKey<SC::F, SC::Digest>>,
    pub trace_height_constraints: Vec<LinearConstraint>,
}

/// Proving key for a single STARK (corresponding to single AIR matrix)
#[derive(Derivative, Serialize, Deserialize)]
#[derivative(Clone(bound = ""))]
#[serde(bound = "")]
pub struct StarkProvingKey<SC: StarkProtocolConfig> {
    /// Type name of the AIR, for display purposes only
    pub air_name: String,
    /// Verifying key
    pub vk: StarkVerifyingKey<SC::F, SC::Digest>,
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
pub struct MultiStarkProvingKey<SC: StarkProtocolConfig> {
    pub per_air: Vec<StarkProvingKey<SC>>,
    pub trace_height_constraints: Vec<LinearConstraint>,
    /// Maximum degree of constraints across all AIRs
    pub max_constraint_degree: usize,
    pub params: SystemParams,
    /// See [MultiStarkVerifyingKey]
    pub vk_pre_hash: SC::Digest,
}

impl<Val, Com> StarkVerifyingKey<Val, Com> {
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

impl<SC: StarkProtocolConfig> MultiStarkProvingKey<SC> {
    pub fn get_vk(&self) -> MultiStarkVerifyingKey<SC> {
        MultiStarkVerifyingKey {
            inner: self.get_vk0(),
            pre_hash: self.vk_pre_hash,
        }
    }

    fn get_vk0(&self) -> MultiStarkVerifyingKey0<SC> {
        MultiStarkVerifyingKey0 {
            params: self.params.clone(),
            per_air: self.per_air.iter().map(|pk| pk.vk.clone()).collect(),
            trace_height_constraints: self.trace_height_constraints.clone(),
        }
    }
}

impl<SC: StarkProtocolConfig> MultiStarkVerifyingKey<SC> {
    /// Global maximum constraint degree across all AIRs and Interactions.
    pub fn max_constraint_degree(&self) -> usize {
        self.inner.max_constraint_degree()
    }
}

impl<SC: StarkProtocolConfig> MultiStarkVerifyingKey0<SC> {
    pub fn max_constraint_degree(&self) -> usize {
        self.params.max_constraint_degree
    }
}
