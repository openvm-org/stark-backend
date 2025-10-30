// NOTE[jpw]: copied from stark-backend but renamed for V2 and without <SC>

// Keygen API for STARK backend
// Changes:
// - All AIRs can be optional
use std::sync::Arc;

use openvm_stark_backend::{
    air_builders::symbolic::SymbolicConstraintsDag,
    keygen::types::{LinearConstraint, TraceWidth},
};
use serde::{Deserialize, Serialize};

use crate::{Digest, F, prover::stacked_pcs::StackedPcsData};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SystemParams {
    pub l_skip: usize,
    pub n_stack: usize,
    pub log_blowup: usize,
    pub k_whir: usize,
    pub num_whir_queries: usize,
    pub log_final_poly_len: usize,
    pub logup_pow_bits: usize,
    pub whir_pow_bits: usize,
}

impl SystemParams {
    #[inline]
    pub fn num_whir_rounds(&self) -> usize {
        (self.n_stack + self.l_skip - self.log_final_poly_len) / self.k_whir
    }

    #[inline]
    pub fn num_whir_sumcheck_rounds(&self) -> usize {
        self.n_stack + self.l_skip - self.log_final_poly_len
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct StarkVerifyingParamsV2 {
    /// Trace sub-matrix widths
    pub width: TraceWidth,
    /// Number of public values for this STARK only
    pub num_public_values: usize,
}

/// Verifier data for preprocessed trace for a single AIR.
///
/// Currently assumes each AIR has it's own preprocessed commitment
#[derive(Clone, Serialize, Deserialize)]
pub struct VerifierSinglePreprocessedData<Digest> {
    /// Commitment to the preprocessed trace.
    pub commit: Digest,
    /// The hypercube dimension of the preprocessed data _before stacking_ (log_height -
    /// vk.l_skip).
    pub hypercube_dim: usize,
    /// The width of the data after stacking.
    pub stacking_width: usize,
}

/// Verifying key for a single STARK (corresponding to single AIR matrix)
#[derive(Clone, Serialize, Deserialize)]
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
}

/// Common verifying key for multiple AIRs.
///
/// This struct contains the necessary data for the verifier to verify proofs generated for
/// multiple AIRs using a single verifying key.
#[derive(Clone, Serialize, Deserialize)]
pub struct MultiStarkVerifyingKeyV2 {
    /// All parts of the verifying key needed by the verifier, except
    /// the `pre_hash` used to initialize the Fiat-Shamir transcript.
    pub inner: MultiStarkVerifyingKey0V2,
    /// The hash of all other parts of the verifying key. The Fiat-Shamir hasher will
    /// initialize by observing this hash.
    pub pre_hash: Digest,
}

/// Everything in [MultiStarkVerifyingKey] except the `pre_hash` used to initialize the Fiat-Shamir
/// transcript.
#[derive(Clone, Serialize, Deserialize)]
pub struct MultiStarkVerifyingKey0V2 {
    pub params: SystemParams,
    pub per_air: Vec<StarkVerifyingKeyV2<F, Digest>>,
    pub trace_height_constraints: Vec<LinearConstraint>,
    pub max_constraint_degree: usize,
}

/// Proving key for a single STARK (corresponding to single AIR matrix)
#[derive(Clone, Serialize, Deserialize)]
pub struct StarkProvingKeyV2 {
    /// Type name of the AIR, for display purposes only
    pub air_name: String,
    /// Verifying key
    pub vk: StarkVerifyingKeyV2<F, Digest>,
    /// Prover only data for preprocessed trace
    pub preprocessed_data: Option<Arc<StackedPcsData<F, Digest>>>,
}

/// Common proving key for multiple AIRs.
///
/// This struct contains the necessary data for the prover to generate proofs for multiple AIRs
/// using a single proving key.
#[derive(Clone, Serialize, Deserialize)]
pub struct MultiStarkProvingKeyV2 {
    pub per_air: Vec<StarkProvingKeyV2>,
    pub trace_height_constraints: Vec<LinearConstraint>,
    /// Maximum degree of constraints across all AIRs
    pub max_constraint_degree: usize,
    pub params: SystemParams,
    /// See [MultiStarkVerifyingKey]
    pub vk_pre_hash: Digest,
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
}

impl MultiStarkProvingKeyV2 {
    pub fn get_vk(&self) -> MultiStarkVerifyingKeyV2 {
        MultiStarkVerifyingKeyV2 {
            inner: self.get_vk0(),
            pre_hash: self.vk_pre_hash,
        }
    }

    fn get_vk0(&self) -> MultiStarkVerifyingKey0V2 {
        MultiStarkVerifyingKey0V2 {
            params: self.params,
            per_air: self.per_air.iter().map(|pk| pk.vk.clone()).collect(),
            trace_height_constraints: self.trace_height_constraints.clone(),
            max_constraint_degree: self.max_constraint_degree,
        }
    }
}
