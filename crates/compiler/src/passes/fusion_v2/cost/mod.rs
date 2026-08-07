//! Cost estimation for the fusion-v2 extractor.
//!
//! `detailed-fusion-plan-v2.md` §5.5 (artifact keys), §12 (KIR
//! estimator), and §12.10 (`KernelCostManager`).
//!
//! The extractor consumes an
//! [`crate::passes::fusion_v2::extract::ExtractionData`] sidecar: one
//! [`GraphNodeCost`] plus an optional [`ArtifactKey`] per alternative
//! graph node. This module supplies both.
//!
//! Public surface:
//!
//! - [`ArtifactKey`] / [`ArtifactContext`] identify compiled artifacts across a run (§5.5).
//! - [`GraphNodeCost`] carries the quantized runtime the ILP objective consumes (§13.5).
//! - [`DeviceModel`] / [`EstimatorConfig`] / [`EstimateContext`] parameterize the estimator
//!   (§12.1).
//! - [`estimator::estimate_kernel`] and [`estimator::estimate_non_kernel`] are the two entry
//!   points.
//! - [`cache::KernelCostManager`] memoizes estimates keyed by module hash + param bindings
//!   (§12.10).

pub mod cache;
pub mod estimator;
pub mod interpreter;
pub mod liveness;
pub mod transactions;

pub use self::{
    cache::{CostError, CostManagerStats, KernelCostManager, KirCostKey},
    estimator::{
        estimate_kernel, estimate_non_kernel, DeviceModel, EstimateContext, EstimatorConfig,
        KernelCostBreakdown,
    },
    interpreter::{CriticalPath, OpLatencyTable},
    liveness::RegisterEstimate,
    transactions::AccessEst,
};

/// A candidate module's byte-identical identity across a compile.
///
/// For the initial implementation the pass keys only on the normalized
/// residual `module_hash` because architecture and compiler flags are
/// constant within one `GraphCompiler` invocation. The wrapper type keeps
/// those fields around so they can be folded in without changing the
/// extraction model (§5.5).
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ArtifactKey {
    pub module_hash: [u8; 32],
    pub target_arch: String,
    pub compiler_flags_hash: [u8; 32],
}

/// Immutable context describing the compile the extractor is planning
/// for. Passed alongside any cost query.
#[derive(Clone, Debug)]
pub struct ArtifactContext {
    pub target_arch: String,
    pub compiler_flags_hash: [u8; 32],
}

impl ArtifactContext {
    /// Wraps a `module_hash` in an [`ArtifactKey`] under this context.
    pub fn key_for(&self, module_hash: [u8; 32]) -> ArtifactKey {
        ArtifactKey {
            module_hash,
            target_arch: self.target_arch.clone(),
            compiler_flags_hash: self.compiler_flags_hash,
        }
    }
}

/// Runtime cost estimate for one alternative-graph node, in the
/// quantized integer units the extractor's ILP objective uses.
///
/// `detailed-fusion-plan-v2.md` §13.5: `runtime_units(a) = max(1,
/// round(total_cycles(a) / cycle_quantum))`. The quantum lives on the
/// [`ExtractOptions`](crate::passes::fusion_v2::extract::ExtractOptions)
/// so bounds can be checked in `i128` before constructing the CP-SAT
/// model.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GraphNodeCost {
    pub runtime_units: i64,
}

impl GraphNodeCost {
    pub const fn new(runtime_units: i64) -> Self {
        Self { runtime_units }
    }

    /// Quantizes a raw cycle count into the extractor's integer units.
    /// The floor is one unit so a "zero-cost" node still forces the
    /// solver to see it (matching the plan's `max(1, ..)` rule in §13.5).
    pub fn from_cycles(total_cycles: f64, cycle_quantum: i64) -> Self {
        let q = cycle_quantum.max(1) as f64;
        let raw = (total_cycles / q).round();
        let clamped = raw.max(1.0).min(i64::MAX as f64) as i64;
        Self {
            runtime_units: clamped,
        }
    }
}
