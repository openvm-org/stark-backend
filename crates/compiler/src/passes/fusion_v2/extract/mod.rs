//! Extraction: pick a subset of alternative-graph nodes that produces
//! every demanded output at minimum lexicographic cost.
//!
//! `detailed-fusion-plan-v2.md` §13. Two extractors live here:
//!
//! - [`brute::extract`] — always-available exhaustive enumerator over feasible node subsets, used
//!   as a correctness oracle and as the M2 exit-gate reference (`brute-force agrees with CP-SAT`).
//! - [`cpsat::extract`] — the production extractor, gated on `planner-ortools` (§4). Same
//!   objective, same constraints, but with a wall-time-bounded CP-SAT solve.
//!
//! Both extractors accept the same [`ExtractionData`] sidecar (§5.4)
//! and [`ExtractOptions`], and return the same [`ExtractionSolution`].
//! The M1 [`ExtractionSolution::original`] remains available as the
//! solver-independent fallback.

pub mod brute;
#[cfg(feature = "planner-ortools")]
pub mod cpsat;

use crate::passes::fusion_v2::{
    cost::{ArtifactKey, GraphNodeCost},
    model::{GraphFuser, NodeId, NodeIdMap},
};

/// The set of alternative-graph nodes an extractor has selected for
/// commit, together with the reason a solver-backed extractor fell back to
/// this solution (if any).
#[derive(Debug, Clone)]
pub struct ExtractionSolution {
    /// Selected alternative-graph nodes.
    pub nodes: Vec<NodeId>,
    /// Reason the solver fell back to the original extraction, or `None`
    /// if this is a solver-produced solution.
    pub fallback: Option<FallbackReason>,
    /// Solver status when this solution came from a solver-backed
    /// extractor.
    pub status: Option<SolverStatus>,
}

/// Terminal status of a solver run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverStatus {
    /// The solver proved optimality within the wall-time cap.
    Optimal,
    /// The solver returned a feasible solution but did not prove
    /// optimality.
    Feasible,
    /// The solver returned no solution before the deadline (or
    /// otherwise gave up).
    Unknown,
    /// The model is infeasible — a v2 bug when no user budget excludes
    /// the original solution.
    Infeasible,
}

/// Enumerates every path that causes v2 to emit the original extraction
/// instead of the solver's answer (§15).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FallbackReason {
    /// Built without `planner-ortools`, or the CP-SAT extractor has not
    /// been enabled for this run.
    SolverUnavailable,
    /// CP-SAT returned `Unknown` within the wall-time limit.
    SolverStatusUnknown,
    /// CP-SAT proved infeasible.
    SolverStatusInfeasible,
    /// CP-SAT wall-time limit expired with no `Feasible` or better status.
    SolverTimeout,
    /// [`crate::passes::fusion_v2::apply::apply_solution`] rejected the
    /// solver solution during pre-commit validation.
    ReconstructBindingMismatch { node: NodeId, kind: String },
    /// The solver's objective did not beat the original within the
    /// configured tolerance.
    NoImprovementOverOriginal,
    /// Catch-all for any other v2-side failure before commit.
    InternalError { message: String },
}

impl ExtractionSolution {
    /// The always-available baseline: every seed node selected in seed
    /// order. Serves as the solver fallback and, in the absence of any
    /// candidates, is the only feasible ILP solution too.
    pub fn original(gf: &GraphFuser) -> Self {
        Self {
            nodes: (0..gf.seed_node_count).map(NodeId).collect(),
            fallback: Some(FallbackReason::SolverUnavailable),
            status: None,
        }
    }
}

/// Per-node data an extractor consumes alongside the alternative graph.
///
/// Kept outside [`GraphFuser`] so estimator revisions and artifact
/// bookkeeping do not mutate the graph itself (§5.4).
#[derive(Debug, Clone)]
pub struct ExtractionData {
    /// One [`GraphNodeCost`] per node, in lockstep with `gf.nodes`.
    pub costs: NodeIdMap<GraphNodeCost>,
    /// Compiled-artifact identity per node. `None` for nodes with no
    /// nvcc-compiled artifact (constants, memcpy/memset, blackbox
    /// kernels).
    pub artifact_keys: NodeIdMap<Option<ArtifactKey>>,
}

impl ExtractionData {
    /// Builds trivial extraction data: every node priced at
    /// `runtime_units = 1` and no artifact keys. Useful for tests that
    /// exercise the constraint model without depending on the estimator.
    pub fn uniform(gf: &GraphFuser) -> Self {
        Self {
            costs: vec![GraphNodeCost::new(1); gf.nodes.len()],
            artifact_keys: vec![None; gf.nodes.len()],
        }
    }
}

/// Configuration knobs shared by both extractors.
#[derive(Debug, Clone)]
pub struct ExtractOptions {
    /// Wall-time cap for the CP-SAT solve, in seconds. Ignored by the
    /// brute-force extractor.
    pub solver_time_limit_secs: f64,
    /// Optional hard cap on the number of selected artifacts
    /// (`sum_m z_m <= max_modules`). If configured below the number of
    /// artifacts required by the original graph the extractor returns
    /// [`FallbackReason::InternalError`] rather than an infeasible model.
    pub max_modules: Option<usize>,
    /// Optional hard cap on the number of newly-introduced artifacts
    /// (`sum_{m not in original_artifacts} z_m <= max_new_modules`).
    pub max_new_modules: Option<usize>,
    /// Cycle quantum used to convert estimated cycles into `i64` runtime
    /// units. Overflow bounds are checked in `i128` before constructing
    /// the model.
    pub cycle_quantum: i64,
    /// PPM slack tolerance for stage-1 (runtime) equality when moving to
    /// stage 2. Zero means strict lexicographic.
    pub runtime_tolerance_ppm: u32,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            solver_time_limit_secs: 5.0,
            max_modules: None,
            max_new_modules: None,
            cycle_quantum: 1,
            runtime_tolerance_ppm: 0,
        }
    }
}
