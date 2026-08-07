//! Kernel fusion v2 — the CP-SAT-extracted rewrite pipeline defined by
//! `detailed-fusion-plan-v2.md`.
//!
//! This module is being built alongside the existing [`crate::passes::fusion`]
//! implementation. Both are selectable at [`crate::graph_exe::GraphCompiler`]
//! configuration time; v2 will replace the existing pass as the default only
//! after the completion criteria in §21 of the plan are met. The v2 pipeline
//! does not import from the existing implementation.
//!
//! Milestone status is tracked in `crates/compiler/fusion-v2-progress.md`.

pub mod access;
pub mod apply;
pub mod cost;
pub mod driver;
pub mod extract;
pub mod fusions;
pub mod model;
pub mod saturate;
pub mod validate;
pub mod version;

pub use self::{
    access::AccessRelation,
    apply::{apply_solution, ApplyError},
    cost::{ArtifactContext, ArtifactKey, GraphNodeCost},
    driver::{fuse_graph_v2, FuseV2Error, FusionOptionsV2, FusionReportV2},
    extract::{ExtractOptions, ExtractionData, ExtractionSolution, FallbackReason, SolverStatus},
    model::{AltGraphNode, GraphFuser, NodeId, NodeIdMap, UseInfo, ValIdMap, ValueClassId},
    saturate::{CandidateKey, SaturationState},
    validate::would_create_cycle,
    version::{take_graph, TakeGraphError},
};

#[cfg(test)]
mod tests;
