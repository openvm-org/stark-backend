//! Kernel fusion — the CP-SAT-extracted rewrite pipeline defined by
//! `old_plans/detailed-fusion-plan-v2.md`.

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
    driver::{fuse_graph, FuseError, FusionOptions, FusionReport},
    extract::{ExtractOptions, ExtractionData, ExtractionSolution, FallbackReason, SolverStatus},
    model::{AltGraphNode, GraphFuser, NodeId, NodeIdMap, UseInfo, ValIdMap, ValueClassId},
    saturate::{CandidateKey, SaturationState},
    validate::would_create_cycle,
    version::{take_graph, TakeGraphError},
};

#[cfg(test)]
mod tests;
