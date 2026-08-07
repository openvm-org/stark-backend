//! Access relations attached to structured-kernel graph nodes.
//!
//! `detailed-fusion-plan-v2.md` §8. This module currently defines the
//! canonical [`AccessRelation`] layout so [`crate::passes::fusion_v2::model::GraphFuser`]
//! can carry it in its `access_relations` sidecar. The extractor
//! (`AccessCollector`) itself is deferred until M0-part 2; every seed
//! `access_relations` entry starts as `None`.

use std::collections::HashMap;

use crate::{ir, passes::fusion_v2::model::ValueClassId, quast::Quast};

/// Read/write summary of one structured-kernel graph node, bound to
/// [`ValueClassId`]s. The extractor (M0-part 2) walks the canonical HIR
/// with the [`crate::passes::fusion_utils`] visitor and produces an
/// unbound raw form; candidate-finalization then binds each read/write
/// position to the candidate's `inputs`/`outputs`.
#[derive(Clone, Debug)]
pub struct AccessRelation {
    pub reads: Vec<ReadRelation>,
    pub writes: Vec<WriteRelation>,
    /// Sparse: only the small subset of [`ir::VarId`]s used as index
    /// variables appears here.
    pub index_bounds: HashMap<ir::VarId, i64>,
    pub grid_index: ir::VarId,
    pub inner_indices: Vec<ir::VarId>,
}

#[derive(Clone, Debug)]
pub struct ReadRelation {
    pub read: Quast,
    pub val: ValueClassId,
    /// The HIR `Node::Index` site this read was extracted from.
    pub node: ir::NodeId,
}

#[derive(Clone, Debug)]
pub struct WriteRelation {
    pub write: Quast,
    pub inv: Quast,
    pub val: ValueClassId,
    pub node: ir::NodeId,
}
