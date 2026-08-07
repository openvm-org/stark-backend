//! Core data model of the fusion-v2 alternative graph.
//!
//! `detailed-fusion-plan-v2.md` §5. The alternative graph is a versioned,
//! DAG-shaped bipartite structure over dense arenas: [`ValueClassId`] indexes
//! logical value classes and [`NodeId`] indexes alternative-graph nodes.
//! Seed nodes and later-synthesized fusion candidates share one `NodeId`
//! namespace; every value class carries a first-instance pointer through
//! [`GraphFuser::re_exported`] that recovers its physical [`BufId`].
//!
//! Sidecar state (origins, seen-candidate keys, candidate costs and artifact
//! keys) is kept outside [`GraphFuser`] so estimator revisions and search
//! bookkeeping do not mutate the alternative graph itself; see plan §5.4.

use crate::{
    graph_ir::{BufId, BufInfo, GraphNode},
    passes::fusion_v2::access::AccessRelation,
};

/// Dense index of a logical value class in [`GraphFuser::bufs`]. Every
/// version of a physical [`BufId`] is a distinct [`ValueClassId`], but every
/// [`ValueClassId`] projects back to a single physical [`BufId`] through
/// [`GraphFuser::re_exported`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ValueClassId(pub usize);

/// Dense index of an alternative-graph node in [`GraphFuser::nodes`]. Seed
/// nodes and fusion-candidate nodes occupy this one namespace.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct NodeId(pub usize);

/// Vector indexed by [`ValueClassId`].
pub type ValIdMap<T> = Vec<T>;

/// Vector indexed by [`NodeId`].
pub type NodeIdMap<T> = Vec<T>;

/// One entry of the alternative graph: a graph-level operation plus its
/// positional logical port bindings.
///
/// `inputs` and `outputs` are aligned with
/// [`GraphNode::get_operands`]/[`GraphNode::get_results`] respectively, per
/// plan §5.1.
pub struct AltGraphNode {
    pub inputs: Vec<ValueClassId>,
    pub outputs: Vec<ValueClassId>,
    pub node: GraphNode,
}

/// One occurrence of a value at a positional port of an alternative-graph
/// node.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct UseInfo {
    pub node: NodeId,
    /// Position in the corresponding `AltGraphNode.inputs`/`outputs` vector.
    pub pos: usize,
}

/// Versioned alternative graph. Owned by the v2 fusion pass for the
/// duration of `fuse_graph_v2`; on error paths the pass restores the
/// original [`crate::graph_ir::GraphBuilder`] via the take-graph guard
/// (§14.3).
pub struct GraphFuser {
    /// Shape/device metadata per logical value. Grows in lockstep with the
    /// other value-indexed arenas.
    pub bufs: ValIdMap<BufInfo>,
    /// Alternative-graph nodes, in insertion order. Seed nodes come first
    /// (see [`GraphFuser::seed_node_count`]); candidates follow in the
    /// order they were inserted.
    pub nodes: NodeIdMap<AltGraphNode>,
    /// Producers of each value class. Empty for graph inputs.
    pub producers: ValIdMap<Vec<UseInfo>>,
    /// Consumers of each value class.
    pub consumers: ValIdMap<Vec<UseInfo>>,
    /// Extracted access relation per node (structured kernels only).
    /// `None` for opaque, memcpy, memset, const, and — until M0-part 2
    /// lands — every seed node.
    pub access_relations: NodeIdMap<Option<AccessRelation>>,
    /// First-instance [`ValueClassId`] for each logical value; the physical
    /// [`BufId`] is `BufId(re_exported[value.0].0)`.
    pub re_exported: ValIdMap<ValueClassId>,
    /// Initial logical value classes of the registered graph inputs, in
    /// interface order.
    pub inputs: Vec<ValueClassId>,
    /// Final logical value classes of the registered graph outputs, in
    /// interface order.
    pub outputs: Vec<ValueClassId>,
    /// Number of seed nodes at the head of `nodes`. Selecting
    /// `NodeId(0)..NodeId(seed_node_count)` reproduces the input graph as
    /// the original-fallback solution (§6.4).
    pub seed_node_count: usize,
}

impl GraphFuser {
    /// Physical [`BufId`] backing a logical value class.
    pub fn physical(&self, value: ValueClassId) -> BufId {
        BufId(self.re_exported[value.0].0)
    }

    /// Number of logical value classes currently allocated.
    pub fn num_values(&self) -> usize {
        self.bufs.len()
    }

    /// Number of alternative-graph nodes currently allocated.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Appends a fusion candidate that reuses existing value classes.
    ///
    /// The MVP fusion passes (§14.2) never introduce new value classes:
    /// a candidate produces semantic values that already exist, so its
    /// `outputs` name previously-allocated [`ValueClassId`]s. This helper
    /// registers the candidate in [`GraphFuser::nodes`], keeps
    /// `access_relations` in lockstep as `None` (extractor is deferred to
    /// M0-part 2), and adds one [`UseInfo`] entry per positional port to
    /// the producer/consumer sidecars.
    ///
    /// The caller is responsible for validating any per-pass legality
    /// requirements (disjoint origins, matching shapes, etc.) *before*
    /// calling this method.
    pub fn insert_candidate(&mut self, alt: AltGraphNode) -> NodeId {
        let node_id = NodeId(self.nodes.len());
        for (pos, &v) in alt.inputs.iter().enumerate() {
            self.consumers[v.0].push(UseInfo { node: node_id, pos });
        }
        for (pos, &v) in alt.outputs.iter().enumerate() {
            self.producers[v.0].push(UseInfo { node: node_id, pos });
        }
        self.nodes.push(alt);
        self.access_relations.push(None);
        node_id
    }
}
