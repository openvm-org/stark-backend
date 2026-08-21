//! Insertion-time validation for fusion candidates.
//!
//! `detailed-fusion-plan-v2.md` §9.1 — the alternative graph is a
//! directed bipartite DAG:
//!
//! ```text
//! ValueClassId -> NodeId       when the value is an input of the node
//! NodeId -> ValueClassId       when the value is an output of the node
//! ```
//!
//! Inserting candidate `a` with input set `I` and output set `O` creates
//! a cycle if and only if the current graph has a directed path from
//! some `o ∈ O` to some `i ∈ I`. Any new cycle must contain `a`;
//! removing `a` from that cycle leaves such a path. Conversely, such a
//! path is closed by `i → a → o`.

use std::collections::{HashMap, HashSet};

use crate::passes::fusion::model::{GraphFuser, NodeId, ValueClassId};

/// Whether inserting an alternative-graph node with the given `inputs`
/// and `outputs` would introduce a cycle into `gf`.
///
/// Traverses forward from `outputs` through the existing bipartite
/// graph. A cycle is reported if the traversal reaches any of `inputs`.
/// The traversal cost is proportional to the portion of the graph
/// reachable from `outputs`.
pub fn would_create_cycle(
    gf: &GraphFuser,
    inputs: &[ValueClassId],
    outputs: &[ValueClassId],
) -> bool {
    let target: HashSet<ValueClassId> = inputs.iter().copied().collect();
    let mut seen_values: HashSet<ValueClassId> = HashSet::new();
    let mut seen_nodes: HashSet<NodeId> = HashSet::new();
    let mut stack: Vec<ValueClassId> = outputs.to_vec();
    while let Some(v) = stack.pop() {
        if target.contains(&v) {
            return true;
        }
        if !seen_values.insert(v) {
            continue;
        }
        for use_info in &gf.consumers[v.0] {
            if !seen_nodes.insert(use_info.node) {
                continue;
            }
            for out_val in &gf.nodes[use_info.node.0].outputs {
                stack.push(*out_val);
            }
        }
    }
    false
}

/// Index of every multi-version storage class, built once per
/// [`crate::passes::fusion::fuse_graph`] invocation right after
/// `take_graph` — value classes never grow afterwards (candidates reuse
/// existing versions), so the index stays valid across saturation
/// rounds.
///
/// A storage class is the canonical (alias-class root) buffer of a value
/// class (see [`GraphFuser::canonical`]); its versions are the
/// graph-input initial version (if any) plus every written version, in
/// write order ([`ValueClassId`] order, which is seed allocation order
/// even across alias siblings). Classes with a single version cannot
/// participate in a storage hazard and are omitted, so the guard costs
/// nothing on pure-SSA graphs.
pub struct StorageHazardIndex {
    versions_by_class: HashMap<usize, Vec<ValueClassId>>,
}

impl StorageHazardIndex {
    pub fn new(gf: &GraphFuser) -> Self {
        let mut versions_by_class: HashMap<usize, Vec<ValueClassId>> = HashMap::new();
        for &v in &gf.inputs {
            versions_by_class
                .entry(gf.canonical(v).0)
                .or_default()
                .push(v);
        }
        for v in 0..gf.producers.len() {
            if !gf.producers[v].is_empty() {
                let v = ValueClassId(v);
                versions_by_class
                    .entry(gf.canonical(v).0)
                    .or_default()
                    .push(v);
            }
        }
        versions_by_class.retain(|_, versions| versions.len() >= 2);
        for versions in versions_by_class.values_mut() {
            versions.sort_unstable_by_key(|v| v.0);
        }
        Self { versions_by_class }
    }

    /// Whether a candidate with the given `inputs`/`outputs` is
    /// storage-unschedulable. Reconstruction orders every reader of a
    /// version before the next writer of the same storage class (the WAR
    /// edges in `apply::topological_order`), so a candidate that reads
    /// version `k` of a class while transitively depending on a later
    /// version can be scheduled neither before nor after that later
    /// writer. Rejecting it at insertion keeps the extractor from
    /// selecting a solution that `apply_solution` would refuse with
    /// `HazardCycle`.
    ///
    /// Versions the candidate writes itself are exempt (the in-place
    /// pattern: read version `k`, write version `k+1` — the WAR
    /// self-edge is vacuous). The check is conservative: a dataflow path
    /// from a later version to a candidate input proves the hazard under
    /// any selection that materializes the later writer, and rejecting a
    /// candidate only forgoes a fusion opportunity.
    pub fn would_create_storage_hazard_cycle(
        &self,
        gf: &GraphFuser,
        inputs: &[ValueClassId],
        outputs: &[ValueClassId],
    ) -> bool {
        if self.versions_by_class.is_empty() {
            return false;
        }
        let mut later: Vec<ValueClassId> = Vec::new();
        for &v in inputs {
            let Some(versions) = self.versions_by_class.get(&gf.canonical(v).0) else {
                continue;
            };
            let pos = versions.partition_point(|&u| u.0 <= v.0);
            later.extend(versions[pos..].iter().filter(|u| !outputs.contains(u)));
        }
        if later.is_empty() {
            return false;
        }
        later.sort_unstable_by_key(|v| v.0);
        later.dedup();
        // A path from a later version to a candidate input closes the
        // cycle: WAR forces candidate -> later-writer, the path forces
        // later-writer -> candidate.
        would_create_cycle(gf, inputs, &later)
    }
}
