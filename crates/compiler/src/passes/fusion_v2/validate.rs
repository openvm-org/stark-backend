//! Insertion-time validation for fusion-v2 candidates.
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

use std::collections::HashSet;

use crate::passes::fusion_v2::model::{GraphFuser, NodeId, ValueClassId};

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
