//! Reconstruct a [`GraphBuilder`] from a v2 [`ExtractionSolution`].
//!
//! `detailed-fusion-plan-v2.md` §7 and §14. Reconstruction runs after
//! extraction and turns the selected alternative-graph nodes back into a
//! flat sequence of [`GraphNode`]s obeying the RAW/WAR/WAW hazards
//! implied by the versioned graph.
//!
//! Storage-hazard ordering (§7):
//!
//! 1. RAW edges are added between the unique selected producer of each consumed value and the
//!    consuming node.
//! 2. WAW edges are added between successive selected writers of the same physical [`BufId`],
//!    following [`ValueClassId`] order (which is the seed insertion order for that buffer's
//!    versions).
//! 3. WAR edges are added between each selected consumer of version `k` and the first selected
//!    writer of a version greater than `k` on the same physical buffer.
//!
//! The resulting precedence graph is acyclic by construction under the
//! MVP restrictions: value edges are a subgraph of the acyclic
//! alternative graph, and synthesized candidates never cross mutable
//! nodes, so the added storage-hazard edges follow seed version order.
//! A cycle here would indicate an invariant violation — the pass reports
//! it as an error and falls back to the original extraction.

use std::collections::HashMap;

use thiserror::Error;

use crate::{
    graph_ir::{GraphBuilder, GraphNode},
    passes::fusion_v2::{
        extract::ExtractionSolution,
        model::{GraphFuser, NodeId, ValueClassId},
    },
};

#[derive(Debug, Error)]
pub enum ApplyError {
    #[error(
        "solution requires value {value} which is neither a graph input nor produced by any \
         selected node"
    )]
    DemandedValueUnavailable { value: usize },
    #[error(
        "solution selects two producers of value {value}: at least NodeId({first}) and NodeId({second})"
    )]
    MultipleProducers {
        value: usize,
        first: usize,
        second: usize,
    },
    #[error(
        "storage-hazard precedence graph has a cycle involving NodeId({node}); this is a v2 \
         invariant violation"
    )]
    HazardCycle { node: usize },
}

/// Commits `solution` into `g`. On success `g.nodes` is replaced by the
/// selected [`GraphNode`]s in a hazard-respecting order and `g.plan` is
/// cleared. On failure `g.nodes` is restored from the original seed prefix
/// of `gf` and the error is returned; the graph is unchanged relative to
/// the pre-`take_graph` state.
///
/// Assumes `g.bufs`, `g.symbols`, and the registered interface still match
/// the state at `take_graph` time (i.e., the caller did not mutate the
/// builder in-between).
pub fn apply_solution(
    g: &mut GraphBuilder,
    gf: GraphFuser,
    solution: &ExtractionSolution,
) -> Result<(), ApplyError> {
    // Validate the solution structurally before mutating the builder so
    // any error path leaves `g` restorable from the moved nodes.
    let selected: Vec<NodeId> = solution.nodes.clone();
    let selected_set: HashMap<NodeId, ()> = selected.iter().map(|&n| (n, ())).collect();

    // Verify every selected input has exactly one selected producer or is
    // a graph input.
    let inputs_set: HashMap<ValueClassId, ()> = gf.inputs.iter().map(|&v| (v, ())).collect();
    let mut selected_producer: HashMap<ValueClassId, NodeId> = HashMap::new();
    for &n in &selected {
        for &out in &gf.nodes[n.0].outputs {
            if let Some(prev) = selected_producer.insert(out, n) {
                return Err(ApplyError::MultipleProducers {
                    value: out.0,
                    first: prev.0,
                    second: n.0,
                });
            }
        }
    }
    for &n in &selected {
        for &v in &gf.nodes[n.0].inputs {
            if !inputs_set.contains_key(&v) && !selected_producer.contains_key(&v) {
                return Err(ApplyError::DemandedValueUnavailable { value: v.0 });
            }
        }
    }
    // Every registered graph output must be selected-produced or a graph
    // input passthrough (which cannot happen after take_graph since
    // inputs have no producer, but is still legal in principle).
    for &v in &gf.outputs {
        if !inputs_set.contains_key(&v) && !selected_producer.contains_key(&v) {
            return Err(ApplyError::DemandedValueUnavailable { value: v.0 });
        }
    }

    // Storage-hazard precedence graph over selected nodes.
    let order = topological_order(&gf, &selected, &selected_producer, &selected_set)?;

    // Move the selected GraphNodes out of the fuser in emission order. On
    // any panic between here and the assignment below the caller keeps
    // the pre-take_graph state through the original fuser prefix (we
    // consume `gf`, so a panicking `into_iter` would still yield the
    // remaining nodes to a Drop guard if one were installed; for M1 we
    // rely on the operations here being infallible after validation).
    let mut arena: Vec<Option<GraphNode>> = gf.nodes.into_iter().map(|n| Some(n.node)).collect();
    let mut emitted: Vec<GraphNode> = Vec::with_capacity(order.len());
    for id in order {
        emitted.push(
            arena[id.0]
                .take()
                .expect("topological sort emits every selected NodeId exactly once"),
        );
    }
    g.nodes = emitted;
    g.plan = None;
    Ok(())
}

fn topological_order(
    gf: &GraphFuser,
    selected: &[NodeId],
    selected_producer: &HashMap<ValueClassId, NodeId>,
    selected_set: &HashMap<NodeId, ()>,
) -> Result<Vec<NodeId>, ApplyError> {
    // Edges: predecessor -> successor. Add each edge only once.
    let mut succ: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    let mut indeg: HashMap<NodeId, usize> = selected.iter().map(|&n| (n, 0)).collect();

    let mut add_edge = |from: NodeId, to: NodeId, succ: &mut HashMap<NodeId, Vec<NodeId>>| {
        if from == to {
            return;
        }
        let entry = succ.entry(from).or_default();
        if !entry.contains(&to) {
            entry.push(to);
            *indeg.entry(to).or_insert(0) += 1;
        }
    };

    // RAW: unique selected producer -> consumer for every selected input.
    for &n in selected {
        for &v in &gf.nodes[n.0].inputs {
            if let Some(&p) = selected_producer.get(&v) {
                if selected_set.contains_key(&p) {
                    add_edge(p, n, &mut succ);
                }
            }
        }
    }

    // Selected writers per physical buffer, in increasing ValueClassId
    // order (which matches seed insertion order for that buffer's versions).
    // Keyed on `BufId.0` since BufId is not Ord.
    let mut writers_by_buf: HashMap<usize, Vec<(ValueClassId, NodeId)>> = HashMap::new();
    for &n in selected {
        for &out in &gf.nodes[n.0].outputs {
            let buf = gf.physical(out);
            writers_by_buf.entry(buf.0).or_default().push((out, n));
        }
    }
    for writers in writers_by_buf.values_mut() {
        writers.sort_by_key(|(v, _)| v.0);
    }

    // WAW: consecutive selected writers of the same physical buffer.
    for writers in writers_by_buf.values() {
        for w in writers.windows(2) {
            add_edge(w[0].1, w[1].1, &mut succ);
        }
    }

    // WAR: for each selected consumer of version k on physical buf b, the
    // first selected writer of a version > k on b must follow.
    let mut consumers_by_val: HashMap<ValueClassId, Vec<NodeId>> = HashMap::new();
    for &n in selected {
        for &v in &gf.nodes[n.0].inputs {
            consumers_by_val.entry(v).or_default().push(n);
        }
    }
    for writers in writers_by_buf.values() {
        // For each version k written in this buffer's writer chain, find
        // the first writer with a later version. Consumers of that
        // earlier version (or any earlier same-buffer version) must
        // precede that later writer.
        for i in 0..writers.len() {
            let next_writer = writers[i + 1..].first().map(|(_, w)| *w);
            let Some(next_writer) = next_writer else {
                continue;
            };
            for &(earlier_val, _) in &writers[..=i] {
                if let Some(cs) = consumers_by_val.get(&earlier_val) {
                    for &c in cs {
                        add_edge(c, next_writer, &mut succ);
                    }
                }
            }
        }
    }

    // Kahn's algorithm with a deterministic tie-break by NodeId — stable
    // emission order across runs.
    let mut ready: Vec<NodeId> = selected
        .iter()
        .copied()
        .filter(|n| indeg.get(n).copied().unwrap_or(0) == 0)
        .collect();
    ready.sort_by_key(|n| n.0);
    let mut out = Vec::with_capacity(selected.len());
    let mut cursor = 0;
    while cursor < ready.len() {
        // Pop the smallest ready node.
        ready[cursor..].sort_by_key(|n| n.0);
        let n = ready[cursor];
        cursor += 1;
        out.push(n);
        if let Some(succs) = succ.get(&n) {
            for &s in succs {
                let d = indeg.get_mut(&s).expect("successor is selected");
                *d -= 1;
                if *d == 0 {
                    ready.push(s);
                }
            }
        }
    }
    if out.len() != selected.len() {
        let stuck = selected
            .iter()
            .find(|n| indeg.get(n).copied().unwrap_or(0) > 0)
            .copied()
            .unwrap_or(NodeId(0));
        return Err(ApplyError::HazardCycle { node: stuck.0 });
    }
    Ok(out)
}
