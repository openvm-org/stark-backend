//! Exhaustive brute-force extractor.
//!
//! Enumerates every subset of alternative-graph nodes, filters to the
//! feasible ones, and returns the lex-min under the four-stage objective
//! defined by [`detailed-fusion-plan-v2.md`] §13.5.
//!
//! Serves two roles: the M2 exit-gate reference (`brute-force matches
//! CP-SAT`), and a solver-free fallback for tiny graphs where the extra
//! CP-SAT setup would be pure overhead. Because the enumeration is
//! exponential the caller must gate this on graph size — the extractor
//! itself hard-limits at [`BRUTE_FORCE_LIMIT`] alternatives and returns
//! `None` above the cap.

use std::collections::{HashMap, HashSet};

use crate::passes::fusion_v2::{
    cost::ArtifactKey,
    extract::{ExtractOptions, ExtractionData, ExtractionSolution, SolverStatus},
    model::{GraphFuser, NodeId, ValueClassId},
};

/// Maximum number of alternative-graph nodes the brute-force extractor
/// will enumerate. `2^32` is already beyond any interactive use; the
/// practical bound for tests and small graphs is `<= 20`.
pub const BRUTE_FORCE_LIMIT: usize = 32;

/// Runs exhaustive extraction. Returns `None` if the graph exceeds
/// [`BRUTE_FORCE_LIMIT`] alternatives.
pub fn extract(
    gf: &GraphFuser,
    data: &ExtractionData,
    options: &ExtractOptions,
) -> Option<ExtractionSolution> {
    let n = gf.nodes.len();
    if n > BRUTE_FORCE_LIMIT {
        return None;
    }
    assert_eq!(data.costs.len(), n, "ExtractionData::costs len mismatch");
    assert_eq!(
        data.artifact_keys.len(),
        n,
        "ExtractionData::artifact_keys len mismatch"
    );

    // Precompute per-value producer lists over the whole alt graph.
    let inputs_set: HashSet<ValueClassId> = gf.inputs.iter().copied().collect();
    let outputs_set: HashSet<ValueClassId> = gf.outputs.iter().copied().collect();

    // Original-artifact set for the max_new_modules cap: artifacts of
    // seed-node kernels.
    let original_artifacts: HashSet<ArtifactKey> = data.artifact_keys[..gf.seed_node_count]
        .iter()
        .flatten()
        .cloned()
        .collect();

    // Iterate every subset. For each, check feasibility and if feasible
    // compute its lex-cost.
    let total_subsets: u64 = 1u64 << n;
    let mut best: Option<(LexCost, Vec<NodeId>, Vec<ValueClassId>)> = None;
    for mask in 0..total_subsets {
        let selected: Vec<NodeId> = (0..n)
            .filter(|i| (mask >> i) & 1 == 1)
            .map(NodeId)
            .collect();

        // Feasibility.
        let Some((materialized, artifacts)) =
            feasibility(gf, &selected, &inputs_set, &outputs_set, data)
        else {
            continue;
        };

        // Optional budgets.
        if let Some(cap) = options.max_modules {
            if artifacts.len() > cap {
                continue;
            }
        }
        if let Some(cap) = options.max_new_modules {
            let new_count = artifacts
                .iter()
                .filter(|k| !original_artifacts.contains(*k))
                .count();
            if new_count > cap {
                continue;
            }
        }

        // Runtime tolerance ppm is honored by the CP-SAT extractor at
        // the stage-1-to-stage-2 hand-off; the brute-force reference
        // uses strict lex, since the exit gate compares under strict
        // lex.
        let runtime: i128 = selected
            .iter()
            .map(|n| data.costs[n.0].runtime_units as i128)
            .sum();
        let cost = LexCost {
            runtime,
            artifact_count: artifacts.len() as u64,
            node_count: selected.len() as u64,
            value_count: materialized.len() as u64,
        };

        match &best {
            None => {
                let materialized: Vec<ValueClassId> = materialized.into_iter().collect();
                best = Some((cost, selected, materialized));
            }
            Some((prev, _, _)) if cost < *prev => {
                let materialized: Vec<ValueClassId> = materialized.into_iter().collect();
                best = Some((cost, selected, materialized));
            }
            _ => {}
        }
    }

    let (_cost, selected, _materialized) = best.expect(
        "the original-seed subset is always feasible (§6.4); brute force must find at least one \
         solution",
    );
    Some(ExtractionSolution {
        nodes: selected,
        fallback: None,
        status: Some(SolverStatus::Optimal),
    })
}

/// Lex-cost tuple corresponding to §13.5's four stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LexCost {
    /// Stage 1: total predicted runtime.
    runtime: i128,
    /// Stage 2: number of unique artifacts required.
    artifact_count: u64,
    /// Stage 3: number of selected alternatives (graph size).
    node_count: u64,
    /// Stage 4: number of materialized values (peak-memory proxy).
    value_count: u64,
}

impl Ord for LexCost {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.runtime
            .cmp(&other.runtime)
            .then(self.artifact_count.cmp(&other.artifact_count))
            .then(self.node_count.cmp(&other.node_count))
            .then(self.value_count.cmp(&other.value_count))
    }
}
impl PartialOrd for LexCost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Returns `Some((materialized_values, artifact_keys))` when `selected` is
/// feasible; `None` otherwise. Feasibility rules mirror §13.3:
///
/// - every registered graph output has a selected producer or is a graph input;
/// - every input used by a selected node has a selected producer or is a graph input;
/// - at most one selected producer per value class.
fn feasibility(
    gf: &GraphFuser,
    selected: &[NodeId],
    inputs_set: &HashSet<ValueClassId>,
    outputs_set: &HashSet<ValueClassId>,
    data: &ExtractionData,
) -> Option<(HashSet<ValueClassId>, HashSet<ArtifactKey>)> {
    let mut producer: HashMap<ValueClassId, NodeId> = HashMap::new();
    for &n in selected {
        for &v in &gf.nodes[n.0].outputs {
            if producer.insert(v, n).is_some() {
                return None; // Two producers for one value.
            }
        }
    }
    let is_available = |v: ValueClassId| producer.contains_key(&v) || inputs_set.contains(&v);
    for &n in selected {
        for &v in &gf.nodes[n.0].inputs {
            if !is_available(v) {
                return None;
            }
        }
    }
    for &v in outputs_set {
        if !is_available(v) {
            return None;
        }
    }
    // Materialized values: every graph input (pinned by §13.3) plus
    // every selected producer's outputs plus every graph output.
    // Graph outputs may be produced by a selected node (so already
    // present) or be a pass-through input.
    let mut materialized: HashSet<ValueClassId> = inputs_set.iter().copied().collect();
    for &n in selected {
        for &v in &gf.nodes[n.0].outputs {
            materialized.insert(v);
        }
    }
    for &v in outputs_set {
        materialized.insert(v);
    }
    let artifacts: HashSet<ArtifactKey> = selected
        .iter()
        .filter_map(|n| data.artifact_keys[n.0].clone())
        .collect();
    Some((materialized, artifacts))
}
