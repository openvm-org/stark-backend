//! Top-level fusion-v2 entry point: [`fuse_graph_v2`].
//!
//! `detailed-fusion-plan-v2.md` §3 pipeline. Runs bounded saturation
//! (§11) on the input graph:
//!
//! 1. Convert to a versioned seed alternative graph
//!    ([`crate::passes::fusion_v2::version::take_graph`]);
//! 2. Loop up to `max_rounds`:
//!    - freeze the current node count as the round's read-only view;
//!    - enumerate each enabled fusion pass over that frozen prefix;
//!    - deduplicate candidates by [`CandidateKey`] across rounds (§9);
//!    - validate acyclicity (§9.1) and insert accepted candidates;
//! 3. Extract with CP-SAT if `planner-ortools`, else brute force, else the original fallback;
//! 4. Apply the selected solution back to the [`GraphBuilder`].
//!
//! Full candidate finalization (§9: canonicalize, monomorphize, launch-
//! schedule validation, boundary pruning) and pattern-key bucketing
//! (§10.0) land in later milestones. What lands here is: multi-round
//! composition, origin tracking, cross-round dedup, and per-round caps.

use std::collections::BTreeSet;

use thiserror::Error;

#[cfg(not(feature = "planner-ortools"))]
use crate::passes::fusion_v2::extract::brute;
use crate::{
    graph_ir::{GraphBuilder, GraphNode},
    module_hash::module_hash,
    passes::fusion_v2::{
        apply::{apply_solution, ApplyError},
        cost::{
            estimate_non_kernel, ArtifactContext, EstimatorConfig, GraphNodeCost, KernelCostManager,
        },
        extract::{ExtractOptions, ExtractionData, ExtractionSolution, FallbackReason},
        fusions::producer_consumer,
        model::{GraphFuser, NodeId},
        saturate::{CandidateKey, SaturationState},
        validate::would_create_cycle,
        version::{take_graph, TakeGraphError},
    },
};

/// Configuration for one invocation of [`fuse_graph_v2`].
///
/// Only the M3/M4-relevant subset of the plan's `FusionOptionsV2` (§15) is
/// wired for now. Additional fields will be added as later milestones
/// consume them.
#[derive(Debug, Clone)]
pub struct FusionOptionsV2 {
    /// Hard cap on the number of candidates inserted into the
    /// alternative graph (§11 `max_total_alternatives`).
    pub max_total_alternatives: usize,
    /// If `true`, every candidate is checked against §9.1's
    /// insertion-time acyclicity guard before being inserted. Turning
    /// it off relies on each fusion pass's legality proof to preserve
    /// the DAG invariant.
    pub validate_alt_graph_acyclicity: bool,
    /// Wall-time cap for the CP-SAT solve, in seconds.
    pub solver_time_limit_secs: f64,
    /// Whether producer-consumer candidates should be enumerated.
    pub enable_producer_consumer: bool,
    /// Whether keep-seam variants should be emitted alongside drop
    /// candidates (§10.2, plan §15). Emission is still gated per-seam
    /// on the trigger conditions in §10.2; setting this to `false`
    /// disables keep-variant emission entirely.
    pub enable_keep_variants: bool,
    /// Diagnostic: emit a keep variant for *every* legal drop candidate
    /// regardless of trigger conditions. Off by default because it
    /// inflates enumeration for the common single-consumer seam. Used
    /// by the M5 tests to exercise the extractor's choice between
    /// materialize/duplicate/keep on a controlled fixture.
    pub enable_all_keep_variants: bool,
    /// M4: KIR estimator configuration (§12.1). Defaults to
    /// `EstimatorConfig::default()`, which uses the synthetic device
    /// profile. Callers pointing at real hardware supply a calibrated
    /// profile through this field.
    pub estimator: EstimatorConfig,
    /// M4: symbolic bindings for `EstimateContext`; typically populated
    /// by the caller with the graph's symbolic sizes. Only bindings
    /// present here can resolve symbolic bounds in `MemcpyNode`,
    /// `MemSetNode`, or grid extents.
    pub graph_symbols: std::collections::BTreeMap<crate::ir::VarId, i64>,
    /// M4: artifact identity context (§5.5). Constant within one
    /// `GraphCompiler` invocation; keeps `target_arch` and
    /// `compiler_flags_hash` on the ILP's `ArtifactKey`s.
    pub artifact: ArtifactContext,
    /// M4: cycle quantum used to convert estimated cycles to i64 runtime
    /// units (§13.5). Larger values compress the objective range at the
    /// cost of tie-detection precision.
    pub cycle_quantum: i64,
    /// M4: caller-supplied cost estimate for `BlackboxKernel` nodes
    /// (§12.8). Zero when unset — the blackbox is the sole producer of
    /// its outputs so the ILP selects it whenever those outputs are
    /// demanded, and the zero cost cannot cause an incorrect drop.
    pub blackbox_hint_cycles: f64,
    /// M6: bound on the number of saturation rounds (§11
    /// `max_rounds`). Each round enumerates the enabled fusion passes
    /// over the frozen node set at the start of the round; new
    /// candidates inserted mid-round become eligible parents in the
    /// next round. The default of `4` matches the plan.
    pub max_rounds: usize,
    /// M6: soft cap on the number of alternatives one pass may emit
    /// per round (§11 `max_alternatives_per_pass_per_round`). When
    /// the enumerator returns more drafts than this the driver
    /// truncates and counts the excess in
    /// [`FusionReportV2::candidates_rejected_pass_cap`]. Set to zero
    /// to disable per-pass truncation.
    pub max_alternatives_per_pass_per_round: usize,
}

impl Default for FusionOptionsV2 {
    fn default() -> Self {
        Self {
            max_total_alternatives: 5000,
            validate_alt_graph_acyclicity: true,
            solver_time_limit_secs: 5.0,
            enable_producer_consumer: true,
            enable_keep_variants: true,
            enable_all_keep_variants: false,
            estimator: EstimatorConfig::default(),
            graph_symbols: std::collections::BTreeMap::new(),
            artifact: ArtifactContext {
                target_arch: "placeholder".into(),
                compiler_flags_hash: [0; 32],
            },
            cycle_quantum: 1,
            blackbox_hint_cycles: 0.0,
            max_rounds: 4,
            max_alternatives_per_pass_per_round: 0,
        }
    }
}

/// Report produced by one call to [`fuse_graph_v2`].
#[derive(Debug, Clone, Default)]
pub struct FusionReportV2 {
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub candidates_generated: usize,
    pub candidates_inserted: usize,
    pub candidates_rejected_cycle: usize,
    pub candidates_rejected_cap: usize,
    /// M6: candidates discarded because their `CandidateKey` was
    /// already in `SaturationState::seen_candidates` (§9 dedup —
    /// typically fires on associative-composition duplicates).
    pub candidates_rejected_dedup: usize,
    /// M6: candidates discarded because a pass emitted more drafts
    /// than `max_alternatives_per_pass_per_round` in that round
    /// (§11).
    pub candidates_rejected_pass_cap: usize,
    pub selected_from_solver: usize,
    /// Fallback reason returned by the extractor, or `None` when the
    /// solver produced a proper solution.
    pub fallback_reason: Option<FallbackReason>,
    /// M4: KIR cost-cache statistics (§12.10). `hits + misses` equals
    /// the number of kernel cost lookups performed by the driver.
    pub cost_cache_hits: u64,
    pub cost_cache_misses: u64,
    /// M4: sum of `runtime_units` across all alternative nodes seen by
    /// the extractor. Useful for regression tracking of the estimator.
    pub total_runtime_units: i64,
    /// M6: number of saturation rounds that actually ran (a round runs
    /// only if the previous round inserted at least one candidate).
    pub rounds_run: usize,
    /// M6: number of candidates inserted per round; `rounds_inserted[r]`
    /// is the count for round `r` (0-indexed). Trailing zero-round
    /// entries are pruned before reporting.
    pub rounds_inserted: Vec<usize>,
    /// M6: `true` if the loop stopped because `max_rounds` fired rather
    /// than reaching a fixed point (`candidates_inserted == 0`).
    pub max_rounds_hit: bool,
}

/// Failure modes of [`fuse_graph_v2`]. Structural errors from
/// [`take_graph`] and [`apply_solution`] surface here.
#[derive(Debug, Error)]
pub enum FuseV2Error {
    #[error(transparent)]
    TakeGraph(#[from] TakeGraphError),
    #[error(transparent)]
    Apply(#[from] ApplyError),
}

/// Runs bounded-saturation fusion v2 on `g`, in place. Preserves the
/// registered interface, invalidates `g.plan`, and returns a diagnostic
/// [`FusionReportV2`].
pub fn fuse_graph_v2(
    g: &mut GraphBuilder,
    options: &FusionOptionsV2,
) -> Result<FusionReportV2, FuseV2Error> {
    let nodes_before = g.nodes.len();

    // Step 1: convert to versioned seed alternative graph.
    let mut gf = take_graph(g)?;
    let mut sat = SaturationState::new(gf.seed_node_count);

    // Step 2: bounded saturation. Each round freezes the current node
    // count, enumerates every enabled pass over that frozen prefix,
    // deduplicates by CandidateKey, validates, and inserts. Stops when
    // a round inserts zero candidates or `max_rounds` fires.
    let max_rounds = options.max_rounds.max(1);
    let mut candidates_generated = 0usize;
    let mut candidates_inserted = 0usize;
    let mut candidates_rejected_cycle = 0usize;
    let mut candidates_rejected_cap = 0usize;
    let mut candidates_rejected_dedup = 0usize;
    let mut candidates_rejected_pass_cap = 0usize;
    let mut rounds_inserted: Vec<usize> = Vec::new();
    let mut max_rounds_hit = false;

    // `min_new_parent_id` starts at 0 (round 1: all seeds are new).
    // After round r ends, it advances to the frozen count of that
    // round + inserted candidates, so round r+1 only enumerates pairs
    // involving at least one node inserted in round r.
    let mut min_new_parent_id = 0usize;
    for round in 0..max_rounds {
        let frozen = gf.nodes.len();

        // Enumerate. Passes see the frozen prefix and current origins.
        let drafts = if options.enable_producer_consumer {
            enumerate_producer_consumer(&gf, &sat, frozen, min_new_parent_id, options)
        } else {
            Vec::new()
        };
        candidates_generated += drafts.len();

        // Per-pass cap: soft-truncate the drafts list to
        // `max_alternatives_per_pass_per_round`.
        let per_pass_cap = options.max_alternatives_per_pass_per_round;
        let (drafts, over_cap) = if per_pass_cap > 0 && drafts.len() > per_pass_cap {
            let over = drafts.len() - per_pass_cap;
            let mut d = drafts;
            d.truncate(per_pass_cap);
            (d, over)
        } else {
            (drafts, 0)
        };
        candidates_rejected_pass_cap += over_cap;

        let mut inserted_this_round = 0usize;
        for draft in drafts {
            if candidates_inserted >= options.max_total_alternatives {
                candidates_rejected_cap += 1;
                continue;
            }
            // CandidateKey dedup across rounds (§9). Compute the key
            // now; if it collides with `seen_candidates` skip without
            // touching the arenas.
            let Some(key) = candidate_key(&draft, &options.artifact) else {
                // Non-kernel candidates have no artifact key today —
                // no producer-consumer pass emits these, but the guard
                // keeps future passes safe.
                candidates_rejected_dedup += 1;
                continue;
            };
            if !sat.note_seen(key) {
                candidates_rejected_dedup += 1;
                continue;
            }
            if options.validate_alt_graph_acyclicity
                && would_create_cycle(&gf, &draft.alt.inputs, &draft.alt.outputs)
            {
                candidates_rejected_cycle += 1;
                continue;
            }
            let parents = draft.parents.clone();
            let node_id = gf.insert_candidate(draft.alt);
            sat.register_origins(node_id, &parents);
            candidates_inserted += 1;
            inserted_this_round += 1;
        }
        rounds_inserted.push(inserted_this_round);

        if inserted_this_round == 0 {
            break;
        }
        if round + 1 == max_rounds {
            max_rounds_hit = true;
        }
        // Round-cap check: if we've hit the total-alt cap there's no
        // point running another round.
        if candidates_inserted >= options.max_total_alternatives {
            break;
        }
        // Advance the "new since last round" watermark so the next
        // enumeration skips pairs entirely inside the previous prefix.
        min_new_parent_id = frozen;
    }
    let rounds_run = rounds_inserted.len();

    // Step 3: build extraction data and extract.
    let mut manager = KernelCostManager::new(
        options.estimator.clone(),
        options.artifact.clone(),
        options.graph_symbols.clone(),
        options.cycle_quantum,
    );
    let data = build_extraction_data(&gf, &g.bufs, &mut manager, options);
    let total_runtime_units: i64 = data
        .costs
        .iter()
        .map(|c| c.runtime_units)
        .fold(0i64, i64::saturating_add);
    let extract_opts = ExtractOptions {
        solver_time_limit_secs: options.solver_time_limit_secs,
        cycle_quantum: options.cycle_quantum,
        ..Default::default()
    };
    let solution = choose_extractor(&gf, &data, &extract_opts);
    let selected_from_solver = solution.nodes.len();
    let fallback_reason = solution.fallback.clone();
    let stats = manager.stats();

    // Step 4: apply solution back to the builder.
    apply_solution(g, gf, &solution)?;

    Ok(FusionReportV2 {
        nodes_before,
        nodes_after: g.nodes.len(),
        candidates_generated,
        candidates_inserted,
        candidates_rejected_cycle,
        candidates_rejected_cap,
        candidates_rejected_dedup,
        candidates_rejected_pass_cap,
        selected_from_solver,
        fallback_reason,
        cost_cache_hits: stats.hits,
        cost_cache_misses: stats.misses,
        total_runtime_units,
        rounds_run,
        rounds_inserted,
        max_rounds_hit,
    })
}

/// Runs the producer-consumer enumerator with the caller's flags
/// applied. `frozen` bounds the eligible parent nodes; `sat.origins`
/// enforces the disjoint-origins check; `min_new_parent_id` skips
/// pairs both of whose parents predate the previous round.
fn enumerate_producer_consumer(
    gf: &GraphFuser,
    sat: &SaturationState,
    frozen: usize,
    min_new_parent_id: usize,
    options: &FusionOptionsV2,
) -> Vec<producer_consumer::CandidateDraft> {
    let enable_all = options.enable_all_keep_variants && options.enable_keep_variants;
    let enum_opts = producer_consumer::EnumerateOptions {
        enable_all_keep_variants: enable_all,
    };
    let ctx = producer_consumer::EnumerateContext {
        frozen_node_count: frozen,
        origins: &sat.origins,
        min_new_parent_id,
        options: enum_opts,
    };
    let mut drafts = producer_consumer::enumerate(gf, &ctx);
    if !options.enable_keep_variants {
        drafts.retain(|d| d.variant != producer_consumer::FusionVariant::Keep);
    }
    drafts
}

/// Computes the [`CandidateKey`] for a draft. Returns `None` if the
/// draft's node is not a `Kernel` (no artifact identity).
fn candidate_key(
    draft: &producer_consumer::CandidateDraft,
    artifact: &ArtifactContext,
) -> Option<CandidateKey> {
    let hash = match &draft.alt.node {
        GraphNode::Kernel(k) => k.hash.unwrap_or_else(|| module_hash(&k.module)),
        _ => return None,
    };
    Some(CandidateKey {
        inputs: draft.alt.inputs.clone(),
        outputs: draft.alt.outputs.clone(),
        artifact: artifact.key_for(hash),
    })
}

// `NodeId` and `BTreeSet` are imported to satisfy potential future
// direct references from doc comments; suppress unused-import lint.
#[allow(dead_code)]
type _MarkerNodeId = NodeId;
#[allow(dead_code)]
type _MarkerOriginSet = BTreeSet<NodeId>;

/// Assembles the [`ExtractionData`] passed to the extractor (M4 §12).
///
/// - `Kernel` nodes are routed through the [`KernelCostManager`]: on a cache hit the cost is served
///   without lowering, otherwise the estimator runs the full HIR→KIR pipeline (§12.10);
/// - `Const`, `Memcpy`, `Memset`, and `BlackboxKernel` nodes get closed-form costs from
///   [`estimate_non_kernel`] (§12.8);
/// - `Kernel` nodes carry an [`ArtifactKey`] under the caller's `ArtifactContext`; non-kernel nodes
///   have `artifact_key == None` and contribute no `z_m` constraints.
fn build_extraction_data(
    gf: &GraphFuser,
    bufs: &[crate::graph_ir::BufInfo],
    manager: &mut KernelCostManager,
    options: &FusionOptionsV2,
) -> ExtractionData {
    let mut costs = Vec::with_capacity(gf.nodes.len());
    let mut artifact_keys = Vec::with_capacity(gf.nodes.len());
    for alt in &gf.nodes {
        match &alt.node {
            GraphNode::Kernel(k) => {
                let hash = k.hash.unwrap_or_else(|| module_hash(&k.module));
                let cost = manager
                    .cost_of(hash, &k.module, &k.param_bindings)
                    .unwrap_or_else(|_| GraphNodeCost::new(1));
                costs.push(cost);
                artifact_keys.push(Some(options.artifact.key_for(hash)));
            }
            other => {
                let cost = estimate_non_kernel(
                    other,
                    bufs,
                    &crate::passes::fusion_v2::cost::EstimateContext {
                        graph_symbols: options.graph_symbols.clone(),
                        param_bindings: Default::default(),
                    },
                    &options.estimator,
                    options.cycle_quantum,
                    options.blackbox_hint_cycles,
                );
                costs.push(cost);
                artifact_keys.push(None);
            }
        }
    }
    ExtractionData {
        costs,
        artifact_keys,
    }
}

/// Picks the strongest available extractor:
///
/// - `planner-ortools`: the CP-SAT extractor;
/// - otherwise: the brute-force extractor if the graph fits its cap, else the original fallback.
fn choose_extractor(
    gf: &GraphFuser,
    data: &ExtractionData,
    options: &ExtractOptions,
) -> ExtractionSolution {
    #[cfg(feature = "planner-ortools")]
    {
        crate::passes::fusion_v2::extract::cpsat::extract(gf, data, options)
    }
    #[cfg(not(feature = "planner-ortools"))]
    {
        brute::extract(gf, data, options).unwrap_or_else(|| ExtractionSolution::original(gf))
    }
}
