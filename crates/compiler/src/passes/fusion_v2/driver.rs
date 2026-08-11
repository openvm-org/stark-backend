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
        fusions::{epilogue, fanout, horizontal, producer_consumer, small_kernel},
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
    /// Number of CP-SAT search workers. The default of `1` keeps the
    /// solve deterministic (plan §2.4); higher values enable the
    /// parallel portfolio, which is faster but breaks run-to-run
    /// reproducibility of tie-broken solutions. `0` lets CP-SAT
    /// decide.
    pub solver_num_workers: usize,
    /// Whether producer-consumer candidates should be enumerated.
    pub enable_producer_consumer: bool,
    /// Whether M7 fanout candidates should be enumerated (§10.5,
    /// §15 default `true`). Fanout emits one candidate per legal
    /// `(producer, k≥2 consumers)` grouping with an identity seam
    /// read; individual `(producer, consumer)` pairs still get
    /// producer-consumer candidates.
    pub enable_fanout: bool,
    /// Whether M8 small-kernel block-fusion candidates should be
    /// enumerated (§10.7, §15 default `true`). Small-kernel fuses
    /// linear chains of concrete-bound kernels into a single kernel
    /// with shared-memory-backed intermediate tiles — different
    /// domain sizes per layer are supported.
    pub enable_small_kernel: bool,
    /// Byte budget for the combined shared-memory footprint of all
    /// small-kernel tiles in one candidate (§10.7). Default: 48 KiB.
    pub small_kernel_shared_bytes: usize,
    /// Maximum kernels a small-kernel chain may contain (§11
    /// `max_region_seed_nodes`). Default: 6.
    pub small_kernel_max_chain: usize,
    /// Whether M9 same-domain horizontal candidates should be
    /// enumerated (§10.6, §15 default `true`). Horizontal fuses pairs
    /// of dataflow-independent flat kernels with equal concrete outer
    /// domain into one kernel returning the concatenated tuple of
    /// outputs; larger groups compose across saturation rounds.
    pub enable_horizontal: bool,
    /// Whether M10 epilogue candidates should be enumerated (§10.4,
    /// §15 default `true`). Epilogue fuses a flat pointwise consumer
    /// into its producer's launch schedule (`par`/`threads`/block
    /// hint retained); producers already covered by the
    /// producer-consumer pass are skipped.
    pub enable_epilogue: bool,
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
    /// M11 (§15): print per-round saturation counters and the selected
    /// extraction (node ids, kinds, costs, fallback reason) to stderr,
    /// mirroring the existing pass's `FusionOptions::verbose`.
    pub verbose: bool,
}

impl Default for FusionOptionsV2 {
    fn default() -> Self {
        Self {
            max_total_alternatives: 5000,
            validate_alt_graph_acyclicity: true,
            solver_time_limit_secs: 5.0,
            solver_num_workers: 1,
            enable_producer_consumer: true,
            enable_fanout: true,
            enable_small_kernel: true,
            small_kernel_shared_bytes: 48 * 1024,
            small_kernel_max_chain: 6,
            enable_horizontal: true,
            enable_epilogue: true,
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
            verbose: false,
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
    let sat_t0 = std::time::Instant::now();
    for round in 0..max_rounds {
        let frozen = gf.nodes.len();

        // Enumerate. Passes see the frozen prefix and current origins.
        // `pass_stats` records `(pass, generated, wall time)` per pass
        // for the verbose dump.
        let mut pass_stats: Vec<(&'static str, usize, std::time::Duration)> = Vec::new();
        let t = std::time::Instant::now();
        let mut drafts = if options.enable_producer_consumer {
            enumerate_producer_consumer(&gf, &sat, frozen, min_new_parent_id, options)
        } else {
            Vec::new()
        };
        pass_stats.push(("producer-consumer", drafts.len(), t.elapsed()));
        if options.enable_fanout {
            let t = std::time::Instant::now();
            let before = drafts.len();
            let fanout_ctx = producer_consumer::EnumerateContext {
                frozen_node_count: frozen,
                origins: &sat.origins,
                min_new_parent_id,
                options: producer_consumer::EnumerateOptions::default(),
            };
            drafts.extend(fanout::enumerate(&gf, &fanout_ctx));
            pass_stats.push(("fanout", drafts.len() - before, t.elapsed()));
        }
        if options.enable_small_kernel {
            let t = std::time::Instant::now();
            let before = drafts.len();
            let sk_ctx = producer_consumer::EnumerateContext {
                frozen_node_count: frozen,
                origins: &sat.origins,
                min_new_parent_id,
                options: producer_consumer::EnumerateOptions::default(),
            };
            let sk_opts = small_kernel::SmallKernelOptions {
                max_shared_bytes: options.small_kernel_shared_bytes,
                max_chain_length: options.small_kernel_max_chain,
            };
            drafts.extend(small_kernel::enumerate(&gf, &sk_ctx, sk_opts));
            pass_stats.push(("small-kernel", drafts.len() - before, t.elapsed()));
        }
        if options.enable_horizontal {
            let t = std::time::Instant::now();
            let before = drafts.len();
            let h_ctx = producer_consumer::EnumerateContext {
                frozen_node_count: frozen,
                origins: &sat.origins,
                min_new_parent_id,
                options: producer_consumer::EnumerateOptions::default(),
            };
            drafts.extend(horizontal::enumerate(&gf, &h_ctx));
            pass_stats.push(("horizontal", drafts.len() - before, t.elapsed()));
        }
        if options.enable_epilogue {
            let t = std::time::Instant::now();
            let before = drafts.len();
            let e_ctx = producer_consumer::EnumerateContext {
                frozen_node_count: frozen,
                origins: &sat.origins,
                min_new_parent_id,
                options: producer_consumer::EnumerateOptions {
                    enable_all_keep_variants: options.enable_all_keep_variants
                        && options.enable_keep_variants,
                },
            };
            let mut e_drafts = epilogue::enumerate(&gf, &e_ctx);
            if !options.enable_keep_variants {
                e_drafts.retain(|d| d.variant != producer_consumer::FusionVariant::Keep);
            }
            drafts.extend(e_drafts);
            pass_stats.push(("epilogue", drafts.len() - before, t.elapsed()));
        }
        let generated_this_round = drafts.len();
        candidates_generated += generated_this_round;
        if options.verbose {
            for (pass, generated, dt) in &pass_stats {
                eprintln!(
                    "[fusion-v2] round {round} pass {pass}: generated={generated} in {:.1} ms",
                    dt.as_secs_f64() * 1e3,
                );
            }
        }

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

        let insert_t0 = std::time::Instant::now();
        let (dedup0, cycle0, cap0) = (
            candidates_rejected_dedup,
            candidates_rejected_cycle,
            candidates_rejected_cap,
        );
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
        if options.verbose {
            eprintln!(
                "[fusion-v2] round {round}: generated={generated_this_round}, \
                 inserted={inserted_this_round}, alt_nodes={}, rejected \
                 dedup={} cycle={} cap={} pass_cap={over_cap}, insert took {:.1} ms",
                gf.nodes.len(),
                candidates_rejected_dedup - dedup0,
                candidates_rejected_cycle - cycle0,
                candidates_rejected_cap - cap0,
                insert_t0.elapsed().as_secs_f64() * 1e3,
            );
        }

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
    if options.verbose {
        eprintln!(
            "[fusion-v2] saturation: {rounds_run} round(s), generated={candidates_generated}, \
             inserted={candidates_inserted}, alt_nodes={}, total {:.1} ms",
            gf.nodes.len(),
            sat_t0.elapsed().as_secs_f64() * 1e3,
        );
    }

    // Step 3: build extraction data and extract.
    let mut manager = KernelCostManager::new(
        options.estimator.clone(),
        options.artifact.clone(),
        options.graph_symbols.clone(),
        options.cycle_quantum,
    );
    let cost_t0 = std::time::Instant::now();
    let data = build_extraction_data(&gf, &g.bufs, &mut manager, options);
    let stats = manager.stats();
    if options.verbose {
        let est_ms = stats.estimate_time.as_secs_f64() * 1e3;
        let runs = stats.misses + stats.failures;
        eprintln!(
            "[fusion-v2] costing: {} nodes in {:.1} ms; kernel cost cache hits={} misses={} \
             failures={}; estimator (HIR→KIR + analysis) {:.1} ms total, {:.2} ms/run",
            gf.nodes.len(),
            cost_t0.elapsed().as_secs_f64() * 1e3,
            stats.hits,
            stats.misses,
            stats.failures,
            est_ms,
            est_ms / runs.max(1) as f64,
        );
    }
    // Failure sentinels would saturate the diagnostic sum; skip them.
    let total_runtime_units: i64 = data
        .costs
        .iter()
        .filter(|c| !c.is_failure())
        .map(|c| c.runtime_units)
        .fold(0i64, i64::saturating_add);
    let extract_opts = ExtractOptions {
        solver_time_limit_secs: options.solver_time_limit_secs,
        solver_num_workers: options.solver_num_workers,
        cycle_quantum: options.cycle_quantum,
        verbose: options.verbose,
        ..Default::default()
    };
    let solve_t0 = std::time::Instant::now();
    let solution = choose_extractor(&gf, &data, &extract_opts);
    let selected_from_solver = solution.nodes.len();
    let fallback_reason = solution.fallback.clone();
    if options.verbose {
        eprintln!(
            "[fusion-v2] solve: {:.1} ms, status={:?}, fallback={:?}, selected={}",
            solve_t0.elapsed().as_secs_f64() * 1e3,
            solution.status,
            solution.fallback,
            solution.nodes.len(),
        );
        dump_extraction(&gf, &data, &solution);
    }

    // Step 4: apply solution back to the builder.
    let apply_t0 = std::time::Instant::now();
    apply_solution(g, gf, &solution)?;
    // Debug-only: statically access-check every selected kernel and dump
    // the HIR of violators (keeps going; the graph compile's own
    // `check_accesses` gate is the enforcing one).
    if std::env::var_os("FUSION_V2_CHECK_SELECTED").is_some() {
        for node in &g.nodes {
            let crate::graph_ir::GraphNode::Kernel(k) = node else {
                continue;
            };
            if let Err(e) = crate::passes::check_accesses::check_module_accesses(
                &k.module,
                &k.param_bindings,
            ) {
                eprintln!(
                    "[fusion-v2-check] module `{}` failed ({e}); bindings={:?}\n{}",
                    k.module.name,
                    k.param_bindings,
                    crate::dump::dump_hir(&k.module)
                );
            }
        }
    }
    if options.verbose {
        eprintln!(
            "[fusion-v2] apply: {:.1} ms, nodes {} -> {}",
            apply_t0.elapsed().as_secs_f64() * 1e3,
            nodes_before,
            g.nodes.len(),
        );
    }

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

/// `verbose` one-line summary of the selected extraction (§15): how
/// many seeds survived unfused, how many fused (alt) nodes replaced
/// how many seeds, and the estimated runtime of the selection.
fn dump_extraction(gf: &GraphFuser, data: &ExtractionData, solution: &ExtractionSolution) {
    let seeds_kept = solution
        .nodes
        .iter()
        .filter(|n| n.0 < gf.seed_node_count)
        .count();
    let fused = solution.nodes.len() - seeds_kept;
    let seeds_fused = gf.seed_node_count - seeds_kept;
    let selected_runtime_units: i64 = solution
        .nodes
        .iter()
        .map(|n| data.costs[n.0])
        .filter(|c| !c.is_failure())
        .map(|c| c.runtime_units)
        .fold(0i64, i64::saturating_add);
    eprintln!(
        "[fusion-v2] extraction: {fused} fused node(s) replace {seeds_fused}/{} seeds \
         ({seeds_kept} seeds kept unfused; selected {}/{} alt-graph nodes, est. runtime \
         {selected_runtime_units} units)",
        gf.seed_node_count,
        solution.nodes.len(),
        gf.nodes.len(),
    );
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
                // Lowering panics on a synthesized module map to the
                // failure sentinel, so the extractor excludes the
                // broken candidate.
                let cost = manager
                    .cost_of(hash, &k.module, &k.param_bindings)
                    .unwrap_or_else(|e| {
                        if std::env::var_os("FUSION_V2_DEBUG").is_some() {
                            eprintln!(
                                "[fusion-v2-debug] cost_of `{}` failed (block_hint={:?}, \
                                 params={:?}, bindings={:?}): {e}",
                                k.module.name,
                                k.module.builder.block_hint(),
                                k.module.builder.params(),
                                k.param_bindings,
                            );
                            if crate::passes::fusion_v2::fusions::debug_reject_level() >= 2 {
                                eprintln!("{}", crate::dump::dump_hir(&k.module));
                            }
                        }
                        GraphNodeCost::FAILED
                    });
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
