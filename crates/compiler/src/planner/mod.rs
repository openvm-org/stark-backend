//! Memory & stream planner.
//!
//! Picks an execution order for graph nodes and a byte offset per buffer
//! on the target device that minimizes execution time under an optional
//! memory + concurrency budget.
//!
//! Backends are selected via [`SchedulerMode`]:
//!
//! - [`SchedulerMode::CpSat`] — joint CP-SAT solve, feature-gated behind `planner-ortools`.
//! - [`SchedulerMode::ListV1`] — profile-guided beam-search list scheduler with depth-`k`
//!   look-ahead. Assigns each node to one of `params.max_concurrency` streams and inserts `WaitOn`
//!   sync instructions.
//! - [`SchedulerMode::ListV2`] — profile-guided persistent-beam list scheduler over
//!   [`AbstractTimingGraph`]; see [`ListSchedulerV2`].
//!
//! Both list schedulers require per-node timings in ms (`params.node_times`);
//! an empty vec is treated as uniform `1.0` for bootstrapping the first
//! compile — subsequent compiles feed timings from
//! [`crate::graph_exe::GraphExe::collect_graph_info`].
//!
//! Feature-gated behind `planner`.

use std::collections::{BTreeMap, HashMap};

use crate::{
    graph_ir::{BufId, BufInfo, DeviceType, GraphBuilder},
    ir::VarId,
};

pub mod abstract_timing;
#[cfg(feature = "planner-ortools")]
pub mod cpsat;
mod ctx;
pub mod list_v1;
pub mod list_v2;
mod plan;
pub mod validate;

#[cfg(feature = "planner-ortools")]
pub use abstract_timing::plan_cpsat_v2;
pub use abstract_timing::{
    load_abstract_timing_graph, perf_est, plan_list_v1_v2, plan_v2, AbstractTimingGraph,
    LoadError as AbstractLoadError, PerfEst, PlanFn,
};
pub use ctx::{
    access_from_node, align_up, eval_size, propagate_alias_offsets, NodeAccess, PlanCtx, PlanError,
};
pub use list_v1::ListSchedulerV1;
pub use plan::{StreamInstr, StreamMemoryPlan};
pub use validate::{validate_plan, ValidationError};

/// Memory-planner backend selector. See [`plan_raw`].
#[derive(Debug, Clone)]
pub enum SchedulerMode {
    /// CP-SAT joint schedule / packing solve, wall-time cap `max_secs`.
    /// Only available with the `planner-ortools` feature.
    #[cfg(feature = "planner-ortools")]
    CpSat { max_secs: f64 },
    /// Profile-guided list scheduler with depth-`k` beam look-ahead.
    /// Consumes per-node timings from `params.node_times`.
    ListV1 { params: ListSchedulerV1 },
    /// Profile-guided list scheduler v2 (persistent-beam, parallel
    /// fan-out). Consumes per-node timings from `params.node_times`
    /// (length must equal the input `nodes.len()` at plan time,
    /// typically populated from a prior
    /// [`crate::graph_exe::GraphExe::collect_graph_info`]).
    ListV2 { params: ListSchedulerV2 },
}

/// Tunables for [`SchedulerMode::ListV2`].
#[derive(Debug, Clone)]
pub struct ListSchedulerV2 {
    pub num_streams: usize,
    pub max_memory_bound: usize,
    pub num_beams: usize,
    pub beam_depth: usize,
    pub frontier_cap: usize,
    pub w_m: f64,
    pub w_t: f64,
    pub w_c: f64,
    /// Per-node runtime cost in milliseconds. Length must equal the
    /// `nodes.len()` at plan time. Empty means "not profiled yet" —
    /// [`plan_raw`] treats an empty vec as uniform 1.0 like v1 so the
    /// scheduler can still return a plan on the first compile.
    pub node_times: Vec<f64>,
}

impl Default for ListSchedulerV2 {
    fn default() -> Self {
        Self {
            num_streams: 8,
            // Effectively unlimited pool: v2 gates candidates against
            // this via a hard `f64::INFINITY` cost. Using `usize::MAX`
            // would wrap `m_bound as i64` to `-1` and reject every
            // candidate — plan_v2 exits with an empty `ready_queue`
            // and a partial schedule.
            max_memory_bound: i64::MAX as usize,
            // Pure-greedy defaults sized for the fractional-sumcheck
            // graphs (~4000 nodes, 8 streams). `num_beams=1`,
            // `beam_depth=1`, `frontier_cap=1` — beam search adds
            // cost quadratic in beam parameters × frontier and the
            // observed win over greedy on these graphs is modest.
            num_beams: 1,
            beam_depth: 1,
            frontier_cap: 1,
            w_m: 1.0,
            w_t: 1.0,
            w_c: -2.0,
            node_times: Vec::new(),
        }
    }
}

impl Default for SchedulerMode {
    fn default() -> Self {
        SchedulerMode::ListV1 {
            params: ListSchedulerV1::default(),
        }
    }
}

/// Jointly plans execution order and buffer offsets on `device`.
///
/// Buffers whose device does not match `device` are ignored (they still
/// affect scheduling through their reads/writes but do not contribute to
/// the packed memory pool).
///
/// Threads `graph.aliases` through so buffers renamed by
/// `passes::restore_ssa` share a pool slot with their canonical.
pub fn plan(
    graph: &GraphBuilder,
    env: &BTreeMap<VarId, i64>,
    device: DeviceType,
) -> Result<StreamMemoryPlan, PlanError> {
    let nodes: Vec<NodeAccess> = graph.nodes.iter().map(access_from_node).collect();
    plan_raw(
        &graph.bufs,
        &nodes,
        env,
        device,
        &[],
        &graph.aliases,
        &SchedulerMode::default(),
    )
}

/// Plans over an explicit `(bufs, nodes)` view. See [`plan`] for the
/// common case where accesses come from a [`GraphBuilder`]; this entry
/// point lets [`crate::graph_exe::GraphCompiler`] inject synthetic
/// per-kernel scratch buffers into the model.
///
/// `pin` lists buffers whose lifetime is pinned to the end of the program.
///
/// `aliases` may be empty (no aliases) or the same length as `bufs`:
/// `aliases[b] = Some(parent)` means `b` and `parent` must share a pool
/// slot.
pub fn plan_raw(
    bufs: &[BufInfo],
    nodes: &[NodeAccess],
    env: &BTreeMap<VarId, i64>,
    device: DeviceType,
    pin: &[crate::graph_ir::BufId],
    aliases: &[Option<crate::graph_ir::BufId>],
    scheduler: &SchedulerMode,
) -> Result<StreamMemoryPlan, PlanError> {
    let ctx = PlanCtx::build_with_aliases(bufs, nodes, env, device, pin, aliases)?;
    if ctx.n_nodes == 0 {
        return Ok(StreamMemoryPlan {
            instructions: Vec::new(),
            stream: Vec::new(),
            record_event: Vec::new(),
            offsets: vec![None; ctx.n_bufs],
            peak_bytes: 0,
            num_streams: 1,
            num_events: 0,
        });
    }
    let canon = ctx.canon.clone();
    let mut plan = match scheduler {
        #[cfg(feature = "planner-ortools")]
        SchedulerMode::CpSat { max_secs } => cpsat::plan_cpsat(bufs, &ctx, *max_secs),
        // Profile-guided list_v1: uses `params.node_times` from a prior
        // `collect_graph_info`. Empty `node_times` degrades to uniform
        // 1.0 for bootstrapping the first compile (chain-depth priority
        // matches the true critical path when it's a long serial chain
        // of launch-latency-bound kernels).
        SchedulerMode::ListV1 { params } => {
            let node_times = params.node_times.clone();
            params.clone().schedule(ctx, move |v| {
                node_times.get(v).copied().unwrap_or(1.0).max(0.0)
            })
        }
        SchedulerMode::ListV2 { params } => {
            // NOTE: [`list_v2::plan_v2`]'s dep model assumes SSA (each
            // BufId has ≤1 writer). Blackbox kernels with
            // `carried_outputs` violate that — they read+write the
            // same BufId, and long carry chains produce multi-writer
            // buffers that make `plan_v2` loop re-committing already-
            // scheduled nodes (see `LIST_V2_TRACE=1`). Until the
            // dep model is fixed to consume the WAW/WAR edges
            // [`PlanCtx::edges`] generates, we route the
            // profile-guided timings through [`ListSchedulerV1`] —
            // same `node_times` input, same critical-path priority,
            // proven correctness on in-place graphs, sub-second plan
            // times. `CC_LIST_V2_STRICT=1` forces the raw v2 beam
            // solver (mainly for planner testing on synthetic ATGs
            // that satisfy the SSA invariant).
            let strict = std::env::var("CC_LIST_V2_STRICT").is_ok();
            if strict {
                let atg = build_atg_for_plan_raw(bufs, nodes, &ctx, &params.node_times);
                list_v2::plan_v2(
                    &atg,
                    params.num_streams,
                    params.max_memory_bound,
                    params.num_beams,
                    params.beam_depth,
                    params.frontier_cap,
                    params.w_m,
                    params.w_t,
                    params.w_c,
                )
            } else {
                let params_v1 = ListSchedulerV1 {
                    max_concurrency: params.num_streams as u32,
                    node_times: params.node_times.clone(),
                    ..ListSchedulerV1::default()
                };
                let node_times = params_v1.node_times.clone();
                params_v1.schedule(ctx, move |v| {
                    node_times.get(v).copied().unwrap_or(1.0).max(0.0)
                })
            }
        }
    }?;
    // Backends assigned offsets only for canonical entries — alias
    // members need to inherit the same slot so mutating blackbox
    // closures and downstream readers hit the same pool address.
    propagate_alias_offsets(&mut plan.offsets, &canon);
    Ok(plan)
}

/// Build an [`AbstractTimingGraph`] view matching [`plan_raw`]'s
/// `(bufs, nodes)` inputs, for [`SchedulerMode::ListV2`] dispatch.
///
/// - **Inputs**: buffers with no writer in the current schedule (matches
///   `GraphBuilder::input_bufs`).
/// - **Outputs**: pinned buffers with at least one writer (`ctx.pinned` excluding inputs).
/// - **Timings**: `node_times` verbatim when its length equals `nodes.len()`; else uniform `1.0`
///   (first-compile bootstrapping).
fn build_atg_for_plan_raw(
    bufs: &[BufInfo],
    nodes: &[NodeAccess],
    ctx: &PlanCtx,
    node_times: &[f64],
) -> AbstractTimingGraph {
    let num_nodes = nodes.len();
    let mut buf_users: HashMap<BufId, Vec<usize>> = HashMap::new();
    let mut buf_producers: HashMap<BufId, Vec<usize>> = HashMap::new();
    let mut node_consumes: Vec<Vec<BufId>> = vec![Vec::new(); num_nodes];
    let mut node_produces: Vec<Vec<BufId>> = vec![Vec::new(); num_nodes];
    // Canonicalize BufIds through the alias table so aliased buffers
    // (post-`restore_ssa`) map to the same key throughout: they must
    // share a pool slot, and `list_v2`'s interference/schedule logic
    // is BufId-keyed. Without this the aliased pair would appear as
    // two distinct buffers, get non-overlapping offsets that the
    // downstream `propagate_alias_offsets` then collapses onto the
    // canonical's — a data race between disjoint canonicals aliased
    // to the same slot.
    let canonize = |b: BufId| -> BufId { BufId(ctx.canon[b.0]) };
    for (v, na) in nodes.iter().enumerate() {
        for &b in &na.reads {
            let cb = canonize(b);
            buf_users.entry(cb).or_default().push(v);
            node_consumes[v].push(cb);
        }
        for &b in &na.writes {
            let cb = canonize(b);
            buf_users.entry(cb).or_default().push(v);
            buf_producers.entry(cb).or_default().push(v);
            node_produces[v].push(cb);
        }
    }
    for users in buf_users.values_mut() {
        users.sort_unstable();
        users.dedup();
    }
    for prods in buf_producers.values_mut() {
        prods.sort_unstable();
        prods.dedup();
    }
    for c in node_consumes.iter_mut() {
        c.sort_unstable_by_key(|b| b.0);
        c.dedup();
    }
    for p in node_produces.iter_mut() {
        p.sort_unstable_by_key(|b| b.0);
        p.dedup();
    }
    let mut inputs: Vec<BufId> = (0..bufs.len())
        .filter(|&b| ctx.canon[b] == b && ctx.writers[b].is_empty())
        .map(BufId)
        .collect();
    inputs.sort_unstable_by_key(|b| b.0);
    inputs.dedup();
    let mut outputs: Vec<BufId> = (0..bufs.len())
        .filter(|&b| ctx.pinned[b] && ctx.canon[b] == b && !ctx.writers[b].is_empty())
        .map(BufId)
        .collect();
    outputs.sort_unstable_by_key(|b| b.0);
    outputs.dedup();
    let inital_ready_nodes: Vec<usize> = (0..num_nodes)
        .filter(|&v| {
            node_consumes[v].iter().all(|bid| {
                buf_producers
                    .get(bid)
                    .map(|ps| ps.iter().all(|&p| p == v))
                    .unwrap_or(true)
            })
        })
        .collect();
    let node_times_vec: Vec<f64> = if node_times.len() == num_nodes {
        node_times.to_vec()
    } else {
        vec![1.0; num_nodes]
    };
    AbstractTimingGraph {
        num_nodes,
        buf_info: bufs.to_vec(),
        node_times: node_times_vec,
        inputs,
        outputs,
        buf_users,
        buf_producers,
        node_consumes,
        node_produces,
        inital_ready_nodes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph_ir::{BufInfo, DeviceType, GraphBuilder},
        quast::Quast,
    };

    fn buf(g: &mut GraphBuilder, name: &str, size: i64) -> BufId {
        g.add_buf(BufInfo {
            name: Some(name.to_string()),
            device_type: DeviceType::Cuda(0),
            size: Quast::cst(size),
            concrete_size: size as usize,
            elem_size: 4,
        })
    }

    use crate::graph_ir::BufId;

    fn sizes(g: &GraphBuilder) -> Vec<u64> {
        g.bufs
            .iter()
            .map(|b| match &b.size {
                Quast::Const(c) => *c as u64,
                _ => panic!("test buffer must have constant size"),
            })
            .collect()
    }

    #[test]
    fn packs_disjoint_lifetimes() {
        let mut g = GraphBuilder::new();
        let a = buf(&mut g, "a", 100);
        let b = buf(&mut g, "b", 200);
        let c = buf(&mut g, "c", 300);
        g.insert_blackbox_kernel(
            "k0",
            std::iter::empty(),
            [a].into_iter(),
            std::iter::empty(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "k1",
            [a].into_iter(),
            [b].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "k2",
            [b].into_iter(),
            [c].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );

        // Pin the scheduler under test — the packing invariant this test
        // covers is scheduler-specific. `SchedulerMode::default()` is a
        // multi-stream `ListV1`, whose stream-multi guard prevents
        // pool-sharing across streams and would not produce a single-stream
        // packed plan.
        let nodes: Vec<NodeAccess> = g.nodes.iter().map(access_from_node).collect();
        let plan = plan_raw(
            &g.bufs,
            &nodes,
            &BTreeMap::new(),
            DeviceType::Cuda(0),
            &[],
            &[],
            &SchedulerMode::ListV1 {
                params: ListSchedulerV1 {
                    max_concurrency: 1,
                    ..ListSchedulerV1::default()
                },
            },
        )
        .unwrap();
        assert_eq!(plan.order(), vec![0, 1, 2]);
        // `c` has no reader and isn't pinned, so `offline_repack` conservatively
        // extends its death to `+inf` — it overlaps with every other buf.
        // Peak = a + b + c = 100 + 200 + 300 = 600.
        assert_eq!(plan.peak_bytes, 600);
        assert!(plan.offsets.iter().all(Option::is_some));
        assert_eq!(plan.num_streams, 1);
        assert!(plan
            .instructions
            .iter()
            .all(|i| matches!(i, StreamInstr::Node(_))));
        let observed = plan
            .offsets
            .iter()
            .enumerate()
            .filter_map(|(i, o)| o.map(|off| off + sizes(&g)[i]))
            .max()
            .unwrap_or(0);
        assert_eq!(observed, plan.peak_bytes);
    }

    #[test]
    fn overlapping_lifetimes_do_not_share_memory() {
        let mut g = GraphBuilder::new();
        let a = buf(&mut g, "a", 128);
        let b = buf(&mut g, "b", 256);
        let out = buf(&mut g, "out", 128);
        g.insert_blackbox_kernel(
            "produce_both",
            std::iter::empty(),
            [a, b].into_iter(),
            std::iter::empty(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "consume_both",
            [a, b].into_iter(),
            [out].into_iter(),
            [false, false].into_iter(),
            |_, _, _| {},
        );

        let plan = plan(&g, &BTreeMap::new(), DeviceType::Cuda(0)).unwrap();
        let oa = plan.offsets[a.0].unwrap();
        let ob = plan.offsets[b.0].unwrap();
        let sa = sizes(&g)[a.0];
        let sb = sizes(&g)[b.0];
        assert!(oa + sa <= ob || ob + sb <= oa, "a and b overlap in memory");
    }

    #[test]
    fn respects_symbol_assignment() {
        let mut g = GraphBuilder::new();
        let n = g.register_symbol("n");
        let a = g.add_buf(BufInfo {
            name: Some("a".into()),
            device_type: DeviceType::Cuda(0),
            size: Quast::sym(n).mul_c(4),
            concrete_size: 0,
            elem_size: 4,
        });
        g.insert_blackbox_kernel(
            "k",
            std::iter::empty(),
            [a].into_iter(),
            std::iter::empty(),
            |_, _, _| {},
        );

        let env = BTreeMap::from([(n, 64)]);
        let plan = plan(&g, &env, DeviceType::Cuda(0)).unwrap();
        assert_eq!(plan.peak_bytes, 256);
        assert_eq!(plan.offsets[a.0], Some(0));
    }

    #[test]
    fn unbound_symbol_is_reported() {
        let mut g = GraphBuilder::new();
        let n = g.register_symbol("n");
        let _a = g.add_buf(BufInfo {
            name: Some("a".into()),
            device_type: DeviceType::Cuda(0),
            size: Quast::sym(n),
            concrete_size: 0,
            elem_size: 4,
        });
        let err = plan(&g, &BTreeMap::new(), DeviceType::Cuda(0)).unwrap_err();
        assert!(matches!(err, PlanError::UnboundSizeSymbol { sym, .. } if sym == n));
    }
}
