//! Memory & stream planner.
//!
//! Picks an execution order for graph nodes and a byte offset per buffer
//! on the target device such that:
//! - peak bytes on that device are minimized (single-stream backends), or
//! - execution time is minimized under an optional memory + concurrency budget (multi-stream list
//!   scheduler).
//!
//! Backends are selected via [`SchedulerMode`]:
//!
//! - [`SchedulerMode::CpSat`] — joint CP-SAT solve, feature-gated behind `planner-ortools`.
//! - [`SchedulerMode::Heuristic`] — solver-free three-phase heuristic (memory-aware greedy topo +
//!   BFD packing + adjacent-swap hill climb).
//! - [`SchedulerMode::ListV1`] — depth-`k` beam-search list scheduler that assigns each node to one
//!   of `max_concurrency` streams and inserts `WaitOn` sync instructions.
//!
//! The single-stream backends and the list scheduler both emit the
//! unified [`StreamMemoryPlan`]. Single-stream plans put every node on
//! stream 0 with no `WaitOn` instructions, so downstream consumers (e.g.
//! `GraphExe`) treat them identically to the pre-refactor `MemoryPlan`.
//!
//! Feature-gated behind `planner`.

use std::collections::BTreeMap;

use crate::{
    graph_ir::{BufInfo, DeviceType, GraphBuilder},
    ir::VarId,
};

#[cfg(feature = "planner-ortools")]
pub mod cpsat;
mod ctx;
pub mod heuristic;
pub mod list_v1;
mod plan;

pub use ctx::{
    access_from_node, align_up, eval_size, propagate_alias_offsets, NodeAccess, PlanCtx, PlanError,
};
pub use list_v1::ListSchedulerV1;
pub use plan::{StreamInstr, StreamMemoryPlan};

/// Memory-planner backend selector. See [`plan_raw`].
#[derive(Debug, Clone)]
pub enum SchedulerMode {
    /// CP-SAT joint schedule / packing solve, wall-time cap `max_secs`.
    /// Only available with the `planner-ortools` feature.
    #[cfg(feature = "planner-ortools")]
    CpSat { max_secs: f64 },
    /// Solver-free heuristic pipeline.
    Heuristic,
    /// Stream-aware list scheduler with depth-`k` look-ahead. Assigns
    /// each node to one of `params.max_concurrency` streams.
    ListV1 { params: ListSchedulerV1 },
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
        SchedulerMode::Heuristic => heuristic::plan_heuristic(bufs, &ctx),
        SchedulerMode::ListV1 { params } => params.clone().schedule(ctx, |_| 1.0),
    }?;
    // Backends assigned offsets only for canonical entries — alias
    // members need to inherit the same slot so mutating blackbox
    // closures and downstream readers hit the same pool address.
    propagate_alias_offsets(&mut plan.offsets, &canon);
    Ok(plan)
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
            &SchedulerMode::Heuristic,
        )
        .unwrap();
        assert_eq!(plan.order(), vec![0, 1, 2]);
        assert_eq!(plan.peak_bytes, 500);
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
            elem_size: 4,
        });
        let err = plan(&g, &BTreeMap::new(), DeviceType::Cuda(0)).unwrap_err();
        assert!(matches!(err, PlanError::UnboundSizeSymbol { sym, .. } if sym == n));
    }
}
