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


pub mod abstract_timing;
pub mod list_v1;
pub mod list_v2;
mod plan;
pub mod validate;

pub use abstract_timing::{
    access_from_node, eval_size, load_abstract_timing_graph, perf_est, plan_list_v1_v2, plan_v2,
    AbstractTimingGraph, AliasInfo, LoadError as AbstractLoadError, NodeAccess, NodeId, PerfEst,
    PlanError, PlanFn,
};
pub use list_v1::ListSchedulerV1;
pub use plan::{StreamInstr, StreamMemoryPlan};
pub use validate::{validate_plan, ValidationError};

/// Memory-planner backend selector. See [`plan_raw`].
#[derive(Debug, Clone)]
pub enum SchedulerMode {
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

/// Dispatch a [`SchedulerMode`] against an already-built
/// [`AbstractTimingGraph`]. Every backend consumes `&AbstractTimingGraph`
/// + its own tunables; this thin wrapper picks the right one.
pub fn plan(
    atg: &AbstractTimingGraph,
    scheduler: &SchedulerMode,
) -> Result<StreamMemoryPlan, PlanError> {
    if atg.num_nodes == 0 {
        return Ok(StreamMemoryPlan {
            instructions: Vec::new(),
            stream: Vec::new(),
            record_event: Vec::new(),
            offsets: vec![None; atg.n_bufs()],
            peak_bytes: 0,
            num_streams: 1,
            num_events: 0,
        });
    }
    match scheduler {
        SchedulerMode::ListV1 { params } => list_v1::plan_list_v1(atg, params),
        SchedulerMode::ListV2 { params } => list_v2::plan_v2(
            atg,
            params.num_streams,
            params.max_memory_bound,
            params.num_beams,
            params.beam_depth,
            params.frontier_cap,
            params.w_m,
            params.w_t,
            params.w_c,
        ),
    }
}


#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{
        graph_ir::{BufInfo, DeviceType, GraphBuilder},
        ir::VarId,
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

    fn atg_for_test(g: &GraphBuilder, device: DeviceType) -> AbstractTimingGraph {
        atg_for_test_with_env(g, device, &BTreeMap::new())
    }

    fn atg_for_test_with_env(
        g: &GraphBuilder,
        device: DeviceType,
        env: &BTreeMap<VarId, i64>,
    ) -> AbstractTimingGraph {
        // Evaluate symbolic sizes into `concrete_size`, matching what
        // the compile pipeline does before it plans memory.
        let mut bufs = g.bufs.clone();
        for (i, info) in bufs.iter_mut().enumerate() {
            if info.device_type == device {
                let s = eval_size(BufId(i), &info.size, env).expect("size eval");
                info.concrete_size = s.max(0) as usize;
            }
        }
        let reads: Vec<Vec<BufId>> = g
            .nodes
            .iter()
            .map(|n| access_from_node(n).reads)
            .collect();
        let writes: Vec<Vec<BufId>> = g
            .nodes
            .iter()
            .map(|n| access_from_node(n).writes)
            .collect();
        AbstractTimingGraph::from_accesses(
            bufs,
            &reads,
            &writes,
            vec![1.0; g.nodes.len()],
            device,
            g.input_bufs().to_vec(),
            g.output_bufs().to_vec(),
        )
    }

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
        let atg = atg_for_test(&g, DeviceType::Cuda(0));
        let plan = plan(
            &atg,
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
        // Every real buffer got an offset. `plan.offsets` also carries
        // an entry per synthetic ordering-edge buf (post-ATG refactor);
        // synthetics have size 0 and aren't packable so their slots are
        // `None`.
        for b in 0..g.bufs.len() {
            assert!(plan.offsets[b].is_some(), "real buf {b} missing offset");
        }
        assert_eq!(plan.num_streams, 1);
        assert!(plan
            .instructions
            .iter()
            .all(|i| matches!(i, StreamInstr::Node(_))));
        let observed = plan
            .offsets
            .iter()
            .take(g.bufs.len())
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

        let atg = atg_for_test(&g, DeviceType::Cuda(0));
        let plan = plan(&atg, &SchedulerMode::default()).unwrap();
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
        let atg = atg_for_test_with_env(&g, DeviceType::Cuda(0), &env);
        let plan = plan(&atg, &SchedulerMode::default()).unwrap();
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
        let err = eval_size(_a, &g.buf_info(_a).size, &BTreeMap::new()).unwrap_err();
        assert!(matches!(err, PlanError::UnboundSizeSymbol { sym, .. } if sym == n));
    }
}
