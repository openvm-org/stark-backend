//! Abstract, timing-annotated view of a compiled graph, and a
//! [`plan_v2`] entry point that consumes it. Purpose: swap in
//! candidate scheduling / memory-planning algorithms without having to
//! carry a full `GraphBuilder` through the planner APIs. Feeds
//! `AbstractTimingGraph` in, gets a [`StreamMemoryPlan`] out — same
//! output surface as the existing [`super::plan_raw`].

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

use super::{
    heuristic::plan_heuristic, list_v1::ListSchedulerV1, NodeAccess, PlanCtx, PlanError,
    SchedulerMode, StreamInstr, StreamMemoryPlan,
};
use crate::{
    graph_info::GraphInfo,
    graph_ir::{BufId, BufInfo, DeviceType, GraphBuilder},
    graph_serializer::SerializableGraphBuilder,
    ir::VarId,
};

/// A simplified DAG view of a compiled graph, suitable for testing
/// candidate scheduling algorithms in isolation.
///
/// Each node is one scheduling unit (kernel dispatch, memcpy, etc.),
/// carrying a measured cost. Each directed edge represents a dataflow
/// dependency labelled with the [`BufId`] transferred along it — that
/// BufId indexes into [`Self::buf_info`], so the plan produced from an
/// `AbstractTimingGraph` maps back to the source graph's buffer table
/// directly.
///
/// The struct is intentionally minimal: it drops per-node
/// symbolic-shape info and side-effect metadata, keeping just enough
/// state (node adjacency, per-edge BufId, per-buffer size, per-node
/// cost) for a scheduler to reason about lifetimes + runtime.
#[derive(Debug, Clone)]
pub struct AbstractTimingGraph {
    pub num_nodes: usize,

    /// Full buffer table, indexed by [`BufId::0`]. Includes buffers
    /// that don't participate in any edge (external inputs, dead
    /// outputs) — the memory planner still needs their sizes.
    pub buf_info: Vec<BufInfo>,
    /// Per-node dispatch cost in milliseconds, indexed by node id.
    /// Typically populated from [`GraphInfo::nodes`]'s `mean_ms`.
    pub node_times: Vec<f64>,
    /// Buffers exposed as *graph inputs* — bytes supplied by the
    /// caller, never written by any node in the DAG. Mirrors
    /// [`GraphBuilder::input_bufs`] on the source builder. Kept
    /// explicit because a graph input has no producer edge, so its
    /// presence isn't derivable from [`Self::edges`] alone.
    pub inputs: Vec<BufId>,
    /// Buffers exposed as *graph outputs* — bytes observed by the
    /// caller after the schedule finishes. Mirrors
    /// [`GraphBuilder::output_bufs`] on the source builder. A
    /// downstream memory planner should pin these so their memory
    /// survives past the last dispatch.
    pub outputs: Vec<BufId>,

    /// Per-buffer node-user index: for each [`BufId`], the list of
    /// node indices that read or write it (deduplicated, sorted
    /// ascending). Buffers that are never touched by any node are
    /// absent from the map. Complements the edge-only representation
    /// by making external inputs / dead outputs discoverable.
    pub buf_users: HashMap<BufId, Vec<usize>>,

    pub buf_producers: HashMap<BufId, Vec<usize>>,

    /// map from node to the BufIds it consumes
    pub node_consumes: Vec<Vec<BufId>>,
    /// map from node to the BufIds it produces
    pub node_produces: Vec<Vec<BufId>>,

    pub inital_ready_nodes: Vec<usize>,
}

impl AbstractTimingGraph {
    /// Build an `AbstractTimingGraph` from a compiled `GraphBuilder`
    /// (post-fuse+dce; the same graph the exe's `ExeNode` list mirrors
    /// index-by-index) and a `GraphInfo` collected off that exe.
    ///
    /// # Panics
    /// Panics if `graph.nodes.len() != info.nodes.len()` — the two
    /// must be index-aligned for per-node timings to attach.
    pub fn from_graph_and_info(graph: &GraphBuilder, info: &GraphInfo) -> Self {
        assert_eq!(
            graph.nodes.len(),
            info.nodes.len(),
            "AbstractTimingGraph::from_graph_and_info: graph has {} nodes but info carries {}",
            graph.nodes.len(),
            info.nodes.len(),
        );
        let num_nodes = graph.nodes.len();

        // Producer per buffer: the most-recent writer's node index.
        // Node insertion order is a valid topological order for
        // graphs that already went through `restore_ssa` (each buffer
        // has ≤1 writer, and if it's rewritten in-place the same node
        // both reads and writes), so a single pass suffices.
        let mut buf_users: HashMap<BufId, Vec<usize>> = HashMap::new();
        let mut buf_producers: HashMap<BufId, Vec<usize>> = HashMap::new();
        let mut node_consumes: Vec<Vec<BufId>> = vec![Vec::new(); num_nodes];
        let mut node_produces: Vec<Vec<BufId>> = vec![Vec::new(); num_nodes];

        for (v, node) in graph.nodes.iter().enumerate() {
            let (reads, writes) = graph.node_reads_writes(node);
            for &(buf, _modifies) in &reads {
                buf_users.entry(buf).or_default().push(v);
                node_consumes[v].push(buf);
            }
            for &buf in &writes {
                buf_users.entry(buf).or_default().push(v);
                buf_producers.entry(buf).or_default().push(v);
                node_produces[v].push(buf);
            }
        }

        // Dedup / sort for deterministic iteration and O(log n) lookups.
        for users in buf_users.values_mut() {
            users.sort_unstable();
            users.dedup();
        }
        for prods in buf_producers.values_mut() {
            prods.sort_unstable();
            prods.dedup();
        }
        for consumes in node_consumes.iter_mut() {
            consumes.sort_unstable_by_key(|b| b.0);
            consumes.dedup();
        }
        for produces in node_produces.iter_mut() {
            produces.sort_unstable_by_key(|b| b.0);
            produces.dedup();
        }

        // A node is initially ready if every buf it consumes either has
        // no producer (graph input) or is produced by itself (in-place
        // read+write). Any cross-node dep prevents it from starting.
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

        let node_times: Vec<f64> = info.nodes.iter().map(|nt| nt.mean_ms).collect();

        let inputs = graph.input_bufs().to_vec();
        let outputs = graph.output_bufs().to_vec();

        Self {
            num_nodes,
            buf_info: graph.bufs.clone(),
            node_times,
            inputs,
            outputs,
            buf_users,
            buf_producers,
            node_consumes,
            node_produces,
            inital_ready_nodes,
        }
    }

    /// Per-node reads/writes reconstructed from
    /// [`Self::node_consumes`] / [`Self::node_produces`] for the
    /// [`PlanCtx`] pipeline.
    pub fn per_node_access(&self) -> Vec<NodeAccess> {
        (0..self.num_nodes)
            .map(|v| NodeAccess {
                reads: self.node_consumes[v].clone(),
                writes: self.node_produces[v].clone(),
            })
            .collect()
    }

    /// Infer the target CUDA device from [`Self::buf_info`]. Post-
    /// `compile` graphs have a uniform device; falls back to
    /// `Cuda(0)` if no CUDA buffer is present.
    fn infer_device(&self) -> DeviceType {
        self.buf_info
            .iter()
            .map(|b| b.device_type)
            .find(|d| matches!(d, DeviceType::Cuda(_)))
            .unwrap_or(DeviceType::Cuda(0))
    }
}

/// Signature every scheduler adapter conforms to. Different candidate
/// scheduling algorithms plug into this shape so benchmarks can drive
/// them uniformly (see [`perf_est`] + the benchmark example in
/// `examples/bench_planners.rs`).
pub type PlanFn = fn(&AbstractTimingGraph) -> Result<StreamMemoryPlan, PlanError>;

/// Build a `PlanCtx` from an `AbstractTimingGraph`. Infers the target
/// device from [`AbstractTimingGraph::buf_info`], threads
/// [`AbstractTimingGraph::outputs`] as the pin list, and passes an
/// empty env (buffer sizes on captured graphs are already concrete —
/// see [`BufInfo::concrete_size`]).
fn build_ctx(graph: &AbstractTimingGraph) -> Result<(PlanCtx, Vec<NodeAccess>), PlanError> {
    let nodes = graph.per_node_access();
    let env: BTreeMap<VarId, i64> = BTreeMap::new();
    let ctx = PlanCtx::build_with_aliases(
        &graph.buf_info,
        &nodes,
        &env,
        graph.infer_device(),
        &graph.outputs,
        &[],
    )?;
    Ok((ctx, nodes))
}

/// Heuristic single-stream packer over an `AbstractTimingGraph`. See
/// [`super::heuristic::plan_heuristic`].
pub fn plan_heuristic_v2(graph: &AbstractTimingGraph) -> Result<StreamMemoryPlan, PlanError> {
    let (ctx, _nodes) = build_ctx(graph)?;
    if ctx.n_nodes == 0 {
        return Ok(empty_plan(ctx.n_bufs));
    }
    let canon = ctx.canon.clone();
    let mut plan = plan_heuristic(&graph.buf_info, &ctx)?;
    super::propagate_alias_offsets(&mut plan.offsets, &canon);
    Ok(plan)
}

/// CP-SAT single-stream packer over an `AbstractTimingGraph`. Feature-
/// gated behind `planner-ortools`. `max_secs` bounds the CP-SAT solve
/// per-call.
#[cfg(feature = "planner-ortools")]
pub fn plan_cpsat_v2(
    graph: &AbstractTimingGraph,
    max_secs: f64,
) -> Result<StreamMemoryPlan, PlanError> {
    let (ctx, _nodes) = build_ctx(graph)?;
    if ctx.n_nodes == 0 {
        return Ok(empty_plan(ctx.n_bufs));
    }
    let canon = ctx.canon.clone();
    let mut plan = super::cpsat::plan_cpsat(&graph.buf_info, &ctx, max_secs)?;
    super::propagate_alias_offsets(&mut plan.offsets, &canon);
    Ok(plan)
}

/// `ListSchedulerV1` variant over an `AbstractTimingGraph`. `params`
/// controls the beam-search knobs and, crucially, `max_concurrency`
/// (the number of concurrent streams the scheduler may assign nodes
/// to). Node cost is read from [`AbstractTimingGraph::node_times`].
pub fn plan_list_v1_v2(
    graph: &AbstractTimingGraph,
    params: ListSchedulerV1,
) -> Result<StreamMemoryPlan, PlanError> {
    let (ctx, _nodes) = build_ctx(graph)?;
    if ctx.n_nodes == 0 {
        return Ok(empty_plan(ctx.n_bufs));
    }
    let canon = ctx.canon.clone();
    let times = graph.node_times.clone();
    let mut plan = params.schedule(ctx, move |n| times[n])?;
    super::propagate_alias_offsets(&mut plan.offsets, &canon);
    Ok(plan)
}

fn empty_plan(n_bufs: usize) -> StreamMemoryPlan {
    StreamMemoryPlan {
        instructions: Vec::new(),
        stream: Vec::new(),
        record_event: Vec::new(),
        offsets: vec![None; n_bufs],
        peak_bytes: 0,
        num_streams: 1,
        num_events: 0,
    }
}

/// Estimated cost of running a `StreamMemoryPlan`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerfEst {
    /// Wall-clock time in milliseconds under the perfect-parallel
    /// stream model (see [`perf_est`]).
    pub time: f64,
    /// Peak byte-count across the packed pool, straight from
    /// [`StreamMemoryPlan::peak_bytes`].
    pub peak_bytes: u64,
}

/// Estimate the execution time of `plan` under a perfectly-parallel
/// stream model.
///
/// # Model
///
/// Each stream advances an independent clock; each node dispatched on
/// stream `s` bumps that clock by [`AbstractTimingGraph::node_times`]
/// for that node. A `WaitOn(s, e)` instruction *only* forces stream
/// `s` to catch up to event `e`'s recorded completion time — the
/// underlying stream itself never blocks on the host or contends for
/// GPU resources.
///
/// The end-of-plan clock max across streams is the reported `time`;
/// with a single stream this degenerates to a plain sum of node
/// times.
pub fn perf_est(graph: &AbstractTimingGraph, plan: &StreamMemoryPlan) -> PerfEst {
    let n_streams = plan.num_streams as usize;
    let n_events = plan.num_events as usize;
    let mut stream_time = vec![0.0f64; n_streams.max(1)];
    let mut event_time = vec![0.0f64; n_events];
    for instr in &plan.instructions {
        match *instr {
            StreamInstr::Node(node_idx) => {
                let s = plan.stream[node_idx] as usize;
                stream_time[s] += graph.node_times.get(node_idx).copied().unwrap_or(0.0);
                if let Some(e) = plan.record_event[node_idx] {
                    event_time[e as usize] = stream_time[s];
                }
            }
            StreamInstr::WaitOn(s, e) => {
                let s = s as usize;
                if let Some(t) = event_time.get(e).copied() {
                    if t > stream_time[s] {
                        stream_time[s] = t;
                    }
                }
            }
        }
    }
    let time = stream_time.iter().copied().fold(0.0f64, f64::max);
    PerfEst {
        time,
        peak_bytes: plan.peak_bytes,
    }
}

/// Legacy `plan_v2` entry point retained for existing callers.
/// Delegates to [`plan_heuristic_v2`] regardless of the passed
/// scheduler. New code should call the specific adapter directly.
#[doc(hidden)]
pub fn plan_v2(
    graph: &AbstractTimingGraph,
    _env: &BTreeMap<VarId, i64>,
    _device: DeviceType,
    _scheduler: &SchedulerMode,
) -> Result<StreamMemoryPlan, PlanError> {
    plan_heuristic_v2(graph)
}

/// Errors returned by [`load_abstract_timing_graph`].
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("read {path:?}: {source}")]
    Read {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("bincode: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("graph and timings node counts differ: graph={graph} timings={timings}")]
    NodeCountMismatch { graph: usize, timings: usize },
}

/// Load a serialized `SerializableGraphBuilder` (bincode) and
/// `GraphInfo` (JSON) from disk and stitch them into an
/// [`AbstractTimingGraph`]. Intended as a test-harness entry point for
/// benchmarking alternative scheduling algorithms against real
/// captured graphs.
///
/// Both inputs are read fully into memory; no GPU context is required
/// (constants come back as `HostBuf` regardless of their original
/// residency — see
/// [`SerializableGraphBuilder::into_graph_builder_offline`]).
pub fn load_abstract_timing_graph(
    graph_path: &Path,
    timing_path: &Path,
) -> Result<AbstractTimingGraph, LoadError> {
    let graph_bytes = std::fs::read(graph_path).map_err(|e| LoadError::Read {
        path: graph_path.to_path_buf(),
        source: e,
    })?;
    let ser: SerializableGraphBuilder = bincode::deserialize(&graph_bytes)?;
    let graph = ser.into_graph_builder_offline();

    let info_str = std::fs::read_to_string(timing_path).map_err(|e| LoadError::Read {
        path: timing_path.to_path_buf(),
        source: e,
    })?;
    let info: GraphInfo = serde_json::from_str(&info_str)?;

    if graph.nodes.len() != info.nodes.len() {
        return Err(LoadError::NodeCountMismatch {
            graph: graph.nodes.len(),
            timings: info.nodes.len(),
        });
    }
    Ok(AbstractTimingGraph::from_graph_and_info(&graph, &info))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph_info::{GraphInfo, NodeKind, NodeTiming},
        graph_ir::GraphBuilder,
        quast::Quast,
    };

    fn buf(g: &mut GraphBuilder, name: &str, size: i64) -> BufId {
        g.add_buf(BufInfo {
            name: Some(name.into()),
            device_type: DeviceType::Cuda(0),
            size: Quast::cst(size),
            concrete_size: size as usize,
            elem_size: 4,
        })
    }

    fn stub_info(mean_ms: &[f64]) -> GraphInfo {
        GraphInfo {
            graph_hash: [0u8; 32],
            num_warmup: 0,
            num_iters: 1,
            nodes: mean_ms
                .iter()
                .map(|&m| NodeTiming {
                    kind: NodeKind::Kernel,
                    name: String::new(),
                    mean_ms: m,
                    std_ms: 0.0,
                })
                .collect(),
            total_ms_mean: mean_ms.iter().sum(),
            total_ms_std: 0.0,
        }
    }

    #[test]
    fn from_graph_and_info_builds_linear_chain() {
        // n0 -> n1 -> n2 through buffers a and b, plus a final output c.
        let mut g = GraphBuilder::new();
        let a = buf(&mut g, "a", 16);
        let b = buf(&mut g, "b", 16);
        let c = buf(&mut g, "c", 16);
        g.register_output(c);
        g.insert_blackbox_kernel(
            "n0",
            std::iter::empty(),
            [a].into_iter(),
            std::iter::empty(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "n1",
            [a].into_iter(),
            [b].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "n2",
            [b].into_iter(),
            [c].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );

        let info = stub_info(&[0.1, 0.2, 0.3]);
        let atg = AbstractTimingGraph::from_graph_and_info(&g, &info);
        assert_eq!(atg.num_nodes, 3);
        assert_eq!(atg.buf_info.len(), 3);
        assert_eq!(atg.node_times, vec![0.1, 0.2, 0.3]);
        assert_eq!(atg.node_consumes[0], Vec::<BufId>::new());
        assert_eq!(atg.node_produces[0], vec![a]);
        assert_eq!(atg.node_consumes[1], vec![a]);
        assert_eq!(atg.node_produces[1], vec![b]);
        assert_eq!(atg.node_consumes[2], vec![b]);
        assert_eq!(atg.node_produces[2], vec![c]);
        // Only n0 has no producer dep; n1 waits on a, n2 waits on b.
        assert_eq!(atg.inital_ready_nodes, vec![0]);
        assert_eq!(atg.buf_producers[&a], vec![0]);
        assert_eq!(atg.buf_producers[&b], vec![1]);
        assert_eq!(atg.buf_producers[&c], vec![2]);
        assert!(atg.inputs.is_empty());
        assert_eq!(atg.outputs, vec![c]);
    }

    #[test]
    fn buf_users_and_node_consumes_produces_are_populated() {
        //   n0        → writes a
        //   n1: a     → writes b
        //   n2: a, b  → writes c   (double reader of a, b)
        //   n3: b     → writes d   (extra reader of b)
        let mut g = GraphBuilder::new();
        let a = buf(&mut g, "a", 16);
        let b = buf(&mut g, "b", 16);
        let c = buf(&mut g, "c", 16);
        let d = buf(&mut g, "d", 16);
        g.register_output(c);
        g.register_output(d);
        g.insert_blackbox_kernel(
            "n0",
            std::iter::empty(),
            [a].into_iter(),
            std::iter::empty(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "n1",
            [a].into_iter(),
            [b].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "n2",
            [a, b].into_iter(),
            [c].into_iter(),
            [false, false].into_iter(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "n3",
            [b].into_iter(),
            [d].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );

        let atg = AbstractTimingGraph::from_graph_and_info(&g, &stub_info(&[0.1; 4]));

        // buf_users: every node touching the buf, sorted+deduped.
        assert_eq!(atg.buf_users[&a], vec![0, 1, 2]); // written by n0, read by n1+n2
        assert_eq!(atg.buf_users[&b], vec![1, 2, 3]); // written by n1, read by n2+n3
        assert_eq!(atg.buf_users[&c], vec![2]); // written by n2, unread
        assert_eq!(atg.buf_users[&d], vec![3]); // written by n3, unread

        // node_consumes / node_produces per-node access sets.
        assert_eq!(atg.node_consumes[0], Vec::<BufId>::new());
        assert_eq!(atg.node_produces[0], vec![a]);
        assert_eq!(atg.node_consumes[1], vec![a]);
        assert_eq!(atg.node_produces[1], vec![b]);
        assert_eq!(atg.node_consumes[2], vec![a, b]);
        assert_eq!(atg.node_produces[2], vec![c]);
        assert_eq!(atg.node_consumes[3], vec![b]);
        assert_eq!(atg.node_produces[3], vec![d]);

        // buf_producers: exactly one writer per buf under SSA.
        assert_eq!(atg.buf_producers[&a], vec![0]);
        assert_eq!(atg.buf_producers[&b], vec![1]);
        assert_eq!(atg.buf_producers[&c], vec![2]);
        assert_eq!(atg.buf_producers[&d], vec![3]);
        assert_eq!(atg.inital_ready_nodes, vec![0]);
    }

    #[test]
    fn perf_est_single_stream_sums_node_times() {
        // 3 nodes all on stream 0 → time = sum of node_times.
        let mut g = GraphBuilder::new();
        let a = buf(&mut g, "a", 8);
        let b = buf(&mut g, "b", 8);
        let c = buf(&mut g, "c", 8);
        g.register_output(c);
        g.insert_blackbox_kernel(
            "n0",
            std::iter::empty(),
            [a].into_iter(),
            std::iter::empty(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "n1",
            [a].into_iter(),
            [b].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "n2",
            [b].into_iter(),
            [c].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );
        let atg = AbstractTimingGraph::from_graph_and_info(&g, &stub_info(&[1.0, 2.0, 4.0]));
        let plan = plan_heuristic_v2(&atg).unwrap();
        let est = perf_est(&atg, &plan);
        assert!((est.time - 7.0).abs() < 1e-9, "time = {}", est.time);
        assert_eq!(est.peak_bytes, plan.peak_bytes);
    }

    #[test]
    fn perf_est_two_streams_take_max_of_stream_times() {
        // Hand-built plan: nodes 0/1 on stream 0, node 2 on stream 1.
        // No WaitOn: each stream advances independently.
        //   stream 0: n0(1.0) + n1(2.0) = 3.0
        //   stream 1: n2(10.0)          = 10.0
        //   time = max(3.0, 10.0) = 10.0
        let atg = AbstractTimingGraph {
            num_nodes: 3,
            buf_info: Vec::new(),
            node_times: vec![1.0, 2.0, 10.0],
            inputs: Vec::new(),
            outputs: Vec::new(),
            buf_users: HashMap::new(),
            buf_producers: HashMap::new(),
            node_consumes: vec![Vec::new(); 3],
            node_produces: vec![Vec::new(); 3],
            inital_ready_nodes: (0..3).collect(),
        };
        let plan = StreamMemoryPlan {
            instructions: vec![
                StreamInstr::Node(0),
                StreamInstr::Node(1),
                StreamInstr::Node(2),
            ],
            stream: vec![0, 0, 1],
            record_event: vec![None; 3],
            offsets: Vec::new(),
            peak_bytes: 0,
            num_streams: 2,
            num_events: 0,
        };
        let est = perf_est(&atg, &plan);
        assert!((est.time - 10.0).abs() < 1e-9, "time = {}", est.time);
    }

    #[test]
    fn perf_est_wait_on_forces_stream_to_catch_up_to_event() {
        // n0 on stream 0 (5.0 ms) records event 0. Stream 1 waits for
        // event 0 before running n1 (1.0 ms). Stream 1's start slips
        // to 5.0, so n1 finishes at 6.0. Stream 0 ends at 5.0.
        //   time = max(5.0, 6.0) = 6.0
        let atg = AbstractTimingGraph {
            num_nodes: 2,
            buf_info: Vec::new(),
            node_times: vec![5.0, 1.0],
            inputs: Vec::new(),
            outputs: Vec::new(),
            buf_users: HashMap::new(),
            buf_producers: HashMap::new(),
            node_consumes: vec![Vec::new(); 2],
            node_produces: vec![Vec::new(); 2],
            inital_ready_nodes: (0..2).collect(),
        };
        let plan = StreamMemoryPlan {
            instructions: vec![
                StreamInstr::Node(0),
                StreamInstr::WaitOn(1, 0),
                StreamInstr::Node(1),
            ],
            stream: vec![0, 1],
            record_event: vec![Some(0), None],
            offsets: Vec::new(),
            peak_bytes: 0,
            num_streams: 2,
            num_events: 1,
        };
        let est = perf_est(&atg, &plan);
        assert!((est.time - 6.0).abs() < 1e-9, "time = {}", est.time);
    }

    #[test]
    fn plan_v2_delegates_to_planner_and_packs_disjoint_lifetimes() {
        let mut g = GraphBuilder::new();
        let a = buf(&mut g, "a", 100);
        let b = buf(&mut g, "b", 200);
        let c = buf(&mut g, "c", 300);
        g.register_output(c);
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
        let info = stub_info(&[0.1, 0.2, 0.3]);
        let atg = AbstractTimingGraph::from_graph_and_info(&g, &info);
        assert_eq!(atg.outputs, vec![c]);

        let plan = plan_v2(
            &atg,
            &BTreeMap::new(),
            DeviceType::Cuda(0),
            &SchedulerMode::Heuristic,
        )
        .expect("plan_v2");
        assert_eq!(plan.order(), vec![0, 1, 2]);
        // a and b overlap during k1 → their two slots can't merge;
        // peak ≥ a + b. Note: the edge-only representation *loses*
        // the fact that n2 writes c (no edge originates from c since
        // nothing reads it), so `per_node_access` doesn't attach a
        // write for c, and pinning c makes the planner treat it as
        // living from start to end. Peak thus lands at a + b + c
        // rather than the tighter a + b or b + c we'd get from a
        // GraphBuilder-level plan. A future non-lossy variant of
        // `AbstractTimingGraph` (adding an explicit `producers` map
        // or per-node writes list) can tighten this.
        assert!(
            plan.peak_bytes >= 100 + 200,
            "peak_bytes = {}",
            plan.peak_bytes
        );
        assert_eq!(plan.num_streams, 1);
        // `c` was registered as a graph output; the planner assigned
        // it an offset (i.e. it stays pinned in the pool).
        assert!(plan.offsets[c.0].is_some());
    }

    #[test]
    fn load_from_paths_round_trip_via_bincode_and_json() {
        // CPU-only round-trip: build a small graph, serialize the
        // `SerializableGraphBuilder` + `GraphInfo` to a tempdir, then
        // hand the paths to `load_abstract_timing_graph` and confirm
        // the reconstructed graph matches the direct one.
        //
        // This is the exact code path a scheduler-benchmarking test
        // would take on artifacts produced by
        // `bench_fractional_sumcheck_eager_vs_ir`.
        use crate::graph_serializer::SerializableGraphBuilder;

        let mut g = GraphBuilder::new();
        let a = buf(&mut g, "a", 128);
        let b = buf(&mut g, "b", 128);
        let c = buf(&mut g, "c", 128);
        g.register_output(c);
        g.insert_blackbox_kernel(
            "n0",
            std::iter::empty(),
            [a].into_iter(),
            std::iter::empty(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "n1",
            [a].into_iter(),
            [b].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "n2",
            [b].into_iter(),
            [c].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );
        let _ = (a, b);
        // Prime original_hash cache so the serializer picks it up
        // without needing a `&mut`.
        let _ = g.original_hash();
        let info = stub_info(&[1.0, 2.0, 3.0]);

        let ser = {
            // No DeviceBuf constants in this graph, so a stub ctx is
            // unused; but the API requires one. Serialize by reading
            // `builder.bufs` directly — this is what the round-trip
            // path uses in the fractional_ir bench.
            let ctx = openvm_cuda_common::stream::GpuDeviceCtx {
                device_id: 0,
                stream: openvm_cuda_common::stream::StreamGuard::new(
                    openvm_cuda_common::stream::CudaStream::new_non_blocking().unwrap(),
                ),
            };
            SerializableGraphBuilder::from_graph_builder(&g, &ctx).unwrap()
        };

        let tmp = tempfile::tempdir().unwrap();
        let graph_path = tmp.path().join("graph.bin");
        let timing_path = tmp.path().join("info.json");
        std::fs::write(&graph_path, bincode::serialize(&ser).unwrap()).unwrap();
        std::fs::write(&timing_path, serde_json::to_string(&info).unwrap()).unwrap();

        let atg = load_abstract_timing_graph(&graph_path, &timing_path).unwrap();
        assert_eq!(atg.num_nodes, 3);
        assert_eq!(atg.buf_info.len(), 3);
        assert_eq!(atg.node_times, vec![1.0, 2.0, 3.0]);
        // Same per-node access sets and dep structure as the in-memory constructor.
        let direct = AbstractTimingGraph::from_graph_and_info(&g, &info);
        assert_eq!(direct.node_consumes, atg.node_consumes);
        assert_eq!(direct.node_produces, atg.node_produces);
        assert_eq!(direct.buf_users, atg.buf_users);
        assert_eq!(direct.buf_producers, atg.buf_producers);
        assert_eq!(direct.inital_ready_nodes, atg.inital_ready_nodes);
        assert_eq!(direct.inputs, atg.inputs);
        assert_eq!(direct.outputs, atg.outputs);
        assert_eq!(atg.outputs, vec![c]);
    }

    #[test]
    #[should_panic(expected = "graph has 2 nodes but info carries 3")]
    fn from_graph_and_info_panics_on_node_count_mismatch() {
        let mut g = GraphBuilder::new();
        let a = buf(&mut g, "a", 16);
        let b = buf(&mut g, "b", 16);
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
        let info = stub_info(&[0.1, 0.2, 0.3]);
        let _ = AbstractTimingGraph::from_graph_and_info(&g, &info);
    }
}
