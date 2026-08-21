//! Abstract, timing-annotated view of a compiled graph, and a
//! [`plan_v2`] entry point that consumes it. Purpose: swap in
//! candidate scheduling / memory-planning algorithms without having to
//! carry a full `GraphBuilder` through the planner APIs. Feeds
//! `AbstractTimingGraph` in, gets a [`StreamMemoryPlan`] out — same
//! output surface as the existing [`super::plan_raw`].

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::Path,
};

use super::{list_v1::ListSchedulerV1, SchedulerMode, StreamInstr, StreamMemoryPlan};
use crate::{
    graph_info::GraphInfo,
    graph_ir::{BufId, BufInfo, DeviceType, GraphBuilder, GraphNode},
    graph_serializer::SerializableGraphBuilder,
    ir::VarId,
    quast::Quast,
};

/// Read/write set for one graph node, as seen by the planner.
#[derive(Debug, Default, Clone)]
pub struct NodeAccess {
    pub reads: Vec<BufId>,
    pub writes: Vec<BufId>,
}

/// Dense node index into an [`AbstractTimingGraph`]'s node arrays
/// (`node_times`, `node_consumes`, `node_produces`, `output_aliased`).
/// Kept as a plain `usize` alias so it can index natively; used to
/// distinguish node identifiers from unrelated `usize` values (stream
/// ids, counts, byte offsets).
pub type NodeId = usize;

/// Per-node alias flags parallel to [`AbstractTimingGraph::node_produces`]:
/// `output_aliased[v][i] == true` iff `node_produces[v][i]` is also present
/// in `node_consumes[v]` (in-place carrier — the write reuses an input
/// buffer's storage). Schedulers use this to skip "new value" bookkeeping
/// for aliased outputs whose ordering is already carried by synthetic
/// edges.
pub type AliasInfo = Vec<bool>;

/// Errors returned by the planner pipeline.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("size expression for buffer {buf:?} references unbound symbol {sym:?}")]
    UnboundSizeSymbol { buf: BufId, sym: VarId },
    #[error("size expression for buffer {buf:?} evaluates to a negative value {value}")]
    NegativeSize { buf: BufId, value: i64 },
    #[error("no legal schedule found: {0}")]
    Infeasible(String),
}

/// Builds a [`NodeAccess`] from a [`GraphNode`].
///
/// `Memcpy` and `Memset` overwrite an existing buffer slot, so both
/// declare the destination as a read too — this lets the ATG's
/// mutation predicate catch WAR hazards uniformly. `Const` is a pure
/// initial producer: no prior value of its output buffer exists to
/// depend on, so we leave it as writes-only.
pub fn access_from_node(node: &GraphNode) -> NodeAccess {
    let mut a = NodeAccess::default();
    match node {
        GraphNode::BlackboxKernel(k) => {
            a.reads.extend(k.inputs.iter().copied());
            a.writes.extend(k.carried_outputs.iter().copied());
            a.writes.extend(k.outputs.iter().copied());
        }
        GraphNode::Kernel(k) => {
            a.reads.extend(k.inputs.iter().copied());
            a.writes.extend(k.outputs.iter().copied());
        }
        GraphNode::Const(c) => a.writes.push(c.buf),
        GraphNode::Memcpy(m) => {
            a.reads.push(m.src);
            // Memcpy overwrites `m.dst`; per the invariant every
            // writer also declares its target as a read.
            a.reads.push(m.dst);
            a.writes.push(m.dst);
        }
        GraphNode::Memset(m) => {
            // Memset overwrites `m.node`; per the invariant every
            // writer also declares its target as a read.
            a.reads.push(m.node);
            a.writes.push(m.node);
        }
    }
    a
}

/// Evaluate a symbolic byte-size expression against `env`. Errors if
/// any referenced symbol is unbound or the result is negative.
pub fn eval_size(buf: BufId, size: &Quast, env: &BTreeMap<VarId, i64>) -> Result<i64, PlanError> {
    let mut syms = std::collections::BTreeSet::new();
    size.syms(&mut syms);
    for s in &syms {
        if !env.contains_key(s) {
            return Err(PlanError::UnboundSizeSymbol { buf, sym: *s });
        }
    }
    let v = size.eval(env);
    if v < 0 {
        return Err(PlanError::NegativeSize { buf, value: v });
    }
    Ok(v)
}

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
/// # Synthetic ordering-edge buffers
///
/// The source `GraphBuilder` may contain multi-writer buffers (an
/// in-place blackbox that carries the same [`BufId`] as both input and
/// output, and downstream blackboxes overwriting it again). Rather
/// than versioning writes, the ATG builder inserts *synthetic*
/// zero-size [`BufInfo`] entries that encode the [WAR / WAW / RAW]
/// ordering constraints as ordinary producer→consumer edges. Post-
/// build invariant: **every buffer in the ATG (real or synthetic) has
/// at most one writer**, so schedulers and validators can walk the
/// graph as if it were SSA.
///
/// Nodes may still both produce and consume the same real BufId (that
/// is the in-place-modify semantics kept from the source graph); the
/// synthetic edges enforce the additional ordering across multi-
/// writer chains.
#[derive(Debug, Clone)]
pub struct AbstractTimingGraph {
    pub num_nodes: usize,

    /// Full buffer table, indexed by [`BufId::0`]. Includes real
    /// buffers from the source graph (may have `concrete_size > 0`)
    /// plus synthetic ordering-edge buffers appended by the builder
    /// (each with `concrete_size = 0` and marked `synthetic[b] =
    /// true`).
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
    /// absent from the map.
    pub buf_users: HashMap<BufId, Vec<NodeId>>,

    /// Per-buffer writer index. Under the ATG's post-fake-edge
    /// invariant every value here has length ≤ 1.
    pub buf_producers: HashMap<BufId, Vec<NodeId>>,

    /// map from node to the BufIds it consumes
    pub node_consumes: Vec<Vec<BufId>>,
    /// map from node to the BufIds it produces
    pub node_produces: Vec<Vec<BufId>>,
    /// Parallel to [`Self::node_produces`]: `output_aliased[v][i]` is
    /// `true` iff `node_produces[v][i]` is also in `node_consumes[v]`
    /// (in-place carrier). See [`AliasInfo`].
    pub output_aliased: Vec<AliasInfo>,

    /// Target planner device. Used by [`Self::on_device`] to decide
    /// which buffers contribute to the packed pool.
    pub device: DeviceType,
    /// Number of "real" buffers copied from the source graph. Buffer
    /// ids `>= n_original_bufs` were appended by the builder as
    /// synthetic ordering-edge entries with `concrete_size = 0`.
    pub n_original_bufs: usize,

    pub inital_ready_nodes: Vec<NodeId>,
}

impl AbstractTimingGraph {
    /// Build an `AbstractTimingGraph` from a compiled `GraphBuilder`
    /// (post-fuse+dce; the same graph the exe's `ExeNode` list mirrors
    /// index-by-index) and a `GraphInfo` collected off that exe.
    ///
    /// The target device is inferred from the graph's buffer table
    /// (first CUDA buffer wins; falls back to `Cuda(0)`).
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
        let node_times: Vec<f64> = info.nodes.iter().map(|nt| nt.mean_ms).collect();
        let device = infer_device(&graph.bufs);
        Self::from_graph(graph, node_times, device)
    }

    /// Build an `AbstractTimingGraph` from a graph builder + per-node
    /// timings + target device, inserting synthetic zero-size ordering
    /// buffers so every synthetic buffer has exactly one writer.
    pub fn from_graph(graph: &GraphBuilder, node_times: Vec<f64>, device: DeviceType) -> Self {
        assert_eq!(
            graph.nodes.len(),
            node_times.len(),
            "AbstractTimingGraph::from_graph: {} nodes vs {} node_times",
            graph.nodes.len(),
            node_times.len(),
        );
        let reads: Vec<Vec<BufId>> = graph
            .nodes
            .iter()
            .map(|node| {
                graph
                    .node_reads_writes(node)
                    .0
                    .into_iter()
                    .map(|(b, _)| b)
                    .collect()
            })
            .collect();
        let writes: Vec<Vec<BufId>> = graph
            .nodes
            .iter()
            .map(|node| graph.node_reads_writes(node).1)
            .collect();
        Self::from_accesses(
            graph.bufs.clone(),
            &reads,
            &writes,
            node_times,
            device,
            graph.input_bufs().to_vec(),
            graph.output_bufs().to_vec(),
        )
    }

    /// Build an `AbstractTimingGraph` from raw per-node access lists.
    /// Same output shape as [`Self::from_graph`]; this variant lets
    /// tests and [`super::plan_raw`] bypass the [`GraphBuilder`]
    /// hop and construct an ATG straight from `NodeAccess`-style
    /// inputs.
    pub fn from_accesses(
        bufs: Vec<BufInfo>,
        reads: &[Vec<BufId>],
        writes: &[Vec<BufId>],
        node_times: Vec<f64>,
        device: DeviceType,
        inputs: Vec<BufId>,
        outputs: Vec<BufId>,
    ) -> Self {
        assert_eq!(
            reads.len(),
            writes.len(),
            "AbstractTimingGraph::from_accesses: reads.len() {} != writes.len() {}",
            reads.len(),
            writes.len(),
        );
        assert_eq!(
            reads.len(),
            node_times.len(),
            "AbstractTimingGraph::from_accesses: {} nodes vs {} node_times",
            reads.len(),
            node_times.len(),
        );
        let num_nodes = reads.len();

        let mut buf_info = bufs;
        let n_original_bufs = buf_info.len();

        let mut buf_users: HashMap<BufId, Vec<NodeId>> = HashMap::new();
        let mut buf_producers: HashMap<BufId, Vec<NodeId>> = HashMap::new();
        let mut node_consumes: Vec<Vec<BufId>> = vec![Vec::new(); num_nodes];
        let mut node_produces: Vec<Vec<BufId>> = vec![Vec::new(); num_nodes];
        for (v, (rs, ws)) in reads.iter().zip(writes.iter()).enumerate() {
            for &buf in rs {
                buf_users.entry(buf).or_default().push(v);
                node_consumes[v].push(buf);
            }
            for &buf in ws {
                buf_users.entry(buf).or_default().push(v);
                buf_producers.entry(buf).or_default().push(v);
                node_produces[v].push(buf);
            }
        }

        // Dedup / sort for deterministic iteration.
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

        // Classify each real buffer as "mutated" iff some node has it
        // in both its reads and its writes — the value is overwritten
        // in place. `access_from_node` enforces the "every writer of
        // a buffer also declares it as a read" invariant across every
        // GraphNode variant, so a buffer with two distinct writers
        // necessarily has at least one node with that buffer in both
        // its reads and writes (namely, each of those writers).
        let mut mutated = vec![false; n_original_bufs];
        for v in 0..num_nodes {
            for b in &node_produces[v] {
                assert!(b.0 < n_original_bufs);
                if node_consumes[v].contains(b) {
                    mutated[b.0] = true;
                }
            }
        }

        // For each mutated buffer, anchor synthetic ordering edges at
        // its writers: every writer syncs with every other user of
        // that buffer (WAR against earlier users, RAW/WAW against
        // later ones). Read-only user pairs get no edge — two readers
        // between the same pair of mutation points see identical
        // bytes and can run in parallel; serialising them into a
        // chain (the old behaviour) is over-conservative.
        //
        // Direction (`u < w` vs `u > w`) follows caller-provided node
        // order, which `GraphBuilder` guarantees respects the graph's
        // RAW/WAW/WAR dependencies — so edges always go earlier→later
        // and the emitted DAG stays acyclic.
        let mut edge_pairs: HashSet<(NodeId, NodeId)> = HashSet::new();
        for (&bid, users) in &buf_users {
            if !mutated[bid.0] {
                continue;
            }
            for &w in users {
                if !node_produces[w].contains(&bid) {
                    continue;
                }
                for &u in users {
                    if u < w {
                        edge_pairs.insert((u, w));
                    } else if u > w {
                        edge_pairs.insert((w, u));
                    }
                }
            }
        }

        // Materialize edges: one synthetic buffer per producer, whose
        // consumers are all the nodes that must sync past it. Merging
        // across original buffers is sound because a synthetic buf is
        // just an ordering signal ("wait for producer") — the reason
        // (RAW vs WAW vs WAR on whichever original buf) is irrelevant
        // downstream.
        let mut per_producer: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
        for (p, c) in edge_pairs {
            per_producer.entry(p).or_default().push(c);
        }
        for consumers in per_producer.values_mut() {
            consumers.sort_unstable();
            consumers.dedup();
        }
        for (producer, consumers) in per_producer {
            let synth_id = BufId(buf_info.len());
            buf_info.push(BufInfo {
                name: Some(format!("__ord{}", producer)),
                device_type: device,
                size: crate::quast::Quast::cst(0),
                concrete_size: 0,
                elem_size: 0,
            });
            node_produces[producer].push(synth_id);
            buf_producers.insert(synth_id, vec![producer]);
            let mut users_list = vec![producer];
            for c in consumers {
                node_consumes[c].push(synth_id);
                users_list.push(c);
            }
            users_list.sort_unstable();
            users_list.dedup();
            buf_users.insert(synth_id, users_list);
        }

        // Re-sort node_produces / node_consumes to keep BufId order
        // deterministic after the synthetic appends.
        for produces in node_produces.iter_mut() {
            produces.sort_unstable_by_key(|b| b.0);
            produces.dedup();
        }
        for consumes in node_consumes.iter_mut() {
            consumes.sort_unstable_by_key(|b| b.0);
            consumes.dedup();
        }

        // A node is initially ready iff every buffer it consumes is a
        // graph input. This runs *after* synthetic ordering-edge
        // buffers are materialized into `node_consumes`, so any node
        // with an incoming synthetic edge sees a synthetic BufId in
        // its consumes list — synthetics are never in `inputs`, so
        // such a node fails the check. Nodes that only consume graph
        // inputs (or nothing at all — Const, Memset-of-input, kernels
        // reading only graph inputs) pass.
        let inital_ready_nodes: Vec<NodeId> = (0..num_nodes)
            .filter(|&v| node_consumes[v].iter().all(|bid| inputs.contains(bid)))
            .collect();

        // Per-output alias flags: `output_aliased[v][i]` = true iff
        // `node_produces[v][i]` also appears in `node_consumes[v]`.
        // Both vectors are BufId-sorted (via `sort_unstable_by_key` +
        // `dedup` above), so we merge them in linear time — no
        // per-lookup binary search on the hot path.
        let output_aliased: Vec<AliasInfo> = (0..num_nodes)
            .map(|v| {
                let (prods, consumes) = (&node_produces[v], &node_consumes[v]);
                let mut out = Vec::with_capacity(prods.len());
                for p in prods {
                    out.push(consumes.contains(p));
                }
                out
            })
            .collect();

        Self {
            num_nodes,
            buf_info,
            node_times,
            inputs,
            outputs,
            buf_users,
            buf_producers,
            node_consumes,
            node_produces,
            output_aliased,
            device,
            n_original_bufs,
            inital_ready_nodes,
        }
    }

    /// Total node count.
    pub fn n_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Total buffer count (real + synthetic).
    pub fn n_bufs(&self) -> usize {
        self.buf_info.len()
    }

    /// Concrete byte size, or `0` for synthetic ordering-edge buffers
    /// and buffers whose size didn't monomorphize.
    pub fn size(&self, b: usize) -> i64 {
        self.buf_info[b].concrete_size as i64
    }

    /// Alignment in bytes for pool packing (`elem_size`, or `1` when
    /// `elem_size == 0`).
    pub fn align(&self, b: usize) -> u64 {
        (self.buf_info[b].elem_size as u64).max(1)
    }

    /// True iff the buffer lives on this planner's target device.
    /// Synthetics count as on-device because the builder stamped them
    /// with `device`; they're excluded from packing by [`Self::size`]
    /// == 0.
    pub fn on_device(&self, b: usize) -> bool {
        self.buf_info[b].device_type == self.device
    }

    /// True iff the buffer is a graph input or output — its lifetime
    /// spans the whole schedule.
    pub fn pinned(&self, b: usize) -> bool {
        let bid = BufId(b);
        self.inputs.iter().any(|p| *p == bid) || self.outputs.iter().any(|p| *p == bid)
    }

    /// True iff the buffer is a synthetic ordering-edge buffer
    /// inserted by [`Self::from_graph`] (BufId index ≥
    /// `n_original_bufs`).
    pub fn synthetic(&self, b: usize) -> bool {
        b >= self.n_original_bufs
    }

    /// Whether the buffer occupies a slot in the packed pool
    /// (on-device with non-zero size). Synthetics fail this because
    /// `size == 0`.
    pub fn packable(&self, b: usize) -> bool {
        self.on_device(b) && self.size(b) > 0
    }

    /// Direct precedence edges implied by the read/write sets.
    ///
    /// Only *non-mutated* buffers contribute natural edges — synthetic
    /// ordering-edge buffers (single-writer by construction) plus real
    /// buffers that have exactly one writer and no in-place carry.
    /// Mutated buffers (multi-writer, or any writer that also reads
    /// the same buf) have their ordering encoded entirely via the
    /// synthetic edges emitted by [`Self::from_accesses`], so `edges()`
    /// skips them to avoid emitting wrong-direction WAR edges to
    /// early readers or blanket writer→writer edges that ignore
    /// carrier semantics.
    ///
    /// Returns `(succ, indeg)` — successor adjacency lists (sorted,
    /// deduped) plus in-degrees.
    pub fn edges(&self) -> (Vec<Vec<NodeId>>, Vec<usize>) {
        let n = self.num_nodes;
        // Recompute the "mutated" mask lazily. Synthetics (BufId ≥
        // n_original_bufs) are single-writer by construction and
        // never appear as an in-place carrier.
        let mut mutated = vec![false; self.buf_info.len()];
        for v in 0..n {
            for b in &self.node_produces[v] {
                if b.0 < self.n_original_bufs && self.node_consumes[v].contains(b) {
                    mutated[b.0] = true;
                }
            }
        }

        let mut succ: Vec<Vec<NodeId>> = vec![Vec::new(); n];
        for (&bid, users) in &self.buf_users {
            if mutated[bid.0] {
                continue;
            }
            let Some(prods) = self.buf_producers.get(&bid) else {
                continue;
            };
            debug_assert!(
                prods.len() <= 1,
                "non-mutated buf {bid:?} has {} writers",
                prods.len(),
            );
            let Some(&p) = prods.first() else { continue };
            for &u in users {
                if u != p {
                    succ[p].push(u);
                }
            }
        }
        for sv in &mut succ {
            sv.sort_unstable();
            sv.dedup();
        }
        let mut indeg = vec![0usize; n];
        for sv in &succ {
            for &v in sv {
                indeg[v] += 1;
            }
        }
        (succ, indeg)
    }

    /// Per-node reads/writes as `usize` (BufId.0) — the shape
    /// [`super::ctx::PlanCtx::per_node_access`] returned.
    pub fn per_node_access(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let n = self.num_nodes;
        let mut writes = vec![Vec::new(); n];
        let mut reads = vec![Vec::new(); n];
        for v in 0..n {
            writes[v] = self.node_produces[v].iter().map(|b| b.0).collect();
            reads[v] = self.node_consumes[v].iter().map(|b| b.0).collect();
        }
        (writes, reads)
    }
}

/// Infer the target CUDA device from a buffer table. Post-`compile`
/// graphs have a uniform device; falls back to `Cuda(0)` if no CUDA
/// buffer is present.
fn infer_device(bufs: &[BufInfo]) -> DeviceType {
    bufs.iter()
        .map(|b| b.device_type)
        .find(|d| matches!(d, DeviceType::Cuda(_)))
        .unwrap_or(DeviceType::Cuda(0))
}

/// Signature every scheduler adapter conforms to. Different candidate
/// scheduling algorithms plug into this shape so benchmarks can drive
/// them uniformly (see [`perf_est`] + the benchmark example in
/// `examples/bench_planners.rs`).
pub type PlanFn = fn(&AbstractTimingGraph) -> Result<StreamMemoryPlan, PlanError>;

/// `ListSchedulerV1` variant over an `AbstractTimingGraph`. `params`
/// controls the beam-search knobs and, crucially, `max_concurrency`
/// (the number of concurrent streams the scheduler may assign nodes
/// to). Node cost is read from [`AbstractTimingGraph::node_times`].
pub fn plan_list_v1_v2(
    graph: &AbstractTimingGraph,
    params: ListSchedulerV1,
) -> Result<StreamMemoryPlan, PlanError> {
    if graph.num_nodes == 0 {
        return Ok(empty_plan(graph.n_bufs()));
    }
    params.schedule(graph)
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
/// Delegates to [`plan_list_v1_v2`] with `ListSchedulerV1::default()`.
/// New code should call the specific adapter directly.
#[doc(hidden)]
pub fn plan_v2(
    graph: &AbstractTimingGraph,
    _env: &BTreeMap<VarId, i64>,
    _device: DeviceType,
    _scheduler: &SchedulerMode,
) -> Result<StreamMemoryPlan, PlanError> {
    plan_list_v1_v2(graph, ListSchedulerV1::default())
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
        assert_eq!(atg.n_original_bufs, 3);
        // Linear chain through non-mutated single-writer bufs — the
        // natural `edges()` output already carries n0→n1 and n1→n2,
        // so no synthetic ordering-edge bufs are inserted.
        assert_eq!(atg.buf_info.len(), 3);
        assert!(!atg.synthetic(a.0) && !atg.synthetic(b.0) && !atg.synthetic(c.0));
        assert_eq!(atg.node_times, vec![0.1, 0.2, 0.3]);
        // Real-buf accesses still show up in node_produces/consumes;
        // synthetic bufs are appended to node_produces of the writer
        // and to node_consumes of the reader.
        let real = |bs: &[BufId]| -> Vec<BufId> {
            bs.iter()
                .copied()
                .filter(|b| b.0 < atg.n_original_bufs)
                .collect()
        };
        assert_eq!(real(&atg.node_consumes[0]), Vec::<BufId>::new());
        assert_eq!(real(&atg.node_produces[0]), vec![a]);
        assert_eq!(real(&atg.node_consumes[1]), vec![a]);
        assert_eq!(real(&atg.node_produces[1]), vec![b]);
        assert_eq!(real(&atg.node_consumes[2]), vec![b]);
        assert_eq!(real(&atg.node_produces[2]), vec![c]);
        // Only n0 has no producer dep; n1 waits on a, n2 waits on b.
        assert_eq!(atg.inital_ready_nodes, vec![0]);
        assert_eq!(atg.buf_producers[&a], vec![0]);
        assert_eq!(atg.buf_producers[&b], vec![1]);
        assert_eq!(atg.buf_producers[&c], vec![2]);
        // Every buf (real + synthetic) has ≤1 writer.
        for (_, w) in &atg.buf_producers {
            assert!(w.len() <= 1);
        }
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

        // buf_users on real buffers: every node touching the buf,
        // sorted+deduped (unchanged by fake-edge insertion).
        assert_eq!(atg.buf_users[&a], vec![0, 1, 2]); // written by n0, read by n1+n2
        assert_eq!(atg.buf_users[&b], vec![1, 2, 3]); // written by n1, read by n2+n3
        assert_eq!(atg.buf_users[&c], vec![2]); // written by n2, unread
        assert_eq!(atg.buf_users[&d], vec![3]); // written by n3, unread

        // Real-buffer view of node_consumes / node_produces (drop
        // synthetic ordering-edge appends).
        let real = |bs: &[BufId]| -> Vec<BufId> {
            bs.iter()
                .copied()
                .filter(|b| b.0 < atg.n_original_bufs)
                .collect()
        };
        assert_eq!(real(&atg.node_consumes[0]), Vec::<BufId>::new());
        assert_eq!(real(&atg.node_produces[0]), vec![a]);
        assert_eq!(real(&atg.node_consumes[1]), vec![a]);
        assert_eq!(real(&atg.node_produces[1]), vec![b]);
        assert_eq!(real(&atg.node_consumes[2]), vec![a, b]);
        assert_eq!(real(&atg.node_produces[2]), vec![c]);
        assert_eq!(real(&atg.node_consumes[3]), vec![b]);
        assert_eq!(real(&atg.node_produces[3]), vec![d]);

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
        let plan = plan_list_v1_v2(
            &atg,
            ListSchedulerV1 {
                max_concurrency: 1,
                ..ListSchedulerV1::default()
            },
        )
        .unwrap();
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
            output_aliased: vec![Vec::new(); 3],
            device: DeviceType::Cuda(0),
            n_original_bufs: 0,
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
            output_aliased: vec![Vec::new(); 2],
            device: DeviceType::Cuda(0),
            n_original_bufs: 0,
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
            &SchedulerMode::ListV1 {
                params: ListSchedulerV1::default(),
            },
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
        // `plan_v2` delegates to `plan_list_v1_v2(ListSchedulerV1::default())`,
        // whose default `max_concurrency` is 8. The graph itself is a
        // 3-node chain so only stream 0 gets used, but the plan's
        // `num_streams` reflects the scheduler's budget.
        assert_eq!(plan.num_streams, 8);
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
        assert_eq!(atg.n_original_bufs, 3);
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
    fn carry_chain_is_ssa_after_fake_edges() {
        // n0 seeds B; n1..n3 carry-mutate B in place; n4 reads B.
        // Post-refactor invariant: every ATG buf has ≤1 writer, and
        // the DAG edges from `AbstractTimingGraph::edges` walk the
        // chain in insertion order.
        let mut g = GraphBuilder::new();
        let bb = buf(&mut g, "carry", 64);
        let out = buf(&mut g, "out", 16);
        g.register_output(out);
        g.insert_blackbox_kernel(
            "seed",
            std::iter::empty(),
            [bb].into_iter(),
            std::iter::empty(),
            |_, _, _| {},
        );
        for carry_name in ["c1", "c2", "c3"] {
            g.insert_blackbox_kernel(
                carry_name,
                [bb].into_iter(),
                std::iter::empty(),
                [true].into_iter(),
                |_, _, _| {},
            );
        }
        g.insert_blackbox_kernel(
            "sink",
            [bb].into_iter(),
            [out].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );
        let atg = AbstractTimingGraph::from_graph_and_info(&g, &stub_info(&[1.0; 5]));

        // 2 real bufs + one synthetic per producer that has a
        // downstream consumer: seed→c1, c1→c2, c2→c3, c3→sink → 4
        // synthetics. n4 (sink) writes out but has no reader, so no
        // synthetic from sink.
        assert_eq!(atg.n_original_bufs, 2);
        assert_eq!(atg.buf_info.len(), 2 + 4);
        // Real carry buffer keeps its multi-writer list; ordering is
        // delegated to the synthetic edges below.
        assert_eq!(atg.buf_producers[&bb], vec![0, 1, 2, 3]);
        // Every synthetic buffer has exactly one writer.
        for b in atg.n_original_bufs..atg.buf_info.len() {
            let bid = BufId(b);
            let prods = atg
                .buf_producers
                .get(&bid)
                .expect("synthetic must have a writer");
            assert_eq!(prods.len(), 1, "synthetic {bid:?} writers = {prods:?}");
        }
        // Every user pair on the mutated buffer gets a synthetic
        // ordering edge (direction-filtered by insertion order), so
        // `edges()` yields the transitively-closed successor list for
        // each writer/reader. The multi-writer real buf is itself
        // skipped by `edges()`; the synthetics carry the ordering.
        let (succ, _) = atg.edges();
        assert_eq!(succ[0], vec![1, 2, 3, 4]);
        assert_eq!(succ[1], vec![2, 3, 4]);
        assert_eq!(succ[2], vec![3, 4]);
        assert_eq!(succ[3], vec![4]);
        assert!(succ[4].is_empty());
    }

    #[test]
    fn readers_between_mutations_run_in_parallel() {
        // Users of the mutated buffer `bb`: w0 seeds it, r1/r2 read
        // the seeded value in parallel (each writing its own scratch
        // buffer), w3 carry-mutates bb, then sink reads the new
        // value. Only `bb` is mutated — x/y/out each have exactly one
        // writer and no reader.
        //
        // The over-conservative version would insert (r1, r2), (r1,
        // sink), and (r2, sink) synthetic edges, chaining every user
        // of `bb` in insertion order. Correct behaviour: r1 and r2
        // see the same version of `bb` and need no ordering; sink
        // depends on the new version via w3, transitively through
        // w3→sink. Expected succ: 0→{1,2,3,4}, 1→{3}, 2→{3},
        // 3→{4}, 4→{}.
        let mut g = GraphBuilder::new();
        let bb = buf(&mut g, "bb", 32);
        let x = buf(&mut g, "x", 16);
        let y = buf(&mut g, "y", 16);
        let out = buf(&mut g, "out", 16);
        g.register_output(out);
        g.insert_blackbox_kernel(
            "w0",
            std::iter::empty(),
            [bb].into_iter(),
            std::iter::empty(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "r1",
            [bb].into_iter(),
            [x].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "r2",
            [bb].into_iter(),
            [y].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "w3",
            [bb].into_iter(),
            std::iter::empty(),
            [true].into_iter(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "sink",
            [bb].into_iter(),
            [out].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );
        let atg = AbstractTimingGraph::from_graph_and_info(&g, &stub_info(&[1.0; 5]));
        let (succ, _) = atg.edges();
        assert_eq!(succ[0], vec![1, 2, 3, 4]);
        assert_eq!(succ[1], vec![3]);
        assert_eq!(succ[2], vec![3]);
        assert_eq!(succ[3], vec![4]);
        assert!(succ[4].is_empty());
    }

    #[test]
    fn non_mutated_bufs_get_no_synthetics() {
        // A DAG whose ordering is entirely captured by natural
        // single-writer edges — no in-place carries, no multi-writer
        // bufs. `from_graph_and_info` should insert zero synthetic
        // ordering-edge buffers.
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
            [a, b].into_iter(),
            [c].into_iter(),
            [false, false].into_iter(),
            |_, _, _| {},
        );
        let atg = AbstractTimingGraph::from_graph_and_info(&g, &stub_info(&[0.1; 3]));
        assert_eq!(atg.buf_info.len(), atg.n_original_bufs);
        let (succ, _) = atg.edges();
        // Natural edges only: n0→n1, n0→n2, n1→n2.
        assert_eq!(succ[0], vec![1, 2]);
        assert_eq!(succ[1], vec![2]);
        assert!(succ[2].is_empty());
    }

    #[test]
    fn in_place_carry_becomes_mutated_even_with_one_writer() {
        // Single blackbox K carries the graph input `bb` in place
        // (reads + writes the same BufId). Even though `bb` has
        // only one writer, `edges()` alone would emit K → itself
        // (skipped by the `u != p` guard), leaving zero edges out
        // of K — which is fine here because no downstream reader
        // exists. The key invariant we assert: the ATG marks `bb`
        // as mutated (single-node carrier), so no wrong-direction
        // natural edge is emitted, and — since K is the only user —
        // no synthetics are needed either.
        let mut g = GraphBuilder::new();
        let bb = buf(&mut g, "carry", 32);
        g.register_input(bb);
        g.insert_blackbox_kernel(
            "K",
            [bb].into_iter(),
            std::iter::empty(),
            [true].into_iter(),
            |_, _, _| {},
        );
        let atg = AbstractTimingGraph::from_graph_and_info(&g, &stub_info(&[1.0]));
        assert_eq!(atg.n_original_bufs, 1);
        // No downstream user of bb → no synthetics needed.
        assert_eq!(atg.buf_info.len(), 1);
        let (succ, indeg) = atg.edges();
        // Zero edges: K is its own reader-and-writer of bb; there
        // are no other users.
        assert!(succ[0].is_empty());
        assert_eq!(indeg, vec![0]);
    }

    #[test]
    fn war_early_reader_then_carrier_gets_synthetic() {
        // n0 reads graph input `bb`, n1 carries `bb` (reads + writes
        // in place) *after* n0. Because n1 mutates `bb`, n0 must
        // finish reading it before n1 rewrites — a WAR edge n0 → n1.
        // No natural non-mutated buffer connects n0 to n1, so a
        // synthetic ordering-edge buf must appear.
        let mut g = GraphBuilder::new();
        let bb = buf(&mut g, "bb", 32);
        let out = buf(&mut g, "out", 16);
        g.register_input(bb);
        g.register_output(out);
        g.insert_blackbox_kernel(
            "n0",
            [bb].into_iter(),
            [out].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "n1",
            [bb].into_iter(),
            std::iter::empty(),
            [true].into_iter(),
            |_, _, _| {},
        );
        let atg = AbstractTimingGraph::from_graph_and_info(&g, &stub_info(&[1.0, 1.0]));
        assert_eq!(atg.n_original_bufs, 2);
        // One synthetic: n0 → n1 (WAR on bb, no natural edge to piggy-back on).
        assert_eq!(atg.buf_info.len(), 3);
        let (succ, _) = atg.edges();
        assert_eq!(succ[0], vec![1]);
        assert!(succ[1].is_empty());
        // n1 consumes `bb` (a graph input) but has an incoming synthetic
        // WAR edge from n0, so only n0 is initially ready — even though
        // both consume the same graph-input buffer.
        assert_eq!(atg.inital_ready_nodes, vec![0]);
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
