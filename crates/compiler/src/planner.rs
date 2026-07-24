//! Memory planner: picks an execution order for graph nodes and a byte
//! offset per buffer on the target device, such that the peak bytes needed
//! on that device are minimized.
//!
//! Two backends are exposed via [`SchedulerMode`]:
//!
//! - [`SchedulerMode::CpSat`] jointly solves for schedule and offsets as one CP-SAT model (see
//!   [`plan_cpsat`]). Optimal-ish but heavy; requires the OR-Tools install described in the crate's
//!   `Cargo.toml`. Best when the graph is small enough to prove optimality within the time budget.
//!
//! - [`SchedulerMode::Heuristic`] runs a lightweight three-phase pipeline — memory-aware greedy
//!   topological order, offline best-fit-decreasing offset assignment, then a hill-climbing local
//!   search over legal adjacent swaps with a repack after each accepted move (see
//!   [`plan_heuristic`]). No external solver. Recommended for large graphs where CP-SAT would time
//!   out.
//!
//! Both backends assume single-stream execution: nodes run sequentially and
//! only stream ordering enforces read-after-write, so a buffer is alive
//! from the earliest schedule time of any of its writers through the
//! latest schedule time of any of its readers.
//!
//! Feature-gated behind `planner` (currently both backends live in this
//! module; only CP-SAT needs OR-Tools at link time).

use std::collections::BTreeMap;

use cp_sat::{
    builder::{CpModelBuilder, IntVar, LinearExpr},
    proto::{CpSolverStatus, SatParameters},
};

use crate::{
    graph_ir::{BufId, BufInfo, DeviceType, GraphBuilder, GraphNode},
    ir::VarId,
};

/// Memory-planner backend selector. See [`plan_raw`].
#[derive(Debug, Clone)]
pub enum SchedulerMode {
    /// CP-SAT joint schedule / packing solve, with an overall wall-time
    /// cap of `max_secs`. The solver returns whatever is best when the
    /// deadline elapses (accepted as `Feasible` on top of `Optimal`).
    CpSat { max_secs: f64 },
    /// Solver-free heuristic pipeline: memory-aware greedy topological
    /// order, offline best-fit-decreasing packing, and adjacent-swap
    /// hill climb.
    Heuristic,
}

impl Default for SchedulerMode {
    fn default() -> Self {
        SchedulerMode::CpSat { max_secs: 30.0 }
    }
}

/// Read/write set for one graph node, as seen by the planner.
///
/// This is the internal shape the CP-SAT model consumes: a node's `reads`
/// contribute to `death[b]` of every read buffer, its `writes` contribute
/// to `birth[b]` of every written buffer, and (writer, reader) pairs across
/// different nodes become precedence edges. A buffer that is both read
/// and written by the same node (e.g. an in-place modify, or a per-node
/// scratch region) has `birth[b] = death[b] = t[n]` and thus a single
/// time-step lifetime.
#[derive(Debug, Default, Clone)]
pub struct NodeAccess {
    pub reads: Vec<BufId>,
    pub writes: Vec<BufId>,
}

#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("size expression for buffer {buf:?} references unbound symbol {sym:?}")]
    UnboundSizeSymbol { buf: BufId, sym: VarId },
    #[error("size expression for buffer {buf:?} evaluates to a negative value {value}")]
    NegativeSize { buf: BufId, value: i64 },
    #[error("CP-SAT returned no solution (status: {0:?})")]
    NoSolution(CpSolverStatus),
}

#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Permutation of node indices in execution order (`order[0]` runs
    /// first).
    pub order: Vec<usize>,
    /// Byte offset per [`BufId`]. `Some(off)` for buffers whose device
    /// matches the planner's target device, `None` otherwise.
    pub offsets: Vec<Option<u64>>,
    /// Peak bytes required on the target device.
    pub peak_bytes: u64,
}

/// Jointly plans execution order and buffer offsets on `device`.
///
/// Buffers whose device does not match `device` are ignored (they still
/// affect scheduling through their reads/writes but do not contribute to
/// the packed memory pool).
pub fn plan(
    graph: &GraphBuilder,
    env: &BTreeMap<VarId, i64>,
    device: DeviceType,
) -> Result<MemoryPlan, PlanError> {
    let nodes: Vec<NodeAccess> = graph.nodes.iter().map(access_from_node).collect();
    plan_raw(
        &graph.bufs,
        &nodes,
        env,
        device,
        &[],
        &SchedulerMode::default(),
    )
}

/// Plans over an explicit `(bufs, nodes)` view. Prefer [`plan`] for the
/// common case where the accesses come straight from a [`GraphBuilder`];
/// this entry point exists so [`crate::graph_exe::GraphCompiler`] can inject
/// synthetic per-kernel scratch buffers into the model.
///
/// `exclude` lists buffers that participate in scheduling (so their reads
/// and writes still induce precedence edges between nodes) but that are
/// **not** packed into the returned pool: they get no offset and their
/// sizes don't count toward `peak_bytes`. This is what
/// [`crate::graph_exe::GraphCompiler`] uses for graph inputs/outputs — the
/// caller supplies their storage at run time so no pool slot is needed.
pub fn plan_raw(
    bufs: &[BufInfo],
    nodes: &[NodeAccess],
    env: &BTreeMap<VarId, i64>,
    device: DeviceType,
    exclude: &[BufId],
    scheduler: &SchedulerMode,
) -> Result<MemoryPlan, PlanError> {
    let ctx = PlanCtx::build(bufs, nodes, env, device, exclude)?;
    if ctx.n_nodes == 0 {
        return Ok(MemoryPlan {
            order: Vec::new(),
            offsets: vec![None; ctx.n_bufs],
            peak_bytes: 0,
        });
    }
    match scheduler {
        SchedulerMode::CpSat { max_secs } => plan_cpsat(bufs, &ctx, *max_secs),
        SchedulerMode::Heuristic => plan_heuristic(bufs, &ctx),
    }
}

/// Shared preprocessing built once and consumed by both scheduler
/// backends: concrete sizes on the target device, per-buffer writer /
/// reader lists, and the in-place-modifier set for each buffer.
struct PlanCtx {
    n_nodes: usize,
    n_bufs: usize,
    /// Concrete byte sizes, `0` for off-device buffers.
    sizes: Vec<i64>,
    on_device: Vec<bool>,
    excluded: Vec<bool>,
    /// Per-buffer node indices that write to it.
    writers: Vec<Vec<usize>>,
    /// Per-buffer node indices that read it.
    readers: Vec<Vec<usize>>,
    /// Per-buffer modifiers (both read and write the buffer). Modifiers of
    /// the same buffer must be serialized in insertion order; see
    /// [`PlanCtx::edges`].
    modifiers: Vec<Vec<usize>>,
}

impl PlanCtx {
    fn build(
        bufs: &[BufInfo],
        nodes: &[NodeAccess],
        env: &BTreeMap<VarId, i64>,
        device: DeviceType,
        exclude: &[BufId],
    ) -> Result<Self, PlanError> {
        let n_nodes = nodes.len();
        let n_bufs = bufs.len();

        let mut excluded = vec![false; n_bufs];
        for &b in exclude {
            excluded[b.0] = true;
        }

        let mut sizes = vec![0i64; n_bufs];
        let mut on_device = vec![false; n_bufs];
        for (idx, info) in bufs.iter().enumerate() {
            if info.device_type == device {
                on_device[idx] = true;
                sizes[idx] = eval_size(BufId(idx), &info.size, env)?;
            }
        }

        let writes: Vec<Vec<usize>> = nodes
            .iter()
            .map(|a| a.writes.iter().map(|b| b.0).collect())
            .collect();
        let reads: Vec<Vec<usize>> = nodes
            .iter()
            .map(|a| a.reads.iter().map(|b| b.0).collect())
            .collect();

        let mut writers: Vec<Vec<usize>> = vec![vec![]; n_bufs];
        let mut readers: Vec<Vec<usize>> = vec![vec![]; n_bufs];
        for n in 0..n_nodes {
            for &b in &writes[n] {
                writers[b].push(n);
            }
            for &b in &reads[n] {
                readers[b].push(n);
            }
        }
        let modifiers: Vec<Vec<usize>> = (0..n_bufs)
            .map(|b| {
                let write_set: std::collections::BTreeSet<usize> = writes
                    .iter()
                    .enumerate()
                    .filter_map(|(n, bufs)| bufs.contains(&b).then_some(n))
                    .collect();
                reads
                    .iter()
                    .enumerate()
                    .filter_map(|(n, bufs)| {
                        (bufs.contains(&b) && write_set.contains(&n)).then_some(n)
                    })
                    .collect()
            })
            .collect();

        Ok(Self {
            n_nodes,
            n_bufs,
            sizes,
            on_device,
            excluded,
            writers,
            readers,
            modifiers,
        })
    }

    /// A packable buffer occupies a slot in the returned plan (on the
    /// target device, non-zero, not excluded).
    fn packable(&self, b: usize) -> bool {
        self.on_device[b] && self.sizes[b] > 0 && !self.excluded[b]
    }

    /// Direct precedence edges implied by the read/write sets:
    ///
    /// - every `writer -> reader` of a buffer, unless both endpoints are in-place modifiers of that
    ///   buffer (in which case the pair would be contradictory in both directions), and
    /// - `M_i -> M_{i+1}` for each pair of consecutive modifiers of the same buffer (serialization
    ///   by insertion order).
    ///
    /// Returns `succ` (adjacency lists, deduped and sorted) and the
    /// corresponding in-degrees.
    fn edges(&self) -> (Vec<Vec<usize>>, Vec<usize>) {
        let mut succ: Vec<Vec<usize>> = vec![vec![]; self.n_nodes];
        for b in 0..self.n_bufs {
            let mods: std::collections::BTreeSet<usize> =
                self.modifiers[b].iter().copied().collect();
            for &w in &self.writers[b] {
                for &r in &self.readers[b] {
                    if w != r && !(mods.contains(&w) && mods.contains(&r)) {
                        succ[w].push(r);
                    }
                }
            }
            for pair in self.modifiers[b].windows(2) {
                succ[pair[0]].push(pair[1]);
            }
        }
        for s in &mut succ {
            s.sort_unstable();
            s.dedup();
        }
        let mut indeg = vec![0usize; self.n_nodes];
        for sv in &succ {
            for &v in sv {
                indeg[v] += 1;
            }
        }
        (succ, indeg)
    }

    /// Convenience: per-node read/write buffer id lists.
    fn per_node_access(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let n = self.n_nodes;
        let mut writes = vec![vec![]; n];
        let mut reads = vec![vec![]; n];
        for b in 0..self.n_bufs {
            for &w in &self.writers[b] {
                writes[w].push(b);
            }
            for &r in &self.readers[b] {
                reads[r].push(b);
            }
        }
        (writes, reads)
    }
}

/// CP-SAT backend for [`plan_raw`]. Jointly solves for execution order
/// and buffer offsets with a wall-time cap of `max_secs`; accepts any
/// `Optimal`/`Feasible` solution the solver returns before the deadline.
///
/// Requires the `planner` feature's OR-Tools install.
fn plan_cpsat(bufs: &[BufInfo], ctx: &PlanCtx, max_secs: f64) -> Result<MemoryPlan, PlanError> {
    let PlanCtx {
        n_nodes,
        n_bufs,
        sizes,
        writers,
        readers,
        modifiers,
        ..
    } = ctx;
    let n_nodes = *n_nodes;
    let n_bufs = *n_bufs;

    let mut m = CpModelBuilder::default();
    let last_time = (n_nodes as i64).saturating_sub(1);

    // Execution time slot per node — an all-different permutation over
    // [0, n_nodes).
    let t: Vec<IntVar> = (0..n_nodes)
        .map(|n| m.new_int_var_with_name([(0, last_time)], format!("t_{n}")))
        .collect();
    m.add_all_different(t.iter().copied());

    // Precedence: every writer of a buffer runs before every reader of it,
    // *except* when both endpoints are in-place modifiers of that buffer —
    // in that case the pair would appear in both directions and yield the
    // contradictory `t[A] < t[B]` / `t[B] < t[A]` clauses. Modifiers of the
    // same buffer are instead serialized by insertion order (see below).
    for b in 0..n_bufs {
        let mods: std::collections::BTreeSet<usize> = modifiers[b].iter().copied().collect();
        for &w in &writers[b] {
            for &r in &readers[b] {
                if w != r && !(mods.contains(&w) && mods.contains(&r)) {
                    m.add_lt(t[w], t[r]);
                }
            }
        }
    }

    // Insertion-order serialization of modifiers of the same buffer. The
    // graph is documented to be built write-before-read (see `graph_ir.rs`),
    // so a modifier's position in the builder's node list defines the intended
    // execution order relative to the buffer's other modifiers.
    for mods in modifiers.iter().take(n_bufs) {
        for pair in mods.windows(2) {
            m.add_lt(t[pair[0]], t[pair[1]]);
        }
    }

    // Device buffers with non-zero size and not in `exclude` are what we're
    // packing. Excluded buffers still participate in precedence (their
    // writers/readers appear in `writers`/`readers` above) but aren't
    // assigned offsets or interfered with in memory.
    let device_bufs: Vec<usize> = (0..n_bufs).filter(|&b| ctx.packable(b)).collect();

    // Live interval [birth, death] per device buffer. Using the sentinel
    // range [-1, n_nodes] lets buffers that are never written (graph
    // inputs) be born at -1 and buffers that are never read (graph
    // outputs) die at n_nodes, so they interfere with everything.
    let mut birth: BTreeMap<usize, IntVar> = BTreeMap::new();
    let mut death: BTreeMap<usize, IntVar> = BTreeMap::new();
    for &b in &device_bufs {
        let birth_v = m.new_int_var_with_name([(-1, n_nodes as i64)], format!("birth_{b}"));
        let death_v = m.new_int_var_with_name([(-1, n_nodes as i64)], format!("death_{b}"));
        if writers[b].is_empty() {
            m.add_eq(birth_v, -1i64);
        } else if writers[b].len() == 1 {
            m.add_eq(birth_v, t[writers[b][0]]);
        } else {
            m.add_min_eq(birth_v, writers[b].iter().map(|&w| t[w]));
        }
        if readers[b].is_empty() {
            m.add_eq(death_v, n_nodes as i64);
        } else if readers[b].len() == 1 {
            m.add_eq(death_v, t[readers[b][0]]);
        } else {
            m.add_max_eq(death_v, readers[b].iter().map(|&r| t[r]));
        }
        birth.insert(b, birth_v);
        death.insert(b, death_v);
    }

    // Upper bound on peak memory = sum of all device buffer sizes.
    let sum_sizes: i64 = device_bufs.iter().map(|&b| sizes[b]).sum();

    let offsets: BTreeMap<usize, IntVar> = device_bufs
        .iter()
        .map(|&b| {
            (
                b,
                m.new_int_var_with_name([(0, sum_sizes)], format!("off_{b}")),
            )
        })
        .collect();

    // Alignment: `offset[b]` must be a multiple of `bufs[b].elem_size` so
    // that a pointer `pool_base + offset[b]` (with `pool_base` returned by
    // `cudaMalloc`, which is 256-byte aligned) is properly aligned for the
    // buffer's element type. We model this via a fresh int var `k` with
    // `offset[b] == k * align`; CP-SAT resolves it as a divisibility clause.
    // `align == 1` needs no constraint.
    for &b in &device_bufs {
        let align = bufs[b].elem_size as i64;
        if align > 1 {
            let ub = if sum_sizes == 0 { 0 } else { sum_sizes / align };
            let k = m.new_int_var_with_name([(0, ub)], format!("align_k_{b}"));
            m.add_eq(offsets[&b], LinearExpr::from((align, k)));
        }
    }

    // Pairwise no-overlap: either the buffers' lifetimes are disjoint, or
    // their memory regions are disjoint. Modeled as a disjunction over four
    // reified linear constraints.
    for (i, &b1) in device_bufs.iter().enumerate() {
        for &b2 in &device_bufs[i + 1..] {
            let lit_t12 = m.new_bool_var(); // death[b1] < birth[b2]
            let lit_t21 = m.new_bool_var(); // death[b2] < birth[b1]
            let lit_m12 = m.new_bool_var(); // off[b1] + size[b1] <= off[b2]
            let lit_m21 = m.new_bool_var(); // off[b2] + size[b2] <= off[b1]
            m.add_or([lit_t12, lit_t21, lit_m12, lit_m21]);

            let c = m.add_lt(death[&b1], birth[&b2]);
            m.only_enforce_if(c, [lit_t12]);
            let c = m.add_lt(death[&b2], birth[&b1]);
            m.only_enforce_if(c, [lit_t21]);
            let c = m.add_le(LinearExpr::from(offsets[&b1]) + sizes[b1], offsets[&b2]);
            m.only_enforce_if(c, [lit_m12]);
            let c = m.add_le(LinearExpr::from(offsets[&b2]) + sizes[b2], offsets[&b1]);
            m.only_enforce_if(c, [lit_m21]);
        }
    }

    // Objective: minimize the peak byte usage.
    let peak = m.new_int_var_with_name([(0, sum_sizes)], "peak");
    for &b in &device_bufs {
        m.add_ge(peak, LinearExpr::from(offsets[&b]) + sizes[b]);
    }
    m.minimize(peak);

    // Cap CP-SAT wall time — for larger graphs (fractional_sumcheck_gpu_ir,
    // ~50–100 buffers) proving optimality is expensive and rarely worth it.
    // `Feasible` is accepted below, so any solution the solver finds within
    // the deadline is used.
    let params = SatParameters {
        max_time_in_seconds: Some(max_secs),
        ..Default::default()
    };
    let response = m.solve_with_parameters(&params);
    match response.status() {
        CpSolverStatus::Optimal | CpSolverStatus::Feasible => {}
        status => return Err(PlanError::NoSolution(status)),
    }

    let times: Vec<i64> = t.iter().map(|v| v.solution_value(&response)).collect();
    let mut order: Vec<usize> = (0..n_nodes).collect();
    order.sort_by_key(|&n| times[n]);

    let mut out_offsets = vec![None; n_bufs];
    for &b in &device_bufs {
        out_offsets[b] = Some(offsets[&b].solution_value(&response) as u64);
    }
    let peak_bytes = peak.solution_value(&response) as u64;

    Ok(MemoryPlan {
        order,
        offsets: out_offsets,
        peak_bytes,
    })
}

/// Heuristic (solver-free) backend for [`plan_raw`].
///
/// Three phases, single stream:
///
/// 1. **Memory-aware greedy topological order.** Iterate a ready set (nodes with all predecessors
///    already scheduled). At each step pick the ready node with the smallest `new_births` (bytes it
///    newly allocates), breaking ties by picking the largest `freed` (bytes dying at its
///    last-reader). Precedence edges come from [`PlanCtx::edges`] so modifier chains are respected.
/// 2. **Offline best-fit-decreasing offset assignment.** With the schedule fixed, [`pack_order`]
///    derives each packable buffer's `[birth, death]` lifetime interval, sorts buffers by size
///    (largest first), and places each at the smallest aligned offset that avoids every already-
///    placed lifetime-overlapping buffer.
/// 3. **Local search with repacking.** Sweep 0..n-1 and try each adjacent swap; a swap is legal iff
///    there is no direct precedence edge from the earlier to the later node (adjacency rules out
///    any indirect path). Re-pack after each accepted swap and keep it only if the new peak is
///    strictly lower. Repeat until a full sweep produces no improvement or `MAX_PASSES` is hit.
fn plan_heuristic(bufs: &[BufInfo], ctx: &PlanCtx) -> Result<MemoryPlan, PlanError> {
    let (succ, indeg) = ctx.edges();

    // O(1) edge lookup for the swap-legality check inside the local search.
    let mut adj = vec![vec![false; ctx.n_nodes]; ctx.n_nodes];
    for (u, sv) in succ.iter().enumerate() {
        for &v in sv {
            adj[u][v] = true;
        }
    }

    // Phase 1: memory-aware topological greedy.
    let initial_order = greedy_topo(ctx, &succ, &indeg);
    // Phase 2: offline best-fit-decreasing packing of that order.
    let (mut best_offsets, mut best_peak) = pack_order(ctx, bufs, &initial_order);
    let mut best_order = initial_order;

    // Phase 3: local search over legal adjacent swaps.
    //
    // Because we only swap adjacent positions, a swap is topologically
    // valid iff there is no direct edge from the earlier to the later
    // node — any indirect path would need an intermediate node between
    // them, and there is no room for one.
    const MAX_PASSES: usize = 50;
    for _ in 0..MAX_PASSES {
        let mut improved = false;
        let mut order = best_order.clone();
        let n = order.len();
        let mut i = 0;
        while i + 1 < n {
            let a = order[i];
            let b = order[i + 1];
            if !adj[a][b] {
                order.swap(i, i + 1);
                let (offs, peak) = pack_order(ctx, bufs, &order);
                if peak < best_peak {
                    best_peak = peak;
                    best_offsets = offs;
                    best_order = order.clone();
                    improved = true;
                } else {
                    // Revert this swap and keep sweeping.
                    order.swap(i, i + 1);
                }
            }
            i += 1;
        }
        if !improved {
            break;
        }
    }

    Ok(MemoryPlan {
        order: best_order,
        offsets: best_offsets,
        peak_bytes: best_peak,
    })
}

/// Memory-aware topological greedy. At each step pick the ready node
/// that minimizes `(new_births, -freed)`: fewer new bytes first, more
/// freed on ties.
fn greedy_topo(ctx: &PlanCtx, succ: &[Vec<usize>], initial_indeg: &[usize]) -> Vec<usize> {
    let n = ctx.n_nodes;
    let mut indeg = initial_indeg.to_vec();
    let mut order = Vec::with_capacity(n);

    let mut remaining_readers: Vec<usize> = ctx.readers.iter().map(|r| r.len()).collect();
    let mut alive: Vec<bool> = vec![false; ctx.n_bufs];
    let (node_writes, node_reads) = ctx.per_node_access();

    let mut ready: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();

    while !ready.is_empty() {
        let mut best_pos = 0;
        let mut best_key: (i64, i64) = (i64::MAX, i64::MIN);
        for (idx, &node) in ready.iter().enumerate() {
            let (nb, fr) = score_node(
                ctx,
                &node_writes,
                &node_reads,
                &alive,
                &remaining_readers,
                node,
            );
            let key = (nb, -fr);
            if key < best_key {
                best_key = key;
                best_pos = idx;
            }
        }
        let picked = ready.swap_remove(best_pos);
        order.push(picked);

        // Apply births first, then reads/deaths — matches the semantics
        // the packer uses so scoring stays consistent with reality.
        for &b in &node_writes[picked] {
            if ctx.packable(b) && !alive[b] {
                alive[b] = true;
            }
        }
        for &b in &node_reads[picked] {
            if remaining_readers[b] > 0 {
                remaining_readers[b] -= 1;
            }
            if remaining_readers[b] == 0 && ctx.packable(b) && alive[b] {
                alive[b] = false;
            }
        }
        for &s in &succ[picked] {
            indeg[s] -= 1;
            if indeg[s] == 0 {
                ready.push(s);
            }
        }
    }

    // Defensive fallback for graphs whose precedence isn't a full DAG
    // through this code path (shouldn't happen — [`PlanCtx::edges`] only
    // produces a DAG on well-formed inputs).
    if order.len() != n {
        for i in 0..n {
            if !order.contains(&i) {
                order.push(i);
            }
        }
    }
    order
}

fn score_node(
    ctx: &PlanCtx,
    node_writes: &[Vec<usize>],
    node_reads: &[Vec<usize>],
    alive: &[bool],
    remaining_readers: &[usize],
    node: usize,
) -> (i64, i64) {
    let mut new_births = 0i64;
    for &b in &node_writes[node] {
        if ctx.packable(b) && !alive[b] {
            new_births += ctx.sizes[b];
        }
    }
    let mut freed = 0i64;
    for &b in &node_reads[node] {
        if remaining_readers[b] == 1 && ctx.packable(b) {
            // Buffer is either currently alive or about to be born by
            // this same node (scratch / in-place modifier pattern): it
            // will be gone after this node runs either way.
            if alive[b] || node_writes[node].contains(&b) {
                freed += ctx.sizes[b];
            }
        }
    }
    (new_births, freed)
}

/// Offline best-fit-decreasing (BFD) packing: given a fixed execution
/// `order`, compute each packable buffer's `[birth, death]` lifetime
/// interval (in time-step positions along `order`), then place buffers
/// in *decreasing size* order, each at the smallest aligned offset that
/// avoids every already-placed lifetime-overlapping buffer.
///
/// Lifetimes use the same strict-less-than disjoint-ness rule the CP-SAT
/// model uses: two buffers overlap unless `death(b1) < birth(b2)` or
/// `death(b2) < birth(b1)`. Buffers with no writer get `birth = -1`
/// (graph inputs live from the start) and buffers with no reader get
/// `death = n` (graph outputs live to the end).
///
/// This offline pass is markedly stronger than an online best-fit walk
/// because it sees every buffer's full lifetime up front and can place
/// larger buffers first, tucking the smaller ones into remaining gaps.
///
/// Returns `(offsets, peak_bytes)`.
fn pack_order(ctx: &PlanCtx, bufs: &[BufInfo], order: &[usize]) -> (Vec<Option<u64>>, u64) {
    let n_bufs = ctx.n_bufs;
    let n = order.len();

    // Position of each node in `order`.
    let mut pos = vec![usize::MAX; ctx.n_nodes];
    for (i, &nid) in order.iter().enumerate() {
        pos[nid] = i;
    }

    // Lifetime intervals per buffer (only meaningful for packable ones).
    let mut birth = vec![i64::MAX; n_bufs];
    let mut death = vec![i64::MIN; n_bufs];
    for b in 0..n_bufs {
        if !ctx.packable(b) {
            continue;
        }
        for &w in &ctx.writers[b] {
            birth[b] = birth[b].min(pos[w] as i64);
        }
        for &r in &ctx.readers[b] {
            death[b] = death[b].max(pos[r] as i64);
        }
        if ctx.writers[b].is_empty() {
            birth[b] = -1;
        }
        if ctx.readers[b].is_empty() {
            death[b] = n as i64;
        }
    }

    let overlaps =
        |b1: usize, b2: usize| -> bool { !(death[b1] < birth[b2] || death[b2] < birth[b1]) };

    // Ordered placement: size desc, then birth asc, then id (stable).
    let mut to_place: Vec<usize> = (0..n_bufs).filter(|&b| ctx.packable(b)).collect();
    to_place.sort_by_key(|&b| (std::cmp::Reverse(ctx.sizes[b]), birth[b], b));

    let mut offsets: Vec<Option<u64>> = vec![None; n_bufs];
    let mut peak: i64 = 0;

    for &b in &to_place {
        let size = ctx.sizes[b];
        let align = (bufs[b].elem_size as i64).max(1);

        // Every already-placed buffer whose lifetime overlaps b's blocks
        // some byte range.
        let mut blocked: Vec<(i64, i64)> = to_place
            .iter()
            .filter_map(|&b2| {
                if b2 == b {
                    return None;
                }
                let off = offsets[b2]? as i64;
                overlaps(b, b2).then_some((off, off + ctx.sizes[b2]))
            })
            .collect();
        blocked.sort_unstable();

        // Sweep from 0 up, jumping past every blocked interval we hit.
        let mut o: i64 = 0;
        let mut placed = false;
        for &(a_off, a_end) in &blocked {
            let o_aligned = align_up(o, align);
            if o_aligned + size <= a_off {
                o = o_aligned;
                placed = true;
                break;
            }
            o = o.max(a_end);
        }
        if !placed {
            o = align_up(o, align);
        }
        offsets[b] = Some(o as u64);
        peak = peak.max(o + size);
    }

    (offsets, peak as u64)
}

#[inline]
fn align_up(off: i64, align: i64) -> i64 {
    if align > 1 {
        (off + align - 1) / align * align
    } else {
        off
    }
}

/// Builds a [`NodeAccess`] from a [`GraphNode`], matching the semantics
/// [`plan_raw`] expects (`modifies` on a `BlackboxKernel` input adds it to
/// both reads and writes; a `Const` writes its target; `Memcpy` reads src
/// and writes dst; etc.).
pub fn access_from_node(node: &GraphNode) -> NodeAccess {
    let mut a = NodeAccess::default();
    match node {
        GraphNode::BlackboxKernel(k) => {
            for (i, b) in k.inputs.iter().enumerate() {
                a.reads.push(*b);
                if k.modifies[i] {
                    a.writes.push(*b);
                }
            }
            a.writes.extend(k.outputs.iter().copied());
        }
        GraphNode::Kernel(k) => {
            a.reads.extend(k.inputs.iter().copied());
            a.writes.extend(k.outputs.iter().copied());
        }
        GraphNode::Const(c) => a.writes.push(c.buf),
        GraphNode::Memcpy(m) => {
            a.reads.push(m.src);
            a.writes.push(m.dst);
        }
        GraphNode::Memset(m) => a.writes.push(m.node),
    }
    a
}

fn eval_size(
    buf: BufId,
    size: &crate::quast::Quast,
    env: &BTreeMap<VarId, i64>,
) -> Result<i64, PlanError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph_ir::{BufInfo, DeviceType},
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

    /// Two blackbox kernels in a chain producing three buffers; peak memory
    /// with optimal reuse should be `max(a) + max(b, c)` after `a` dies.
    #[test]
    fn packs_disjoint_lifetimes() {
        let mut g = GraphBuilder::new();
        let a = buf(&mut g, "a", 100);
        let b = buf(&mut g, "b", 200);
        let c = buf(&mut g, "c", 300);
        // k0: produces a
        g.insert_blackbox_kernel(
            "k0",
            std::iter::empty(),
            [a].into_iter(),
            std::iter::empty(),
            |_, _, _| {},
        );
        // k1: reads a, produces b
        g.insert_blackbox_kernel(
            "k1",
            [a].into_iter(),
            [b].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );
        // k2: reads b, produces c (a is dead by now)
        g.insert_blackbox_kernel(
            "k2",
            [b].into_iter(),
            [c].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );

        let plan = plan(&g, &BTreeMap::new(), DeviceType::Cuda(0)).unwrap();

        // Topological order: k0, k1, k2.
        assert_eq!(plan.order, vec![0, 1, 2]);

        // a is only live during k0..k1, then dies; b and c overlap
        // (b is read by k2 which produces c) so they must be disjoint.
        // Optimal peak = 500 (b at [0,200) + c at [200,500)) or reuse a's
        // space when it dies. Since a and c never coexist, they can share
        // the same offset. Optimal peak is max(a's slot, b + c) = 500.
        assert_eq!(plan.peak_bytes, 500);

        // Every device buffer got an offset.
        assert!(plan.offsets.iter().all(Option::is_some));
        // Peak equals max(offset + size).
        let observed = plan
            .offsets
            .iter()
            .enumerate()
            .filter_map(|(i, o)| o.map(|off| off + sizes(&g)[i]))
            .max()
            .unwrap_or(0);
        assert_eq!(observed, plan.peak_bytes);
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

    /// Buffers with overlapping live ranges must not share memory.
    #[test]
    fn overlapping_lifetimes_do_not_share_memory() {
        let mut g = GraphBuilder::new();
        let a = buf(&mut g, "a", 128);
        let b = buf(&mut g, "b", 256);
        let out = buf(&mut g, "out", 128);
        // k0 produces both a and b in parallel.
        g.insert_blackbox_kernel(
            "produce_both",
            std::iter::empty(),
            [a, b].into_iter(),
            std::iter::empty(),
            |_, _, _| {},
        );
        // k1 consumes both and produces out.
        g.insert_blackbox_kernel(
            "consume_both",
            [a, b].into_iter(),
            [out].into_iter(),
            [false, false].into_iter(),
            |_, _, _| {},
        );

        let plan = plan(&g, &BTreeMap::new(), DeviceType::Cuda(0)).unwrap();
        // a and b are simultaneously alive so their regions must be disjoint.
        let oa = plan.offsets[a.0].unwrap();
        let ob = plan.offsets[b.0].unwrap();
        let sa = sizes(&g)[a.0];
        let sb = sizes(&g)[b.0];
        assert!(oa + sa <= ob || ob + sb <= oa, "a and b overlap in memory");
    }

    /// Symbolic sizes are resolved via the passed assignment.
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

    fn plan_with(
        g: &GraphBuilder,
        env: &BTreeMap<VarId, i64>,
        device: DeviceType,
        mode: SchedulerMode,
    ) -> Result<MemoryPlan, PlanError> {
        let nodes: Vec<NodeAccess> = g.nodes.iter().map(access_from_node).collect();
        plan_raw(&g.bufs, &nodes, env, device, &[], &mode)
    }

    /// Same chain as `packs_disjoint_lifetimes`, but planned through the
    /// heuristic backend. Since the dependency is linear the schedule is
    /// forced (k0 → k1 → k2); the heuristic still has to reuse `a`'s slot
    /// once it dies to match CP-SAT's peak of 500.
    #[test]
    fn heuristic_packs_disjoint_lifetimes() {
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

        let plan = plan_with(
            &g,
            &BTreeMap::new(),
            DeviceType::Cuda(0),
            SchedulerMode::Heuristic,
        )
        .unwrap();
        assert_eq!(plan.order, vec![0, 1, 2]);
        // Best-fit should reuse a's slot when it dies before c is born.
        assert_eq!(plan.peak_bytes, 500);
        let observed = plan
            .offsets
            .iter()
            .enumerate()
            .filter_map(|(i, o)| o.map(|off| off + sizes(&g)[i]))
            .max()
            .unwrap_or(0);
        assert_eq!(observed, plan.peak_bytes);
    }

    /// Heuristic must not overlap simultaneously-alive buffers.
    #[test]
    fn heuristic_overlapping_lifetimes_do_not_share_memory() {
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
        let plan = plan_with(
            &g,
            &BTreeMap::new(),
            DeviceType::Cuda(0),
            SchedulerMode::Heuristic,
        )
        .unwrap();
        let oa = plan.offsets[a.0].unwrap();
        let ob = plan.offsets[b.0].unwrap();
        let sa = sizes(&g)[a.0];
        let sb = sizes(&g)[b.0];
        assert!(oa + sa <= ob || ob + sb <= oa, "a and b overlap in memory");
    }

    /// Local search must reorder two independent producers so the first
    /// 600-byte buffer dies before the second one is born; the
    /// theoretical minimum is 800 (b(600) + mid(100) + out(100) alive at
    /// the final step, which no reordering can avoid).
    ///
    /// Graph: two independent kernels k0, k1 each produce a buffer of
    /// size 600. k2 consumes k0's output and produces `mid` (100).
    /// k3 consumes `mid` and k1's output to produce `out` (100).
    ///
    /// Adversarial insertion (k0, k1, k2, k3) keeps both 600-byte buffers
    /// simultaneously alive → peak 1200 or worse under BFD. Reordering
    /// to (k0, k2, k1, k3) drops the first 600-byte buffer before the
    /// second is born and hits the 800-byte lower bound.
    #[test]
    fn heuristic_local_search_reorders_independent_producers() {
        let mut g = GraphBuilder::new();
        let a = buf(&mut g, "a", 600);
        let b = buf(&mut g, "b", 600);
        let mid = buf(&mut g, "mid", 100);
        let out = buf(&mut g, "out", 100);
        g.insert_blackbox_kernel(
            "k0",
            std::iter::empty(),
            [a].into_iter(),
            std::iter::empty(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "k1",
            std::iter::empty(),
            [b].into_iter(),
            std::iter::empty(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "k2",
            [a].into_iter(),
            [mid].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );
        g.insert_blackbox_kernel(
            "k3",
            [mid, b].into_iter(),
            [out].into_iter(),
            [false, false].into_iter(),
            |_, _, _| {},
        );
        let plan = plan_with(
            &g,
            &BTreeMap::new(),
            DeviceType::Cuda(0),
            SchedulerMode::Heuristic,
        )
        .unwrap();
        assert_eq!(
            plan.peak_bytes, 800,
            "expected peak 800 (theoretical minimum) after local search, got {}",
            plan.peak_bytes
        );
    }

    /// Heuristic respects symbolic sizes just like the CP-SAT backend.
    #[test]
    fn heuristic_respects_symbol_assignment() {
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
        let plan = plan_with(&g, &env, DeviceType::Cuda(0), SchedulerMode::Heuristic).unwrap();
        assert_eq!(plan.peak_bytes, 256);
        assert_eq!(plan.offsets[a.0], Some(0));
    }
}
