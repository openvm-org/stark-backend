use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use im::{HashMap as ImHashMap, HashSet as ImHashSet};
use itertools::Itertools;
use rayon::prelude::*;

use crate::{
    graph_ir::{BufId, DeviceType},
    planner::{
        plan::{StreamInstr, StreamMemoryPlan},
        AbstractTimingGraph, PlanError,
    },
};

/// Bottom-level per node: longest weighted path from `v` to any sink
/// (including `v`'s own duration). Classic critical-path priority for
/// list scheduling.
fn bottom_levels(g: &AbstractTimingGraph) -> Vec<f64> {
    let n = g.num_nodes;
    // Successor list from `buf_producers` / `buf_users`: v's successors
    // are the nodes that consume any buf v produces.
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for v in 0..n {
        for &bid in &g.node_produces[v] {
            if let Some(users) = g.buf_users.get(&bid) {
                for &u in users {
                    if u != v {
                        succ[v].push(u);
                    }
                }
            }
        }
        succ[v].sort_unstable();
        succ[v].dedup();
    }
    let mut indeg = vec![0usize; n];
    for sv in &succ {
        for &u in sv {
            indeg[u] += 1;
        }
    }
    let mut ready: Vec<usize> = (0..n).filter(|&v| indeg[v] == 0).collect();
    let mut cur = 0;
    let mut fwd = Vec::with_capacity(n);
    while cur < ready.len() {
        let v = ready[cur];
        cur += 1;
        fwd.push(v);
        for &u in &succ[v] {
            indeg[u] -= 1;
            if indeg[u] == 0 {
                ready.push(u);
            }
        }
    }
    let mut bl = vec![0.0f64; n];
    for &v in fwd.iter().rev() {
        let mut m = 0.0f64;
        for &u in &succ[v] {
            if bl[u] > m {
                m = bl[u];
            }
        }
        bl[v] = g.node_times[v] + m;
    }
    bl
}

/// Undirected interference between buffers whose lifetime intervals
/// overlap in the final multi-stream schedule. Two buffers with no
/// interference edge may share a memory offset.
struct MemoryInterferenceGraph {
    /// Symmetric adjacency: `interf[a]` contains every `b` that `a`
    /// overlaps with. `[birth, death]` is a global-position interval
    /// derived from the linearised instruction stream.
    interf: HashMap<BufId, Vec<BufId>>,
    /// `[birth, death]` *wall-clock* times per BufId, indexed by
    /// `BufId::0`. Both endpoints come from the same simulated
    /// stream clocks that `perf_est` uses. Graph inputs get
    /// `birth = 0.0` and outputs get `death = f64::INFINITY`.
    birth: Vec<f64>,
    death: Vec<f64>,
}

/// Persistent (structurally-shared) beam-search state.
///
/// `stream_of` and `node_start_times` are stored as `ImHashMap` and
/// grow one entry at a time as nodes are placed — states early in
/// the search hold only a handful of entries instead of `n_nodes`
/// slots initialised to zero. The `live_bufs` map likewise holds
/// only *currently allocated* buffers (entries appear on produce
/// and disappear when the last remaining consumer schedules), so
/// the tracked state is bounded by the working set, not by
/// `n_bufs` / `n_nodes`.
///
/// `remaining_deps` is a small persistent map (only entries for
/// nodes that still have unresolved data deps) — nodes drop out of
/// it once they hit ready.
///
/// `bl`, `s_norm`, `b_norm`, and the three weights are constant
/// across the entire search; they're precomputed in
/// [`ScheduleState::new`] and shared via `Arc` (bl) or `Copy` (the
/// scalars).
#[derive(Clone)]
struct ScheduleState<'a> {
    cur_mem_used: usize,
    max_peak_mem: usize,
    cur_max_time: f64,
    /// Currently-live packable bufs → set of *unscheduled* consumer
    /// node ids. A buf enters when its writer schedules; it exits
    /// when the last remaining consumer schedules. Graph inputs are
    /// *not* tracked here — their memory sits outside the scheduler's
    /// budget (matches the pre-refactor `cur_mem_used = 0` at init).
    live_bufs: ImHashMap<BufId, ImHashSet<usize>>,
    /// Nodes that have been placed → stream index. Grows one entry
    /// per `put_on` call.
    stream_of: ImHashMap<usize, usize>,
    stream_end_t: Vec<f64>,
    ready_queue: ImHashSet<usize>,
    /// Lazy-grown counter: entries appear only when a producer of one
    /// of `v`'s reads schedules for the first time (initial value =
    /// `initial_deps[v] - 1`), decrement on subsequent producers, and
    /// drop out when the count reaches 0 (`v` moves into
    /// `ready_queue`). Under SSA `v` never produces something it
    /// consumes, so `initial_deps[v] = |{ bid ∈ node_consumes[v] :
    /// buf_producers.contains_key(bid) }|` — the number of reads
    /// with an external producer. Nodes with `initial_deps == 0` are
    /// initially ready and never touch this map.
    remaining_deps: ImHashMap<usize, usize>,
    /// Placed nodes → simulated start time on their assigned stream.
    /// Grows one entry per `put_on` call.
    node_start_times: ImHashMap<usize, f64>,
    /// Immutable per-node initial dep count (see [`remaining_deps`]).
    /// Precomputed once and shared across every state via `Arc`.
    initial_deps: Arc<Vec<usize>>,
    /// Immutable per-node bottom-level (longest downstream chain
    /// length). Shared across every state via `Arc` — clone is a
    /// refcount bump.
    bl: Arc<Vec<f64>>,
    /// Sum of `g.node_times` — the single-stream sequential-time
    /// upper bound. Used to normalise the `time_cost` term.
    s_norm: f64,
    /// `max(bl)` — the critical-path length lower bound on makespan.
    /// Used to normalise the critical-path term.
    b_norm: f64,
    /// Absolute byte cap. Feasibility is a hard filter: candidates
    /// that would push `cur_mem_used + produced_bytes` above `m_bound`
    /// return `INF` from [`Self::cost`] and get dropped from the beam.
    m_bound: usize,
    /// Weights for the three-term cost:
    /// `w_m * mem_delta/M + w_t * time_cost/S + w_c * bl[node]/B`.
    /// Sign convention: positive `w_m`/`w_t` = prefer memory-freeing
    /// / gap-filling moves; **negative** `w_c` = classic
    /// critical-path priority (higher `bl` → lower cost → picked first).
    w_m: f64,
    w_t: f64,
    w_c: f64,
    g: &'a AbstractTimingGraph,
}

impl<'a> ScheduleState<'a> {
    fn new(
        g: &'a AbstractTimingGraph,
        streams: usize,
        m_bound: usize,
        w_m: f64,
        w_t: f64,
        w_c: f64,
    ) -> Self {
        let bl = bottom_levels(g);
        let s_norm: f64 = g
            .node_times
            .iter()
            .copied()
            .sum::<f64>()
            .max(f64::MIN_POSITIVE);
        let b_norm: f64 = bl
            .iter()
            .copied()
            .fold(0.0f64, f64::max)
            .max(f64::MIN_POSITIVE);

        // Under SSA every consumed buf has either a real external
        // producer or no producer at all (graph input) — a node never
        // produces something it consumes. So the initial dep count
        // collapses to `|{ bid ∈ node_consumes[v] :
        // buf_producers.contains_key(bid) }|` (no self-producer
        // filter needed).
        let initial_deps: Vec<usize> = (0..g.num_nodes)
            .map(|v| {
                g.node_consumes[v]
                    .iter()
                    .filter(|bid| g.buf_producers.contains_key(bid))
                    .count()
            })
            .collect();

        // `remaining_deps` starts empty and grows lazily — a node
        // gets an entry only when the first producer of one of its
        // reads schedules (see `put_on`). This keeps the map size
        // proportional to the "active frontier" (nodes whose deps
        // are partially resolved) rather than to all unscheduled
        // nodes.
        let remaining_deps: ImHashMap<usize, usize> = ImHashMap::new();

        let ready_queue: ImHashSet<usize> = g.inital_ready_nodes.iter().cloned().collect();

        ScheduleState {
            cur_mem_used: 0,
            max_peak_mem: 0,
            cur_max_time: 0.0,
            live_bufs: ImHashMap::new(),
            stream_of: ImHashMap::new(),
            stream_end_t: vec![0.0; streams],
            ready_queue,
            remaining_deps,
            node_start_times: ImHashMap::new(),
            initial_deps: Arc::new(initial_deps),
            bl: Arc::new(bl),
            s_norm,
            b_norm,
            m_bound,
            w_m,
            w_t,
            w_c,
            g,
        }
    }

    /// Linearise the scheduled nodes into a per-stream order via
    /// Kahn's topological sort. Ties (multiple ready nodes on the
    /// same stream) are broken by lowest node id — deterministic and
    /// good enough for downstream lifetime accounting.
    fn per_stream_orders(&self) -> Vec<Vec<usize>> {
        let g = self.g;
        let n_streams = self.stream_end_t.len();
        let mut orders: Vec<Vec<usize>> = vec![Vec::new(); n_streams];
        let mut remaining: Vec<usize> = (0..g.num_nodes)
            .map(|v| {
                g.node_consumes[v]
                    .iter()
                    .filter(|bid| {
                        g.buf_producers
                            .get(bid)
                            .map(|ps| ps.iter().any(|&p| p != v))
                            .unwrap_or(false)
                    })
                    .count()
            })
            .collect();
        let mut ready: HashSet<usize> = g.inital_ready_nodes.iter().copied().collect();
        while !ready.is_empty() {
            let v = *ready.iter().min().expect("ready non-empty");
            ready.remove(&v);
            orders[*self.stream_of.get(&v).unwrap_or(&0)].push(v);
            for &bid in &g.node_produces[v] {
                if let Some(users) = g.buf_users.get(&bid) {
                    for &u in users {
                        if u == v {
                            continue;
                        }
                        // Only decrement for consumers (u reads bid, u ≠ producer).
                        if g.node_consumes[u].contains(&bid) && remaining[u] > 0 {
                            remaining[u] -= 1;
                            if remaining[u] == 0 {
                                ready.insert(u);
                            }
                        }
                    }
                }
            }
        }
        orders
    }

    /// Build the interference graph over BufIds using *wall-clock*
    /// simulated times as `[birth, death]` endpoints — the same
    /// timeline that `perf_est` and the validator use.
    ///
    /// Rationale: two nodes at different positions on *different*
    /// streams can still execute at the same wall-clock time, so a
    /// position-based lifetime check underestimates concurrency and
    /// permits unsafe pool sharing. Using simulated times matches
    /// `list_v1::offline_repack` and lets `validate_plan` sign off
    /// on the packing.
    ///
    /// `start[v]` and `finish[v]` are the simulated launch and
    /// completion times of node `v` on its assigned stream (with
    /// cross-stream `WaitOn`s already applied). `NaN` entries mean
    /// the node wasn't emitted; its buffers are dropped from the
    /// pack list.
    fn make_interference_graph(&self, start: &[f64], finish: &[f64]) -> MemoryInterferenceGraph {
        let g = self.g;
        let n_bufs = g.buf_info.len();

        let input_set: HashSet<BufId> = g.inputs.iter().copied().collect();
        let output_set: HashSet<BufId> = g.outputs.iter().copied().collect();
        let mut birth = vec![f64::INFINITY; n_bufs];
        let mut death = vec![f64::NEG_INFINITY; n_bufs];
        for bid in 0..n_bufs {
            let bid_id = BufId(bid);
            if let Some(prods) = g.buf_producers.get(&bid_id) {
                for &v in prods {
                    if !start[v].is_nan() {
                        if start[v] < birth[bid] {
                            birth[bid] = start[v];
                        }
                        if finish[v] > death[bid] {
                            death[bid] = finish[v];
                        }
                    }
                }
            }
            if let Some(users) = g.buf_users.get(&bid_id) {
                for &v in users {
                    if !finish[v].is_nan() && finish[v] > death[bid] {
                        death[bid] = finish[v];
                    }
                }
            }
            if input_set.contains(&bid_id) {
                birth[bid] = birth[bid].min(0.0);
            }
            if output_set.contains(&bid_id) {
                death[bid] = f64::INFINITY;
            }
        }

        // Adjacency by pairwise interval-overlap check on packable
        // (Cuda) buffers reachable through at least one node.
        let packable: Vec<usize> = (0..n_bufs)
            .filter(|&b| {
                matches!(g.buf_info[b].device_type, DeviceType::Cuda(_))
                    && birth[b].is_finite()
                    && death[b] > birth[b] - 1.0
            })
            .collect();

        // Per-buffer stream signature: `Some(s)` if every producer and
        // user of the buffer runs on stream `s`; `None` if multiple
        // streams touch it. Wall-clock-disjoint slot reuse *across*
        // streams is a data race at the CUDA layer — `make_schedule`
        // only emits `WaitOn`s for same-canonical RAW deps, not for
        // slot handoffs between different canonicals that happen to
        // share an offset. Same-stream slot reuse is safe because
        // stream ordering guarantees release-before-acquire.
        let mut buf_stream: Vec<Option<usize>> = vec![None; n_bufs];
        for (b, slot) in buf_stream.iter_mut().enumerate() {
            let bid = BufId(b);
            let mut chosen: Option<usize> = None;
            let mut multi = false;
            let touches = g
                .buf_producers
                .get(&bid)
                .into_iter()
                .flatten()
                .chain(g.buf_users.get(&bid).into_iter().flatten())
                .copied();
            for v in touches {
                let s = *self.stream_of.get(&v).unwrap_or(&0);
                match chosen {
                    None => chosen = Some(s),
                    Some(prev) if prev != s => {
                        multi = true;
                        break;
                    }
                    _ => {}
                }
            }
            *slot = if multi { None } else { chosen };
        }

        let mut interf: HashMap<BufId, Vec<BufId>> = HashMap::new();
        for i in 0..packable.len() {
            let a = packable[i];
            for &b in &packable[i + 1..] {
                let intersects = match (buf_stream[a], buf_stream[b]) {
                    (Some(sa), Some(sb)) if sa == sb => {
                        // Exactly-touching intervals (a's death == b's birth) don't overlap.
                        birth[a].max(birth[b]) + 1e-9 < death[a].min(death[b])
                    }
                    _ => true,
                };
                if intersects {
                    interf.entry(BufId(a)).or_default().push(BufId(b));
                    interf.entry(BufId(b)).or_default().push(BufId(a));
                }
            }
        }
        MemoryInterferenceGraph {
            interf,
            birth,
            death,
        }
    }

    /// Convert the completed schedule into a [`StreamMemoryPlan`].
    ///
    /// Steps:
    /// 1. Recover per-stream ordered node lists (Kahn's topo sort).
    /// 2. Emit an interleaved instruction stream. For each cross-stream data dep `P → N` we insert
    ///    `WaitOn(N.stream, event_of(P))` before `N`, unless the current stream has already synced
    ///    past `P`'s position on `P`'s stream via an earlier `WaitOn` — in which case the sync is
    ///    implied by transitive event ordering.
    /// 3. Build the interference graph and assign each packable BufId a memory offset via
    ///    best-fit-decreasing placement.
    fn make_schedule(&self) -> StreamMemoryPlan {
        let g = self.g;
        let n_nodes = g.num_nodes;
        let n_bufs = g.buf_info.len();
        let n_streams = self.stream_end_t.len();

        // (1) Per-stream ordered node lists + intra-stream positions.
        let stream_orders = self.per_stream_orders();
        let mut node_pos_in_stream = vec![usize::MAX; n_nodes];
        for seq in stream_orders.iter() {
            for (i, &v) in seq.iter().enumerate() {
                node_pos_in_stream[v] = i;
            }
        }

        // (2a) A producer needs an event iff any of its outputs is
        // consumed by a node on a *different* stream. Assign event ids
        // in visit order for determinism.
        let mut producer_event: HashMap<usize, u32> = HashMap::new();
        for v in 0..n_nodes {
            if node_pos_in_stream[v] == usize::MAX {
                continue;
            }
            let s_v = *self.stream_of.get(&v).unwrap_or(&0);
            let mut needs = false;
            for &bid in &g.node_produces[v] {
                if let Some(users) = g.buf_users.get(&bid) {
                    if users
                        .iter()
                        .any(|&u| u != v && *self.stream_of.get(&u).unwrap_or(&0) != s_v)
                    {
                        needs = true;
                        break;
                    }
                }
            }
            if needs {
                let e = producer_event.len() as u32;
                producer_event.insert(v, e);
            }
        }
        let num_events = producer_event.len() as u32;

        // (2b) Interleave: dispatch nodes round-robin across streams
        // as their cross-stream deps become emitted. `last_synced[s][s']`
        // = highest position in stream `s'` that stream `s` has already
        // waited past; a WaitOn is skipped when a producer at that
        // position is `≤ last_synced`.
        let mut instructions: Vec<StreamInstr> = Vec::new();
        let mut ptr: Vec<usize> = vec![0; n_streams];
        let mut emitted = vec![false; n_nodes];
        let mut last_synced: Vec<Vec<i64>> = vec![vec![-1; n_streams]; n_streams];

        loop {
            let mut progressed = false;
            for s in 0..n_streams {
                if ptr[s] >= stream_orders[s].len() {
                    continue;
                }
                let v = stream_orders[s][ptr[s]];

                // Cross-stream data-dep readiness check + collect the
                // set of syncs we'd need if we dispatch v now.
                let mut needed: HashMap<usize, (u32, i64)> = HashMap::new();
                let mut ready = true;
                for &bid in &g.node_consumes[v] {
                    if let Some(prods) = g.buf_producers.get(&bid) {
                        for &p_node in prods {
                            let s_p = *self.stream_of.get(&p_node).unwrap_or(&0);
                            if s_p == s {
                                continue;
                            }
                            let pos_p = node_pos_in_stream[p_node] as i64;
                            if last_synced[s][s_p] >= pos_p {
                                continue; // already synced past this producer
                            }
                            if !emitted[p_node] {
                                ready = false;
                                break;
                            }
                            // Merge per-target-stream; keep the max pos_p.
                            let e = *producer_event
                                .get(&p_node)
                                .expect("cross-stream producer must have event");
                            let cur = needed.entry(s_p).or_insert((e, -1));
                            if pos_p > cur.1 {
                                *cur = (e, pos_p);
                            }
                        }
                        if !ready {
                            break;
                        }
                    }
                }
                if !ready {
                    continue;
                }

                for (s_p, (e, pos_p)) in needed {
                    instructions.push(StreamInstr::WaitOn(s, e as usize));
                    last_synced[s][s_p] = pos_p;
                }
                instructions.push(StreamInstr::Node(v));
                emitted[v] = true;
                ptr[s] += 1;
                progressed = true;
            }
            if !progressed {
                break;
            }
        }

        // (3) Simulate the emitted plan under the same perfect-parallel
        // model that `perf_est` / `validate_plan` use, to derive
        // per-node start/finish times for lifetime accounting.
        let mut sim_stream_time = vec![0.0f64; n_streams.max(1)];
        let mut sim_event_time = vec![0.0f64; num_events as usize];
        let mut sim_start = vec![f64::NAN; n_nodes];
        let mut sim_finish = vec![f64::NAN; n_nodes];
        for instr in &instructions {
            match *instr {
                StreamInstr::Node(v) => {
                    let s = *self.stream_of.get(&v).unwrap_or(&0);
                    sim_start[v] = sim_stream_time[s];
                    sim_stream_time[s] += g.node_times[v];
                    sim_finish[v] = sim_stream_time[s];
                    if let Some(&e) = producer_event.get(&v) {
                        sim_event_time[e as usize] = sim_finish[v];
                    }
                }
                StreamInstr::WaitOn(s, e) => {
                    if s < n_streams
                        && e < sim_event_time.len()
                        && sim_stream_time[s] < sim_event_time[e]
                    {
                        sim_stream_time[s] = sim_event_time[e];
                    }
                }
            }
        }

        // (4) Interference + best-fit-decreasing offset assignment.
        let mig = self.make_interference_graph(&sim_start, &sim_finish);
        let mut offsets: Vec<Option<u64>> = vec![None; n_bufs];
        let mut peak: u64 = 0;
        // Order: largest first, then earliest birth for tie-break — same
        // policy as heuristic::pack_order.
        let mut to_place: Vec<usize> = (0..n_bufs)
            .filter(|&b| {
                matches!(g.buf_info[b].device_type, DeviceType::Cuda(_))
                    && mig.birth[b].is_finite()
                    && mig.death[b] > mig.birth[b] - 1.0
            })
            .collect();
        to_place.sort_by(|&a, &b| {
            let sa = g.buf_info[a].concrete_size;
            let sb = g.buf_info[b].concrete_size;
            sb.cmp(&sa)
                .then_with(|| mig.birth[a].total_cmp(&mig.birth[b]))
                .then_with(|| a.cmp(&b))
        });
        for &b in &to_place {
            let size = g.buf_info[b].concrete_size as u64;
            let align = g.buf_info[b].elem_size.max(1) as u64;
            let mut blocked: Vec<(u64, u64)> = mig
                .interf
                .get(&BufId(b))
                .map(|nbrs| {
                    nbrs.iter()
                        .filter_map(|nbr| {
                            offsets[nbr.0]
                                .map(|off| (off, off + g.buf_info[nbr.0].concrete_size as u64))
                        })
                        .collect()
                })
                .unwrap_or_default();
            blocked.sort_unstable();
            let mut o: u64 = 0;
            let mut placed = false;
            for (a_off, a_end) in &blocked {
                let o_aligned = o.div_ceil(align) * align;
                if o_aligned + size <= *a_off {
                    o = o_aligned;
                    placed = true;
                    break;
                }
                o = o.max(*a_end);
            }
            if !placed {
                o = o.div_ceil(align) * align;
            }
            offsets[b] = Some(o);
            peak = peak.max(o + size);
        }

        // Per-node stream & record_event vectors indexed by node id.
        let mut stream_out = vec![0u32; n_nodes];
        for v in 0..n_nodes {
            stream_out[v] = self.stream_of.get(&v).copied().unwrap_or(0) as u32;
        }
        let mut record_event_out = vec![None; n_nodes];
        for (&v, &e) in &producer_event {
            record_event_out[v] = Some(e);
        }

        StreamMemoryPlan {
            instructions,
            stream: stream_out,
            record_event: record_event_out,
            offsets,
            peak_bytes: peak,
            num_streams: n_streams as u32,
            num_events,
        }
    }

    /// Normalised action cost for placing `node` on stream `on_stream`.
    ///
    /// Returns `f64::INFINITY` when the placement would push
    /// `cur_mem_used + produced_bytes` above `m_bound` (hard
    /// feasibility filter — the beam-search discards these).
    /// Otherwise:
    ///   `cost = w_m * mem_delta / M  +  w_t * time_cost / S  +  w_c * bl[node] / B`
    fn cost(&self, node: usize, on_stream: usize) -> f64 {
        let mut max_producer_end_t = 0.0;
        for bid in self.g.node_consumes[node].iter() {
            if let Some(prods) = self.g.buf_producers.get(bid) {
                for n in prods {
                    let start = self.node_start_times.get(n).copied().unwrap_or(0.0);
                    let end_t = start + self.g.node_times[*n];
                    if end_t > max_producer_end_t {
                        max_producer_end_t = end_t;
                    }
                }
            }
        }
        let time_cost = max_producer_end_t.max(self.stream_end_t[on_stream]) - self.cur_max_time;

        let mut killed_bytes: i64 = 0;
        let mut produced_bytes: i64 = 0;
        for bid in self.g.node_consumes[node].iter() {
            if let Some(set) = self.live_bufs.get(bid) {
                if set.len() == 1 && set.contains(&node) {
                    killed_bytes += self.g.buf_info[bid.0].concrete_size as i64;
                }
            }
        }
        for bid in self.g.node_produces[node].iter() {
            produced_bytes += self.g.buf_info[bid.0].concrete_size as i64;
        }
        let mem_delta = produced_bytes - killed_bytes;

        if (self.cur_mem_used as i64) + produced_bytes > self.m_bound as i64 {
            return f64::INFINITY;
        }

        self.w_m * (mem_delta as f64) / (self.m_bound as f64)
            + self.w_t * time_cost / self.s_norm
            + self.w_c * self.bl[node] / self.b_norm
    }

    fn put_on(&mut self, node: usize, on_stream: usize) {
        let mut max_producer_end_t = 0.0;
        for bid in self.g.node_consumes[node].iter() {
            if let Some(prods) = self.g.buf_producers.get(bid) {
                for n in prods {
                    let start = self.node_start_times.get(n).copied().unwrap_or(0.0);
                    let end_t = start + self.g.node_times[*n];
                    if end_t > max_producer_end_t {
                        max_producer_end_t = end_t;
                    }
                }
            }
        }

        let start_t = max_producer_end_t.max(self.stream_end_t[on_stream]);
        self.stream_of.insert(node, on_stream);
        self.node_start_times.insert(node, start_t);
        self.stream_end_t[on_stream] = start_t + self.g.node_times[node];

        let mut killed_bytes: usize = 0;
        let mut produced_bytes: usize = 0;

        // Consume: remove `node` from each live buf's remaining-consumer
        // set. When the set empties, the buf dies. Bufs not in `live_bufs`
        // (graph inputs) aren't tracked — consuming them is a no-op for
        // memory accounting.
        for bid in self.g.node_consumes[node].iter() {
            let now_dead = if let Some(set) = self.live_bufs.get_mut(bid) {
                set.remove(&node);
                set.is_empty()
            } else {
                false
            };
            if now_dead {
                killed_bytes += self.g.buf_info[bid.0].concrete_size;
                self.live_bufs.remove(bid);
            }
        }

        // Produce: each newly-produced buf enters `live_bufs` with the
        // set of its unscheduled readers (from `g.buf_users` minus
        // `node` itself, the writer under SSA). Each such reader has
        // one fewer unresolved dep — first visit inserts
        // `initial_deps[c] - 1`; subsequent visits decrement the
        // existing counter. When a count reaches 0 the node moves
        // into `ready_queue`. This lazy growth keeps `remaining_deps`
        // proportional to the active dep-partial-resolution frontier
        // rather than to all unscheduled nodes.
        for bid in self.g.node_produces[node].iter() {
            produced_bytes += self.g.buf_info[bid.0].concrete_size;
            let consumers: ImHashSet<usize> = self
                .g
                .buf_users
                .get(bid)
                .map(|users| users.iter().copied().filter(|&u| u != node).collect())
                .unwrap_or_default();
            for &c in consumers.iter() {
                let new_count = match self.remaining_deps.get(&c) {
                    Some(&cur) => cur - 1,
                    None => self.initial_deps[c].saturating_sub(1),
                };
                if new_count == 0 {
                    self.remaining_deps.remove(&c);
                    self.ready_queue.insert(c);
                } else {
                    self.remaining_deps.insert(c, new_count);
                }
            }
            self.live_bufs.insert(*bid, consumers);
        }

        self.cur_mem_used += produced_bytes;
        self.cur_mem_used = self.cur_mem_used.saturating_sub(killed_bytes);
        self.max_peak_mem = self.max_peak_mem.max(self.cur_mem_used);
        self.cur_max_time = self.cur_max_time.max(self.stream_end_t[on_stream]);
        self.ready_queue.remove(&node);
    }

    fn done(&self) -> bool {
        self.ready_queue.is_empty()
    }
}

/// Timing statistics for a synthetic run of the greedy inner loop
/// on a freshly-constructed [`ScheduleState`]. Returned by
/// [`bench_ops`].
#[derive(Debug, Clone, Copy)]
pub struct OpBench {
    /// Total nodes scheduled during the greedy walk.
    pub nodes_scheduled: usize,
    /// Total `cost()` calls made (≈ `nodes_scheduled × ready × streams`).
    pub cost_calls: usize,
    /// Total `put_on()` calls (== `nodes_scheduled`).
    pub put_on_calls: usize,
    /// Total `state.clone()` calls (one per considered candidate).
    pub clone_calls: usize,
    /// Average `cost()` time in ns.
    pub cost_ns_avg: f64,
    /// Average `put_on()` time in ns.
    pub put_on_ns_avg: f64,
    /// Average `state.clone()` time in ns.
    pub clone_ns_avg: f64,
    /// Total wall-clock time in ns.
    pub total_ns: u128,
}

/// Runs a greedy schedule (no beam search — just pick min-cost per
/// step and commit) while timing the three per-step primitives.
/// The greedy step at each iteration mirrors the innermost loop of
/// [`plan_v2`]: for each ready node, clone the state, put_on, and
/// evaluate — then commit the min-cost placement.
pub fn bench_ops(
    g: &AbstractTimingGraph,
    num_streams: usize,
    max_memory_bound: usize,
    w_m: f64,
    w_t: f64,
    w_c: f64,
) -> OpBench {
    use std::time::Instant;

    let mut state = ScheduleState::new(g, num_streams, max_memory_bound, w_m, w_t, w_c);

    let mut cost_calls: usize = 0;
    let mut put_on_calls: usize = 0;
    let mut clone_calls: usize = 0;
    let mut cost_ns: u128 = 0;
    let mut put_on_ns: u128 = 0;
    let mut clone_ns: u128 = 0;
    let mut nodes_scheduled: usize = 0;

    let t_total = Instant::now();
    while !state.done() {
        // Enumerate (ready, stream) candidates. For each, time a
        // `clone + put_on` to observe the beam-expand primitive.
        // (The greedy commit itself uses a separate `put_on`.)
        let mut best: Option<(usize, usize, f64)> = None;
        let ready: Vec<usize> = state.ready_queue.iter().copied().collect();
        for &node in &ready {
            for s in 0..num_streams {
                let t = Instant::now();
                let c = state.cost(node, s);
                cost_ns += t.elapsed().as_nanos();
                cost_calls += 1;
                if !c.is_finite() {
                    continue;
                }
                // Time clone + put_on as an expansion probe.
                let t = Instant::now();
                let mut trial = state.clone();
                clone_ns += t.elapsed().as_nanos();
                clone_calls += 1;
                let t = Instant::now();
                trial.put_on(node, s);
                put_on_ns += t.elapsed().as_nanos();
                put_on_calls += 1;
                if best.as_ref().map(|(_, _, bc)| c < *bc).unwrap_or(true) {
                    best = Some((node, s, c));
                }
            }
        }
        if let Some((node, s, _)) = best {
            let t = Instant::now();
            state.put_on(node, s);
            put_on_ns += t.elapsed().as_nanos();
            put_on_calls += 1;
            nodes_scheduled += 1;
        } else {
            break;
        }
    }
    let total_ns = t_total.elapsed().as_nanos();

    OpBench {
        nodes_scheduled,
        cost_calls,
        put_on_calls,
        clone_calls,
        cost_ns_avg: cost_ns as f64 / cost_calls.max(1) as f64,
        put_on_ns_avg: put_on_ns as f64 / put_on_calls.max(1) as f64,
        clone_ns_avg: clone_ns as f64 / clone_calls.max(1) as f64,
        total_ns,
    }
}

/// Depth-D beam-search planner. At each outer step the current
/// committed state is expanded through up to `beam_depth` rollout
/// levels; at each level every surviving state fans out over
/// `ready_queue × 0..num_streams`, states exceeding `max_memory_bound`
/// are discarded, and the top `num_beams` by cumulative action cost
/// survive. Only the *first* action of the best resulting rollout
/// is committed to the current state; the search then restarts from
/// the new state.
///
/// Cost per action (see [`ScheduleState::cost`]):
/// `w_m * mem_delta/M + w_t * time_cost/S + w_c * bl[node]/B`.
/// Sign convention: `w_c < 0` gives classic critical-path priority
/// (higher `bl` → lower cost → picked first).
pub fn plan_v2(
    g: &AbstractTimingGraph,
    num_streams: usize,
    max_memory_bound: usize,
    num_beams: usize,
    beam_depth: usize,
    frontier_cap: usize,
    w_m: f64,
    w_t: f64,
    w_c: f64,
) -> Result<StreamMemoryPlan, PlanError> {
    let mut current = ScheduleState::new(g, num_streams, max_memory_bound, w_m, w_t, w_c);

    // Beam entry: (state after applying trajectory, actions committed
    // so far in this rollout, cumulative action cost).
    type Beam<'a> = (ScheduleState<'a>, Vec<(usize, usize)>, f64);

    let trace = std::env::var("LIST_V2_TRACE").is_ok();
    let t_plan_start = std::time::Instant::now();
    let mut commits_since_report: u64 = 0;
    let mut expand_ns_since: u128 = 0;
    let mut sort_ns_since: u128 = 0;
    let mut commit_ns_since: u128 = 0;
    let mut cartprod_ns_since: u128 = 0;
    let mut max_ready_seen: usize = 0;

    while !current.done() {
        let ready_len = current.ready_queue.len();
        if ready_len > max_ready_seen {
            max_ready_seen = ready_len;
        }
        let t_clone = std::time::Instant::now();
        let mut beams: Vec<Beam<'_>> = vec![(current.clone(), Vec::new(), 0.0)];
        let clone_ns = t_clone.elapsed().as_nanos();

        for _ in 0..beam_depth.max(1) {
            let t_cart = std::time::Instant::now();
            let this_beams = beams
                .drain(..)
                .cartesian_product(0..num_streams)
                .collect::<Vec<_>>();
            cartprod_ns_since += t_cart.elapsed().as_nanos();

            let t_expand = std::time::Instant::now();
            let expanded: Vec<Beam<'_>> = this_beams
                .into_par_iter()
                .flat_map(|((state, traj, cum), stream)| -> Vec<Beam<'_>> {
                    if state.done() {
                        return vec![(state, traj, cum)];
                    }
                    let mut ready: Vec<usize> = state.ready_queue.iter().copied().collect();
                    if frontier_cap > 0 && ready.len() > frontier_cap {
                        let bl = &state.bl;
                        ready.sort_by(|&a, &b| bl[b].total_cmp(&bl[a]));
                        ready.truncate(frontier_cap);
                    }
                    let mut children: Vec<Beam<'_>> = Vec::with_capacity(ready.len());
                    for &node in &ready {
                        let action_cost = state.cost(node, stream);
                        if !action_cost.is_finite() {
                            continue;
                        }
                        let mut new_state = state.clone();
                        new_state.put_on(node, stream);
                        let mut new_traj = traj.clone();
                        new_traj.push((node, stream));
                        children.push((new_state, new_traj, cum + action_cost));
                    }
                    children
                })
                .collect();
            expand_ns_since += t_expand.elapsed().as_nanos();

            if expanded.is_empty() {
                break;
            }
            let t_sort = std::time::Instant::now();
            let mut expanded = expanded;
            expanded.sort_by(|a, b| a.2.total_cmp(&b.2));
            if expanded.len() > num_beams {
                expanded.truncate(num_beams);
            }
            sort_ns_since += t_sort.elapsed().as_nanos();
            beams = expanded;
        }

        let t_commit = std::time::Instant::now();
        let best = beams.get(0);
        match best {
            Some((_, traj, _)) if !traj.is_empty() => {
                let (node, s) = traj[0];
                let already = current.stream_of.contains_key(&node);
                if trace && already && commits_since_report < 5 {
                    eprintln!(
                        "[list_v2] RE-COMMIT node={node} (already scheduled), ready_queue.len={}",
                        current.ready_queue.len()
                    );
                }
                current.put_on(node, s);
            }
            _ => break,
        }
        commit_ns_since += t_commit.elapsed().as_nanos() + clone_ns;

        commits_since_report += 1;
        if trace && commits_since_report >= 100 {
            let scheduled = current.stream_of.len();
            let total_ms = t_plan_start.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "[list_v2] scheduled {scheduled}/{} in {total_ms:.1} ms | last 100 commits: \
                 cart={:.2}ms expand={:.2}ms sort={:.2}ms commit+clone={:.2}ms | max_ready={max_ready_seen}",
                g.num_nodes,
                cartprod_ns_since as f64 / 1e6,
                expand_ns_since as f64 / 1e6,
                sort_ns_since as f64 / 1e6,
                commit_ns_since as f64 / 1e6,
            );
            commits_since_report = 0;
            expand_ns_since = 0;
            sort_ns_since = 0;
            commit_ns_since = 0;
            cartprod_ns_since = 0;
        }
    }
    if trace {
        eprintln!(
            "[list_v2] finished {} commits in {:.2} s (max_ready {max_ready_seen})",
            current.stream_of.len(),
            t_plan_start.elapsed().as_secs_f64(),
        );
    }

    Ok(current.make_schedule())
}
