//! Stream-aware list scheduler V1 with depth-`k` beam look-ahead.
//!
//! Emits a [`StreamMemoryPlan`] where each graph node is assigned to a
//! stream in `[0, max_concurrency)` and cross-stream data dependencies
//! are enforced by [`StreamInstr::WaitOn`] instructions interleaved with
//! node launches.
//!
//! Signature:
//! ```ignore
//! ListSchedulerV1 { max_concurrency, max_memory, k, beam }
//!     .schedule(ctx, est_time) -> StreamMemoryPlan
//! ```
//!
//! `est_time(node_idx) -> f64` provides a per-node runtime estimate (ns
//! or any consistent unit) used for the critical-path priority and beam
//! scoring. Passing `|_| 1.0` degenerates the score to depth-only ranking.

use std::collections::HashMap;

use crate::{
    graph_ir::BufId,
    planner::{
        abstract_timing::AbstractTimingGraph,
        plan::{StreamInstr, StreamMemoryPlan},
        NodeId, PlanError,
    },
};

/// Nodes that write buffer `b`. Post-fake-edge insertion this is `0` for
/// graph inputs and `1` for most buffers; multi-writer real bufs
/// (in-place carry chains) may exceed 1. The scheduler only reads the
/// list, never mutates it.
#[inline]
fn writers_of<'a>(atg: &'a AbstractTimingGraph, b: usize) -> &'a [usize] {
    atg.buf_producers
        .get(&BufId(b))
        .map(|v| v.as_slice())
        .unwrap_or(&[])
}

/// Tunable parameters for the list scheduler.
#[derive(Debug, Clone)]
pub struct ListSchedulerV1 {
    /// Number of concurrent CUDA streams the scheduler may use.
    pub max_concurrency: u32,
    /// Byte cap on the total pool. `u64::MAX` for "no cap".
    pub max_memory: u64,
    /// Look-ahead depth. `0` or `1` is pure greedy; 3..5 is typical.
    pub lookahead_k: usize,
    /// Beam width per level. `1` is pure greedy.
    pub beam: usize,
    /// Score weights.
    pub w_cp: f64,
    pub w_mem: f64,
    /// Fraction of `max_memory` above which the memory-pressure penalty
    /// starts contributing (`0.9` = last 10% of the pool is expensive).
    pub mem_target_frac: f64,
}

impl Default for ListSchedulerV1 {
    fn default() -> Self {
        Self {
            max_concurrency: 8,
            max_memory: u64::MAX,
            lookahead_k: 3,
            beam: 2,
            w_cp: 1.0,
            w_mem: 1e-3,
            mem_target_frac: 0.9,
        }
    }
}

/// Run the list_v1 scheduler over `atg` with `params`. Node runtime
/// timings come from [`AbstractTimingGraph::node_times`] — a zero-
/// filled vec degrades to depth-only priority for first-compile
/// bootstrapping.
pub fn plan_list_v1(
    atg: &AbstractTimingGraph,
    params: &ListSchedulerV1,
) -> Result<StreamMemoryPlan, PlanError> {
    params.schedule(atg)
}

impl ListSchedulerV1 {
    /// Runs the scheduler. Node timings are read from
    /// [`AbstractTimingGraph::node_times`].
    pub fn schedule(&self, atg: &AbstractTimingGraph) -> Result<StreamMemoryPlan, PlanError> {
        if atg.num_nodes == 0 {
            return Ok(StreamMemoryPlan {
                instructions: Vec::new(),
                stream: Vec::new(),
                record_event: Vec::new(),
                offsets: vec![None; atg.n_bufs()],
                peak_bytes: 0,
                num_streams: self.max_concurrency.max(1),
                num_events: 0,
            });
        }

        let n = atg.num_nodes;
        let rt: Vec<f64> = atg.node_times.iter().map(|t| t.max(0.0)).collect();
        debug_assert_eq!(rt.len(), n, "ATG node_times length must equal num_nodes");

        let (succ, indeg) = atg.edges();
        let mut preds: Vec<Vec<NodeId>> = vec![vec![]; n];
        for (u, sv) in succ.iter().enumerate() {
            for &v in sv {
                preds[v].push(u);
            }
        }

        let bl = bottom_levels(&succ, &rt);
        let (node_writes, node_reads) = atg.per_node_access();

        let m = self.max_concurrency.max(1) as usize;
        let beam = self.beam.max(1);
        let k = self.lookahead_k;

        let mut state = SchedState::new(atg, &indeg, &node_reads, m);

        // Pre-place *only* bufs with no writer (real graph inputs supplied
        // by the caller): they must occupy a stable pool slot from t=0
        // because nothing in the schedule ever produces them, and the
        // memory-pressure guard needs to know they're consuming budget
        // from the start.
        //
        // Pinned bufs *with* a writer (graph outputs) are handled by the
        // regular `commit()` allocation path: their writer takes an
        // ordinary pool slot when it runs, and `offline_repack` at the
        // end sees `death = INF` (via `atg.pinned(b)`) so the slot stays
        // reserved to end-of-plan. Pre-placing them at t=0 forces the
        // full output footprint (950 bufs × their concrete sizes in this
        // graph) to live in the pool for the entire schedule, which
        // blows even multi-GB `max_memory` budgets on graphs with many
        // outputs.
        for b in 0..atg.n_bufs() {
            if !atg.packable(b) {
                continue;
            }
            let no_writer = writers_of(atg, b).is_empty();
            if !no_writer {
                continue;
            }
            let size = atg.size(b) as u64;
            let align = atg.align(b).max(1);
            let off = state
                .best_fit(size, align, self.max_memory)
                .ok_or_else(|| {
                    PlanError::Infeasible(format!(
                        "list scheduler: cannot pre-place input BufId({b}) \
                         of size {size} bytes within max_memory={}",
                        self.max_memory
                    ))
                })?;
            state.offsets[b] = Some(off);
            state.buf_placed[b] = true;
            state.live_bytes += size;
            state.live.push(LiveInterval {
                buf: b,
                offset: off,
                size,
                expiry: f64::INFINITY,
            });
            if (off + size) > state.peak_bytes {
                state.peak_bytes = off + size;
            }
        }

        while state.order.len() < n {
            let picked = self.look_ahead_pick(
                &state,
                atg,
                &rt,
                &bl,
                &succ,
                &node_writes,
                &node_reads,
                k,
                beam,
                m,
            );

            match picked {
                Some(v) => {
                    self.commit(
                        &mut state,
                        v,
                        atg,
                        &rt,
                        &succ,
                        &node_writes,
                        &node_reads,
                        m,
                    )?;
                }
                None => {
                    // No candidate can be placed right now — advance
                    // time to the next expiry / stream-free event that
                    // is *strictly* after `state.now`. Without the strict
                    // filter we may loop forever at `t=0` (stream_free
                    // starts at 0 and never gets past it).
                    let now = state.now;
                    let next = state
                        .live
                        .iter()
                        .filter(|iv| iv.expiry.is_finite() && iv.expiry > now)
                        .map(|iv| iv.expiry)
                        .chain(state.stream_free.iter().copied().filter(|&t| t > now))
                        .fold(f64::INFINITY, f64::min);
                    if !next.is_finite() {
                        return Err(PlanError::Infeasible(
                            "list scheduler: no legal placement and no pending expiry".to_string(),
                        ));
                    }
                    state.now = next;
                    state.expire(state.now);
                }
            }
        }

        // Offline BFD repack over the wall-clock lifetime intervals the
        // scheduler produced. This mirrors what the heuristic backend does
        // after fixing its order: sort buffers by size descending, place
        // each in the smallest gap that avoids every already-placed
        // lifetime-overlapping buffer. It's much stronger than the online
        // best-fit inside `commit()` because it sees every buffer at once.
        offline_repack(&mut state, atg);

        Ok(state.into_plan(m as u32))
    }

    #[allow(clippy::too_many_arguments)]
    fn look_ahead_pick(
        &self,
        state: &SchedState,
        atg: &AbstractTimingGraph,
        rt: &[f64],
        bl: &[f64],
        succ: &[Vec<NodeId>],
        node_writes: &[Vec<usize>],
        node_reads: &[Vec<usize>],
        k: usize,
        beam: usize,
        m: usize,
    ) -> Option<usize> {
        let mut candidates: Vec<(f64, usize)> = state
            .ready
            .iter()
            .map(|&v| {
                (
                    self.score_step(state, atg,rt, bl, node_writes, node_reads, v, m),
                    v,
                )
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        candidates.truncate(beam);

        if k <= 1 || candidates.len() == 1 {
            for (_, v) in candidates {
                if self.feasible(state, atg,node_writes, v) {
                    return Some(v);
                }
            }
            return None;
        }

        let mut best_score = f64::INFINITY;
        let mut best_first = None;
        for (_, v) in candidates {
            if !self.feasible(state, atg,node_writes, v) {
                continue;
            }
            let mut trial = state.clone();
            if self
                .commit(&mut trial, v, atg,rt, succ, node_writes, node_reads, m)
                .is_err()
            {
                continue;
            }
            let rollout = self.rollout_score(
                &trial,
                atg,
                rt,
                bl,
                succ,
                node_writes,
                node_reads,
                k - 1,
                beam,
                m,
            );
            if rollout < best_score {
                best_score = rollout;
                best_first = Some(v);
            }
        }
        best_first
    }

    #[allow(clippy::too_many_arguments)]
    fn rollout_score(
        &self,
        state: &SchedState,
        atg: &AbstractTimingGraph,
        rt: &[f64],
        bl: &[f64],
        succ: &[Vec<NodeId>],
        node_writes: &[Vec<usize>],
        node_reads: &[Vec<usize>],
        depth: usize,
        beam: usize,
        m: usize,
    ) -> f64 {
        if depth == 0 || state.order.len() == atg.num_nodes {
            return leaf_score(state);
        }
        let mut candidates: Vec<(f64, usize)> = state
            .ready
            .iter()
            .map(|&v| {
                (
                    self.score_step(state, atg,rt, bl, node_writes, node_reads, v, m),
                    v,
                )
            })
            .collect();
        if candidates.is_empty() {
            return leaf_score(state);
        }
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        candidates.truncate(beam);

        let mut best = f64::INFINITY;
        for (_, v) in candidates {
            if !self.feasible(state, atg,node_writes, v) {
                continue;
            }
            let mut trial = state.clone();
            if self
                .commit(&mut trial, v, atg,rt, succ, node_writes, node_reads, m)
                .is_err()
            {
                continue;
            }
            let s = self.rollout_score(
                &trial,
                atg,
                rt,
                bl,
                succ,
                node_writes,
                node_reads,
                depth - 1,
                beam,
                m,
            );
            if s < best {
                best = s;
            }
        }
        if best.is_infinite() {
            leaf_score(state)
        } else {
            best
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn score_step(
        &self,
        state: &SchedState,
        atg: &AbstractTimingGraph,
        rt: &[f64],
        bl: &[f64],
        node_writes: &[Vec<usize>],
        node_reads: &[Vec<usize>],
        v: usize,
        m: usize,
    ) -> f64 {
        let t_reads = node_reads[v]
            .iter()
            .filter_map(|&b| state.buf_ready.get(&b).map(|(t, _)| *t))
            .fold(0.0, f64::max);
        let t_streams = state
            .stream_free
            .iter()
            .take(m)
            .copied()
            .fold(f64::INFINITY, f64::min);
        let est_start = t_reads.max(t_streams);
        let est_finish = est_start + rt[v];

        let mut mem_delta: i64 = 0;
        for &b in &node_writes[v] {
            if atg.packable(b) && !state.buf_placed[b] {
                mem_delta += atg.size(b);
            }
        }
        for &b in &node_reads[v] {
            if state.remaining_readers[b] == 1
                && atg.packable(b)
                && !atg.pinned(b)
                && state.buf_placed[b]
            {
                mem_delta -= atg.size(b);
            }
        }
        let mem_after = (state.live_bytes as i64 + mem_delta).max(0) as u64;
        let target = if self.max_memory == u64::MAX {
            u64::MAX
        } else {
            (self.max_memory as f64 * self.mem_target_frac) as u64
        };
        let mem_pen = if mem_after > target {
            (mem_after - target) as f64
        } else {
            0.0
        };

        self.w_cp * (bl[v] + est_finish) + self.w_mem * mem_pen
    }

    fn feasible(
        &self,
        state: &SchedState,
        atg: &AbstractTimingGraph,
        node_writes: &[Vec<usize>],
        v: usize,
    ) -> bool {
        // Quick capacity screen: sum of new-birth sizes must fit in
        // whatever pool budget remains (best-fit will discover the exact
        // gap in `commit`).
        if self.max_memory == u64::MAX {
            return true;
        }
        let mut need = 0i64;
        for &b in &node_writes[v] {
            if atg.packable(b) && !state.buf_placed[b] {
                need += atg.size(b);
            }
        }
        state.live_bytes.saturating_add(need as u64) <= self.max_memory
    }

    #[allow(clippy::too_many_arguments)]
    fn commit(
        &self,
        state: &mut SchedState,
        v: NodeId,
        atg: &AbstractTimingGraph,
        rt: &[f64],
        succ: &[Vec<NodeId>],
        node_writes: &[Vec<usize>],
        node_reads: &[Vec<usize>],
        m: usize,
    ) -> Result<(), PlanError> {
        // `pred_stream` is the stream of the latest-arriving input's
        // producer — the node's chain predecessor.
        let mut t_reads = 0.0f64;
        let mut pred_stream = None;
        for &b in &node_reads[v] {
            if let Some(&(t, ps)) = state.buf_ready.get(&b) {
                if t >= t_reads {
                    t_reads = t;
                    pred_stream = Some(ps as usize);
                }
            }
        }

        // Pick the stream that minimizes start time. Ties prefer the
        // chain predecessor's stream, then lowest index for
        // determinism. Chain affinity matters because cost estimates
        // are coarse (uniform): on a start-time tie against a stream
        // holding a queued heavy kernel, the old lowest-index pick
        // would FIFO-queue this node behind that kernel, serializing
        // independent work at runtime — observed with transcript
        // sponge chains queued behind precompute-M builds in the GKR
        // pipelined driver.
        let mut s = 0usize;
        let mut best_start = f64::INFINITY;
        for i in 0..m {
            let start = state.stream_free[i].max(t_reads);
            if start < best_start || (start == best_start && Some(i) == pred_stream) {
                best_start = start;
                s = i;
            }
        }
        let t_start = best_start;
        let t_finish = t_start + rt[v];

        // Expire against `t_start` (this node's actual birth time), not
        // the stale `state.now`. A buffer whose last reader finished on a
        // stream before `t_start` is dead by the time this node's outputs
        // are born and its pool slot should be reusable.
        state.expire(t_start);

        // Allocate outputs (best-fit, aligned) against currently-live intervals.
        let mut new_placements: Vec<(usize, u64)> = Vec::new();
        for &b in &node_writes[v] {
            if !atg.packable(b) || state.buf_placed[b] {
                continue;
            }
            let size = atg.size(b) as u64;
            let align = atg.align(b).max(1);
            let off = state
                .best_fit(size, align, self.max_memory)
                .ok_or_else(|| {
                    PlanError::Infeasible(format!(
                        "list scheduler: cannot fit buffer BufId({b}) of size {size} bytes \
                     within max_memory={} (live_bytes={})",
                        self.max_memory, state.live_bytes
                    ))
                })?;
            new_placements.push((b, off));
            state.offsets[b] = Some(off);
            state.buf_placed[b] = true;
            state.live_bytes += size;
            state.live.push(LiveInterval {
                buf: b,
                offset: off,
                size,
                expiry: f64::INFINITY,
            });
            if (off + size) > state.peak_bytes {
                state.peak_bytes = off + size;
            }
        }

        // Emit cross-stream RAW wait instructions before this node's launch.
        for &b in &node_reads[v] {
            let Some(&(_t_ready, producer_stream)) = state.buf_ready.get(&b) else {
                continue;
            };
            if producer_stream as usize != s {
                let Some(&(producer_node, event_idx)) = state.event_of_buf.get(&b) else {
                    // Producer is off-device or has no event yet (rare):
                    // skip. This can happen for graph inputs (no writer).
                    continue;
                };
                let _ = producer_node;
                state.push_wait(s, event_idx);
            }
        }

        // Cross-stream WAR/WAW waits: a writer of `b` must not launch
        // until every earlier reader of the current version (WAR) and the
        // previous writer (WAW) have finished on their streams. The
        // in-order edges from `PlanCtx::edges` only constrain *commit*
        // order; without an event the GPU streams still race. This
        // matters for in-place SSA alias classes (`restore_ssa` carried
        // outputs), where the new version overwrites the exact bytes the
        // old version's readers are consuming.
        for &b in &node_writes[v] {
            if let Some(&(_t, ws)) = state.buf_ready.get(&b) {
                if ws as usize != s {
                    if let Some(&(_pn, event_idx)) = state.event_of_buf.get(&b) {
                        state.push_wait(s, event_idx);
                    }
                }
            }
            if let Some(readers) = state.readers_since_write.remove(&b) {
                for r in readers {
                    if state.stream[r] as usize != s {
                        if let Some(event_idx) = state.record_event[r] {
                            state.push_wait(s, event_idx);
                        }
                    }
                }
            }
        }

        // Emit the node launch.
        state.instructions.push(StreamInstr::Node(v));
        state.stream[v] = s as u32;
        state.stream_free[s] = t_finish;
        state.now = state.now.max(t_start);
        state.t_start[v] = t_start;
        state.t_finish[v] = t_finish;

        // Every write becomes ready at finish on this stream; the writer
        // count drops so `remaining_writers` means "writers not yet
        // scheduled" below.
        for &b in &node_writes[v] {
            state.buf_ready.insert(b, (t_finish, s as u32));
            state.remaining_writers[b] = state.remaining_writers[b].saturating_sub(1);
        }

        // Assign an event to this node if a downstream node on a
        // different stream may need it: an unscheduled reader of a
        // written buffer (RAW), an unscheduled writer of a written buffer
        // (WAW), or an unscheduled writer of a read buffer (WAR — a
        // future in-place mutation must wait for this read). On a
        // single-stream plan (`m == 1`) no cross-stream sync is ever
        // possible, so gate the whole block on `m > 1` to avoid emitting
        // ~n_nodes worth of no-op `cudaEventRecord`s.
        let mut needs_event = false;
        if m > 1 {
            for &b in &node_writes[v] {
                if state.remaining_readers[b] > 0 || state.remaining_writers[b] > 0 {
                    needs_event = true;
                    break;
                }
            }
        }
        if !needs_event {
            for &b in &node_reads[v] {
                if state.remaining_writers[b] > 0 {
                    needs_event = true;
                    break;
                }
            }
        }
        if needs_event {
            let event_idx = state.num_events;
            state.num_events += 1;
            state.record_event[v] = Some(event_idx);
            for &b in &node_writes[v] {
                if state.remaining_readers[b] > 0 || state.remaining_writers[b] > 0 {
                    state.event_of_buf.insert(b, (v, event_idx));
                }
            }
        }

        // Register this node as a reader of the current version of every
        // buffer it reads, so a future writer can WAR-wait on it. (For
        // in-place nodes the write above already cleared the list; adding
        // `v` here just makes future writers wait on `v` itself, which
        // the WAW path covers anyway.)
        for &b in &node_reads[v] {
            state.readers_since_write.entry(b).or_default().push(v);
        }

        // Decrement reader counts; mark expiries for buffers whose last
        // reader is `v`.
        for &b in &node_reads[v] {
            if state.remaining_readers[b] > 0 {
                state.remaining_readers[b] -= 1;
                if state.remaining_readers[b] == 0
                    && atg.packable(b)
                    && !atg.pinned(b)
                    && state.buf_placed[b]
                {
                    if let Some(iv) = state
                        .live
                        .iter_mut()
                        .find(|iv| iv.buf == b && iv.expiry.is_infinite())
                    {
                        iv.expiry = t_finish;
                    }
                }
            }
        }

        // Release successors whose predecessors are all placed.
        for &u in &succ[v] {
            state.indeg[u] = state.indeg[u].saturating_sub(1);
            if state.indeg[u] == 0 && !state.done[u] {
                state.ready.insert(u);
            }
        }
        state.done[v] = true;
        state.ready.remove(&v);
        state.order.push(v);

        Ok(())
    }
}

fn bottom_levels(succ: &[Vec<NodeId>], rt: &[f64]) -> Vec<f64> {
    let n = succ.len();
    let mut bl = vec![0.0f64; n];
    let order = reverse_topo_order(succ);
    for v in order {
        let mut m = 0.0f64;
        for &u in &succ[v] {
            if bl[u] > m {
                m = bl[u];
            }
        }
        bl[v] = rt[v] + m;
    }
    bl
}

fn reverse_topo_order(succ: &[Vec<NodeId>]) -> Vec<NodeId> {
    let n = succ.len();
    let mut indeg = vec![0usize; n];
    for sv in succ {
        for &v in sv {
            indeg[v] += 1;
        }
    }
    let mut fwd: Vec<NodeId> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut cursor = 0;
    let mut fwd_order: Vec<NodeId> = Vec::with_capacity(n);
    while cursor < fwd.len() {
        let u = fwd[cursor];
        cursor += 1;
        fwd_order.push(u);
        for &v in &succ[u] {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                fwd.push(v);
            }
        }
    }
    fwd_order.reverse();
    fwd_order
}

/// Offline best-fit-decreasing repack over the wall-clock lifetime
/// intervals the scheduler produced.
///
/// This is the same algorithm the heuristic backend runs in `pack_order`,
/// but with continuous-time (`t_start` / `t_finish`) intervals instead of
/// integer positions in a linear order — so it stays correct for the
/// multi-stream case. Two buffers can share memory iff their `[birth,
/// death]` half-open intervals don't overlap on the timeline.
///
/// Sorts buffers by `(size desc, birth asc, id)` and places each at the
/// smallest aligned offset that avoids every already-placed
/// lifetime-overlapping buffer. Overwrites `state.offsets` and
/// `state.peak_bytes` with the tighter packing; the schedule (order,
/// stream assignment, instructions, events) is untouched.
fn offline_repack(state: &mut SchedState, atg: &AbstractTimingGraph) {
    let n_bufs = atg.n_bufs();
    let n_nodes = atg.num_nodes;

    // Per-buffer lifetime in wall-clock time.
    //
    // * Buffers with no writer (graph inputs / pinned inputs) are alive from time `-inf` — they
    //   must reserve their slot from the start.
    // * Buffers pinned to the end of the program die at `+inf`.
    // * A regular buffer's lifetime spans the earliest writer's `t_start` through the latest
    //   access's `t_finish`. Writers count as accesses (a pure overwrite must keep the previous
    //   slot valid through the overwrite itself, matching CP-SAT semantics).
    let mut birth = vec![f64::INFINITY; n_bufs];
    let mut death = vec![f64::NEG_INFINITY; n_bufs];
    let mut has_writer = vec![false; n_bufs];
    let mut has_reader = vec![false; n_bufs];
    // Single stream that touches each buf, or `Some(None)` if the buf
    // is touched by more than one stream. Outer option tracks "seen
    // any touch at all"; inner tracks the single-stream tag.
    let mut buf_stream: Vec<Option<u32>> = vec![None; n_bufs];
    let mut buf_multi: Vec<bool> = vec![false; n_bufs];
    for v in 0..n_nodes {
        let s = state.stream[v];
        let t_start = state.t_start[v];
        let t_finish = state.t_finish[v];
        for b in &atg.node_produces[v] {
            let b = b.0;
            if !atg.packable(b) {
                continue;
            }
            has_writer[b] = true;
            if t_start < birth[b] {
                birth[b] = t_start;
            }
            if t_finish > death[b] {
                death[b] = t_finish;
            }
            match buf_stream[b] {
                None => buf_stream[b] = Some(s),
                Some(prev) if prev != s => buf_multi[b] = true,
                _ => {}
            }
        }
        for b in &atg.node_consumes[v] {
            let b = b.0;
            if !atg.packable(b) {
                continue;
            }
            has_reader[b] = true;
            if t_finish > death[b] {
                death[b] = t_finish;
            }
            match buf_stream[b] {
                None => buf_stream[b] = Some(s),
                Some(prev) if prev != s => buf_multi[b] = true,
                _ => {}
            }
        }
    }
    for b in 0..n_bufs {
        if !has_writer[b] {
            birth[b] = f64::NEG_INFINITY;
        }
        if atg.pinned(b) || !has_reader[b] {
            death[b] = f64::INFINITY;
        }
        if buf_multi[b] {
            buf_stream[b] = None;
        }
    }

    // Two buffers "overlap" (cannot share pool bytes) if EITHER:
    //   (a) at least one of them is touched by more than one stream; or
    //   (b) both are single-stream but on *different* streams; or
    //   (c) both are on the same single stream AND their wall-clock lifetime intervals overlap.
    //
    // A cross-stream release/acquire without an injected sync edge is
    // a data race at the CUDA layer: `commit()` emits `WaitOn`s for
    // data deps on the *same* canonical BufId, not for slot reuse
    // across different canonicals that happen to alias into one
    // offset. The pipelined GKR driver at `max_concurrency >= 2` was
    // producing wrong claims until this guard was restored (previously
    // relaxed to lifetime-only overlap for peak-memory reasons).
    let overlaps = |b1: usize, b2: usize| -> bool {
        match (buf_stream[b1], buf_stream[b2]) {
            (Some(s1), Some(s2)) if s1 == s2 => !(death[b1] < birth[b2] || death[b2] < birth[b1]),
            _ => true,
        }
    };

    let mut to_place: Vec<usize> = (0..n_bufs).filter(|&b| atg.packable(b)).collect();
    to_place.sort_by(|&a, &b| {
        let sa = atg.size(a);
        let sb = atg.size(b);
        sb.cmp(&sa)
            .then_with(|| birth[a].total_cmp(&birth[b]))
            .then_with(|| a.cmp(&b))
    });

    let mut offsets: Vec<Option<u64>> = vec![None; n_bufs];
    let mut peak: u64 = 0;

    for &b in &to_place {
        let size = atg.size(b) as u64;
        let align = atg.align(b).max(1);

        let mut blocked: Vec<(u64, u64)> = to_place
            .iter()
            .filter_map(|&b2| {
                if b2 == b {
                    return None;
                }
                let off = offsets[b2]?;
                overlaps(b, b2).then_some((off, off + atg.size(b2) as u64))
            })
            .collect();
        blocked.sort_unstable();

        let mut o: u64 = 0;
        let mut placed = false;
        for &(a_off, a_end) in &blocked {
            let o_aligned = align_up_u(o, align);
            if o_aligned + size <= a_off {
                o = o_aligned;
                placed = true;
                break;
            }
            if a_end > o {
                o = a_end;
            }
        }
        if !placed {
            o = align_up_u(o, align);
        }
        offsets[b] = Some(o);
        if o + size > peak {
            peak = o + size;
        }
    }

    state.offsets = offsets;
    state.peak_bytes = peak;
}

fn leaf_score(state: &SchedState) -> f64 {
    let makespan = state.stream_free.iter().copied().fold(0.0f64, f64::max);
    makespan + 1e-3 * (state.peak_bytes as f64)
}

/// A currently-allocated interval in the memory pool.
#[derive(Debug, Clone)]
struct LiveInterval {
    buf: usize,
    offset: u64,
    size: u64,
    /// Simulated time at which this interval frees. `+inf` if the last
    /// reader hasn't been scheduled yet.
    expiry: f64,
}

#[derive(Clone)]
struct SchedState {
    /// Order of committed node launches (excluding WaitOns).
    order: Vec<NodeId>,
    /// Flat instruction stream — the eventual `plan.instructions`.
    instructions: Vec<StreamInstr>,
    /// Stream assignment per graph node (0 for unplaced).
    stream: Vec<u32>,
    /// Event recorded after node completes; `None` if no downstream
    /// cross-stream consumer.
    record_event: Vec<Option<u32>>,
    /// Placed offset per BufId; `None` for unplaced/off-device.
    offsets: Vec<Option<u64>>,
    buf_placed: Vec<bool>,

    /// Simulated wall-clock. Advances only when no candidate fits.
    now: f64,
    /// Next-free time per stream.
    stream_free: Vec<f64>,
    /// `(finish_time, stream)` when each written buffer becomes readable.
    buf_ready: HashMap<usize, (f64, u32)>,
    /// `(producer_node, event_idx)` for each buffer that a consumer on
    /// another stream may need to wait for.
    event_of_buf: HashMap<usize, (NodeId, u32)>,
    num_events: u32,

    /// Currently-allocated intervals in the memory pool.
    live: Vec<LiveInterval>,
    live_bytes: u64,
    peak_bytes: u64,
    /// Per-node wall-clock start / finish times, populated as `commit`
    /// places each node. Used by the offline BFD post-pass to derive
    /// buffer lifetimes independent of the online allocator's decisions.
    t_start: Vec<f64>,
    t_finish: Vec<f64>,

    /// Reader countdown for buffer death.
    remaining_readers: Vec<usize>,
    /// Writer countdown per buffer (canonical ids). Non-zero means a
    /// future writer exists that must WAR/WAW-wait on current accesses.
    remaining_writers: Vec<usize>,
    /// Nodes that read each buffer since its last writer was scheduled —
    /// the WAR-wait set for the next writer of that buffer.
    readers_since_write: HashMap<usize, Vec<NodeId>>,
    /// In-degree per node (predecessors still to schedule).
    indeg: Vec<usize>,
    ready: BTreeSetLike,
    done: Vec<bool>,
}

/// A tiny ordered set backed by a sorted Vec, used to iterate `ready`
/// candidates deterministically.
#[derive(Clone)]
struct BTreeSetLike {
    v: Vec<NodeId>,
}
impl BTreeSetLike {
    fn new() -> Self {
        Self { v: Vec::new() }
    }
    fn insert(&mut self, x: NodeId) {
        if let Err(pos) = self.v.binary_search(&x) {
            self.v.insert(pos, x);
        }
    }
    fn remove(&mut self, x: &NodeId) {
        if let Ok(pos) = self.v.binary_search(x) {
            self.v.remove(pos);
        }
    }
    fn iter(&self) -> impl Iterator<Item = &NodeId> {
        self.v.iter()
    }
}

impl SchedState {
    fn new(atg: &AbstractTimingGraph, indeg: &[usize], _node_reads: &[Vec<usize>], m: usize) -> Self {
        let n = atg.num_nodes;
        let mut ready = BTreeSetLike::new();
        for (i, &d) in indeg.iter().enumerate().take(n) {
            if d == 0 {
                ready.insert(i);
            }
        }
        Self {
            order: Vec::with_capacity(n),
            instructions: Vec::new(),
            stream: vec![0; n],
            record_event: vec![None; n],
            offsets: vec![None; atg.n_bufs()],
            buf_placed: vec![false; atg.n_bufs()],
            now: 0.0,
            stream_free: vec![0.0; m],
            buf_ready: HashMap::new(),
            event_of_buf: HashMap::new(),
            num_events: 0,
            live: Vec::new(),
            live_bytes: 0,
            peak_bytes: 0,
            remaining_readers: {
                let mut r = vec![0usize; atg.n_bufs()];
                for consumes in &atg.node_consumes {
                    for b in consumes {
                        r[b.0] += 1;
                    }
                }
                r
            },
            remaining_writers: {
                let mut w = vec![0usize; atg.n_bufs()];
                for produces in &atg.node_produces {
                    for b in produces {
                        w[b.0] += 1;
                    }
                }
                w
            },
            readers_since_write: HashMap::new(),
            indeg: indeg.to_vec(),
            ready,
            done: vec![false; n],
            t_start: vec![0.0; n],
            t_finish: vec![0.0; n],
        }
    }

    /// Emit a de-duplicated `WaitOn(s, event)` instruction.
    fn push_wait(&mut self, s: usize, event_idx: u32) {
        let dup = self.instructions.iter().any(|instr| {
            matches!(instr, StreamInstr::WaitOn(ss, ee) if *ss == s && *ee == event_idx as usize)
        });
        if !dup {
            self.instructions
                .push(StreamInstr::WaitOn(s, event_idx as usize));
        }
    }

    /// Expire (deallocate) any live intervals whose expiry is `<= now`.
    fn expire(&mut self, now: f64) {
        let mut i = 0;
        while i < self.live.len() {
            if self.live[i].expiry.is_finite() && self.live[i].expiry <= now {
                let iv = self.live.swap_remove(i);
                self.live_bytes = self.live_bytes.saturating_sub(iv.size);
                // Note: we keep `offsets[iv.buf]` set (the buffer keeps its
                // planned slot in the output plan; expiry only makes the
                // range available for another placement).
            } else {
                i += 1;
            }
        }
    }

    /// Best-fit placement of `size` bytes at `align` alignment, avoiding
    /// every live interval. Returns the chosen offset.
    fn best_fit(&self, size: u64, align: u64, max_memory: u64) -> Option<u64> {
        // Sort blocked intervals by offset.
        let mut blocked: Vec<(u64, u64)> = self
            .live
            .iter()
            .map(|iv| (iv.offset, iv.offset + iv.size))
            .collect();
        blocked.sort_unstable();

        let cap = max_memory;

        // Sweep gaps: [0..blocked[0].0), [blocked[i].1 .. blocked[i+1].0), ...
        let mut best: Option<(u64, u64)> = None; // (gap_size, offset)
        let mut cursor: u64 = 0;
        for &(lo, hi) in &blocked {
            let start = align_up_u(cursor, align);
            if start + size <= lo {
                let gap = lo - start;
                if best.is_none_or(|(g, _)| gap < g) {
                    best = Some((gap, start));
                }
            }
            cursor = cursor.max(hi);
        }
        let start = align_up_u(cursor, align);
        if start.saturating_add(size) <= cap {
            let gap = cap - start;
            if best.is_none_or(|(g, _)| gap < g) {
                best = Some((gap, start));
            }
        }
        best.map(|(_, o)| o)
    }

    fn into_plan(self, num_streams: u32) -> StreamMemoryPlan {
        StreamMemoryPlan {
            instructions: self.instructions,
            stream: self.stream,
            record_event: self.record_event,
            offsets: self.offsets,
            peak_bytes: self.peak_bytes,
            num_streams,
            num_events: self.num_events,
        }
    }
}

#[inline]
fn align_up_u(off: u64, align: u64) -> u64 {
    if align > 1 {
        off.div_ceil(align) * align
    } else {
        off
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph_ir::{BufId, BufInfo, DeviceType},
        quast::Quast,
    };

    fn make_bufs(sizes: &[i64]) -> Vec<BufInfo> {
        sizes
            .iter()
            .enumerate()
            .map(|(i, &s)| BufInfo {
                name: Some(format!("b{i}")),
                device_type: DeviceType::Cuda(0),
                size: Quast::cst(s),
                concrete_size: s as usize,
                elem_size: 4,
            })
            .collect()
    }

    #[test]
    fn single_node_schedules() {
        let bufs = make_bufs(&[64]);
        let atg = AbstractTimingGraph::from_accesses(
            bufs,
            &[vec![]],
            &[vec![BufId(0)]],
            vec![1.0],
            DeviceType::Cuda(0),
            vec![],
            vec![],
        );
        let sched = ListSchedulerV1::default();
        let plan = sched.schedule(&atg).unwrap();
        assert_eq!(plan.instructions.len(), 1);
        assert!(matches!(plan.instructions[0], StreamInstr::Node(0)));
        assert_eq!(plan.stream[0], 0);
    }

    #[test]
    fn two_independent_nodes_get_different_streams() {
        // Two independent producers, one consumer of both. With M=2 we
        // want the producers on different streams and the consumer to
        // WaitOn one of them.
        let bufs = make_bufs(&[64, 64, 128]);
        let a = BufId(0);
        let b = BufId(1);
        let c = BufId(2);
        let reads = vec![vec![], vec![], vec![a, b]];
        let writes = vec![vec![a], vec![b], vec![c]];
        let atg = AbstractTimingGraph::from_accesses(
            bufs,
            &reads,
            &writes,
            vec![1.0; 3],
            DeviceType::Cuda(0),
            vec![],
            vec![],
        );
        let sched = ListSchedulerV1 {
            max_concurrency: 2,
            ..Default::default()
        };
        let plan = sched.schedule(&atg).unwrap();
        // The two producers should be on different streams.
        assert_ne!(plan.stream[0], plan.stream[1]);
        // A WaitOn must be emitted before the consumer.
        assert!(plan
            .instructions
            .iter()
            .any(|i| matches!(i, StreamInstr::WaitOn(_, _))));
    }
}
