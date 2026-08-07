# Stream & Memory co-scheduling

Currently the graph IR compiler runs everything on one stream, and plans the memory. A more optimal scheduling would consider memory + stream. 

More precisely, in the graph IR, suppose we have approximate run-times for each node. Each node also has input and output edges that are buffers with size and alignment requirements. We set a `max_concurrency` threshold and `max_memory` threshold. `max_concurrency` is the maximum number of streams allowed and `max_memory` is the maximum bytes of memory allowed. 


## List Scheduling V1

A greedy list scheduler with depth-`k` look-ahead. It produces a `StreamMemoryPlan`:

```rust
pub struct StreamMemoryPlan {
    /// Execution order the runtime should issue launches in (topological).
    pub order: Vec<NodeId>,
    /// Stream assignment per node, in `[0, max_concurrency)`.
    pub stream: Vec<u32>,
    /// Byte offset in `[0, max_memory)` per BufId; `None` if off-device.
    pub offset: Vec<Option<u64>>,
    /// Cross-stream sync edges: `(producer_node, consumer_node)` where
    /// `stream[producer] != stream[consumer]`. The runtime records an
    /// event after `producer` and makes `consumer`'s stream wait on it.
    pub sync_edges: Vec<(NodeId, NodeId)>,
    pub peak_bytes: u64,
    pub makespan: f64,
}
```

### Inputs

- Graph `G = (V, E)` from `GraphBuilder`: nodes `V`, precedence edges `E` derived from RAW/WAW/WAR on `BufId`s (as in `planner::PlanCtx`).
- Runtime estimate `rt: V → f64` (nanoseconds).
- Buffer sizes `size: BufId → u64` (evaluated from `Quast` under the current env).
- Buffer lifetimes derived from node reads/writes; `pin` set kept alive to end of program (same semantics as `planner::plan_raw`).
- Knobs: `max_concurrency: u32`, `max_memory: u64`, look-ahead depth `k: usize`.

### State

Maintained during scheduling:

- `ready: BinaryHeap<NodeId>` — nodes with all predecessors placed, ordered by static priority (see below).
- `stream_free_at: [f64; max_concurrency]` — earliest time each stream can start its next node.
- `buf_ready_at: HashMap<BufId, (f64, u32)>` — when each written buffer becomes readable, and on which stream it was produced (for sync-edge emission).
- `pending_readers: HashMap<BufId, usize>` — outstanding reader count; buffer frees at max completion time of its readers once this hits 0.
- `live: IntervalTree<u64>` — currently reserved byte ranges in `[0, max_memory)` keyed by expiry time; supports best-fit query and lazy expiry.
- `now: f64` — simulated wall-clock, advanced only when we must wait (all streams busy, or memory too tight to place any candidate).

### Static priorities

Compute once, before the main loop:

- `bl(v) = rt(v) + max_{u ∈ succ(v)} bl(u)` (bottom-level / longest weighted path to a sink). Standard critical-path heuristic — schedules the node that most limits makespan first.
- Tiebreak by `memory_delta(v) = sum(size of v's outputs) - sum(size of v's inputs that die at v)`. Prefer negative deltas when memory is pressured, positive when it's slack (see selection score below).

These are lower bounds under infinite concurrency; they don't change as scheduling progresses.

### Selection score (single-step)

For a candidate `v` under the current state, compute:

```
est_start(v)   = max(  buf_ready_at[b].0  for b in reads(v),
                       min(stream_free_at) )
est_finish(v)  = est_start(v) + rt(v)
mem_after(v)   = live_bytes(now) + births(v) - deaths(v)
score_step(v)  = w_cp   * (bl(v) + est_finish(v))          // pushes critical path
               + w_mem  * max(0, mem_after(v) - target_mem) // penalise pressure
               + w_frag * fragmentation_after_placing(v)    // best-fit residuals
```

Lower is better. `target_mem = 0.9 * max_memory` gives headroom; `w_*` are constants tuned empirically (start with `w_cp = 1.0`, `w_mem = 1e-3 / byte`, `w_frag = 1e-6`).

### Look-ahead of depth `k`

At each pick, do not commit to `argmin score_step` directly. Instead:

1. Take the top `b` candidates from `ready` by `score_step` (`b` a small branching factor, e.g. 4).
2. For each candidate `v`, snapshot the scheduler state, tentatively place `v`, and recursively expand `k-1` more picks, again keeping the top `b` at each level (beam search with beam width `b`, depth `k`).
3. Score each leaf state by `α * simulated_makespan + β * peak_memory_in_rollout`.
4. Commit only the *first* move of the best rollout, then re-plan from the new state.

`k` trades solution quality for compile time: `k = 1` is pure greedy; `k = 3..5` is usually enough to escape local traps like "greedy picked a small-output node and starved a large-output node into a memory cliff". Cost per decision is `O(b^k · (place + score))`.

Terminate look-ahead early if a rollout exhausts `ready` (schedule complete) or if `est_finish` at some level exceeds the current best makespan by more than a slack.

### Placement of a node `v`

1. **Free expired intervals.** Drop everything in `live` whose expiry `≤ now`.
2. **Pick a stream.** Choose `s = argmin stream_free_at[s]` subject to `stream_free_at[s] ≤ est_start(v)`. If none satisfies, either advance `now` to the earliest such time (V1: block) or, if look-ahead prefers, keep `v` in `ready` and pick a different candidate.
3. **Allocate memory for outputs.** For each output `b`:
   - Best-fit search over free gaps in `[0, max_memory)`, honoring `align(b)`.
   - If no gap fits, either (a) sink lower-priority live buffers by delaying their late reader (not allowed in V1 — dependencies are fixed), or (b) fail this candidate and let look-ahead pick another. If the whole `ready` set fails, advance `now` to the next expiry event and retry.
4. **Emit sync edges.** For each `b ∈ reads(v)` with `buf_ready_at[b].1 != s`, push `(writer_of_b, v)` onto `sync_edges`.
5. **Commit.**
   - `order.push(v); stream[v] = s;`
   - `offset[b] = Some(chosen_offset)` for new outputs `b`.
   - `stream_free_at[s] = est_finish(v);`
   - `buf_ready_at[b] = (est_finish(v), s)` for outputs.
   - Decrement `pending_readers[b]` for each read; when zero, mark `b`'s interval expiring at `max_reader_finish(b)`.
   - Enqueue any successor whose predecessors are all placed.

### Main loop (pseudocode)

```
init ready = { v ∈ V : preds(v) = ∅ }
while ready not empty:
    v = look_ahead_pick(ready, state, depth=k, beam=b)
    if v is None:
        now = next_expiry_or_stream_free()      // no legal placement now
        continue
    place(v)                                     // updates state as above
finalize:
    peak_bytes  = max live_bytes(t) over the run
    makespan    = max stream_free_at[s]
return StreamMemoryPlan { order, stream, offset, sync_edges, peak_bytes, makespan }
```

### Invariants and edge cases

- Nodes with `carried_outputs` (in-place writes on `KernelNode`) treat the carried `BufId` as both a read and a write; its lifetime is unchanged by the node, but its version bumps so subsequent readers depend on `v`.
- `MemcpyNode` / `MemSetNode` are single-node placements just like kernels; runtime estimate uses `num_bytes / bandwidth`.
- Pinned buffers get their offset once and never expire; they participate in `live` for the entire schedule.
- If `max_concurrency = 1`, the scheduler degenerates into the current single-stream planner but keeps the same memory allocator.
- If placement of an output buffer never fits (buffer is larger than `max_memory` minus the pinned footprint), return an error rather than looping.

### What V1 does *not* do

- No spilling to host memory when device memory is tight — placement failure blocks until an expiry frees space, and errors if the graph can never fit.
- No reordering of already-placed nodes (no local search / kernel swap). Look-ahead is the only backtracking mechanism.
- No use of `NodeAccess.reads` in-place flag beyond treating the buffer as a write; no analysis of partial overlap between `MemcpyNode` src/dst ranges.
- Sync edges are emitted per (producer, consumer) pair; V1 does not coalesce multiple consumers of the same producer on the same stream into one event wait. (Runtime can dedupe on lowering.)

