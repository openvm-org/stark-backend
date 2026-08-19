The interface is as follows: every scheduler takes `AbstractTimingGraph` and produces `StreamMemoryPlan`.

`AbstractTimingGraph` is a graph with nodes and edges that has profile guided information about the time it takes for each node to run. 

```rust
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
```

`StreamMemoryPlan` assigns each buf an offset and each node a stream, along with an ordering of execution for each stream.

```rust
pub struct StreamMemoryPlan {
    pub instructions: Vec<StreamInstr>,
    pub stream: Vec<u32>,
    pub record_event: Vec<Option<u32>>,
    pub offsets: Vec<Option<u64>>,
    pub peak_bytes: u64,
    pub num_streams: u32,
    pub num_events: u32,
}

pub enum StreamInstr {
    /// Execute graph node `node_idx` on its assigned stream
    /// (`StreamMemoryPlan::stream[node_idx]`). Immediately after the launch
    /// completes on the host side, if
    /// `StreamMemoryPlan::record_event[node_idx]` is `Some(e)`, event `e`
    /// is recorded on that stream so downstream waiters can synchronize.
    Node(usize),
    /// `WaitOn(stream_idx, event_idx)`: enqueue on stream `stream_idx` a
    /// wait for event `event_idx`. The event is recorded elsewhere in the
    /// plan by the producer node whose `record_event` slot equals
    /// `event_idx`.
    WaitOn(usize, usize),
}
```

Running moderate graph sizes by formulating the optimization problem in CP-SAT takes too long. The current v2 scheduler runs a beam search with look-ahead on the DAG, which formulates the problem as a markov decision process. The action is `(n, s)`, which puts node `n` on stream `s` and the state is `(S, N_t, N_s, B, L, P)` where
- `S[j]` is the end time of stream `j` for all nodes scheduled so far.
- `N_t[i]` is the end time of node `i`
- `N_s[i]` is the stream of node `i`
- `B` is set of live buffers
- `L` is the set of ready nodes to be scheduled
- `P` is the current peak memory

Then the cost for action `(n, s)` is `w_m * mem_delta / M + w_t * time_cost / T + w_c * depth_cost / D`
where:
- `mem_delta = bytes_produced_by(n) - bytes_killed_by(n)`
- `M` is the max memory bound set as a parameter
- `time_cost = max(max_of_dep_end_times, end_t_on_stream(s))`
- `T` is the critical path time (or some other norm, like sum of times)
- `depth_cost = number_of_nodes_to_end(n)`
- `D` is the max graph depth
- `w_m, w_t, w_c` are all weight parameters

Then given a `ready_set`of nodes and a current schedule state `St`, the beam search runs as follows:
1. expand every possible action, which is the cartesian product `ready_set x {1,...,num_streams}` to form new states `St_1, ..., St_n`
2. take the top `num_beams` states
3. repeat above for `beam_depth`
4. take the lowest cost action, set the `ready_set` and `St` accordingly as the result of taking this action

## Concrete algorithm (`list_v2.rs`)

The implementation splits into two phases: a beam search that assigns every node to a stream (`plan_v2`), and a post-pass that materialises the assignment into a `StreamMemoryPlan` (`make_schedule`).

### Precomputation

Before the search starts, `ScheduleState::new` computes three things that stay constant for the whole run and are shared across every beam via `Arc`:

- `bl[v]` — the *bottom level* of each node: the longest weighted path from `v` to any sink (including `v`'s own `node_times[v]`). Classic list-scheduling priority; higher `bl` means `v` sits on a longer downstream chain. Used both as the `w_c` term in `cost` and as the frontier-cap ordering key.
- `s_norm = sum(node_times)` — the sequential single-stream time. Normaliser for the `time_cost` term.
- `b_norm = max(bl)` — the critical-path lower bound on makespan. Normaliser for the `bl[node]` term.
- `initial_deps[v]` — the count of `v`'s consumed buffers that have an in-graph producer (i.e. non-input reads). This is the initial value of the dependency counter used to promote nodes into `ready_queue`.

### State representation

`ScheduleState` uses the persistent (structurally-shared) collections from the `im` crate so that `state.clone()` — called once per beam expansion — is O(1) refcount bumps rather than a full copy. The persistent fields are:

- `live_bufs: ImHashMap<BufId, ImHashSet<usize>>` — for each currently-allocated buffer, the set of its still-unscheduled consumers. A buffer enters on produce and disappears when its consumer set empties. Graph inputs are deliberately *not* tracked (their memory sits outside the scheduler's budget).
- `stream_of: ImHashMap<usize, usize>` and `node_start_times: ImHashMap<usize, f64>` — placement records that grow one entry per `put_on`.
- `ready_queue: ImHashSet<usize>` — nodes with all producers scheduled.
- `remaining_deps: ImHashMap<usize, usize>` — a *lazy* dep counter. A node only gets an entry once one of its producers first schedules (initial value = `initial_deps[v] - 1`); subsequent producer visits decrement; hitting 0 moves the node into `ready_queue`. This keeps the map size proportional to the "partially resolved" frontier rather than to `num_nodes`.

Alongside these, `stream_end_t: Vec<f64>` is a plain `Vec` (cheap to clone at `num_streams` entries), and `cur_mem_used`, `max_peak_mem`, `cur_max_time` are scalar accumulators.

### `put_on(node, stream)`

Applies an action in place on a (freshly cloned) state:

1. Compute the earliest start on `stream` as `max(max_producer_end_t, stream_end_t[stream])` and record `node_start_times[node]`, `stream_of[node]`, and the new `stream_end_t[stream]`.
2. For each consumed buffer, remove `node` from its live-consumer set. If the set empties, the buffer dies and its bytes are credited to `killed_bytes`.
3. For each produced buffer, insert it into `live_bufs` with its unscheduled-consumer set and, for each such consumer, decrement `remaining_deps` (lazily, per the initial-count rule above). Consumers that hit 0 move into `ready_queue`.
4. Update `cur_mem_used`, `max_peak_mem`, and `cur_max_time`; remove `node` from `ready_queue`.

### `cost(node, stream)`

Returns `f64::INFINITY` if placing `node` would push `cur_mem_used + produced_bytes` above `m_bound` (hard feasibility filter — the beam drops these). Otherwise computes the normalised three-term score exactly as described earlier:

```
cost = w_m * mem_delta / m_bound
     + w_t * time_cost / s_norm
     + w_c * bl[node]  / b_norm
```

where `time_cost = max(max_producer_end_t, stream_end_t[stream]) - cur_max_time` — i.e. the *increase* in wall-clock makespan the action would cause. `w_c` is conventionally negative so that higher `bl` (more downstream work) lowers cost.

### The `plan_v2` outer loop

While `ready_queue` is non-empty:

1. Seed a beam list with a single entry: `(current.clone(), [], 0.0)` — state, trajectory of actions taken so far, cumulative cost.
2. For `beam_depth` levels:
   a. Cartesian-product every beam with `0..num_streams`. Each `(beam, stream)` pair will fan out over the ready set on that beam's state.
   b. In parallel (`rayon`), expand each pair: for each ready node, evaluate `cost(node, stream)`, skip if infinite, otherwise `clone + put_on` and push the child beam.
   c. Sort all children by cumulative cost and keep the top `num_beams`.
3. Commit only the *first* action of the best resulting trajectory to `current` via `put_on`, then restart the outer loop from the new state.

Two knobs bound the fan-out:

- `frontier_cap` — before expanding a beam, if `ready_queue.len() > frontier_cap`, keep only the top `frontier_cap` nodes by `bl` (critical-path priority). Prevents the branching factor from blowing up on wide graphs.
- `num_beams` — cap after each depth level.

Committing only the first action (rather than the entire best rollout) means the look-ahead informs the *choice* of first move but the search always re-plans from the true committed state — closer in spirit to MPC than to trajectory optimisation.

### `make_schedule`: turning the assignment into a `StreamMemoryPlan`

Once every node has a stream, `make_schedule` produces the linear instruction stream, event assignments, and buffer offsets:

1. **Per-stream order.** `per_stream_orders` runs a Kahn's topological sort filtered by `stream_of`, tie-breaking by lowest node id, to recover the intra-stream sequence for each stream.
2. **Event assignment.** A producer needs an event iff any of its outputs is consumed by a node on a different stream. Event ids are handed out in visit order.
3. **Instruction interleaving.** Sweep the per-stream cursors round-robin. Before dispatching a node, check every cross-stream data dep: if the target stream has already synced past the producer's position (tracked in `last_synced[dst_stream][src_stream]`), the sync is implied by transitive event ordering and skipped. Otherwise emit one `WaitOn` per source stream (merging multiple producers on the same source stream into a single wait at the max position), then the `Node` instruction.
4. **Simulation for lifetimes.** Replay the emitted instruction stream under the same perfect-parallel timing model that `perf_est` and `validate_plan` use, deriving `sim_start[v]` and `sim_finish[v]` for every node.
5. **Interference graph.** For each packable (CUDA) buffer, `[birth, death]` are the simulated launch of its earliest producer and the simulated finish of its last user, with graph inputs pinned to `birth = 0` and outputs pinned to `death = ∞`. Two buffers interfere unless they both live entirely on the *same* stream and their wall-clock intervals are disjoint. The cross-stream restriction is deliberate: `make_schedule` only emits `WaitOn`s for same-canonical RAW deps, so slot handoffs between different canonicals on different streams would be data races even if wall-clock-disjoint. Same-stream reuse is safe because stream ordering guarantees release-before-acquire.
6. **Best-fit-decreasing packing.** Sort packable buffers by size (descending), then earliest birth, then id. For each buffer, gather the offset intervals of already-placed interfering neighbours, walk them in ascending order, and pick the lowest aligned offset that fits before the next blocked region.

The result is the `StreamMemoryPlan`: `instructions`, per-node `stream` and `record_event`, per-buffer `offsets`, and the final `peak_bytes`.

