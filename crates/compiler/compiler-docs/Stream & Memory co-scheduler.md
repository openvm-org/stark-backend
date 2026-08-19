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
- `T` is the critical path time
- `depth_cost = number_of_nodes_to_end(n)`
- `D` is the max graph depth
- `w_m, w_t, w_c` are all weight parameters

Then given a `ready_set`of nodes and a current schedule state `St`, the beam search runs as follows:
1. expand every possible action, which is the cartesian product `ready_set x {1,...,num_streams}` to form new states `St_1, ..., St_n`
2. take the top `num_beams` states
3. repeat above for `beam_depth`
4. take the lowest cost action, set the `ready_set` and `St` accordingly as the result of taking this action



