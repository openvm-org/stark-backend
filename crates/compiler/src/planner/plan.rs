//! Output of every planner backend.
//!
//! A `StreamMemoryPlan` describes both the execution order and the
//! memory pool layout, with optional multi-stream scheduling: each graph
//! node is assigned an integer stream index, and cross-stream data
//! dependencies are enforced by [`StreamInstr::WaitOn`] instructions
//! interleaved with node launches.
//!
//! Single-stream planners (heuristic, CP-SAT) emit `num_streams = 1`,
//! `stream[..] = 0`, and no `WaitOn` instructions — equivalent to the
//! prior `MemoryPlan` shape.

/// One issued instruction in a plan's execution sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Joint execution schedule + memory layout.
///
/// * `instructions` is the flat sequence the runtime issues in order.
/// * `stream[b]` is the stream index every graph node runs on
///   (`0..num_streams`).
/// * `record_event[b]` is `Some(e)` for nodes whose completion downstream
///   waiters synchronize on; `None` for nodes with no cross-stream
///   consumer.
/// * `offsets[b]` is the byte offset per `BufId` in the unified pool
///   (`None` for off-device buffers).
///
/// For a single-stream plan: `num_streams = 1`, `num_events = 0`,
/// `stream = vec![0; n_nodes]`, `record_event = vec![None; n_nodes]`, and
/// `instructions` is just `Node(order[i])` in execution order.
#[derive(Debug, Clone)]
pub struct StreamMemoryPlan {
    pub instructions: Vec<StreamInstr>,
    pub stream: Vec<u32>,
    pub record_event: Vec<Option<u32>>,
    pub offsets: Vec<Option<u64>>,
    pub peak_bytes: u64,
    pub num_streams: u32,
    pub num_events: u32,
}

impl StreamMemoryPlan {
    /// Builds a single-stream plan from a linear execution `order` and
    /// buffer `offsets`. Every node runs on stream 0, no events, no waits.
    /// This is the shape the CP-SAT and heuristic backends emit.
    pub fn single_stream(
        order: Vec<usize>,
        offsets: Vec<Option<u64>>,
        peak_bytes: u64,
        n_nodes: usize,
    ) -> Self {
        let instructions = order.into_iter().map(StreamInstr::Node).collect();
        Self {
            instructions,
            stream: vec![0; n_nodes],
            record_event: vec![None; n_nodes],
            offsets,
            peak_bytes,
            num_streams: 1,
            num_events: 0,
        }
    }

    /// Linear order of node launches, skipping `WaitOn` instructions.
    /// Convenience for debug prints / consumers that only need the
    /// permutation.
    pub fn order(&self) -> Vec<usize> {
        self.instructions
            .iter()
            .filter_map(|i| match i {
                StreamInstr::Node(n) => Some(*n),
                _ => None,
            })
            .collect()
    }
}
