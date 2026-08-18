//! Correctness validator for [`StreamMemoryPlan`].
//!
//! Verifies that a plan is a legal schedule of an
//! [`AbstractTimingGraph`]:
//!
//! * **Structural** — every node scheduled exactly once, stream and
//!   event indices in range, exactly one recorded producer per event.
//! * **Data-dependency ordering** — for every RAW edge `P → C` the
//!   simulated finish time of `P` is `≤` the simulated start time of
//!   `C` on `C`'s stream (missing / mis-placed `WaitOn`s show up
//!   here because the simulator advances a stream's clock only when
//!   it explicitly waits).
//! * **Cross-stream sync existence** — for every cross-stream RAW
//!   edge, some `WaitOn(C.stream, record_event[P])` instruction
//!   appears in the flat instruction stream at a global position
//!   strictly after `Node(P)` and strictly before `Node(C)`.
//! * **Memory-pool safety** — no two buffers share pool bytes while
//!   their simulated lifetimes overlap. Lifetime of a buf =
//!   `[birth, death]` where `birth = producer's simulated start`
//!   (or 0 for graph inputs) and `death = max simulated finish over
//!   readers` (or `f64::INFINITY` for graph outputs).
//!
//! The validator runs entirely on the plan + [`AbstractTimingGraph`]
//! (no live GPU state / kernel launches), so it's safe to call from
//! any scheduler test or benchmark.
use std::collections::HashSet;

use crate::{
    graph_ir::BufId,
    planner::{
        abstract_timing::AbstractTimingGraph,
        plan::{StreamInstr, StreamMemoryPlan},
    },
};

/// One detected inconsistency between a plan and its graph.
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// `plan.stream.len()` or `plan.record_event.len()` didn't match
    /// `atg.num_nodes`.
    LengthMismatch {
        field: &'static str,
        expected: usize,
        found: usize,
    },
    /// Some node id didn't appear as any `Node(_)` instruction.
    NodeUnscheduled(usize),
    /// Some node id appeared as `Node(_)` more than once.
    NodeScheduledTwice(usize),
    /// `plan.stream[node]` is `>= num_streams`.
    StreamOutOfRange {
        node: usize,
        stream: u32,
        num_streams: u32,
    },
    /// `plan.record_event[node] = Some(e)` with `e >= num_events`.
    RecordEventOutOfRange {
        node: usize,
        event: u32,
        num_events: u32,
    },
    /// A `WaitOn(_, e)` referenced `e >= num_events`.
    WaitEventOutOfRange {
        instruction_idx: usize,
        event: usize,
        num_events: u32,
    },
    /// A `WaitOn(s, _)` referenced `s >= num_streams`.
    WaitStreamOutOfRange {
        instruction_idx: usize,
        stream: usize,
        num_streams: u32,
    },
    /// A `WaitOn(_, e)` fired for an event no `Node` records.
    EventNeverRecorded { event: u32 },
    /// Two different nodes both claim `record_event = Some(e)`.
    EventRecordedTwice {
        event: u32,
        node_a: usize,
        node_b: usize,
    },
    /// A dataflow edge `producer -> consumer` (via `buf`) is not
    /// respected: the simulator sees `producer_finish > consumer_start`.
    /// For a same-stream pair this means the plan orders `consumer`
    /// before `producer` in the emitted stream; for a cross-stream
    /// pair it means the required `WaitOn` is missing or mis-placed.
    DataDepRace {
        producer: usize,
        consumer: usize,
        buf: BufId,
        producer_finish: f64,
        consumer_start: f64,
    },
    /// A cross-stream dataflow edge `producer -> consumer` is not
    /// synchronised: at the point `Node(consumer)` is issued, the
    /// consumer's stream has not `WaitOn`'d any event on the
    /// producer's stream at a position ≥ the producer's own
    /// position. Any `WaitOn(consumer_stream, e)` where the recording
    /// producer sits at or after `producer` on its stream would
    /// satisfy this — the check is *transitive*, matching how
    /// `list_v2::make_schedule` emits collapsed syncs.
    MissingCrossStreamSync {
        producer: usize,
        consumer: usize,
        producer_stream: u32,
        consumer_stream: u32,
        producer_pos_in_stream: usize,
        last_synced_pos: i64,
    },
    /// Two buffers with overlapping pool byte-ranges also have
    /// overlapping simulated lifetimes — writing one while the other
    /// is still live races.
    PoolLifetimeOverlap {
        buf_a: BufId,
        buf_b: BufId,
        offset_a: u64,
        offset_b: u64,
        size_a: u64,
        size_b: u64,
        birth_a: f64,
        death_a: f64,
        birth_b: f64,
        death_b: f64,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ValidationError::*;
        match self {
            LengthMismatch { field, expected, found } => {
                write!(f, "length mismatch on `{field}`: expected {expected}, got {found}")
            }
            NodeUnscheduled(v) => write!(f, "node {v} is not scheduled"),
            NodeScheduledTwice(v) => write!(f, "node {v} appears more than once"),
            StreamOutOfRange { node, stream, num_streams } => {
                write!(f, "stream[{node}] = {stream} >= num_streams {num_streams}")
            }
            RecordEventOutOfRange { node, event, num_events } => {
                write!(f, "record_event[{node}] = {event} >= num_events {num_events}")
            }
            WaitEventOutOfRange { instruction_idx, event, num_events } => {
                write!(f, "instructions[{instruction_idx}]: WaitOn event {event} >= num_events {num_events}")
            }
            WaitStreamOutOfRange { instruction_idx, stream, num_streams } => {
                write!(f, "instructions[{instruction_idx}]: WaitOn stream {stream} >= num_streams {num_streams}")
            }
            EventNeverRecorded { event } => write!(f, "event {event} is waited on but never recorded"),
            EventRecordedTwice { event, node_a, node_b } => {
                write!(f, "event {event} recorded by both node {node_a} and node {node_b}")
            }
            DataDepRace { producer, consumer, buf, producer_finish, consumer_start } => write!(
                f,
                "RAW race on {buf:?}: node {producer} finishes at {producer_finish:.3} but node {consumer} starts at {consumer_start:.3}"
            ),
            MissingCrossStreamSync {
                producer,
                consumer,
                producer_stream,
                consumer_stream,
                producer_pos_in_stream,
                last_synced_pos,
            } => write!(
                f,
                "cross-stream edge {producer} (s={producer_stream}, pos={producer_pos_in_stream}) -> {consumer} (s={consumer_stream}): consumer's stream has only synced past pos {last_synced_pos} on producer's stream"
            ),
            PoolLifetimeOverlap {
                buf_a, buf_b, offset_a, offset_b, size_a, size_b, birth_a, death_a, birth_b, death_b,
            } => write!(
                f,
                "pool race: {buf_a:?}=[{offset_a}, {}) [{birth_a:.3}, {death_a:.3}] overlaps with {buf_b:?}=[{offset_b}, {}) [{birth_b:.3}, {death_b:.3}]",
                offset_a + size_a, offset_b + size_b,
            ),
        }
    }
}

/// Validate `plan` against `atg`. Returns a (possibly empty) list of
/// every distinct inconsistency found. Structural checks short-circuit
/// the deeper simulation-based passes when the plan is malformed.
pub fn validate_plan(
    atg: &AbstractTimingGraph,
    plan: &StreamMemoryPlan,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let n = atg.num_nodes;
    let m = plan.num_streams as usize;
    let ne = plan.num_events as usize;

    // Structural checks.
    if plan.stream.len() != n {
        errors.push(ValidationError::LengthMismatch {
            field: "stream",
            expected: n,
            found: plan.stream.len(),
        });
    }
    if plan.record_event.len() != n {
        errors.push(ValidationError::LengthMismatch {
            field: "record_event",
            expected: n,
            found: plan.record_event.len(),
        });
    }
    if plan.offsets.len() != atg.buf_info.len() {
        errors.push(ValidationError::LengthMismatch {
            field: "offsets",
            expected: atg.buf_info.len(),
            found: plan.offsets.len(),
        });
    }
    if plan.stream.len() < n || plan.record_event.len() < n {
        // Later passes index into these — bail early if we'd panic.
        return errors;
    }
    for v in 0..n {
        if plan.stream[v] as usize >= m {
            errors.push(ValidationError::StreamOutOfRange {
                node: v,
                stream: plan.stream[v],
                num_streams: plan.num_streams,
            });
        }
        if let Some(e) = plan.record_event[v] {
            if (e as usize) >= ne {
                errors.push(ValidationError::RecordEventOutOfRange {
                    node: v,
                    event: e,
                    num_events: plan.num_events,
                });
            }
        }
    }

    // Node schedule coverage + event-recording uniqueness.
    let mut scheduled = vec![false; n];
    let mut recorded_by: Vec<Option<usize>> = vec![None; ne];
    for &e in plan.record_event.iter().flatten() {
        let e = e as usize;
        if e >= ne {
            continue;
        }
    }
    for (v, &re) in plan.record_event.iter().enumerate() {
        if let Some(e) = re {
            let e = e as usize;
            if e >= ne {
                continue;
            }
            match recorded_by[e] {
                None => recorded_by[e] = Some(v),
                Some(other) if other != v => {
                    errors.push(ValidationError::EventRecordedTwice {
                        event: e as u32,
                        node_a: other,
                        node_b: v,
                    });
                }
                _ => {}
            }
        }
    }
    for (i, instr) in plan.instructions.iter().enumerate() {
        match *instr {
            StreamInstr::Node(v) => {
                if v >= n {
                    // Guard against out-of-range Node — treat as
                    // structural error; skip further per-node work.
                    errors.push(ValidationError::NodeUnscheduled(v));
                    continue;
                }
                if scheduled[v] {
                    errors.push(ValidationError::NodeScheduledTwice(v));
                }
                scheduled[v] = true;
            }
            StreamInstr::WaitOn(s, e) => {
                if s >= m {
                    errors.push(ValidationError::WaitStreamOutOfRange {
                        instruction_idx: i,
                        stream: s,
                        num_streams: plan.num_streams,
                    });
                }
                if e >= ne {
                    errors.push(ValidationError::WaitEventOutOfRange {
                        instruction_idx: i,
                        event: e,
                        num_events: plan.num_events,
                    });
                    continue;
                }
                if recorded_by[e].is_none() {
                    errors.push(ValidationError::EventNeverRecorded { event: e as u32 });
                }
            }
        }
    }
    for v in 0..n {
        if !scheduled[v] {
            errors.push(ValidationError::NodeUnscheduled(v));
        }
    }
    // Bail before simulation if the plan is structurally broken.
    if errors.iter().any(|e| {
        matches!(
            e,
            ValidationError::NodeUnscheduled(_)
                | ValidationError::NodeScheduledTwice(_)
                | ValidationError::StreamOutOfRange { .. }
        )
    }) {
        return errors;
    }

    // Simulate the plan under the perfect-parallel stream model:
    // `stream_time[s]` advances by `node_times[v]` per node; a
    // `WaitOn(s, e)` bumps `stream_time[s]` up to `event_time[e]`.
    // We also track per-node position *within its stream* and the
    // running `last_synced[s][s']` = highest position on `s'` that
    // `s` has waited past. That lets us match the transitive-sync
    // model that `list_v2::make_schedule` emits (one `WaitOn` can
    // cover many earlier cross-stream producers on the same source
    // stream).
    let mut stream_time = vec![0.0f64; m.max(1)];
    let mut event_time = vec![0.0f64; ne];
    let mut start = vec![f64::NAN; n];
    let mut finish = vec![f64::NAN; n];
    let mut node_pos_in_stream = vec![usize::MAX; n];
    let mut event_prod: Vec<Option<(usize, u32, usize)>> = vec![None; ne];
    let mut last_synced: Vec<Vec<i64>> = vec![vec![-1i64; m.max(1)]; m.max(1)];
    let mut stream_counter = vec![0usize; m.max(1)];

    // First pass: assign positions-in-stream + collect event
    // producers (so on the main pass we can resolve WaitOns against
    // producer positions regardless of instruction order).
    for instr in &plan.instructions {
        if let StreamInstr::Node(v) = *instr {
            let s = plan.stream[v] as usize;
            node_pos_in_stream[v] = stream_counter[s];
            stream_counter[s] += 1;
            if let Some(e) = plan.record_event[v] {
                event_prod[e as usize] = Some((v, plan.stream[v], node_pos_in_stream[v]));
            }
        }
    }

    // Second pass: simulate + validate.
    let mut checked_edges: HashSet<(usize, usize)> = HashSet::new();
    for instr in &plan.instructions {
        match *instr {
            StreamInstr::WaitOn(s, e) => {
                if s < m && e < ne {
                    if stream_time[s] < event_time[e] {
                        stream_time[s] = event_time[e];
                    }
                    if let Some((_, prod_stream, prod_pos)) = event_prod[e] {
                        let sp = prod_stream as usize;
                        let pp = prod_pos as i64;
                        if last_synced[s][sp] < pp {
                            last_synced[s][sp] = pp;
                        }
                    }
                }
            }
            StreamInstr::Node(v) => {
                let s_v = plan.stream[v] as usize;
                start[v] = stream_time[s_v];
                stream_time[s_v] += atg.node_times[v];
                finish[v] = stream_time[s_v];
                if let Some(e) = plan.record_event[v] {
                    event_time[e as usize] = finish[v];
                }
                // Check every RAW dep P → v at this exact global
                // position (before further WaitOns arrive).
                for &bid in &atg.node_consumes[v] {
                    let producers = match atg.buf_producers.get(&bid) {
                        Some(p) => p,
                        None => continue,
                    };
                    for &p in producers {
                        if p == v {
                            continue;
                        }
                        if !checked_edges.insert((p, v)) {
                            continue;
                        }
                        if start[v] + 1e-9 < finish[p] {
                            errors.push(ValidationError::DataDepRace {
                                producer: p,
                                consumer: v,
                                buf: bid,
                                producer_finish: finish[p],
                                consumer_start: start[v],
                            });
                        }
                        let s_p = plan.stream[p] as usize;
                        if s_p != s_v {
                            let synced = last_synced[s_v][s_p];
                            let pp = node_pos_in_stream[p] as i64;
                            if synced < pp {
                                errors.push(ValidationError::MissingCrossStreamSync {
                                    producer: p,
                                    consumer: v,
                                    producer_stream: plan.stream[p],
                                    consumer_stream: plan.stream[v],
                                    producer_pos_in_stream: node_pos_in_stream[p],
                                    last_synced_pos: synced,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // Memory pool safety. For every packable buf compute its simulated
    // [birth, death]; then any pair whose pool byte-ranges overlap
    // must have disjoint lifetimes.
    let n_bufs = atg.buf_info.len();
    let input_set: HashSet<BufId> = atg.inputs.iter().copied().collect();
    let output_set: HashSet<BufId> = atg.outputs.iter().copied().collect();
    let mut birth = vec![f64::INFINITY; n_bufs];
    let mut death = vec![f64::NEG_INFINITY; n_bufs];
    for bid in 0..n_bufs {
        let bid_id = BufId(bid);
        if plan.offsets[bid].is_none() {
            continue;
        }
        if let Some(prods) = atg.buf_producers.get(&bid_id) {
            for &p in prods {
                if !start[p].is_nan() {
                    if start[p] < birth[bid] {
                        birth[bid] = start[p];
                    }
                    if finish[p] > death[bid] {
                        death[bid] = finish[p];
                    }
                }
            }
        }
        if let Some(users) = atg.buf_users.get(&bid_id) {
            for &u in users {
                if !finish[u].is_nan() && finish[u] > death[bid] {
                    death[bid] = finish[u];
                }
            }
        }
        if input_set.contains(&bid_id) {
            birth[bid] = 0.0f64.min(birth[bid]);
        }
        if output_set.contains(&bid_id) {
            death[bid] = f64::INFINITY;
        }
    }
    let packed: Vec<usize> = (0..n_bufs)
        .filter(|&b| plan.offsets[b].is_some() && birth[b].is_finite() && death[b] > birth[b] - 1.0)
        .collect();
    for i in 0..packed.len() {
        let a = packed[i];
        let oa = plan.offsets[a].unwrap();
        let sa = atg.buf_info[a].concrete_size as u64;
        for &b in &packed[i + 1..] {
            let ob = plan.offsets[b].unwrap();
            let sb = atg.buf_info[b].concrete_size as u64;
            let byte_overlap = oa < ob + sb && ob < oa + sa;
            if !byte_overlap {
                continue;
            }
            // Life-time overlap = intervals [birth, death] intersect
            // with non-zero measure. Use a small epsilon to allow
            // exactly-touching intervals to count as disjoint.
            let time_overlap =
                birth[a].max(birth[b]) + 1e-9 < death[a].min(death[b]);
            if time_overlap {
                errors.push(ValidationError::PoolLifetimeOverlap {
                    buf_a: BufId(a),
                    buf_b: BufId(b),
                    offset_a: oa,
                    offset_b: ob,
                    size_a: sa,
                    size_b: sb,
                    birth_a: birth[a],
                    death_a: death[a],
                    birth_b: birth[b],
                    death_b: death[b],
                });
            }
        }
    }

    errors
}
