//! Consumer-side schema for the segment-shape profile JSONL.
//!
//! These two structs are the deserialization targets for each line of a
//! profile captured by the upstream `SHADOW_BENCH_PROFILE_PATH` probe
//! (idea 0008). The producer that writes the JSONL lives in
//! `stark-backend`'s prover module; nothing in this directory captures
//! profiles — we only consume them.
//!
//! `BusIndex` and `TraceWidth` come from `openvm-stark-backend`. When
//! integrating these source files into a workspace, either re-use the
//! upstream definitions or substitute the underlying primitives directly
//! (`BusIndex` is a `u16`; `TraceWidth` carries `preprocessed: Option<usize>`,
//! `cached_mains: Vec<usize>`, `common_main: usize`, `after_challenge: Vec<usize>`).

use openvm_stark_backend::{interaction::BusIndex, keygen::types::TraceWidth};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AirShapeRecord {
    pub air_name: String,
    pub air_id: usize,
    pub log_height: usize,
    pub height: usize,
    pub width: TraceWidth,
    pub num_constraints: usize,
    pub num_interactions: usize,
    pub max_constraint_degree: usize,
    /// Per-interaction `bus_index`. `len() == num_interactions`.
    pub buses: Vec<BusIndex>,
    /// Per-interaction message length (number of field expressions).
    /// `len() == num_interactions`. Added in schema v2.
    pub interaction_message_lens: Vec<usize>,
    /// Per-interaction `count_weight` (logup soundness parameter).
    /// `len() == num_interactions`. Added in schema v2.
    pub interaction_count_weights: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentProfile {
    pub schema: String,
    pub segment_idx: usize,
    pub global_max_constraint_degree: usize,
    pub airs: Vec<AirShapeRecord>,
}
