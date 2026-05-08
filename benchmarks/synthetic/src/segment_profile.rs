//! Consumer-side schema for the segment-shape profile JSONL.
//!
//! These structs are the deserialization targets for each line of a
//! profile captured by the upstream `SHADOW_BENCH_PROFILE_PATH` probe.
//! The producer that writes the JSONL lives in `stark-backend`'s prover
//! module; nothing in this directory captures profiles — we only
//! consume them.
//!
//! [`ProfileTraceWidth`] is intentionally a *local* mirror of the
//! captured wire shape rather than a re-use of `keygen::types::TraceWidth`:
//! the captured JSONL is a stable on-disk format, while the upstream type
//! evolves (e.g. the `after_challenge` field was removed upstream after
//! these profiles were captured). Decoupling lets old captures keep
//! deserializing against newer backends. `#[serde(default)]` on optional
//! fields gives forward/backward schema tolerance.

use openvm_stark_backend::interaction::BusIndex;
use serde::{Deserialize, Serialize};

/// Width fields as captured in the profile JSONL — independent of the
/// upstream `TraceWidth` so schema evolution upstream doesn't break
/// replay of older captures.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileTraceWidth {
    #[serde(default)]
    pub preprocessed: Option<usize>,
    #[serde(default)]
    pub cached_mains: Vec<usize>,
    pub common_main: usize,
    #[serde(default)]
    pub after_challenge: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AirShapeRecord {
    pub air_name: String,
    pub air_id: usize,
    pub log_height: usize,
    pub height: usize,
    pub width: ProfileTraceWidth,
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
