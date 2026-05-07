//! Synthetic AIR generator.
//!
//! Reads a shape atlas (produced by `scripts/analyze_profile.py` from a
//! profile JSONL captured by the `SHADOW_BENCH_PROFILE_PATH` probe) and
//! provides [`SyntheticAir`], a parametric AIR matching captured shape
//! statistics: trace height, common-main width, num_constraints,
//! max_constraint_degree, num_interactions, num_distinct_buses.
//!
//! ## Trick
//!
//! Column 0 of the trace is treated as a "kill column": it is filled
//! with zeros. Every constraint multiplies by it (so the constraint is
//! trivially zero everywhere), and every interaction uses it as the
//! count column (so multiplicities are zero, making the interactions
//! trivially balanced regardless of send/receive distribution). The
//! prover still iterates the same number of trace cells and constraint
//! / interaction terms as the real AIR, so kernel timing is preserved.
//!
//! v1 limitations: ignores preprocessed columns, cached_main partitions,
//! and after-challenge widths. The captured profile distribution is
//! dominated by AIRs with these set to small or zero values, so this is
//! sufficient to start. Extend if validation shows drift.

use std::path::Path;

use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder},
    p3_air::{Air, BaseAir, BaseAirWithPublicValues},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    PartitionedBaseAir,
};
use serde::{Deserialize, Serialize};

/// One entry in a shape atlas — shape statistics for a single AIR
/// observed in the captured profile, deduplicated and counted by
/// `scripts/analyze_profile.py`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyntheticShape {
    pub air_name: String,
    pub log_height: usize,
    pub preprocessed_width: usize,
    pub cached_main_widths: Vec<usize>,
    pub common_main_width: usize,
    pub after_challenge_widths: Vec<usize>,
    pub num_constraints: usize,
    pub num_interactions: usize,
    pub num_distinct_buses: usize,
    pub max_constraint_degree: usize,
    /// Per-interaction message length. Schema v2 atlases populate this;
    /// older atlases leave it empty and a heuristic is used.
    #[serde(default)]
    pub interaction_message_lens: Vec<usize>,
    /// Per-interaction `count_weight` (logup soundness param).
    /// Schema v2 atlases populate this; older atlases leave it empty.
    #[serde(default)]
    pub interaction_count_weights: Vec<u32>,
    pub occurrences: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShapeAtlas {
    pub source: String,
    pub shapes: Vec<SyntheticShape>,
}

impl ShapeAtlas {
    pub fn read(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let f = std::fs::File::open(path)?;
        Ok(serde_json::from_reader(f)?)
    }
}

/// A parametric AIR that matches the prover-observable shape of a
/// captured AIR record. See module docs for the construction trick.
#[derive(Clone, Debug)]
pub struct SyntheticAir {
    width: usize,
    num_constraints: usize,
    max_constraint_degree: usize,
    /// `(bus_index, is_send, message_len, count_weight)` per interaction.
    interactions: Vec<(BusIndex, bool, usize, u32)>,
}

impl SyntheticAir {
    pub fn from_shape(s: &SyntheticShape) -> Self {
        // Width must accommodate (a) the kill column (col 0) and (b)
        // the largest captured interaction message — every field in a
        // message references a real column. If common_main_width is too
        // narrow (notably width=1 AIRs like ProgramAir whose 9 columns
        // live in cached_mains, which v1 ignores), widen to fit the
        // largest message + the kill column. Otherwise message lengths
        // would silently clamp to 0 and the prover's per-interaction
        // work would diverge from the captured shape.
        let max_captured_msg_len = s
            .interaction_message_lens
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let width = s.common_main_width.max(max_captured_msg_len + 1).max(1);
        let num_buses = s.num_distinct_buses.max(1);
        // Each captured length is clamped to (width - 1). For v1 atlases
        // (empty lens), fall back to min(width - 1, 4) — a typical real
        // value, comfortably below the keygen's 128-element message cap.
        let max_field_count = width - 1;
        let fallback_len = max_field_count.min(4);
        let interactions = (0..s.num_interactions)
            .map(|i| {
                let bus = (i % num_buses) as BusIndex;
                let is_send = i % 2 == 0;
                let captured_len = s
                    .interaction_message_lens
                    .get(i)
                    .copied()
                    .unwrap_or(fallback_len);
                let msg_len = captured_len.min(max_field_count);
                let cw = s.interaction_count_weights.get(i).copied().unwrap_or(0);
                (bus, is_send, msg_len, cw)
            })
            .collect();
        Self {
            width,
            num_constraints: s.num_constraints,
            max_constraint_degree: s.max_constraint_degree.max(1),
            interactions,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn num_constraints(&self) -> usize {
        self.num_constraints
    }
    pub fn num_interactions(&self) -> usize {
        self.interactions.len()
    }

    /// Generate an all-zeros trace at the given log_height. Width matches
    /// `self.width()`. Trace is trivially valid (every constraint and
    /// interaction count multiplies by column 0 = 0).
    pub fn generate_trace<F: Field>(&self, log_height: usize) -> RowMajorMatrix<F> {
        let h = 1usize << log_height;
        RowMajorMatrix::new(vec![F::ZERO; h * self.width], self.width)
    }
}

impl<F: Field> BaseAir<F> for SyntheticAir {
    fn width(&self) -> usize {
        self.width
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for SyntheticAir {}
impl<F: Field> PartitionedBaseAir<F> for SyntheticAir {}

impl<AB: InteractionBuilder> Air<AB> for SyntheticAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("trace has at least one row");
        let next = main.row_slice(1).expect("trace has at least two rows");

        // Build `num_constraints` STRUCTURALLY DISTINCT degree-D
        // monomials over the column variables. Distinctness matters
        // because `build_symbolic_constraints_dag` dedupes DAG nodes
        // (dag.rs constraint_idx.dedup()), so duplicate symbolic
        // expressions would collapse and the keygen-reported constraint
        // count would underrun the captured shape's `num_constraints`.
        //
        // Encoding: the constraint index `c` is interpreted as a
        // base-(2*width) integer of length max_constraint_degree. Each
        // digit picks a column slot from the union of `local[0..width]`
        // and `next[0..width]`, giving (2*width)^max_constraint_degree
        // distinct monomials. For width >= 2 and degree >= 2 this
        // comfortably exceeds any captured `num_constraints` (max
        // observed in captured profiles: ~25 with much larger widths).
        //
        // Trace correctness: `generate_trace` fills every cell with 0,
        // so any polynomial in the column variables evaluates to 0
        // regardless of which columns the constraint references. Kernel
        // workload (cells touched, monomial degree) still matches the
        // captured shape.
        let degree = self.max_constraint_degree;
        let slot_count = 2 * self.width;
        for c in 0..self.num_constraints {
            let mut idx = c;
            let mut term: Option<AB::Expr> = None;
            for _ in 0..degree {
                let slot = idx % slot_count;
                idx /= slot_count;
                let col = if slot < self.width {
                    local[slot]
                } else {
                    next[slot - self.width]
                };
                term = Some(match term {
                    None => col.into(),
                    Some(t) => t * col.into(),
                });
            }
            let term = term.unwrap_or_else(|| local[0].into());
            builder.assert_zero(term);
        }

        // Interactions: each one reads `field_count` columns (all >=
        // 1 in the captured data when width > 1) plus the count column
        // (always col 0). Trace is all-zero so multiplicities are zero,
        // making interactions trivially balanced regardless of how many
        // are sends vs receives. Bus indices are renumbered to
        // `0..num_distinct_buses` — actual identity doesn't affect the
        // prover's per-bus batching cost.
        for (bus, is_send, msg_len, count_weight) in &self.interactions {
            let count = local[0];
            let fields: Vec<AB::Expr> = (0..*msg_len)
                .map(|i| local[(i + 1) % self.width].into())
                .collect();
            if *is_send {
                builder.push_interaction(*bus, fields, count, *count_weight);
            } else {
                builder.push_interaction(
                    *bus,
                    fields,
                    AB::Expr::NEG_ONE * count.into(),
                    *count_weight,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape_for_test() -> SyntheticShape {
        // Arbitrary but representative numbers — not from a real
        // capture; just exercises the constructor + AIR impl.
        SyntheticShape {
            air_name: "TestSynthetic".into(),
            log_height: 4,
            preprocessed_width: 0,
            cached_main_widths: vec![],
            common_main_width: 6,
            after_challenge_widths: vec![],
            num_constraints: 7,
            num_interactions: 3,
            num_distinct_buses: 2,
            max_constraint_degree: 3,
            interaction_message_lens: vec![3, 3, 3],
            interaction_count_weights: vec![0, 0, 0],
            occurrences: 1,
        }
    }

    #[test]
    fn synthetic_air_metadata_matches_shape() {
        let s = shape_for_test();
        let air = SyntheticAir::from_shape(&s);
        assert_eq!(air.width(), s.common_main_width);
        assert_eq!(air.num_constraints(), s.num_constraints);
        assert_eq!(air.num_interactions(), s.num_interactions);
    }

    #[test]
    fn width_widens_to_fit_captured_message_len() {
        // Mirrors the ProgramAir-shaped records in the bundled profile:
        // common_main_width=1 with a captured 9-field interaction. The
        // synthetic AIR must widen to width=10 (kill col + 9 fields) so
        // the interaction's per-field kernel cost survives.
        let s = SyntheticShape {
            air_name: "WidthOneWithMsg".into(),
            log_height: 4,
            preprocessed_width: 0,
            cached_main_widths: vec![],
            common_main_width: 1,
            after_challenge_widths: vec![],
            num_constraints: 0,
            num_interactions: 1,
            num_distinct_buses: 1,
            max_constraint_degree: 1,
            interaction_message_lens: vec![9],
            interaction_count_weights: vec![0],
            occurrences: 1,
        };
        let air = SyntheticAir::from_shape(&s);
        assert_eq!(air.width(), 10, "kill col + 9 message fields");
    }

    #[test]
    fn synthetic_air_keygen_prove_verify_round_trip() {
        use std::sync::Arc;

        use eyre::eyre;
        use openvm_stark_backend::{
            prover::{AirProvingContext, DeviceDataTransporter, ProvingContext},
            StarkEngine,
        };
        use openvm_stark_sdk::config::{
            app_params_with_100_bits_security, baby_bear_poseidon2::BabyBearPoseidon2CpuEngine,
        };
        use p3_baby_bear::BabyBear;

        let s = shape_for_test();
        let air = SyntheticAir::from_shape(&s);
        let trace = air.generate_trace::<BabyBear>(s.log_height);

        // Stacked height must be >= log_height of any AIR; use a
        // value comfortably larger than the test shape's log_height.
        let params = app_params_with_100_bits_security(15);
        let engine: BabyBearPoseidon2CpuEngine = StarkEngine::new(params);

        let air_arc = Arc::new(air);
        let (pk, vk) = engine.keygen(&[air_arc]);

        let inner_pk = &pk.per_air[0];
        assert_eq!(
            inner_pk
                .vk
                .symbolic_constraints
                .constraints
                .constraint_idx
                .len(),
            s.num_constraints,
            "keygen-reported num_constraints must match the shape's num_constraints",
        );
        assert_eq!(
            inner_pk.vk.symbolic_constraints.interactions.len(),
            s.num_interactions,
            "keygen-reported num_interactions must match the shape's num_interactions",
        );

        let trace_ctx = AirProvingContext::simple_no_pis(trace);
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let proof = engine
            .prove(&d_pk, ProvingContext::new(vec![(0, trace_ctx)]))
            .map_err(|e| eyre!("Proving failed: {e:?}"))
            .unwrap();
        engine
            .verify(&vk, &proof)
            .expect("synthetic proof verifies");
    }
}
