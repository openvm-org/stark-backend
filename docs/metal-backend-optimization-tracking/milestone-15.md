# Milestone 15: Final 2x Validation and Residual Backlog

- Status: `not_started`
- Owner:
- Branch:
- PR:
- Started:
- Completed:
- Last Updated: 2026-02-20

## Objective

Validate final performance against the 2x target and publish residual bottlenecks with prioritized follow-up work.

## Scope

- Run:
  - `cargo check -p openvm-metal-backend`
  - `cargo test -p openvm-metal-backend --test ntt_roundtrip`
  - `cargo run -p openvm-metal-backend --release --example keccakf` (5 warm runs)
  - `cargo test -p openvm-metal-backend --release test_batch_constraints_with_interactions -- --nocapture` (5 warm runs)
- Compare results against Milestone 8 baseline medians.
- Document final speedup, span deltas, and remaining bottlenecks.

## Exit Criteria

- `keccakf` median `stark_prove_excluding_trace <= 1.18s` or documented gap-to-target analysis.
- LogUp benchmark spans (`logup_gkr`, `round0`, `mle_rounds`) show measurable improvement or documented limits.
- Final report includes prioritized residual optimization backlog.

## Validation Log

- Pending.

## Metrics

- Pending.

## Notes

- This is the final gate for the Phase 2 optimization program.
