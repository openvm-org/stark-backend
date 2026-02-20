# Milestone 6: Device-Aware `batch_ntt_small` Tuning

- Status: `completed`
- Owner: codex
- Branch: `metal-backend`
- PR:
- Started: 2026-02-19
- Completed: 2026-02-19
- Last Updated: 2026-02-19

## Objective

Improve occupancy and portability by removing hard-coded threadgroup assumptions.

## Scope

- Files:
  - `crates/metal-backend/src/metal/batch_ntt_small.rs`
  - `crates/metal-backend/metal/src/batch_ntt_small.metal`
- Work:
  - Query pipeline/device limits.
  - Derive threadgroup shape and shared-memory usage from limits.
  - Keep algorithm and twiddle behavior unchanged.

## Exit Criteria

- Launch config adapts to device limits.
- No validation failures from threadgroup/shared-memory mismatch.

## Validation Log

- `cargo check -p openvm-metal-backend`: pass (existing warnings in `mle_round.rs` and `stacked_reduction.rs`)
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`: pass (`0` tests, `0` failed)
- `cargo run -p openvm-metal-backend --example keccakf`: pass; post-change trace captured

## Metrics

- Before:
  - `stark_prove_excluding_trace = 24.9s`
  - `rs_code_matrix = 950ms`
  - `prover.main_trace_commit > stacked_commit > merkle_tree = 233ms`
  - `rap_constraints.round0 = 6.79s`
  - `rap_constraints.mle_rounds = 414ms`
- After:
  - `stark_prove_excluding_trace = 25.6s`
  - `rs_code_matrix = 1.18s`
  - `prover.main_trace_commit > stacked_commit > merkle_tree = 239ms`
  - `rap_constraints.round0 = 7.00s`
  - `rap_constraints.mle_rounds = 438ms`
- Delta:
  - `stark_prove_excluding_trace: +0.7s`
  - `rs_code_matrix: +230ms`
  - `prover.main_trace_commit > stacked_commit > merkle_tree: +6ms`
  - `rap_constraints.round0: +0.21s`
  - `rap_constraints.mle_rounds: +24ms`

## Notes

- `batch_ntt_small` now derives threadgroup shape from `max_total_threads_per_threadgroup`, per-axis device limits, and dynamic threadgroup memory limits.
- The Metal kernel now uses `threads_per_threadgroup.y` for block indexing so host-selected launch geometry remains correct when threadgroup size is not `1024`.
- Additional post-change `prover.openings.whir > merkle_tree` spans: `51.8ms`, `33.5ms`, `16.9ms` (baseline: `48.8ms`, `26.5ms`, `14.5ms`).
