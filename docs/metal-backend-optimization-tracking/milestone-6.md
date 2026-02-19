# Milestone 6: Device-Aware `batch_ntt_small` Tuning

- Status: `not_started`
- Owner:
- Branch:
- PR:
- Started:
- Completed:
- Last Updated:

## Objective

Improve occupancy and portability by removing hard-coded threadgroup assumptions.

## Scope

- Files:
  - `crates/metal-backend/src/metal/batch_ntt_small.rs`
- Work:
  - Query pipeline/device limits.
  - Derive threadgroup shape and shared-memory usage from limits.
  - Keep algorithm and twiddle behavior unchanged.

## Exit Criteria

- Launch config adapts to device limits.
- No validation failures from threadgroup/shared-memory mismatch.

## Validation Log

- `cargo check -p openvm-metal-backend`:
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`:
- keccakf trace run:

## Metrics

- Before:
- After:
- Delta:

## Notes

