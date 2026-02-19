# Milestone 1: Batch Synchronization and Dispatch Plumbing

- Status: `not_started`
- Owner:
- Branch:
- PR:
- Started:
- Completed:
- Last Updated:

## Objective

Remove avoidable CPU/GPU synchronization overhead from hot sequential kernels.

## Scope

- Files:
  - `crates/metal-backend/src/metal/mod.rs`
  - `crates/metal-backend/src/metal/stacked_reduction.rs`
  - `crates/metal-backend/src/metal/logup_zerocheck.rs`
- Work:
  - Add or expand staged dispatch API.
  - Replace selected back-to-back `dispatch_sync` chains with staged dispatch.
  - Preserve ordering, determinism, and error behavior.

## Exit Criteria

- Fewer sync points in round0/reduction hot paths.
- Correctness preserved.

## Validation Log

- `cargo check -p openvm-metal-backend`:
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`:
- keccakf trace run:

## Metrics

- Before:
- After:
- Delta:

## Notes

