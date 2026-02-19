# Milestone 3: Unified-Memory Copy Elimination

- Status: `not_started`
- Owner:
- Branch:
- PR:
- Started:
- Completed:
- Last Updated:

## Objective

Reduce redundant host-device copies by taking advantage of unified memory.

## Scope

- Files:
  - `crates/metal-backend/src/merkle_tree.rs`
  - `crates/metal-backend/src/data_transporter.rs`
  - `crates/metal-backend/src/logup_zerocheck/mod.rs`
- Work:
  - Remove avoidable `to_host()`/`to_device()` in hot paths.
  - Keep APIs correct for call sites that require owned host buffers.
  - Preserve proof/transcript behavior.

## Exit Criteria

- Fewer hot-path copy conversions.
- No behavior or transcript change.

## Validation Log

- `cargo check -p openvm-metal-backend`:
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`:
- keccakf trace run:

## Metrics

- Before:
- After:
- Delta:

## Notes

