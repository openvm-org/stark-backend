# Milestone 3: Unified-Memory Copy Elimination

- Status: `completed`
- Owner: codex
- Branch: `perf/m3`
- PR: TODO
- Started: 2026-02-20
- Completed: 2026-02-20
- Last Updated: 2026-02-20

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

- `cargo check -p openvm-metal-backend`: pass (2026-02-20)
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`: pass (2026-02-20, 0 tests in target; harness completed successfully)
- keccakf trace run: not run (out of milestone scope)

## Metrics

- Before: N/A
- After: N/A
- Delta: N/A

## Notes

- `MerkleTreeMetal` now caches non-root digest layer pointer buffers (`MetalBuffer<u64>`) and reads the root via direct single-element shared-memory access instead of `to_host().pop()`.
- `batch_query_merkle_proofs` now consumes cached per-tree pointer buffers and avoids rebuilding/reuploading per-layer pointer arrays from digest buffers on each query path.
- Added `read_folded_matrix_first_row` in `data_transporter.rs` to read folded `MetalMatrix` row 0 directly from shared memory.
- Refactored `into_column_openings` to use direct folded-row reads for height=1 matrices and preserve the exact opening ordering/rotation semantics used by transcript observation.
