# Milestone 2: Stacked Trace Copy Path Optimization

- Status: `completed`
- Owner: codex
- Branch: `perf/m2`
- PR:
- Started: 2026-02-19
- Completed: 2026-02-19
- Last Updated: 2026-02-20

## Objective

Remove per-column blit synchronization in stacked trace assembly.

## Scope

- Files:
  - `crates/metal-backend/src/stacked_pcs.rs`
- Work:
  - Refactor `stack_traces_into_expanded` to batch blit copies.
  - Keep output layout and zero-fill semantics unchanged.
  - Preserve behavior for `s.log_height() < l_skip`.

## Exit Criteria

- No per-segment sync in copy path.
- `rs_code_matrix` span improves in profile.

## Validation Log

- `cargo check -p openvm-metal-backend`: pass
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`: pass (test target executed, 0 tests)
- keccakf trace run: pass (captured before/after `rs_code_matrix`)

## Metrics

- Before `rs_code_matrix`: `1.12s`
- After `rs_code_matrix`: `669ms`
- Delta: `-451ms` (~`-40.3%`)

## Notes
- `stack_traces_into_expanded` now batches blit copies and flushes once per contiguous blit run (instead of per copy), preserving order around `batch_expand_pad_wide` calls.
- Added tracing counters at the batching point:
  - `blit_copy_count`
  - `blit_sync_count`
  - `expand_pad_dispatch_count`
