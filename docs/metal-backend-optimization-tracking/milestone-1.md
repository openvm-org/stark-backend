# Milestone 1: Batch Synchronization and Dispatch Plumbing

- Status: `completed`
- Owner: codex
- Branch: `perf/m1`
- PR:
- Started: 2026-02-19
- Completed: 2026-02-19
- Last Updated: 2026-02-19

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

- `cargo check -p openvm-metal-backend`: pass (existing warnings in `mle_round.rs` and `stacked_reduction.rs`)
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`: pass (`0` tests, `0` failed)
- keccakf trace run: pass (`cargo run -p openvm-metal-backend --example keccakf`)

## Metrics

- Dispatch/sync structure changes:
  - `stacked_reduction_sumcheck_round0`: `2` kernels, sync points reduced `2 -> 1` per call.
  - `zerocheck_ntt_eval_constraints` (per coset): `2` kernels, sync points reduced `2 -> 1`.
  - `logup_bary_eval_interactions_round0` (per coset): `3` kernels, sync points reduced `3 -> 1`.
- Span timings (`keccakf`, baseline from Milestone 0 vs first post-change run):
  - `stark_prove_excluding_trace`: `24.9s -> 25.8s` (`+0.9s`)
  - `prover.main_trace_commit > stacked_commit > rs_code_matrix`: `950ms -> 1.07s` (`+120ms`)
  - `prover.main_trace_commit > stacked_commit > merkle_tree`: `233ms -> 233ms` (`~0ms`)
  - `prover.rap_constraints.round0`: `6.79s -> 7.04s` (`+250ms`)
  - `prover.rap_constraints.mle_rounds`: `414ms -> 439ms` (`+25ms`)
  - `prover.openings.whir > merkle_tree`: `48.8ms/26.5ms/14.5ms -> 65.7ms/34.5ms/16.3ms`

## Notes

- Added staged dispatch tracing (`metal_dispatch_stage`) with `stage`, `kernel_count`, and `sync_count`.
- A second post-change `keccakf` sample showed higher total time (`30.4s`), indicating high run-to-run variance on this host.
