# Milestone 13: Openings and WHIR Pipeline Tightening

- Status: `completed`
- Owner: codex
- Branch: `perf/m13`
- PR:
- Started: 2026-02-20
- Completed: 2026-02-20
- Last Updated: 2026-02-20

## Objective

Tighten the openings path by reducing WHIR-round allocation churn and cutting stacked-reduction MLE synchronization/readback overhead, while preserving protocol semantics.

## Scope

- Files:
  - `crates/metal-backend/src/whir.rs`
  - `crates/metal-backend/src/stacked_reduction.rs`
  - `crates/metal-backend/src/metal/stacked_reduction.rs`
- Work:
  - Reuse WHIR fold scratch buffers across rounds instead of allocating per round.
  - Batch stacked-reduction MLE window kernels into one staged dispatch and do one readback per round.
  - Add safety docs/comments for unsafe buffer/pointer invariants in touched paths.

## Exit Criteria

- WHIR/reduction behavior remains protocol-equivalent.
- `openvm-metal-backend` check/test commands pass.
- New batching points are instrumented and safety invariants documented.

## Validation Log

- `cargo check -p openvm-metal-backend`: pass (2026-02-20)
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`: pass (2026-02-20, `0` tests in target; harness succeeded)
- `cargo run -p openvm-metal-backend --release --example keccakf`: pass (2026-02-20)
- `cargo test -p openvm-metal-backend --release test_batch_constraints_with_interactions -- --nocapture`: pass (2026-02-20, `1` passed)

## Metrics

- WHIR fold output allocations:
  - Before: `2` `MetalBuffer` allocations per WHIR inner round (`new_f_coeffs`, `new_w_moments`).
  - After: `0` per-round allocations (ping-pong scratch buffers reused).
- Stacked-reduction MLE sumcheck (`batch_sumcheck_poly_eval`) synchronization/readback:
  - Before: `O(num_windows)` syncs and `O(num_windows)` readbacks per round (one per window).
  - After: `1` staged sync and `1` batched readback per round.
- Instrumentation added at batching points:
  - `stacked_reduction_sumcheck_mle_batch` (round-level window batching summary)
  - `metal_dispatch_stage` stage `stacked_reduction_sumcheck_mle_round.batch`
- Observed release benchmark spans from `keccakf` run:
  - `prover.openings.whir`: `434ms`
  - `prover.openings.stacked_reduction`: `105ms`
  - `prover.openings.stacked_reduction.mle_rounds`: `39.1ms`

## Notes

- Added `/// # Safety` sections for unsafe stacked-reduction Metal wrapper entry points, including the new batched MLE-round dispatcher.
- Added explicit `SAFETY` comments around new batched accumulator readback and dispatch invariants.
- Degenerate-window `eq_ub` payloads are packed once on host and copied to device in a single H2D transfer before batch dispatch.
