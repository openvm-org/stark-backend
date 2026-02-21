# Milestone 14: Cross-Phase Scheduling and Buffer Pool

- Status: `completed`
- Owner: codex
- Branch: `metal-backend`
- PR:
- Started: 2026-02-21
- Completed: 2026-02-21
- Last Updated: 2026-02-21

## Objective

Add cross-phase scheduling and reusable temporary buffer pooling to cut overhead across major proving phases.

## Scope

- Added phase-level staged scheduling helper:
  - `dispatch_phase_staged_sync` in `crates/metal-backend/src/metal/mod.rs`.
- Added reusable shared scratch-buffer pooling:
  - `SharedMetalBufferPool` + `PooledMetalBuffer` in `crates/metal-common/src/d_buffer.rs`.
  - backend convenience accessors in `crates/metal-backend/src/base.rs`.
- Adopted in round0/logup path:
  - `crates/metal-backend/src/metal/logup_zerocheck.rs`
  - `crates/metal-backend/src/logup_zerocheck/round0.rs`
- Adopted in stacked commit path:
  - `crates/metal-backend/src/stacked_pcs.rs`
  - `crates/metal-backend/src/merkle_tree.rs`
- Adopted in openings path:
  - `crates/metal-backend/src/whir.rs`
  - `crates/metal-backend/src/metal/stacked_reduction.rs`

## Exit Criteria

- Reduced allocation churn and sync count across at least three phases.
- Positive end-to-end performance gain relative to the Milestones 9-13 baseline.

## Validation Log

- `cargo check -p openvm-metal-backend`: pass (2026-02-21)
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`: pass (2026-02-21, 0 tests in target; harness succeeded)
- `cargo run -p openvm-metal-backend --release --example keccakf`: pass (2026-02-21; warm runs captured)
- `cargo test -p openvm-metal-backend --release test_batch_constraints_with_interactions -- --nocapture`: pass (2026-02-21, 1 passed)

## Metrics

- Cross-phase dispatch instrumentation:
  - `metal_dispatch_stage` now includes explicit `phase` values for staged batches.
  - Adopted phases: `prover.rap_constraints.round0`, `prover.main_trace_commit`, `prover.openings.stacked_reduction`.
- Buffer pool instrumentation:
  - Added `metal.buffer_pool.{reuse,miss,return,drop}` counters.
  - Observed reuse in release LogUp path (e.g. round0 `tmp_p/tmp_q` and commit `stacked_mixed` scopes emitted reuse logs).
- Release `keccakf` warm run spans (2026-02-21):
  - `stark_prove_excluding_trace = 2.60s`
  - `prover.main_trace_commit = 935ms`
  - `rs_code_matrix = 707ms`
  - `prover.rap_constraints.round0 = 1.01s`
  - `prover.openings.whir = 422ms`
- Release LogUp benchmark (`n_logup > 0`) spans (2026-02-21):
  - `prover.rap_constraints = 35.5ms`
  - `prover.rap_constraints.logup_gkr = 24.8ms`
  - `prover.rap_constraints.round0 = 6.61ms`
  - `prover.rap_constraints.mle_rounds = 4.01ms`

## Notes

- Pool reuse is exact-size (`(TypeId, len)`) to preserve existing length invariants in debug assertions and avoid protocol-side behavior drift.
- New staged scheduler is a strict wrapper over existing command-buffer semantics (single staged command buffer + one sync), with additional phase tagging only.
- Residual bottlenecks from current release run remain `prover.rap_constraints.round0`, `prover.main_trace_commit`, and `prover.openings.whir`.
