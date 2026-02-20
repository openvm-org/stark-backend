# Metal Backend Performance Optimization Milestones

This plan tracks the next optimization phase for `crates/metal-backend` on Apple GPUs with unified memory.

## Program Goal (Phase 2)

Primary KPI (end-to-end):
- On February 20, 2026, release `keccakf` warm runs measured `stark_prove_excluding_trace` at `2.36s`, `2.37s`, and `2.39s` (median `2.37s`).
- Target: `<= 1.18s` median over 5 warm release runs (`2x` speedup).

Secondary KPI (LogUp path):
- Use a release run with `n_logup > 0`:
  - `cargo test -p openvm-metal-backend --release test_batch_constraints_with_interactions -- --nocapture`
- On February 20, 2026, sample release trace measured:
  - `prover.rap_constraints = 31.4ms`
  - `prover.rap_constraints.logup_gkr = 17.1ms`
  - `prover.rap_constraints.round0 = 6.18ms`
  - `prover.rap_constraints.mle_rounds = 8.09ms`

Guardrails:
- No protocol or transcript behavior changes.
- No cryptographic output changes.
- Per-crate validation only; no workspace-wide check/test commands.

## Historical Milestones (Completed)

Milestones `0` through `7` are complete and preserved for traceability in:
- `docs/metal-backend-optimization-tracking/milestone-0.md`
- `docs/metal-backend-optimization-tracking/milestone-7.md`

These established the current baseline and removed major known overheads, including staged sync improvements, stacked copy batching, unified-memory copy cleanup, and strided transport simplification.

## Phase 2 Milestones (Active Plan)

### Milestone 8: Benchmark Harness Expansion and KPI Contract

Objective:
- Lock in reproducible baseline capture for both `keccakf` and LogUp-heavy paths.

Primary files:
- `docs/metal-backend-optimization-tracking/baselines/`
- `docs/metal-backend-optimization-tracking/milestone-8.md`
- optional helper in `crates/metal-backend/examples/` (only if needed to improve repeatability)

Tasks:
- Capture 5 warm release runs for:
  - `cargo run -p openvm-metal-backend --release --example keccakf`
- Capture 5 warm release runs for:
  - `cargo test -p openvm-metal-backend --release test_batch_constraints_with_interactions -- --nocapture`
- Standardize required reported spans:
  - `stark_prove_excluding_trace`
  - `prover.main_trace_commit`
  - `rs_code_matrix`
  - `prover.rap_constraints.round0`
  - `prover.rap_constraints.mle_rounds`
  - `prover.openings.whir`
  - `prover.rap_constraints.logup_gkr` (LogUp benchmark)
- Store raw logs and median table in the milestone tracking file.

Exit criteria:
- Baseline logs committed for both benchmarks.
- Median KPI table is recorded and referenced by later milestones.

### Milestone 9: Round0/LogUp Coset Dispatch Fusion

Objective:
- Reduce round0 and LogUp round0 command-buffer churn and allocation overhead.

Primary files:
- `crates/metal-backend/src/metal/logup_zerocheck.rs`
- `crates/metal-backend/src/metal/mod.rs`

Tasks:
- Fuse per-coset dispatch loops into fewer staged command-buffer submissions.
- Reuse persistent scratch buffers across cosets for round0 evaluation.
- Reduce repeated pipeline/buffer binding work where argument layouts are stable.
- Keep reduction ordering and deterministic behavior unchanged.

Exit criteria:
- Measurable drop in sync/dispatch count in round0 spans.
- `keccakf` `prover.rap_constraints.round0` improves by at least `20%` vs Milestone 8 median, or rationale is documented if not reached.
- LogUp benchmark has no correctness regressions.

### Milestone 10: LogUp MLE Batching and Fallback Elimination

Objective:
- Improve LogUp scalability under larger interaction workloads and avoid single-trace serialization.

Primary files:
- `crates/metal-backend/src/logup_zerocheck/batch_mle.rs`
- `crates/metal-backend/src/logup_zerocheck/batch_mle_monomial.rs`
- `crates/metal-backend/src/metal/logup_zerocheck.rs`

Tasks:
- Replace or reduce `evaluate_single_logup` fallback usage with chunked/sub-batch execution.
- Tune `memory_limit_bytes` heuristics for balanced GPU occupancy and memory usage.
- Cut dual-pass monomial overhead (numerator/denominator) where safe to fuse or co-schedule.
- Add tracing counters for fallback frequency and batch composition.

Exit criteria:
- LogUp benchmark shows reduced `mle_rounds` and/or `logup_gkr` spans.
- Oversized-trace fallback frequency is reduced to zero for tracked benchmark inputs or explicitly justified.

### Milestone 11: Main Trace Commit and RS Pipeline Optimization

Objective:
- Reduce `prover.main_trace_commit` time by improving stacked commit dataflow.

Primary files:
- `crates/metal-backend/src/stacked_pcs.rs`
- `crates/metal-backend/src/merkle_tree.rs`

Tasks:
- Reuse recurring intermediate buffers across stacked commit stages.
- Further batch copy/pack operations and limit command-buffer breaks.
- Evaluate kernel fusion opportunities in copy + pad + layout setup without changing output layout.
- Minimize Merkle layer materialization overhead in commit path.

Exit criteria:
- `keccakf` `prover.main_trace_commit` improves by at least `20%` vs Milestone 8 median.
- `rs_code_matrix` and commit-path `merkle_tree` spans show directional improvement.

### Milestone 12: Trace Transport and Layout Path Optimization

Objective:
- Reduce host-side work and temporary allocations in data transport.

Primary files:
- `crates/metal-backend/src/data_transporter.rs`
- `crates/metal-backend/src/metal/matrix.rs`

Tasks:
- Move remaining strided-collapse style work to direct layout generation or GPU gather path.
- Eliminate avoidable temporary host allocations in transport hot paths.
- Keep `stride == 1` and `stride > 1` behavior compatible with existing invariants.
- Add per-path dispatch/allocation counters for verification.

Exit criteria:
- Fewer allocations and dispatches in transport traces.
- No regression in keccakf or LogUp benchmark correctness.

### Milestone 13: Openings and WHIR Pipeline Tightening

Objective:
- Reduce `prover.openings` cost, especially `prover.openings.whir`.

Primary files:
- `crates/metal-backend/src/whir.rs`
- `crates/metal-backend/src/stacked_reduction.rs`
- `crates/metal-backend/src/metal/stacked_reduction.rs`

Tasks:
- Batch/fuse opening reduction dispatches where ordering allows.
- Reuse round buffers across opening rounds and Merkle-related stages.
- Minimize synchronous boundaries in WHIR path while preserving challenge flow.

Exit criteria:
- `keccakf` `prover.openings.whir` improves by at least `25%` vs Milestone 8 median.
- No transcript or proof-output changes.

### Milestone 14: Cross-Phase Scheduling and Buffer Pool

Objective:
- Introduce reusable infrastructure that lowers overhead across round0, commit, and openings.

Primary files:
- `crates/metal-backend/src/metal/mod.rs`
- `crates/metal-backend/src/base.rs`
- `crates/metal-common/src/d_buffer.rs`

Tasks:
- Add phase-level scheduling helpers for multi-dispatch batches with explicit sync points.
- Add reusable temporary buffer pooling/arena strategy for common hot sizes.
- Adopt infrastructure in at least:
  - round0/logup path
  - stacked commit path
  - openings path

Exit criteria:
- Reduced allocation churn and sync count across all three target phases.
- Net end-to-end gain is positive versus Milestones 9-13 combined baseline.

### Milestone 15: Final 2x Validation and Residual Backlog

Objective:
- Validate final performance against the 2x target and document any remaining bottlenecks.

Tasks:
- Run:
  - `cargo check -p openvm-metal-backend`
  - `cargo test -p openvm-metal-backend --test ntt_roundtrip`
  - `cargo run -p openvm-metal-backend --release --example keccakf` (5 warm runs)
  - `cargo test -p openvm-metal-backend --release test_batch_constraints_with_interactions -- --nocapture` (5 warm runs)
- Compare against Milestone 8 baseline medians.
- Report:
  - final median and p95-style spread for both benchmarks
  - per-span deltas
  - achieved speedup ratio
  - residual bottlenecks and ranked follow-up backlog

Exit criteria:
- `keccakf` median `<= 1.18s` or a documented gap-to-target analysis with next-step ROI.
- LogUp benchmark shows clear improvement in `logup_gkr`, `round0`, and `mle_rounds` or rationale for bottleneck limits.

## Implementation and Sequencing Notes

- Keep each milestone reviewable and independently benchmarked.
- Recommended order:
  1. Milestone 8
  2. Milestones 9, 11, 12, 13 (parallel where conflict-free)
  3. Milestone 10
  4. Milestone 14
  5. Milestone 15
- Update tracking files on every milestone start and completion:
  - `docs/metal-backend-optimization-tracking/STATUS.md`
  - `docs/metal-backend-optimization-tracking/milestone-<id>.md`
