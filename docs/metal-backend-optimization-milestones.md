# Metal Backend Performance Optimization Milestones

This plan is for implementing all currently identified performance optimizations in `crates/metal-backend` on Apple GPUs with unified memory.

## Baseline

Reference trace (keccakf example):
- `stark_prove_excluding_trace`: 2.55s
- `prover.main_trace_commit > stacked_commit > rs_code_matrix`: 922ms
- `prover.main_trace_commit > stacked_commit > merkle_tree`: 226ms
- `prover.rap_constraints.round0`: 889ms
- `prover.rap_constraints.mle_rounds`: 123ms
- `prover.openings.whir`: 307ms

## Milestone 0: Baseline Harness and Guardrails

Objective:
- Ensure every performance change is comparable and does not regress correctness.

Tasks:
- Capture and store baseline traces for:
  - `cargo run -p openvm-metal-backend --release --example keccakf`
- Confirm local correctness checks:
  - `cargo check -p openvm-metal-backend`
  - `cargo test -p openvm-metal-backend --test ntt_roundtrip`
- Add a short `PERF_NOTES` section to PR description template used by the agent.

Exit criteria:
- Baseline numbers recorded in PR notes.
- Checks/tests pass.

## Milestone 1: Batch Synchronization and Dispatch Plumbing

Objective:
- Remove avoidable CPU/GPU synchronization overhead from hot sequential kernels.

Primary files:
- `crates/metal-backend/src/metal/mod.rs`
- `crates/metal-backend/src/metal/stacked_reduction.rs`
- `crates/metal-backend/src/metal/logup_zerocheck.rs`

Tasks:
- Introduce or expand staged dispatch API so multiple kernels can be encoded and synced once.
- Replace obvious back-to-back `dispatch_sync` chains in hot paths with staged dispatch.
- Preserve phase ordering and error propagation semantics.

Exit criteria:
- Fewer command-buffer sync points in round0 and reduction paths.
- No correctness changes in tests.

## Milestone 2: Stacked Trace Copy Path (`rs_code_matrix`) Optimization

Objective:
- Remove per-column blit synchronization in stacked trace assembly.

Primary files:
- `crates/metal-backend/src/stacked_pcs.rs`

Tasks:
- Refactor `stack_traces_into_expanded` copy loop to batch blit copies.
- Keep expand/pad behavior for `s.log_height() < l_skip`.
- Keep exact layout and zeroed padding behavior unchanged.

Exit criteria:
- Copy path no longer syncs per segment.
- `rs_code_matrix` trace span decreases in keccakf profiling.

## Milestone 3: Unified-Memory Copy Elimination (H2D/D2H cleanup)

Objective:
- Reduce redundant host-device copies by taking advantage of shared memory.

Primary files:
- `crates/metal-backend/src/merkle_tree.rs`
- `crates/metal-backend/src/data_transporter.rs`
- `crates/metal-backend/src/logup_zerocheck/mod.rs`

Tasks:
- Remove unnecessary `to_host()` for merkle root extraction when scalar read can come from shared buffer.
- Reduce repeated `to_device()` / `to_host()` conversions for digest layers and matrix readback paths.
- Rework `into_column_openings` to avoid full matrix D2H copies when only folded single-row values are needed.

Exit criteria:
- Fewer `to_host()` / `to_device()` calls in hot phases.
- No change to proof outputs or transcript interactions.

## Milestone 4: Trace Transport Layout Path Simplification

Objective:
- Eliminate strided upload then collapse sequence.

Primary files:
- `crates/metal-backend/src/data_transporter.rs`
- `crates/metal-backend/src/metal/matrix.rs`

Tasks:
- Remove double-buffered strided upload path in `transport_and_unstack_single_data_h2d`.
- Produce contiguous target layout directly where feasible.
- Keep compatibility with both `stride == 1` and `stride > 1`.

Exit criteria:
- One less allocation and one less kernel launch in strided trace transport path.
- Existing transport invariants still hold.

## Milestone 5: LogUp Round0 Scratch Reuse and Reduction Batching

Objective:
- Lower allocation churn and sync overhead in coset loop.

Primary files:
- `crates/metal-backend/src/metal/logup_zerocheck.rs`

Tasks:
- Preallocate `tmp_p` / `tmp_q` scratch buffers once per call and reuse across cosets.
- Batch or co-schedule final numerator/denominator reductions to minimize sync boundaries.
- Validate memory footprint does not exceed practical limits for target devices.

Exit criteria:
- Reduced per-coset allocations.
- Reduced reduction dispatch/sync count.
- Round0 span improves or remains stable with lower CPU overhead.

## Milestone 6: `batch_ntt_small` Device-Aware Occupancy Tuning

Objective:
- Improve portability and occupancy by removing hard-coded threadgroup assumptions.

Primary files:
- `crates/metal-backend/src/metal/batch_ntt_small.rs`

Tasks:
- Query pipeline/device limits for max threads per threadgroup.
- Compute threadgroup dimensions and shared memory usage accordingly.
- Keep algorithm and twiddle logic unchanged.

Exit criteria:
- Launch configuration adapts to device limits.
- No kernel validation failures from threadgroup/shared-memory mismatch.

## Milestone 7: End-to-End Validation and Profiling Report

Objective:
- Confirm optimizations are complete, correct, and measurable.

Tasks:
- Run:
  - `cargo check -p openvm-metal-backend`
  - `cargo test -p openvm-metal-backend --test ntt_roundtrip`
  - `cargo run -p openvm-metal-backend --release --example keccakf`
- Compare baseline vs final spans:
  - `rs_code_matrix`
  - `merkle_tree` (main trace + whir openings)
  - `rap_constraints.round0`
  - `rap_constraints.mle_rounds`
- Document:
  - total prove-time delta
  - per-span deltas
  - remaining top bottlenecks

Exit criteria:
- All required optimizations implemented (Milestones 1-6).
- Checks/tests pass.
- Before/after trace comparison included in final summary.

## Implementation Notes

- Keep each milestone in a reviewable commit or tightly scoped PR.
- Avoid workspace-wide commands; only run crate-specific check/test commands.
- If a milestone requires API changes outside `crates/metal-backend`, keep changes minimal and justified in PR notes.
