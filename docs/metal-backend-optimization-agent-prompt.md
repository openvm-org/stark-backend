You are working in:
/Users/jpw/github/openvm-org/stark-backend-metal

Goal:
Implement exactly one milestone from `docs/metal-backend-optimization-milestones.md` in this run, with no proof-system behavior changes.

Milestone selection input (required):
- `MILESTONE_ID` must be provided by the caller (for example: `8`, `9`, `10`, ...).
- Execute only the selected milestone.
- Do not make code changes for any other milestone, even if nearby.

Context:
- Apple GPU uses unified memory (StorageModeShared semantics matter).
- Phase 1 milestones (`0` through `7`) are historical and complete.
- Phase 2 target (as of February 20, 2026): reduce release `keccakf` median from `2.37s` to `<= 1.18s`.
- LogUp path must be tracked separately using `test_batch_constraints_with_interactions` (n_logup > 0 path).

Hard constraints:
- Keep correctness and transcript behavior unchanged.
- Do not change protocol semantics or cryptographic outputs.
- Do not upgrade Plonky3 (`=0.4.1` remains pinned).
- Always run per-crate checks/tests only (never workspace-wide checks).
- Prefer `cargo check -p openvm-metal-backend`.
- Prefer targeted `cargo test -p openvm-metal-backend ...` for touched paths.
- Treat all non-selected milestones as out of scope for this run.
- Do not edit `docs/metal-backend-optimization-tracking/STATUS.md` in this workflow.

Pre-flight process:
1. Read `docs/metal-backend-optimization-milestones.md`.
2. Read `docs/metal-backend-optimization-tracking/STATUS.md`.
3. Read `docs/metal-backend-optimization-tracking/milestone-<MILESTONE_ID>.md`.
4. If selected milestone is marked `completed`, stop and report no-op.
5. If selected milestone is `in_progress` by another owner, stop and report conflict.

Phase 2 optimization catalog (do not implement all in one run):

8) Benchmark harness expansion and KPI contract
- Capture and store reproducible warm-run baselines for:
  - `cargo run -p openvm-metal-backend --release --example keccakf`
  - `cargo test -p openvm-metal-backend --release test_batch_constraints_with_interactions -- --nocapture`
- Report required spans and median values used by later milestones.

9) Round0/LogUp coset dispatch fusion
- Focus files:
  - `crates/metal-backend/src/metal/logup_zerocheck.rs`
  - `crates/metal-backend/src/metal/mod.rs`
- Goal:
  - reduce coset-loop dispatch/sync overhead and scratch allocation churn.

10) LogUp MLE batching and fallback elimination
- Focus files:
  - `crates/metal-backend/src/logup_zerocheck/batch_mle.rs`
  - `crates/metal-backend/src/logup_zerocheck/batch_mle_monomial.rs`
  - `crates/metal-backend/src/metal/logup_zerocheck.rs`
- Goal:
  - reduce `evaluate_single_logup` fallback use and improve MLE throughput.

11) Main trace commit and RS pipeline optimization
- Focus files:
  - `crates/metal-backend/src/stacked_pcs.rs`
  - `crates/metal-backend/src/merkle_tree.rs`
- Goal:
  - lower `prover.main_trace_commit` and `rs_code_matrix` spans.

12) Trace transport and layout path optimization
- Focus files:
  - `crates/metal-backend/src/data_transporter.rs`
  - `crates/metal-backend/src/metal/matrix.rs`
- Goal:
  - reduce host-side layout work, temporary allocations, and transport dispatch count.

13) Openings and WHIR pipeline tightening
- Focus files:
  - `crates/metal-backend/src/whir.rs`
  - `crates/metal-backend/src/stacked_reduction.rs`
  - `crates/metal-backend/src/metal/stacked_reduction.rs`
- Goal:
  - reduce `prover.openings.whir` and reduction sync overhead.

14) Cross-phase scheduling and buffer pool
- Focus files:
  - `crates/metal-backend/src/metal/mod.rs`
  - `crates/metal-backend/src/base.rs`
  - `crates/metal-common/src/d_buffer.rs`
- Goal:
  - introduce shared scheduling/buffer infrastructure used by multiple phases.

15) Final 2x validation and residual backlog
- No new optimization scope.
- Validate final result against Milestone 8 baselines and publish residual bottlenecks.

Milestone-to-work mapping:
- Milestone 8 -> work item 8
- Milestone 9 -> work item 9
- Milestone 10 -> work item 10
- Milestone 11 -> work item 11
- Milestone 12 -> work item 12
- Milestone 13 -> work item 13
- Milestone 14 -> work item 14
- Milestone 15 -> work item 15

Execution rule:
- Implement only the work item mapped to `MILESTONE_ID`.
- Ignore all other work items.

Implementation requirements:
- Add tracing around new batching points (kernel count / sync count / fallback count where relevant).
- Keep unsafe invariants documented when moving pointer/buffer code.
- Minimize extra allocations in hot loops.

Validation required before finishing:
1. `cargo check -p openvm-metal-backend`
2. Run milestone-relevant tests (at minimum `cargo test -p openvm-metal-backend --test ntt_roundtrip` if touched code can affect it)
3. If milestone includes profiling goals:
   - `cargo run -p openvm-metal-backend --release --example keccakf`
4. If milestone targets LogUp (`9`, `10`, `14`, `15`):
   - `cargo test -p openvm-metal-backend --release test_batch_constraints_with_interactions -- --nocapture`
5. Update only `docs/metal-backend-optimization-tracking/milestone-<MILESTONE_ID>.md`:
   - Set status to `completed`
   - Set/refresh `Completed` and `Last Updated` dates
   - Record validation commands and measured metrics
6. Create a git commit before final response:
   - Stage all intended milestone changes
   - Use a non-empty commit message that references `MILESTONE_ID`
   - Do not amend prior commits unless explicitly instructed
7. Include in final report:
   - `MILESTONE_ID` and scope completed
   - commit hash
   - changed files
   - optimization summary for selected milestone only
   - measured timing deltas from tracked spans
   - residual bottlenecks

Non-goals:
- No protocol redesign.
- No cross-crate large refactors outside metal-backend unless strictly needed for API support.
- No implementation of unselected milestones.

PR description template:
- Use `.github/PULL_REQUEST_TEMPLATE.md`.
- Always fill `PERF_NOTES` with baseline/post-change spans for milestone work.
