You are working in:
/Users/jpw/github/openvm-org/stark-backend-metal

Goal:
Implement exactly one milestone from `docs/metal-backend-optimization-milestones.md` in this run, with no proof-system behavior changes.

Milestone selection input (required):
- `MILESTONE_ID` must be provided by the caller (e.g. `1`, `2`, `3`, ...).
- Execute only the selected milestone.
- Do not make code changes for any other milestone, even if nearby.

Context:
- Apple GPU has unified memory (StorageModeShared semantics matter).
- Baseline trace from `keccakf` example shows dominant time in:
  - `prover.main_trace_commit > stacked_commit > rs_code_matrix` (~36%)
  - `prover.main_trace_commit > stacked_commit > merkle_tree` (~9%)
  - `prover.rap_constraints.round0` (~35%)
  - `prover.openings.whir > merkle_tree` (~2-3% across calls)

Hard constraints:
- Keep correctness and transcript behavior unchanged.
- Do not change protocol semantics or cryptographic outputs.
- Do not upgrade Plonky3 (`=0.4.1` remains pinned).
- Always run per-crate checks/tests only (never workspace-wide checks).
- Prefer `cargo check -p openvm-metal-backend`.
- Prefer `cargo test -p openvm-metal-backend --test ntt_roundtrip`.
- Treat all non-selected milestones as out of scope for this run.

Pre-flight process:
1. Read `docs/metal-backend-optimization-milestones.md`.
2. Read `docs/metal-backend-optimization-tracking/STATUS.md`.
3. Read `docs/metal-backend-optimization-tracking/milestone-<MILESTONE_ID>.md`.
4. If selected milestone is marked `completed`, stop and report no-op.
5. If selected milestone is `in_progress` by another owner, stop and report conflict.

Optimization catalog (global list; do not implement all in one run):

1) Remove per-copy synchronization in stacked trace assembly (`rs_code_matrix` path)
- Current hotspot:
  - `crates/metal-backend/src/stacked_pcs.rs:110`
  - `crates/metal-backend/src/stacked_pcs.rs:129`
  - `crates/metal-backend/src/stacked_pcs.rs:146`
- Problem:
  - `command::blit_operation` is called repeatedly in a loop, forcing repeated command-buffer sync.
- Target:
  - Batch all blit copies into one command buffer with one sync, or replace with one packing kernel.
  - Keep same output layout and zero-fill semantics.

2) Reduce synchronous kernel dispatch overhead across hot loops
- Current hotspot:
  - `crates/metal-backend/src/metal/mod.rs:166`
  - repeated use in `crates/metal-backend/src/metal/stacked_reduction.rs:77`
  - repeated use in `crates/metal-backend/src/metal/logup_zerocheck.rs:1319`
  - repeated reductions in `crates/metal-backend/src/metal/logup_zerocheck.rs:1406`
- Problem:
  - New command buffer + immediate wait for many kernels.
- Target:
  - Introduce stage-level batching (multi-dispatch/async + single barrier per stage where legal).
  - Preserve ordering and determinism.

3) Exploit unified memory to remove avoidable D2H/H2D copies
- Current hotspots:
  - Root extraction: `crates/metal-backend/src/merkle_tree.rs:97`
  - Per-layer H2D digest copies: `crates/metal-backend/src/data_transporter.rs:170`
  - Generic matrix D2H path: `crates/metal-backend/src/data_transporter.rs:217`
  - Column openings conversion path: `crates/metal-backend/src/logup_zerocheck/mod.rs:2222`
- Problem:
  - Unnecessary `to_host()` / `to_device()` copies in shared-memory environment.
- Target:
  - Keep data in shared buffers and avoid full copies when only scalar/sliced reads are needed.
  - Preserve API behavior for callers that require owned host buffers.

4) Remove strided-upload double buffering during trace transport
- Current hotspot:
  - `crates/metal-backend/src/data_transporter.rs:110`
  - `crates/metal-backend/src/data_transporter.rs:131`
  - `crates/metal-backend/src/data_transporter.rs:136`
  - `crates/metal-backend/src/metal/matrix.rs:145`
- Problem:
  - Upload to `strided_trace`, then allocate again and run `collapse_strided_matrix`.
- Target:
  - Upload in contiguous layout directly when possible, removing extra allocation and kernel dispatch.
  - Keep compatibility with `stride == 1` and `stride > 1`.

5) Reuse temporary buffers and batch reductions in logup round0
- Current hotspot:
  - `crates/metal-backend/src/metal/logup_zerocheck.rs:1254`
  - `crates/metal-backend/src/metal/logup_zerocheck.rs:1315`
  - `crates/metal-backend/src/metal/logup_zerocheck.rs:1414`
  - `crates/metal-backend/src/metal/logup_zerocheck.rs:69`
- Problem:
  - Per-coset `tmp_p/tmp_q` allocations and separate synced reductions.
- Target:
  - Preallocate scratch buffers per call and reuse across cosets.
  - Batch numerator/denominator final reductions where safe.

6) Make `batch_ntt_small` threadgroup/shared-memory sizing device-aware
- Current hotspot:
  - `crates/metal-backend/src/metal/batch_ntt_small.rs:57`
  - `crates/metal-backend/src/metal/batch_ntt_small.rs:67`
- Problem:
  - Hard-coded 1024 assumptions can reduce occupancy or break on stricter limits.
- Target:
  - Derive threadgroup shape and shared memory from pipeline/device limits.
  - Keep existing correctness checks and twiddle usage.

Milestone-to-work mapping:
- Milestone 1 -> work item 2
- Milestone 2 -> work item 1
- Milestone 3 -> work item 3
- Milestone 4 -> work item 4
- Milestone 5 -> work item 5
- Milestone 6 -> work item 6
- Milestone 7 -> end-to-end validation/reporting only (no new optimization scope)
- Milestone 0 -> baseline harness/guardrails only

Execution rule:
- Implement only the work items mapped to `MILESTONE_ID`.
- Ignore all other work items.

Implementation requirements:
- Add tracing around new batching points (kernel count / sync count per major phase).
- Keep unsafe invariants documented when moving pointer/buffer code.
- Minimize extra allocations in hot loops.

Validation required before finishing:
1. `cargo check -p openvm-metal-backend`
2. Run milestone-relevant tests (at least `cargo test -p openvm-metal-backend --test ntt_roundtrip` if touched code affects it)
3. If milestone includes profiling goals, run keccakf with tracing:
   - `cargo run -p openvm-metal-backend --example keccakf`
4. Update tracking files:
   - `docs/metal-backend-optimization-tracking/milestone-<MILESTONE_ID>.md`
   - `docs/metal-backend-optimization-tracking/STATUS.md`
5. Include in final report:
   - `MILESTONE_ID` and scope completed
   - changed files
   - optimization summary for selected milestone only
   - measured timing deltas from trace spans
   - any residual bottlenecks

Non-goals:
- No protocol redesign.
- No cross-crate large refactors outside metal-backend unless strictly needed for API support.
- No implementation of unselected milestones.
