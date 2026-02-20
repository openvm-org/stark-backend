# Milestone Parallelism Guidance

This is a practical guide for running multiple Codex agents concurrently on different milestones.

## Phase 1 (Historical)

- Milestones `0` through `7` are completed.
- Keep previous notes for historical context only; current scheduling applies to Phase 2 milestones `8` through `15`.

## Phase 2 Generally Parallel-Safe

- Milestone 9 (`metal/logup_zerocheck.rs`) with Milestone 11 (`stacked_pcs.rs`, `merkle_tree.rs`)
- Milestone 9 with Milestone 12 (`data_transporter.rs`, `metal/matrix.rs`)
- Milestone 10 (`batch_mle*.rs`) with Milestone 11
- Milestone 10 with Milestone 12
- Milestone 11 with Milestone 13 (`whir.rs`, `stacked_reduction*.rs`)

## Coordination Rules

- Claim milestone in `STATUS.md` before editing.
- One milestone per agent per run.
- Rebase/merge frequently to reduce conflict windows.

## Phase 2 Potential Conflict (Prefer Serial)

- Milestone 9 with Milestone 10
  - Both can touch `crates/metal-backend/src/metal/logup_zerocheck.rs`.
- Milestone 12 with Milestone 14
  - Milestone 14 may update shared allocation/scheduling infra used by transporter paths.
- Milestone 13 with Milestone 14
  - Both may touch shared reduction dispatch infrastructure.
- Milestone 14 with any still-open milestone
  - Cross-phase infra changes are safest after feature milestones merge.
- Milestone 8 and Milestone 15
  - Milestone 15 depends on Milestone 8 baseline definitions.

## Phase 2 Recommended Parallel Plan

1. Run Milestone 8 first.
2. Parallel wave A:
   - Agent A: Milestone 9
   - Agent B: Milestone 11
   - Agent C: Milestone 12
   - Agent D: Milestone 13
3. Parallel wave B after wave A merge:
   - Agent E: Milestone 10
4. Run Milestone 14 after wave B.
5. Run Milestone 15 last.
