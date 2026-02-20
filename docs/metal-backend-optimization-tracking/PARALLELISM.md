# Milestone Parallelism Guidance

This is a practical guide for running multiple Codex agents concurrently on different milestones.

## Generally Parallel-Safe

- Milestone 2 (`stacked_pcs.rs`) with Milestone 6 (`batch_ntt_small.rs`)
- Milestone 2 with Milestone 3
- Milestone 2 with Milestone 5
- Milestone 3 with Milestone 6
- Milestone 5 with Milestone 6

## Potential Conflict (Prefer Serial)

- Milestone 1 with Milestone 5
  - Both touch `crates/metal-backend/src/metal/logup_zerocheck.rs`.
- Milestone 3 with Milestone 4
  - Both touch `crates/metal-backend/src/data_transporter.rs`.
- Milestone 0 and Milestone 7
  - Milestone 7 should run after optimization milestones complete.

## Current State (2026-02-20)

- Milestone 0 completed.
- Wave A completed and merged into `metal-backend`: Milestones 1, 2, 3, and 6.
- Wave B completed and merged into `metal-backend`: Milestones 4 and 5.
- Milestone 7 is the next and final optimization milestone.

## Recommended Parallel Plan

1. Run Milestone 0 first. (completed)
2. Parallel wave A: (completed)
   - Agent A: Milestone 1
   - Agent B: Milestone 2
   - Agent C: Milestone 3
   - Agent D: Milestone 6
3. Parallel wave B (completed after wave A merge):
   - Agent E: Milestone 4
   - Agent F: Milestone 5
4. Run Milestone 7 last (next).

## Coordination Rules

- Claim milestone in `STATUS.md` before editing.
- One milestone per agent per run.
- Rebase/merge frequently to reduce conflict windows.
