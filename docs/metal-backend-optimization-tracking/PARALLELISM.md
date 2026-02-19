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

## Recommended Parallel Plan

1. Run Milestone 0 first.
2. Parallel wave A:
   - Agent A: Milestone 1
   - Agent B: Milestone 2
   - Agent C: Milestone 3
   - Agent D: Milestone 6
3. Parallel wave B (after wave A merge):
   - Agent E: Milestone 4
   - Agent F: Milestone 5
4. Run Milestone 7 last.

## Coordination Rules

- Claim milestone in `STATUS.md` before editing.
- One milestone per agent per run.
- Rebase/merge frequently to reduce conflict windows.
