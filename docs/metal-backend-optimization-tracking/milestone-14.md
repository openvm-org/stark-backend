# Milestone 14: Cross-Phase Scheduling and Buffer Pool

- Status: `not_started`
- Owner:
- Branch:
- PR:
- Started:
- Completed:
- Last Updated: 2026-02-20

## Objective

Add cross-phase scheduling and reusable temporary buffer pooling to cut overhead across major proving phases.

## Scope

- Add phase-level staged scheduling helpers in `crates/metal-backend/src/metal/mod.rs`.
- Add reusable temporary buffer pooling in `crates/metal-backend/src/base.rs` and/or `crates/metal-common/src/d_buffer.rs`.
- Adopt across round0/logup, stacked commit, and openings paths.

## Exit Criteria

- Reduced allocation churn and sync count across at least three phases.
- Positive end-to-end performance gain relative to the Milestones 9-13 baseline.

## Validation Log

- Pending.

## Metrics

- Pending.

## Notes

- Prefer landing this milestone after feature milestones merge to minimize integration conflicts.
