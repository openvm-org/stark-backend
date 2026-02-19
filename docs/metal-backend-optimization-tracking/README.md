# Metal Backend Optimization Tracking

This directory tracks milestone ownership and completion for `docs/metal-backend-optimization-milestones.md`.

## Workflow

1. Pick one milestone.
2. Claim it by editing:
   - `docs/metal-backend-optimization-tracking/STATUS.md`
   - `docs/metal-backend-optimization-tracking/milestone-<id>.md`
3. Set status to `in_progress` with owner, branch, and start date.
4. Implement only that milestone.
5. Record validation and performance results in the milestone file.
6. Set status to `completed` when done, and update completion date.

## Status Values

- `not_started`
- `in_progress`
- `blocked`
- `completed`

## Files

- `STATUS.md`: global board across milestones.
- `milestone-0.md` ... `milestone-7.md`: detailed per-milestone logs and results.
- `PARALLELISM.md`: guidance for running multiple milestone agents concurrently.
