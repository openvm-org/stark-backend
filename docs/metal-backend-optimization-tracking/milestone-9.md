# Milestone 9: Round0/LogUp Coset Dispatch Fusion

- Status: `not_started`
- Owner:
- Branch:
- PR:
- Started:
- Completed:
- Last Updated: 2026-02-20

## Objective

Reduce round0/LogUp command-buffer churn and allocation overhead by fusing coset dispatch and reusing scratch space.

## Scope

- Refactor round0 coset loops in `crates/metal-backend/src/metal/logup_zerocheck.rs`.
- Reuse per-call scratch buffers across cosets.
- Reduce repeated dispatch/sync boundaries via staged submissions.
- Preserve deterministic ordering and transcript behavior.

## Exit Criteria

- Round0 sync/dispatch count reduced vs Milestone 8 baseline.
- `keccakf` `prover.rap_constraints.round0` improves by at least 20%, or residual bottleneck analysis is documented.
- LogUp benchmark remains correct.

## Validation Log

- Pending.

## Metrics

- Pending.

## Notes

- Coordinate with Milestone 10 to avoid overlapping edits in `metal/logup_zerocheck.rs`.
