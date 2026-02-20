# Milestone 13: Openings and WHIR Pipeline Tightening

- Status: `not_started`
- Owner:
- Branch:
- PR:
- Started:
- Completed:
- Last Updated: 2026-02-20

## Objective

Reduce `prover.openings` latency, especially `prover.openings.whir`, by tightening reduction and batching flow.

## Scope

- Optimize `crates/metal-backend/src/whir.rs` dispatch and buffering.
- Optimize `crates/metal-backend/src/stacked_reduction.rs` and `crates/metal-backend/src/metal/stacked_reduction.rs` batching/sync boundaries.
- Preserve challenge ordering and proof semantics.

## Exit Criteria

- `keccakf` `prover.openings.whir` improves by at least 25% vs Milestone 8 median.
- No correctness or transcript changes.

## Validation Log

- Pending.

## Metrics

- Pending.

## Notes

- Coordinate with Milestone 14 if shared dispatch helpers are introduced.
