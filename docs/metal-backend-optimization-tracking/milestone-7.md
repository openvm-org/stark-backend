# Milestone 7: End-to-End Validation and Profiling Report

- Status: `not_started`
- Owner:
- Branch:
- PR:
- Started:
- Completed:
- Last Updated:

## Objective

Confirm optimization milestones are complete, correct, and measurable.

## Scope

- Run:
  - `cargo check -p openvm-metal-backend`
  - `cargo test -p openvm-metal-backend --test ntt_roundtrip`
  - `cargo run -p openvm-metal-backend --example keccakf`
- Compare baseline vs final spans:
  - `rs_code_matrix`
  - `merkle_tree` (main trace + whir openings)
  - `rap_constraints.round0`
  - `rap_constraints.mle_rounds`
- Produce final summary with deltas and remaining bottlenecks.

## Exit Criteria

- Milestones 1-6 are complete.
- Checks/tests pass.
- Before/after trace comparison is documented.

## Validation Log

- `cargo check -p openvm-metal-backend`:
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`:
- `cargo run -p openvm-metal-backend --example keccakf`:

## Metrics

- Before total:
- After total:
- Before/after spans:

## Notes

