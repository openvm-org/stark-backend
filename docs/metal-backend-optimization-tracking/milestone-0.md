# Milestone 0: Baseline Harness and Guardrails

- Status: `not_started`
- Owner:
- Branch:
- PR:
- Started:
- Completed:
- Last Updated:

## Objective

Ensure every performance change is comparable and does not regress correctness.

## Scope

- Capture baseline trace for:
  - `cargo run -p openvm-metal-backend --example keccakf`
- Confirm local correctness checks:
  - `cargo check -p openvm-metal-backend`
  - `cargo test -p openvm-metal-backend --test ntt_roundtrip`
- Add baseline numbers to notes.

## Exit Criteria

- Baseline numbers are recorded.
- Required check/test commands pass.

## Validation Log

- `cargo check -p openvm-metal-backend`:
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`:
- `cargo run -p openvm-metal-backend --example keccakf`:

## Metrics

- Baseline total:
- Baseline `rs_code_matrix`:
- Baseline `merkle_tree`:
- Baseline `rap_constraints.round0`:
- Baseline `rap_constraints.mle_rounds`:

## Notes

