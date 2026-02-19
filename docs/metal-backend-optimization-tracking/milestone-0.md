# Milestone 0: Baseline Harness and Guardrails

- Status: `completed`
- Owner: codex
- Branch: `metal-backend`
- PR:
- Started: 2026-02-19
- Completed: 2026-02-19
- Last Updated: 2026-02-19

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

- `cargo check -p openvm-metal-backend`: pass (existing warnings in `mle_round.rs` and `stacked_reduction.rs`)
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`: pass (`0` tests, `0` failed)
- `cargo run -p openvm-metal-backend --example keccakf`: pass; trace captured at `docs/metal-backend-optimization-tracking/baselines/keccakf-2026-02-19.log`

## Metrics

- Baseline total: `stark_prove_excluding_trace = 24.9s`
- Baseline `rs_code_matrix`: `950ms`
- Baseline `merkle_tree`: `233ms` (`prover.main_trace_commit > stacked_commit > merkle_tree`)
- Baseline `rap_constraints.round0`: `6.79s`
- Baseline `rap_constraints.mle_rounds`: `414ms`

## Notes

- Additional `prover.openings.whir > merkle_tree` spans from baseline: `48.8ms`, `26.5ms`, `14.5ms`.
- Added PR template guardrail with `PERF_NOTES` section at `.github/PULL_REQUEST_TEMPLATE.md`.
