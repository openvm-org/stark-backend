# Milestone 7: End-to-End Validation and Profiling Report

- Status: `completed`
- Owner: codex
- Branch: `metal-backend`
- PR:
- Started: 2026-02-20
- Completed: 2026-02-20
- Last Updated: 2026-02-20

## Objective

Confirm optimization work is correct and measurable under release-mode benchmarking.

## Scope

- Run:
  - `cargo check -p openvm-metal-backend`
  - `cargo test -p openvm-metal-backend --test ntt_roundtrip`
  - `cargo run -p openvm-metal-backend --release --example keccakf`
- Compare baseline vs final spans:
  - `rs_code_matrix`
  - `merkle_tree` (main trace + whir openings)
  - `rap_constraints.round0`
  - `rap_constraints.mle_rounds`
- Produce final summary with deltas and remaining bottlenecks.

## Exit Criteria

- Milestones 1-6 executed and validated.
- Checks/tests pass.
- Before/after release trace comparison is documented.

## Validation Log

- `cargo check -p openvm-metal-backend`: pass
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`: pass (`0` tests, `0` failed)
- `cargo run -p openvm-metal-backend --release --example keccakf`: pass; final trace at `docs/metal-backend-optimization-tracking/baselines/keccakf-2026-02-20-release-post-revert.log`

## Metrics

- Baseline total (Milestone 0 commit `ba4e010f`, release): `stark_prove_excluding_trace = 3.35s`
- Final total (post-revert `metal-backend`, release): `stark_prove_excluding_trace = 2.42s`
- Total delta: `-930ms`
- Baseline vs final spans:
  - `rs_code_matrix`: `1.12s` -> `660ms` (delta `-460ms`)
  - `prover.main_trace_commit > stacked_commit > merkle_tree`: `457ms` -> `244ms` (delta `-213ms`)
  - `prover.openings.whir > merkle_tree` aggregate: `90.9ms` -> `89.9ms` (delta `-1.0ms`)
  - `prover.rap_constraints.round0`: `1.16s` -> `928ms` (delta `-232ms`)
  - `prover.rap_constraints.mle_rounds`: `164ms` -> `162ms` (delta `-2ms`)

## Notes

- Release-mode commit sweep identified a regression at Milestone 6 (`ddfa41ed`): `2.46s` -> `2.54s` (`+80ms`) from Milestone 5 baseline.
- The regression was reverted by `fb16bd32`; post-revert `keccakf` is `2.42s`.
- Remaining top bottlenecks in the final trace:
  - `prover.rap_constraints.round0 = 928ms`
  - `prover.main_trace_commit = 904ms`
  - `prover.openings.whir = 323ms`
