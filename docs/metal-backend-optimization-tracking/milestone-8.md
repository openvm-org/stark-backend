# Milestone 8: Benchmark Harness Expansion and KPI Contract

- Status: `completed`
- Owner: codex
- Branch: `metal-backend`
- PR:
- Started: 2026-02-20
- Completed: 2026-02-20
- Last Updated: 2026-02-20

## Objective

Lock in reproducible baseline capture for `keccakf` and LogUp benchmarks and define milestone KPI reporting requirements.

## Scope

- Capture 5 warm release runs for:
  - `cargo run -p openvm-metal-backend --release --example keccakf`
- Capture 5 warm release runs for:
  - `cargo test -p openvm-metal-backend --release test_batch_constraints_with_interactions -- --nocapture`
- Record required metrics:
  - `stark_prove_excluding_trace`
  - `prover.main_trace_commit`
  - `rs_code_matrix`
  - `prover.rap_constraints.round0`
  - `prover.rap_constraints.mle_rounds`
  - `prover.openings.whir`
  - `prover.rap_constraints.logup_gkr` (LogUp benchmark)

## Exit Criteria

- Baseline logs committed under `docs/metal-backend-optimization-tracking/baselines/`.
- Median KPI table recorded in this milestone file and referenced by Milestones 9-15.

## Validation Log

- `cargo run -p openvm-metal-backend --release --example keccakf`:
  - `1` warmup run + `5` measured runs: pass
  - raw log: `docs/metal-backend-optimization-tracking/baselines/keccakf-2026-02-20-m8.log`
- `cargo test -p openvm-metal-backend --release test_batch_constraints_with_interactions -- --nocapture`:
  - `1` warmup run + `5` measured runs: pass
  - raw log: `docs/metal-backend-optimization-tracking/baselines/test_batch_constraints_with_interactions-2026-02-20-m8.log`
- Median extraction:
  - measured-run span parsing done from both logs; medians computed in milliseconds over `n=5` measured runs per benchmark.

## Metrics

| Span | Keccakf median (ms, n=5) | LogUp test median (ms, n=5) |
| --- | ---: | ---: |
| `stark_prove_excluding_trace` | `2440.00` | `N/A` |
| `prover.main_trace_commit` | `891.00` | `N/A` |
| `rs_code_matrix` | `655.00` | `4.38` |
| `prover.rap_constraints.round0` | `949.00` | `3.03` |
| `prover.rap_constraints.mle_rounds` | `178.00` | `3.25` |
| `prover.openings.whir` | `314.00` | `N/A` |
| `prover.rap_constraints.logup_gkr` | `1.54` | `8.74` |

## Notes

- The LogUp benchmark command emits `prover.rap_constraints.*` spans and does not emit top-level prover spans (`stark_prove_excluding_trace`, `prover.main_trace_commit`, `prover.openings.whir`); these are recorded as `N/A` for that benchmark.
- This table is the KPI baseline contract for Milestones 9-15.
