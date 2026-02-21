# Milestone 15: Final 2x Validation and Residual Backlog

- Status: `completed`
- Owner: codex
- Branch: `metal-backend`
- PR:
- Started: 2026-02-21
- Completed: 2026-02-21
- Last Updated: 2026-02-21

## Objective

Validate final performance against the 2x target and publish residual bottlenecks with prioritized follow-up work.

## Scope

- Run:
  - `cargo check -p openvm-metal-backend`
  - `cargo test -p openvm-metal-backend --test ntt_roundtrip`
  - `cargo run -p openvm-metal-backend --release --example keccakf` (5 warm runs)
  - `cargo test -p openvm-metal-backend --release test_batch_constraints_with_interactions -- --nocapture` (5 warm runs)
- Compare results against Milestone 8 baseline medians.
- Document final speedup, span deltas, and remaining bottlenecks.

## Exit Criteria

- `keccakf` median `stark_prove_excluding_trace <= 1.18s` or documented gap-to-target analysis.
- LogUp benchmark spans (`logup_gkr`, `round0`, `mle_rounds`) show measurable improvement or documented limits.
- Final report includes prioritized residual optimization backlog.

## Validation Log

- `cargo check -p openvm-metal-backend`: pass
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`: pass (`0` tests discovered in this target)
- `cargo run -p openvm-metal-backend --release --example keccakf`:
  - warmup run + `5` measured runs: pass
  - measured span capture source: `/tmp/keccakf-2026-02-21-m15.log`
- `cargo test -p openvm-metal-backend --release test_batch_constraints_with_interactions -- --nocapture`:
  - warmup run + `5` measured runs: pass
  - measured span capture source: `/tmp/test_batch_constraints_with_interactions-2026-02-21-m15.log`
- Span parsing:
  - converted `s`/`ms`/`us`/`µs` to milliseconds
  - medians computed over measured runs (`n=5`)
  - p95-style spread reported as nearest-rank p95 over `n=5` (`max` observed run)

## Metrics

### Benchmark headline

| Benchmark | Headline span | M15 median (ms, n=5) | p95-style (ms) | min..max (ms) |
| --- | --- | ---: | ---: | ---: |
| `keccakf` | `stark_prove_excluding_trace` | `2460.00` | `2470.00` | `2370.00..2470.00` |
| `test_batch_constraints_with_interactions` | `prover.rap_constraints` | `23.50` | `25.30` | `22.30..25.30` |

### Keccakf spans vs Milestone 8 baseline

| Span | M8 median (ms) | M15 median (ms, n=5) | Delta (ms) | Delta (%) |
| --- | ---: | ---: | ---: | ---: |
| `stark_prove_excluding_trace` | `2440.00` | `2460.00` | `+20.00` | `+0.82%` |
| `prover.main_trace_commit` | `891.00` | `833.00` | `-58.00` | `-6.51%` |
| `rs_code_matrix` | `655.00` | `613.00` | `-42.00` | `-6.41%` |
| `prover.rap_constraints.round0` | `949.00` | `934.00` | `-15.00` | `-1.58%` |
| `prover.rap_constraints.mle_rounds` | `178.00` | `176.00` | `-2.00` | `-1.12%` |
| `prover.openings.whir` | `314.00` | `408.00` | `+94.00` | `+29.94%` |
| `prover.rap_constraints.logup_gkr` | `1.54` | `1.53` | `-0.01` | `-0.65%` |

### LogUp benchmark spans vs Milestone 8 baseline

| Span | M8 median (ms) | M15 median (ms, n=5) | Delta (ms) | Delta (%) |
| --- | ---: | ---: | ---: | ---: |
| `rs_code_matrix` | `4.38` | `9.18` | `+4.80` | `+109.59%` |
| `prover.rap_constraints.round0` | `3.03` | `5.88` | `+2.85` | `+94.06%` |
| `prover.rap_constraints.mle_rounds` | `3.25` | `2.10` | `-1.15` | `-35.38%` |
| `prover.rap_constraints.logup_gkr` | `8.74` | `15.40` | `+6.66` | `+76.20%` |

### Target gate result

- M8→M15 keccakf speedup ratio on `stark_prove_excluding_trace`: `2440.00 / 2460.00 = 0.992x` (regression)
- 2x target check (`<= 1180ms`): **not met**
  - gap to target: `2460.00 - 1180.00 = 1280.00ms`

## Notes

- Residual bottlenecks (ranked by expected ROI):
  1. `prover.openings.whir` regression (`+29.94%` vs M8; median `408ms`) dominates the negative keccakf delta despite gains in commit/RS spans.
  2. `prover.rap_constraints.round0` remains a large absolute cost (`934ms`) and only improved modestly vs M8 (`-1.58%`).
  3. LogUp path `prover.rap_constraints.logup_gkr` and `round0` regressed materially on the tracked interaction test (`+76.20%`, `+94.06%`), indicating residual phase-level scheduling/dispatch inefficiency under `n_logup > 0`.
  4. `rs_code_matrix` improved in keccakf (`-6.41%`) but still carries a large absolute share (`613ms`) and remains a secondary optimization target after WHIR/LogUp stabilization.
