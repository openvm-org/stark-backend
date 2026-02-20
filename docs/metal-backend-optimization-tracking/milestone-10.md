# Milestone 10: LogUp MLE Batching and Fallback Elimination

- Status: `not_started`
- Owner:
- Branch:
- PR:
- Started:
- Completed:
- Last Updated: 2026-02-20

## Objective

Improve LogUp throughput under larger interaction workloads and eliminate serialized single-trace fallback where possible.

## Scope

- Update `crates/metal-backend/src/logup_zerocheck/batch_mle.rs` batching strategy.
- Update `crates/metal-backend/src/logup_zerocheck/batch_mle_monomial.rs` to reduce dual-pass overhead.
- Tune `memory_limit_bytes` behavior and add fallback-frequency metrics.

## Exit Criteria

- Reduced fallback frequency for tracked LogUp workloads.
- Improved `prover.rap_constraints.mle_rounds` and/or `prover.rap_constraints.logup_gkr` on LogUp benchmark.
- No proof correctness regressions.

## Validation Log

- Pending.

## Metrics

- Pending.

## Notes

- Run after Milestone 9 merges if overlapping files are touched.
