# Milestone 9: Round0/LogUp Coset Dispatch Fusion

- Status: `completed`
- Owner: codex
- Branch: `perf/m9`
- PR:
- Started: 2026-02-21
- Completed: 2026-02-21
- Last Updated: 2026-02-21

## Objective

Reduce round0 and LogUp round0 command-buffer churn and scratch allocation overhead without changing proof behavior.

## Scope

- Files:
  - `crates/metal-backend/src/metal/logup_zerocheck.rs`
  - `crates/metal-backend/src/metal/mod.rs`
- Work:
  - Fused zerocheck round0 coset execution into one staged command-buffer submission per round0 call.
  - Reused a persistent zerocheck round0 scratch buffer (`tmp`) across cosets.
  - Kept LogUp round0 staged coset batching and reduced encoder churn in dual final-reduce by encoding both reductions in a single compute encoder.
  - Added/expanded batch tracing fields (`expected_kernel_count`, `command_buffer_count`, `kernel_count`, `sync_count`) for round0 stages.

## Exit Criteria

- Round0 staged dispatch/sync count reduced.
- LogUp benchmark correctness preserved.

## Validation Log

- `cargo check -p openvm-metal-backend`: pass (existing warnings only)
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`: pass (`0` tests, `0` failed)
- `cargo run -p openvm-metal-backend --release --example keccakf`: pass
- `cargo test -p openvm-metal-backend --release test_batch_constraints_with_interactions -- --nocapture`: pass (`1` passed, `0` failed)

## Metrics

### Dispatch/Sync Structure

- Before (pre-M9 code structure):
  - `zerocheck_ntt_eval_constraints` dispatched/synced once per coset (`dispatch_staged_sync` inside the coset loop), with per-coset temporary scratch allocation.
- After:
  - `zerocheck_ntt_eval_constraints` dispatches all cosets in one staged submit (`command_buffer_count=1`, `kernel_count=4`, `sync_count=1` for a `coset_count=2` run) with one shared scratch allocation.
  - `logup_bary_eval_interactions_round0` remains one staged submit per call (`command_buffer_count=1`, `kernel_count=3`, `sync_count=1` for `coset_count=1`), and dual reductions are encoded through one encoder per coset.

### Keccakf Release Spans

- Reference (latest committed release baseline, post-revert):
  - `stark_prove_excluding_trace = 2.42s`
  - `prover.main_trace_commit = 904ms`
  - `rs_code_matrix = 660ms`
  - `prover.rap_constraints.round0 = 928ms`
  - `prover.rap_constraints.mle_rounds = 162ms`
  - `prover.openings.whir = 323ms`
- Post-change (M9 run):
  - `stark_prove_excluding_trace = 2.27s`
  - `prover.main_trace_commit = 855ms`
  - `rs_code_matrix = 634ms`
  - `prover.rap_constraints.round0 = 876ms`
  - `prover.rap_constraints.mle_rounds = 150ms`
  - `prover.openings.whir = 301ms`
- Delta:
  - `stark_prove_excluding_trace: -150ms`
  - `prover.main_trace_commit: -49ms`
  - `rs_code_matrix: -26ms`
  - `prover.rap_constraints.round0: -52ms` (`-5.6%`)
  - `prover.rap_constraints.mle_rounds: -12ms`
  - `prover.openings.whir: -22ms`

### LogUp Path Spans (Release Interaction Test, `n_logup > 0`)

- Phase-2 reference sample (2026-02-20 plan):
  - `prover.rap_constraints = 31.4ms`
  - `prover.rap_constraints.logup_gkr = 17.1ms`
  - `prover.rap_constraints.round0 = 6.18ms`
  - `prover.rap_constraints.mle_rounds = 8.09ms`
- Post-change run:
  - `prover.rap_constraints = 27.8ms`
  - `prover.rap_constraints.logup_gkr = 16.0ms`
  - `prover.rap_constraints.round0 = 7.78ms`
  - `prover.rap_constraints.mle_rounds = 4.09ms`
- Delta vs phase-2 sample:
  - `prover.rap_constraints: -3.6ms`
  - `prover.rap_constraints.logup_gkr: -1.1ms`
  - `prover.rap_constraints.round0: +1.60ms`
  - `prover.rap_constraints.mle_rounds: -4.00ms`

## Notes

- The `>=20%` round0 improvement target vs Milestone 8 median is not reached in this run (`~5.6%` vs latest committed release baseline). Milestone 8 baseline artifacts are not yet committed in this branch, so this milestone records directional improvement with rationale.
- No protocol/transcript behavior changes were made.
