# Milestone 5: LogUp Round0 Scratch Reuse and Reduction Batching

- Status: `completed`
- Owner: codex
- Branch: `perf/m5`
- PR:
- Started: 2026-02-20
- Completed: 2026-02-20
- Last Updated: 2026-02-20

## Objective

Lower allocation churn and sync overhead in the LogUp round0 coset loop.

## Scope

- Files:
  - `crates/metal-backend/src/metal/logup_zerocheck.rs`
- Work:
  - Preallocate `tmp_p`/`tmp_q` once per call and reuse across cosets.
  - Batch or co-schedule final numerator/denominator reductions where safe.
  - Keep memory usage within practical limits.

## Exit Criteria

- Reduced per-coset allocations.
- Reduced reduction dispatch/sync count.
- Round0 span improves or stays stable with lower CPU overhead.

## Validation Log

- `cargo check -p openvm-metal-backend`: pass (existing warnings in `mle_round.rs` and `stacked_reduction.rs`)
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`: pass (`0` tests, `0` failed)
- keccakf trace run: pass (`cargo run -p openvm-metal-backend --example keccakf`), log at `docs/metal-backend-optimization-tracking/baselines/keccakf-2026-02-20-m5.log`

## Metrics

- Before:
  - Round0 staged dispatch structure (`logup_bary_eval_interactions_round0`): per-coset stage (`3` kernels per coset, `sync_count=1` per coset; total syncs = `num_cosets`).
  - Baseline spans (Milestone 0): `stark_prove_excluding_trace=24.9s`, `rs_code_matrix=950ms`, `stacked_commit > merkle_tree=233ms`, `prover.rap_constraints.round0=6.79s`, `prover.rap_constraints.mle_rounds=414ms`.
- After:
  - Round0 staged dispatch structure (`logup_bary_eval_interactions_round0`): single staged command buffer across all cosets (`kernel_count = 3 * num_cosets`, `sync_count=1` per call) plus reused `tmp_p/tmp_q` scratch allocation for the full call.
  - Post-change spans (2026-02-20 run): `stark_prove_excluding_trace=25.0s`, `rs_code_matrix=652ms`, `stacked_commit > merkle_tree=243ms`, `prover.rap_constraints.round0=6.85s`, `prover.rap_constraints.mle_rounds=438ms`.
- Delta:
  - Dispatch/sync: Round0 sync points reduced from `num_cosets` to `1` per call.
  - Trace spans vs baseline: total `+0.1s`; `rs_code_matrix -298ms`; `stacked_commit > merkle_tree +10ms`; `prover.rap_constraints.round0 +60ms`; `prover.rap_constraints.mle_rounds +24ms`.

## Notes

- `tmp_p/tmp_q` are now preallocated once per `logup_bary_eval_interactions_round0` call and reused across all cosets.
- Added dual final-reduction encoder helper and debug tracing around the reduction batching point (`metal_dispatch_stage` with per-coset reduction kernel counts).
