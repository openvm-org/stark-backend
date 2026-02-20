# Milestone 6: Device-Aware `batch_ntt_small` Tuning

- Status: `completed`
- Owner: codex
- Branch: `metal-backend`
- PR:
- Started: 2026-02-19
- Completed: 2026-02-20
- Last Updated: 2026-02-20

## Objective

Evaluate device-aware `batch_ntt_small` launch sizing and keep only changes that do not regress end-to-end prover performance.

## Scope

- Files:
  - `crates/metal-backend/src/metal/batch_ntt_small.rs`
  - `crates/metal-backend/metal/src/batch_ntt_small.metal`
- Work:
  - Implement device-aware threadgroup/shared-memory sizing.
  - Benchmark against pre-M6 state using release-mode `keccakf`.
  - Revert if regression is observed.

## Exit Criteria

- Either keep a non-regressing device-aware launch configuration, or document and revert a regressing change.
- Validation commands pass on the final branch state.

## Validation Log

- `cargo check -p openvm-metal-backend`: pass
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`: pass
- `cargo run -p openvm-metal-backend --release --example keccakf`: pass

## Metrics

- Pre-M6 (`d191fe47`, release `keccakf`):
  - `stark_prove_excluding_trace = 2.46s`
  - `rs_code_matrix = 682ms`
  - `prover.rap_constraints.round0 = 956ms`
  - `prover.rap_constraints.mle_rounds = 143ms`
- M6 candidate (`ddfa41ed`, release `keccakf`):
  - `stark_prove_excluding_trace = 2.54s`
  - `rs_code_matrix = 703ms`
  - `prover.rap_constraints.round0 = 1.00s`
  - `prover.rap_constraints.mle_rounds = 168ms`
- Delta (M6 minus pre-M6):
  - `stark_prove_excluding_trace: +80ms`
  - `rs_code_matrix: +21ms`
  - `prover.rap_constraints.round0: +44ms`
  - `prover.rap_constraints.mle_rounds: +25ms`

## Notes

- The device-aware launch commit `ddfa41ed` regressed release-mode prover time.
- The regression was reverted by `fb16bd32` (`revert: undo batch_ntt_small device-aware tuning regression`).
- Final branch state keeps the pre-M6 `batch_ntt_small` launch behavior.
