# Milestone 11: Main Trace Commit and RS Pipeline Optimization

- Status: `completed`
- Owner: codex
- Branch: `perf/m11`
- PR:
- Started: 2026-02-20
- Completed: 2026-02-20
- Last Updated: 2026-02-20

## Objective

Reduce `prover.main_trace_commit` overhead by tightening stacked commit dataflow, with specific focus on `rs_code_matrix` and Merkle tree layer materialization.

## Scope

- Files:
  - `crates/metal-backend/src/stacked_pcs.rs`
  - `crates/metal-backend/src/merkle_tree.rs`
- Work:
  - Added a compact non-cached RS path for `l_skip > 0 && log_blowup > 0`:
    - stack/evaluate at base `height`
    - run one `batch_ntt_small` on the compact buffer
    - run one `batch_expand_pad` to `codeword_height`
  - Switched Merkle adjacent layer construction to a staged command-buffer flow:
    - preallocate all digest layers up front
    - dispatch all adjacent layer compress kernels in one sync point
  - Added tracing counters for the new batching points:
    - `rs_code_matrix_dispatch_stats`
    - `merkle_tree_adjacent_layer_dispatch_stats`
    - `metal_dispatch_stage` for `merkle_tree.adjacent_layers`

## Exit Criteria

- Commit-path batching changes landed without protocol or transcript changes.
- `prover.main_trace_commit`, `rs_code_matrix`, and commit-path `merkle_tree` moved in the right direction.

## Validation Log

- `cargo check -p openvm-metal-backend`: pass (existing warnings unchanged)
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`: pass (`0` tests, `0` failed)
- `cargo test -p openvm-metal-backend test_stacked_opening_reduction -- --nocapture`: pass
- `cargo run -p openvm-metal-backend --release --example keccakf`: pass
- `RUST_LOG=debug cargo run -p openvm-metal-backend --release --example keccakf`: pass; new dispatch stats confirmed in logs

## Metrics

- Baseline reference (latest tracked release before Phase 2; `milestone-7.md`):
  - `stark_prove_excluding_trace = 2.42s`
  - `prover.main_trace_commit = 904ms`
  - `rs_code_matrix = 660ms`
  - `prover.main_trace_commit > merkle_tree = 244ms`
- Post-change warm run (2026-02-20):
  - `stark_prove_excluding_trace = 2.29s`
  - `prover.main_trace_commit = 858ms`
  - `rs_code_matrix = 639ms`
  - `prover.main_trace_commit > merkle_tree = 220ms`
- Delta vs tracked baseline:
  - `stark_prove_excluding_trace`: `-130ms` (`-5.4%`)
  - `prover.main_trace_commit`: `-46ms` (`-5.1%`)
  - `rs_code_matrix`: `-21ms` (`-3.2%`)
  - commit-path `merkle_tree`: `-24ms` (`-9.8%`)

## Notes

- The new compact RS path removes expanded-height `batch_ntt_small` work on zero-padded tails.
- Merkle construction now materially lowers command-buffer break frequency per tree build.
- The milestone target of `>=20%` main-commit improvement was not reached in this slice; this change is directional and leaves substantial headroom for later milestones.

## Residual Bottlenecks

- `prover.rap_constraints.round0` remains the largest span (`~883ms`).
- `prover.main_trace_commit` remains a primary bottleneck (`~858ms`), with `rs_code_matrix` as the dominant sub-span.
- `prover.openings.whir` is still significant (`~292ms`), including repeated Merkle-tree work in opening rounds.
