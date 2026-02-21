# Milestone 12: Trace Transport and Layout Path Optimization

- Status: `completed`
- Owner: codex
- Branch: `smb-12`
- PR:
- Started: 2026-02-20
- Completed: 2026-02-20
- Last Updated: 2026-02-20

## Objective

Reduce host-side transport/layout overhead by removing avoidable matrix uploads, avoiding unnecessary layout kernel dispatches for degenerate shapes, and skipping zero-work layout kernels.

## Scope

- Files:
  - `crates/metal-backend/src/data_transporter.rs`
  - `crates/metal-backend/src/metal/matrix.rs`
- Work:
  - Gate stacked-matrix H2D upload behind `cache_stacked_matrix` in transport paths.
  - Add transport tracing counters for cache upload activity and transpose dispatch/allocation counts.
  - Add degenerate row/column transpose fast paths (`width <= 1 || height <= 1`) to avoid transpose dispatches and temporary buffers.
  - Add zero-work guards in matrix dispatch wrappers to avoid empty kernel/sync submissions.

## Exit Criteria

- Fewer avoidable transport allocations and layout dispatches in touched paths.
- `stride == 1` and `stride > 1` transport behavior preserved.
- No correctness regressions in keccakf path and LogUp interaction path.

## Validation Log

- `cargo check -p openvm-metal-backend`: pass
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`: pass (`0` tests, `0` failed)
- `cargo test -p openvm-metal-backend collapse_strided_trace`: pass (`2` tests, `0` failed)
- `cargo test -p openvm-metal-backend transpose_layout_identity_only_for_degenerate_shapes`: pass (`1` test, `0` failed)
- `cargo test -p openvm-metal-backend test_batch_constraints_with_interactions -- --nocapture`: pass (`1` test, `0` failed)
- `cargo run -p openvm-metal-backend --release --example keccakf`: pass

## Metrics

- Baseline reference (Milestone 7 post-revert release trace):
  - `stark_prove_excluding_trace = 2.42s`
  - `prover.main_trace_commit = 904ms`
  - `rs_code_matrix = 660ms`
  - `prover.rap_constraints.round0 = 928ms`
  - `prover.rap_constraints.mle_rounds = 162ms`
  - `prover.openings.whir = 323ms`
- Post-change release trace (`cargo run -p openvm-metal-backend --release --example keccakf`):
  - `stark_prove_excluding_trace = 2.27s`
  - `prover.main_trace_commit = 853ms`
  - `rs_code_matrix = 631ms`
  - `prover.rap_constraints.round0 = 872ms`
  - `prover.rap_constraints.mle_rounds = 143ms`
  - `prover.openings.whir = 311ms`
- Delta vs Milestone 7 reference:
  - `stark_prove_excluding_trace`: `-150ms`
  - `prover.main_trace_commit`: `-51ms`
  - `rs_code_matrix`: `-29ms`
  - `prover.rap_constraints.round0`: `-56ms`
  - `prover.rap_constraints.mle_rounds`: `-19ms`
  - `prover.openings.whir`: `-12ms`

## Notes

- `transport_and_unstack_single_data_h2d` now avoids full stacked-matrix upload when `cache_stacked_matrix = false` (default), eliminating one full H2D allocation/copy per call in that mode.
- `transport_pcs_data_h2d` applies the same cache-gated upload behavior.
- Degenerate transpose layouts now bypass transpose kernels:
  - row-major H2D: `width <= 1 || height <= 1` avoids transpose dispatch and temporary transpose output allocation.
  - row-major D2H: same shape rule avoids transpose dispatch and temporary transpose buffer allocation.
- Matrix wrapper guards skip dispatches when dimensions imply zero work (`width == 0`, `height == 0`, zero rows queried, etc.).

## Residual Bottlenecks

- `prover.rap_constraints.round0 = 872ms`
- `prover.main_trace_commit = 853ms`
- `rs_code_matrix = 631ms`
- `prover.openings.whir = 311ms`
