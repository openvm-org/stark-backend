# Milestone 4: Trace Transport Layout Simplification

- Status: `completed`
- Owner: codex
- Branch: `perf/m5`
- PR:
- Started: 2026-02-20
- Completed: 2026-02-20
- Last Updated: 2026-02-20

## Objective

Eliminate strided-upload then collapse sequence in trace transport.

## Scope

- Files:
  - `crates/metal-backend/src/data_transporter.rs`
  - `crates/metal-backend/src/metal/matrix.rs`
- Work:
  - Remove double-buffered strided upload path in `transport_and_unstack_single_data_h2d`.
  - Produce contiguous target layout directly where feasible.
  - Keep compatibility for `stride == 1` and `stride > 1`.

## Exit Criteria

- One less allocation and one less kernel launch in transport path.
- Existing transport invariants hold.

## Validation Log

- `cargo check -p openvm-metal-backend`: pass (existing warnings in `mle_round.rs` and `stacked_reduction.rs`)
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`: pass (`0` tests, `0` failed)
- `cargo test -p openvm-metal-backend collapse_strided_trace`: pass (`2` tests, `0` failed)
- `cargo run -p openvm-metal-backend --example keccakf`: pass
- `RUST_LOG=openvm_metal_backend::data_transporter=debug cargo run -p openvm-metal-backend --example keccakf`: pass; transporter debug fields emitted, but `transport_and_unstack_single_data_h2d` was not exercised by this benchmark input (no preprocessed trace transport on this path)

## Metrics

- Before:
  - Strided trace transport (`stride > 1`) allocated two device buffers (`strided_trace`, `trace_buffer`) and launched one `collapse_strided_matrix` kernel with one sync boundary.
  - Baseline trace (from Milestone 0): `stark_prove_excluding_trace=24.9s`, `rs_code_matrix=950ms`, `main_trace_commit.merkle_tree=233ms`, `rap_constraints.round0=6.79s`, `rap_constraints.mle_rounds=414ms`, `openings.whir.merkle_tree=48.8ms/26.5ms/14.5ms`.
- After:
  - Strided trace transport allocates only the final contiguous `trace_buffer`; no collapse kernel dispatch, no additional sync point.
  - Post-change trace: `stark_prove_excluding_trace=25.2s`, `rs_code_matrix=687ms`, `main_trace_commit.merkle_tree=237ms`, `rap_constraints.round0=6.98s`, `rap_constraints.mle_rounds=462ms`, `openings.whir.merkle_tree=64.7ms/34.7ms/15.1ms`.
- Delta:
  - Per strided transport call: `-1` device allocation, `-1` kernel dispatch, `-1` sync point.
  - Keccakf top-level spans did not show a focused improvement signal for this milestone path because the benchmark run does not materially exercise preprocessed trace transport.

## Notes

- `transport_and_unstack_single_data_h2d` now collapses strided columns directly into the destination shared buffer during upload, preserving the exact contiguous column-major layout previously produced by `collapse_strided_matrix`.
- Added unit coverage in `data_transporter.rs` to validate stride-1 identity and stride-2 column-major collapse ordering.
