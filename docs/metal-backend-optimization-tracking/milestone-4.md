# Milestone 4: Trace Transport Layout Simplification

- Status: `not_started`
- Owner:
- Branch:
- PR:
- Started:
- Completed:
- Last Updated:

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

- `cargo check -p openvm-metal-backend`:
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`:
- keccakf trace run:

## Metrics

- Before:
- After:
- Delta:

## Notes

