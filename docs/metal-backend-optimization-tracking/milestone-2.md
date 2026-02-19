# Milestone 2: Stacked Trace Copy Path Optimization

- Status: `not_started`
- Owner:
- Branch:
- PR:
- Started:
- Completed:
- Last Updated:

## Objective

Remove per-column blit synchronization in stacked trace assembly.

## Scope

- Files:
  - `crates/metal-backend/src/stacked_pcs.rs`
- Work:
  - Refactor `stack_traces_into_expanded` to batch blit copies.
  - Keep output layout and zero-fill semantics unchanged.
  - Preserve behavior for `s.log_height() < l_skip`.

## Exit Criteria

- No per-segment sync in copy path.
- `rs_code_matrix` span improves in profile.

## Validation Log

- `cargo check -p openvm-metal-backend`:
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`:
- keccakf trace run:

## Metrics

- Before `rs_code_matrix`:
- After `rs_code_matrix`:
- Delta:

## Notes

