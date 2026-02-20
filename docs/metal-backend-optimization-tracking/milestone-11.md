# Milestone 11: Main Trace Commit and RS Pipeline Optimization

- Status: `not_started`
- Owner:
- Branch:
- PR:
- Started:
- Completed:
- Last Updated: 2026-02-20

## Objective

Lower `prover.main_trace_commit` runtime by reducing overhead in stacked commit and RS pipeline setup.

## Scope

- Optimize `crates/metal-backend/src/stacked_pcs.rs` copy/layout flow.
- Reduce temporary buffer churn and command-buffer breaks.
- Improve commit-path `merkle_tree` dataflow in `crates/metal-backend/src/merkle_tree.rs` where possible.

## Exit Criteria

- `keccakf` `prover.main_trace_commit` improves by at least 20% vs Milestone 8 median.
- `rs_code_matrix` and commit-path `merkle_tree` spans show measurable reductions.

## Validation Log

- Pending.

## Metrics

- Pending.

## Notes

- Keep stack layout and padding semantics unchanged.
