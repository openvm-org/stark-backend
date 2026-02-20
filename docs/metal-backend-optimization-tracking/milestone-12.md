# Milestone 12: Trace Transport and Layout Path Optimization

- Status: `not_started`
- Owner:
- Branch:
- PR:
- Started:
- Completed:
- Last Updated: 2026-02-20

## Objective

Reduce transport overhead by removing remaining host-side layout work and temporary allocations.

## Scope

- Optimize `crates/metal-backend/src/data_transporter.rs` transport paths.
- Optimize layout/gather behavior in `crates/metal-backend/src/metal/matrix.rs`.
- Keep compatibility for both `stride == 1` and `stride > 1` paths.

## Exit Criteria

- Dispatch count and allocation count drop in transport hot paths.
- No regressions in `keccakf` and LogUp benchmark correctness.

## Validation Log

- Pending.

## Metrics

- Pending.

## Notes

- Coordinate with Milestone 14 on any shared allocation infrastructure changes.
