# Milestone 5: LogUp Round0 Scratch Reuse and Reduction Batching

- Status: `not_started`
- Owner:
- Branch:
- PR:
- Started:
- Completed:
- Last Updated:

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

- `cargo check -p openvm-metal-backend`:
- `cargo test -p openvm-metal-backend --test ntt_roundtrip`:
- keccakf trace run:

## Metrics

- Before:
- After:
- Delta:

## Notes

