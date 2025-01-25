# Metrics

We use the [`metrics`](https://docs.rs/metrics/latest/metrics/) crate to collect metrics for the STARK prover. We refer to [reth docs](https://github.com/paradigmxyz/reth/blob/main/docs/design/metrics.md) for more guidelines on how to use metrics.

Each invocation of `MultiTraceStarkProver::prove` will collect the following metrics, all as `Gauge`s. We use gauge instead of histogram because these metrics are not frequently sampled and we care about the exact value. Any application that uses this backend is responsible for adding additional namespace labels if they wish to distinguish between different proof invocations.

- `stark_prove_excluding_trace_time_ms`: The total elapsed time in milliseconds of `prove`. This excludes the main trace generation because that is not done by `stark-backend`.

The following metrics comprise the main breakdown of the components of `prove`. They are disjoint and _expected_ to sum up to almost `stark_prove_excluding_trace_time_ms` (if it does not, an issue should be opened as it means there is an unexpected source of slowdown).

- `main_trace_commit_time_ms`: The time to commit the main trace matrices, depending on the PCS.
- `generate_perm_trace_time_ms`: When FRI is used for the log up argument, this is the time to generate the permutation trace.
- `perm_trace_commit_time_ms`: When FRI is used for the log up argument, this is the time to commit the permutation trace.
- `quotient_poly_compute_time_ms`: The time to compute the quotient polynomials from the trace matrices according to AIR constraints.
- `quotient_poly_commit_time_ms`: The time to commit the quotient polynomials.
- `pcs_opening_time_ms`: The time to compute all polynomial commitment scheme (PCS) opening proofs necessary for the proof. Currently the PCS is FRI over a base field with high `2`-adicity.
