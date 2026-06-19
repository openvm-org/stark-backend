# Metrics

We use the [`metrics`](https://docs.rs/metrics/latest/metrics/) crate to collect metrics for the STARK prover. We
refer to [reth docs](https://github.com/paradigmxyz/reth/blob/main/docs/design/metrics.md) for more guidelines on how
to use metrics.

Timing metrics are collected by using a custom tracing layer
[`TimingMetricsLayer`](../crates/stark-sdk/src/metrics_tracing.rs). This layer emits a gauge metric named
`${name}_time_ms` for the elapsed time of each `INFO` or higher tracing span named `${name}`, in milliseconds. String
span fields are emitted as metric labels; for prover spans this commonly includes `phase = "prover"`.

Each invocation of [`Coordinator::prove`](../crates/stark-backend/src/prover/mod.rs), the implementation of the
[`Prover`](../crates/stark-backend/src/prover/mod.rs) trait, collects timing gauges. We use gauges instead of
histograms because these metrics are not frequently sampled and we care about the exact value. Any application that uses
this backend is responsible for adding additional namespace labels if it needs to distinguish proof invocations.

- `stark_prove_excluding_trace_time_ms`: The total elapsed time in milliseconds of `prove`. This excludes the main trace generation because that is not done by `stark-backend`.

## Prover Breakdown

The prover uses the stacked PCS and WHIR. LogUp is proved through the GKR fractional sumcheck and the batched
zerocheck.

The commonly useful prover metrics are:

- `prover.main_trace_commit_time_ms`: Time to commit the common main trace matrices with the stacked PCS.
- `prove_zerocheck_and_logup_time_ms` or `prove_zerocheck_and_logup_gpu_time_ms`: combined LogUp GKR and batched zerocheck proof.
- `fractional_sumcheck_time_ms` and `prover.rap_constraints.logup_gkr_time_ms`: LogUp GKR fractional sumcheck spans.
- `prover.batch_constraints.mle_rounds_time_ms` and `prover.rap_constraints.mle_rounds_time_ms`: batched constraint sumcheck MLE rounds.
- `prove_stacked_opening_reduction_time_ms` and `prover.openings.stacked_reduction_time_ms`: stacked opening reduction.
- `prove_whir_time_ms`, `prove_whir_opening_cpu_time_ms`, and `prover.openings.whir_time_ms`: WHIR opening proof generation.

Backend implementations also expose nested spans such as `stacked_round0_time_ms`,
`stacked_fold_mle_time_ms`, `whir_sumcheck_time_ms`, `whir_dft_merkle_time_ms`, and
`whir_mle_conversion_time_ms`.

These spans are nested and backend-dependent, so the metrics listed above are not disjoint and should not be summed as a
flat decomposition of `stark_prove_excluding_trace_time_ms`. For example, `trace_commit_cpu_time_ms` or
`prover.commit_time_ms` is nested inside `prover.main_trace_commit_time_ms`, and WHIR subspans are nested inside the
backend's opening stage.
