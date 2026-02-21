# How Metrics Work

## Architecture Overview

There are **two layers** of metrics in this codebase:

### 1. The `metrics` crate (global metric recording)

The repo uses the [`metrics`](https://docs.rs/metrics) crate ecosystem as a global metric facade. Code throughout the backend emits metrics via macros like `counter!()` and `gauge!()`:

- **`crates/stark-backend/src/keygen/mod.rs`** — emits `constraint_deg`, `constraints`, and `interactions` counters during keygen.
- **`crates/stark-backend/src/prover/metrics.rs`** — `TraceMetrics::emit()` and `SingleTraceMetrics::emit()` emit counters for `rows`, `cells`, `total_cells`, `prep_cols`, `main_cols`, `perm_cols`, etc., all gated behind `#[cfg(feature = "metrics")]`.

### 2. `TimingMetricsLayer` (automatic span timing)

`crates/stark-sdk/src/metrics_tracing.rs` defines `TimingMetricsLayer`, a custom `tracing` subscriber layer. It **automatically converts tracing spans into timing metrics**:

- On `on_new_span`: records the start time and any string attributes as labels for `INFO`-level or higher spans.
- On `on_close` (or `on_event` with a `return` field): computes `elapsed` time and emits a `gauge!("{span_name}_time_ms", labels)`.

This means any `#[instrument(level = "info")]` function in the prover automatically gets a duration gauge without explicit metric code.

---

## How `run_with_metric_collection` works

Defined at `crates/stark-sdk/src/bench/mod.rs`, it's the main harness for running benchmarks with metrics. Step by step:

### 1. Output file setup

```rust
let file = std::env::var(&output_path_envar).map(|path| std::fs::File::create(path).unwrap());
```

Reads an env var (e.g. `METRIC_OUTPUT`) to get a file path for JSON output. If unset, no file is written.

### 2. Tracing subscriber setup

Builds a layered tracing subscriber:

- **`EnvFilter`** — respects `RUST_LOG`, defaults to `info,p3_=warn` (quiets Plonky3).
- **`ForestLayer`** — pretty-prints tracing spans as a tree in the terminal.
- **`MetricsLayer`** — bridges tracing span context into the `metrics` crate (so metrics get labeled with span context).
- **`TimingMetricsLayer`** (if `metrics` feature) — auto-emits `{span_name}_time_ms` gauges for all info-level spans.

### 3. Metrics recorder setup

```rust
let recorder = DebuggingRecorder::new();
let snapshotter = recorder.snapshotter();
let recorder = TracingContextLayer::all().layer(recorder);
metrics::set_global_recorder(recorder).unwrap();
```

- **`DebuggingRecorder`** — an in-memory recorder from `metrics-util` that captures all metrics into a snapshot-able buffer.
- **`TracingContextLayer`** — decorates the recorder so metrics inherit labels from the active tracing span context.
- The `snapshotter` handle is kept to read all recorded metrics later.

### 4. Run the actual work

```rust
let res = f();
```

All `counter!()`, `gauge!()` calls made during `f()` are captured by the `DebuggingRecorder`.

### 5. OTLP upload (gated by `metrics-upload` feature)

If `METRICS_ENDPOINT` env var is set:

- Takes a snapshot of all recorded metrics.
- Converts them to OTLP protobuf format (`otlp_upload.rs`) — each metric becomes an OTLP `Gauge` data point with labels as attributes and a `run_id` attribute.
- POSTs the protobuf to `{endpoint}/v1/metrics` via HTTP.
- Falls back gracefully if upload fails.

### 6. JSON file output

If the output file env var was set, serializes the snapshot as JSON with structure:

```json
{
  "gauge": [{"metric": "...", "labels": [...], "value": "..."}],
  "counter": [{"metric": "...", "labels": [...], "value": "..."}]
}
```

### 7. Return result

---

## Data flow summary

```
Code in prover/keygen          TimingMetricsLayer
  | counter!(), gauge!()          | gauge!("{span}_time_ms")
  +---------------+---------------+
                  |
                  v
       TracingContextLayer  (adds span labels)
                  |
                  v
          DebuggingRecorder  (in-memory buffer)
                  |
             snapshotter.snapshot()
                +-+-+
                |   |
                v   v
         JSON file  OTLP upload
```

The `metrics` feature gates all emission code, and `metrics-upload` additionally gates the OTLP upload path.
