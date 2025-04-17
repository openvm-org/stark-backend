use std::{collections::BTreeMap, ffi::OsStr};

use metrics_exporter_prometheus::PrometheusBuilder;
use metrics_tracing_context::{MetricsLayer, TracingContextLayer};
use metrics_util::{
    debugging::{DebugValue, DebuggingRecorder, Snapshot},
    layers::Layer,
    CompositeKey, MetricKind,
};
use serde_json::json;
use tracing_forest::ForestLayer;
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};

/// Run a function with metric collection enabled. The metrics will be written to a file specified
/// by an environment variable which name is `output_path_envar`.
pub fn run_with_metric_collection<R>(
    output_path_envar: impl AsRef<OsStr>,
    f: impl FnOnce() -> R,
) -> R {
    let file = std::env::var(output_path_envar).map(|path| std::fs::File::create(path).unwrap());
    // Set up tracing:
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,p3_=warn"));
    // Plonky3 logging is more verbose, so we set default to debug.
    let subscriber = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .with(MetricsLayer::new());
    // Prepare tracing.
    tracing::subscriber::set_global_default(subscriber).unwrap();

    // Prepare metrics.
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let recorder = TracingContextLayer::all().layer(recorder);
    // Install the registry as the global recorder
    metrics::set_global_recorder(recorder).unwrap();
    let res = f();

    if let Ok(file) = file {
        serde_json::to_writer_pretty(&file, &serialize_metric_snapshot(snapshotter.snapshot()))
            .unwrap();
    }
    res
}

/// Run a function with metric exporter enabled. The metrics will be served on the port specified
/// by an environment variable which name is `metrics_port_envar`.
pub fn run_with_metric_exporter<R>(
    metrics_port_envar: impl AsRef<OsStr>,
    f: impl FnOnce() -> R,
) -> R {
    // Get the port from environment variable or use a default
    let metrics_port = std::env::var(metrics_port_envar)
        .map(|port| port.parse::<u16>().unwrap_or(9091))
        .unwrap();
    let endpoint = format!("http://127.0.0.1:{}/metrics/job/stark-sdk", metrics_port);

    // Clear metrics before pushing to the push gateway
    let status = std::process::Command::new("curl")
        .args(["-X", "DELETE", &endpoint])
        .status()
        .expect("Failed to clear metrics");
    if status.success() {
        println!("Metrics cleared successfully");
    }

    // Set up Prometheus recorder and exporter
    let builder = PrometheusBuilder::new()
        .with_push_gateway(endpoint, std::time::Duration::from_secs(5), None, None)
        .expect("Push gateway endpoint should be valid");

    let recorder = if let Ok(handle) = tokio::runtime::Handle::try_current() {
        let (recorder, exporter) = {
            let _g = handle.enter();
            builder.build().unwrap()
        };
        handle.spawn(exporter);
        recorder
    } else {
        let thread_name = "metrics-exporter-prometheus-push-gateway";
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let (recorder, exporter) = {
            let _g = runtime.enter();
            builder.build().unwrap()
        };
        std::thread::Builder::new()
            .name(thread_name.to_string())
            .spawn(move || runtime.block_on(exporter))
            .unwrap();
        recorder
    };

    // Set up tracing:
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,p3_=warn"));
    // Plonky3 logging is more verbose, so we set default to debug.
    let subscriber = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .with(MetricsLayer::new());
    // Prepare tracing.
    tracing::subscriber::set_global_default(subscriber).unwrap();

    // Prepare metrics
    let recorder = TracingContextLayer::all().layer(recorder);
    // Install the registry as the global recorder
    metrics::set_global_recorder(recorder).unwrap();

    // Run the actual function
    let res = f();
    std::thread::sleep(std::time::Duration::from_secs(10));
    println!(
        "Metrics available at http://127.0.0.1:{}/metrics/job/stark-sdk",
        metrics_port
    );
    res
}

/// Serialize a gauge/counter metric into a JSON object. The object has the following structure:
/// {
///    "metric": <Metric Name>,
///    "labels": [
///       (<key1>, <value1>),
///       (<key2>, <value2>),
///     ],
///    "value": <float value if gauge | integer value if counter>
/// }
///
fn serialize_metric(ckey: CompositeKey, value: DebugValue) -> serde_json::Value {
    let (_kind, key) = ckey.into_parts();
    let (key_name, labels) = key.into_parts();
    let value = match value {
        DebugValue::Gauge(v) => v.into_inner().to_string(),
        DebugValue::Counter(v) => v.to_string(),
        DebugValue::Histogram(_) => todo!("Histograms not supported yet."),
    };
    let labels = labels
        .into_iter()
        .map(|label| {
            let (k, v) = label.into_parts();
            (k.as_ref().to_owned(), v.as_ref().to_owned())
        })
        .collect::<Vec<_>>();

    json!({
        "metric": key_name.as_str(),
        "labels": labels,
        "value": value,
    })
}

/// Serialize a metric snapshot into a JSON object. The object has the following structure:
/// {
///   "gauge": [
///     {
///         "metric": <Metric Name>,
///         "labels": [
///             (<key1>, <value1>),
///             (<key2>, <value2>),
///         ],
///         "value": <float value>
///     },
///     ...
///   ],
///   ...
/// }
///
pub fn serialize_metric_snapshot(snapshot: Snapshot) -> serde_json::Value {
    let mut ret = BTreeMap::<_, Vec<serde_json::Value>>::new();
    for (ckey, _, _, value) in snapshot.into_vec() {
        match ckey.kind() {
            MetricKind::Gauge => {
                ret.entry("gauge")
                    .or_default()
                    .push(serialize_metric(ckey, value));
            }
            MetricKind::Counter => {
                ret.entry("counter")
                    .or_default()
                    .push(serialize_metric(ckey, value));
            }
            MetricKind::Histogram => todo!(),
        }
    }
    json!(ret)
}
