use std::time::{SystemTime, UNIX_EPOCH};

use metrics_util::{
    debugging::{DebugValue, Snapshot},
    MetricKind,
};
use opentelemetry_proto::tonic::{
    collector::metrics::v1::ExportMetricsServiceRequest,
    common::v1::{any_value::Value as AnyValueValue, AnyValue, KeyValue},
    metrics::v1::{
        number_data_point::Value as NumberValue, Gauge, Metric, NumberDataPoint, ResourceMetrics,
        ScopeMetrics,
    },
    resource::v1::Resource,
};
use prost::Message;

/// Convert a DebuggingRecorder snapshot to an OTLP ExportMetricsServiceRequest.
fn snapshot_to_otlp(snapshot: Snapshot, run_id: &str) -> (ExportMetricsServiceRequest, usize) {
    let timestamp_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let mut metrics = Vec::new();

    for (ckey, _unit, _description, value) in snapshot.into_vec() {
        let (kind, key) = ckey.into_parts();
        let (key_name, labels) = key.into_parts();

        let numeric_value = match (&kind, &value) {
            (MetricKind::Gauge, DebugValue::Gauge(v)) => NumberValue::AsDouble(v.into_inner()),
            (MetricKind::Counter, DebugValue::Counter(v)) => NumberValue::AsInt(*v as i64),
            _ => continue,
        };

        let mut attributes: Vec<KeyValue> = labels
            .into_iter()
            .map(|label| {
                let (k, v) = label.into_parts();
                KeyValue {
                    key: k.as_ref().to_owned(),
                    value: Some(AnyValue {
                        value: Some(AnyValueValue::StringValue(v.as_ref().to_owned())),
                    }),
                }
            })
            .collect();

        // Add run_id as an attribute
        attributes.push(KeyValue {
            key: "run_id".to_string(),
            value: Some(AnyValue {
                value: Some(AnyValueValue::StringValue(run_id.to_string())),
            }),
        });

        let data_point = NumberDataPoint {
            attributes,
            time_unix_nano: timestamp_nanos,
            value: Some(numeric_value),
            ..Default::default()
        };

        let metric = match kind {
            MetricKind::Gauge => Metric {
                name: key_name.as_str().to_owned(),
                data: Some(opentelemetry_proto::tonic::metrics::v1::metric::Data::Gauge(
                    Gauge {
                        data_points: vec![data_point],
                    },
                )),
                ..Default::default()
            },
            MetricKind::Counter => Metric {
                name: key_name.as_str().to_owned(),
                data: Some(opentelemetry_proto::tonic::metrics::v1::metric::Data::Gauge(
                    Gauge {
                        data_points: vec![data_point],
                    },
                )),
                ..Default::default()
            },
            MetricKind::Histogram => continue,
        };

        metrics.push(metric);
    }

    let count = metrics.len();

    let request = ExportMetricsServiceRequest {
        resource_metrics: vec![ResourceMetrics {
            resource: Some(Resource {
                attributes: vec![KeyValue {
                    key: "service.name".to_string(),
                    value: Some(AnyValue {
                        value: Some(AnyValueValue::StringValue(
                            "openvm-benchmark".to_string(),
                        )),
                    }),
                }],
                ..Default::default()
            }),
            scope_metrics: vec![ScopeMetrics {
                metrics,
                ..Default::default()
            }],
            ..Default::default()
        }],
    };

    (request, count)
}

/// Upload a metric snapshot as OTLP protobuf to the given endpoint.
///
/// Returns the number of metrics uploaded on success.
pub fn upload_snapshot_as_otlp(
    snapshot: Snapshot,
    endpoint: &str,
    api_key: &str,
    run_id: &str,
) -> eyre::Result<usize> {
    let (otlp_request, count) = snapshot_to_otlp(snapshot, run_id);

    if count == 0 {
        return Ok(0);
    }

    let body = otlp_request.encode_to_vec();

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let url = format!("{}/v1/metrics", endpoint.trim_end_matches('/'));

    let mut request = client
        .post(&url)
        .header("Content-Type", "application/x-protobuf")
        .body(body);

    if !api_key.is_empty() {
        request = request.header("X-API-Key", api_key);
    }

    let resp = request.send()?;

    if resp.status().is_success() {
        Ok(count)
    } else {
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        eyre::bail!("OTLP upload failed: {} {}", status, text)
    }
}
