use std::{sync::Arc, time::Instant};

use dashmap::DashMap;
use tracing::{
    field::{Field, Visit},
    Id, Subscriber,
};
use tracing_subscriber::{registry::LookupSpan, Layer};

/// A tracing layer that automatically emits metric gauges for all span durations.
/// This replaces the need for manual metrics_span calls by leveraging the tracing infrastructure.
#[derive(Clone, Default)]
pub struct TimingMetricsLayer {
    /// Store span timings indexed by span ID
    span_timings: Arc<DashMap<Id, SpanTiming>>,
}

#[derive(Debug)]
struct SpanTiming {
    name: String,
    start_time: Instant,
    labels: Vec<(String, String)>,
}

/// A visitor to extract the return value from span events
struct ReturnValueVisitor {
    has_return: bool,
}

impl Visit for ReturnValueVisitor {
    fn record_debug(&mut self, field: &Field, _value: &dyn std::fmt::Debug) {
        if field.name() == "return" {
            self.has_return = true;
        }
    }

    fn record_i64(&mut self, _field: &Field, _value: i64) {}
    fn record_u64(&mut self, _field: &Field, _value: u64) {}
    fn record_bool(&mut self, _field: &Field, _value: bool) {}
    fn record_str(&mut self, _field: &Field, _value: &str) {}
}

/// A visitor to extract all string fields from span attributes as metric labels
#[derive(Default)]
struct LabelVisitor {
    labels: Vec<(String, String)>,
}

impl Visit for LabelVisitor {
    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
    fn record_i64(&mut self, _field: &Field, _value: i64) {}
    fn record_u64(&mut self, _field: &Field, _value: u64) {}
    fn record_bool(&mut self, _field: &Field, _value: bool) {}
    fn record_str(&mut self, field: &Field, value: &str) {
        self.labels
            .push((field.name().to_string(), value.to_string()));
    }
}

impl TimingMetricsLayer {
    /// Create a new TimingMetricsLayer
    pub fn new() -> Self {
        Self::default()
    }

    fn emit_metric(name: &str, duration_ms: f64, labels: &[(String, String)]) {
        let metric_name = format!("{}_time_ms", name);
        let labels: Vec<metrics::Label> = labels
            .iter()
            .map(|(k, v)| metrics::Label::new(k.clone(), v.clone()))
            .collect();
        metrics::gauge!(metric_name, labels).set(duration_ms);
    }
}

impl<S> Layer<S> for TimingMetricsLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &Id,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if let Some(span) = ctx.span(id) {
            let metadata = span.metadata();
            let name = metadata.name();

            // Only track spans at INFO level or higher to match metrics_span behavior
            if metadata.level() <= &tracing::Level::INFO {
                // Start with labels inherited from parent span
                let mut labels = if let Some(parent) = span.parent() {
                    if let Some((_, parent_timing)) = self.span_timings.get(&parent.id()) {
                        parent_timing.labels.clone()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                // Extract all string fields from span attributes as labels
                // These will override any inherited labels with the same key
                let mut label_visitor = LabelVisitor::default();
                attrs.record(&mut label_visitor);

                // Merge: own labels override inherited ones
                for (key, value) in label_visitor.labels {
                    if let Some(pos) = labels.iter().position(|(k, _)| k == &key) {
                        labels[pos] = (key, value);
                    } else {
                        labels.push((key, value));
                    }
                }

                self.span_timings.insert(
                    id.clone(),
                    SpanTiming {
                        name: name.to_string(),
                        start_time: Instant::now(),
                        labels,
                    },
                );
            }
        }
    }

    fn on_event(&self, event: &tracing::Event<'_>, ctx: tracing_subscriber::layer::Context<'_, S>) {
        // Check if this is a return event in an instrumented function
        let mut visitor = ReturnValueVisitor { has_return: false };
        event.record(&mut visitor);

        if visitor.has_return {
            // Get the current span
            if let Some(span) = ctx.event_span(event) {
                let span_id = span.id();

                // Emit metric for the span that's returning
                if let Some((_, timing)) = self.span_timings.remove(&span_id) {
                    let duration_ms = timing.start_time.elapsed().as_millis() as f64;
                    Self::emit_metric(&timing.name, duration_ms, &timing.labels);
                }
            }
        }
    }

    fn on_close(&self, id: Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        // Clean up any spans that weren't emitted via return events
        // This handles spans that don't have instrumented return values
        if let Some((_, timing)) = self.span_timings.remove(&id) {
            let duration_ms = timing.start_time.elapsed().as_millis() as f64;
            Self::emit_metric(&timing.name, duration_ms, &timing.labels);
        }
    }
}

#[cfg(test)]
mod tests {
    use tracing::instrument;
    use tracing_subscriber::{layer::SubscriberExt, Registry};

    use super::*;

    #[instrument(level = "info")]
    fn example_function() -> i32 {
        std::thread::sleep(std::time::Duration::from_millis(10));
        42
    }

    #[test]
    fn test_metrics_layer() {
        let subscriber = Registry::default().with(TimingMetricsLayer::new());

        tracing::subscriber::with_default(subscriber, || {
            let result = example_function();
            assert_eq!(result, 42);
        });
    }
}
