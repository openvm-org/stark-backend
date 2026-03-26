use std::sync::Arc;

use tracing::{
    field::{Field, Visit},
    Id, Subscriber,
};
use tracing_subscriber::{registry::LookupSpan, Layer};

/// A tracing layer that maps tracing spans to NVTX ranges for Nsight Systems profiling.
///
/// When active, each span enter/exit maps to an NVTX range_push/range_pop, making
/// proving phases visible in `nsys profile --trace=cuda,nvtx` timelines.
///
/// Only spans at or above the configured level are exported.
pub struct NvtxLayer {
    config: NvtxConfig,
}

pub struct NvtxConfig {
    pub max_level: tracing::Level,
}

impl Default for NvtxConfig {
    fn default() -> Self {
        Self {
            max_level: tracing::Level::INFO,
        }
    }
}

impl NvtxLayer {
    pub fn new(config: NvtxConfig) -> Self {
        Self { config }
    }
}

/// Per-span cached NVTX label, stored in span extensions.
/// Only inserted for spans that pass the level filter.
struct NvtxSpanData {
    label: Arc<str>,
}

/// Visitor that collects all span fields into a compact label suffix.
struct FieldCollector {
    parts: Vec<String>,
}

impl FieldCollector {
    fn new() -> Self {
        Self { parts: Vec::new() }
    }

    fn into_suffix(self) -> Option<String> {
        if self.parts.is_empty() {
            None
        } else {
            Some(self.parts.join(","))
        }
    }
}

impl Visit for FieldCollector {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.parts.push(format!("{}={:?}", field.name(), value));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.parts.push(format!("{}={}", field.name(), value));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.parts.push(format!("{}={}", field.name(), value));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.parts.push(format!("{}={}", field.name(), value));
    }
}

fn build_label(meta: &tracing::Metadata<'_>, attrs: &tracing::span::Attributes<'_>) -> Arc<str> {
    let mut collector = FieldCollector::new();
    attrs.record(&mut collector);

    let name = meta.name();
    match collector.into_suffix() {
        Some(suffix) => Arc::from(format!("{name}{{{suffix}}}")),
        None => Arc::from(name),
    }
}

impl<S> Layer<S> for NvtxLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &Id,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let Some(span) = ctx.span(id) else { return };
        let meta = span.metadata();

        if meta.level() > &self.config.max_level {
            return;
        }

        let label = build_label(meta, attrs);
        span.extensions_mut().insert(NvtxSpanData { label });
    }

    fn on_record(
        &self,
        id: &Id,
        values: &tracing::span::Record<'_>,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let Some(span) = ctx.span(id) else { return };
        let mut exts = span.extensions_mut();
        let Some(data) = exts.get_mut::<NvtxSpanData>() else {
            return;
        };
        // Append newly recorded fields to the existing label.
        // If the label already has fields ("name{a=1}"), strip the closing brace and
        // append with a comma. If no fields yet ("name"), open a new brace group.
        let mut collector = FieldCollector::new();
        values.record(&mut collector);
        if let Some(suffix) = collector.into_suffix() {
            let base = data.label.trim_end_matches('}');
            let sep = if base.len() < data.label.len() {
                ","
            } else {
                "{"
            };
            data.label = Arc::from(format!("{base}{sep}{suffix}}}"));
        }
    }

    fn on_enter(&self, id: &Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let exts = span.extensions();
        let Some(data) = exts.get::<NvtxSpanData>() else {
            return;
        };
        nvtx::range_push!("{}", data.label);
    }

    fn on_exit(&self, id: &Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let exts = span.extensions();
        if exts.get::<NvtxSpanData>().is_some() {
            nvtx::range_pop!();
        }
    }
}

#[cfg(test)]
mod tests {
    use tracing::info_span;
    use tracing_subscriber::{layer::SubscriberExt, Registry};

    use super::*;

    #[test]
    fn test_nvtx_layer_with_fields() {
        let subscriber = Registry::default().with(NvtxLayer::new(NvtxConfig::default()));
        tracing::subscriber::with_default(subscriber, || {
            // Should produce "prove_segment{segment=42,phase=prover}"
            let span = info_span!("prove_segment", segment = 42, phase = "prover");
            let _guard = span.enter();
            // No fields -> just the name
            let inner = info_span!("stark_prove");
            let _inner_guard = inner.enter();
        });
    }

    #[test]
    fn test_nvtx_layer_late_bound_fields() {
        let subscriber = Registry::default().with(NvtxLayer::new(NvtxConfig::default()));
        tracing::subscriber::with_default(subscriber, || {
            // Late-bound field via span.record()
            let span = info_span!("step", idx = tracing::field::Empty);
            let _guard = span.enter();
            span.record("idx", 7);
            // Also test recording on a span that already has fields
            let span2 = info_span!("work", phase = "init", result = tracing::field::Empty);
            let _guard2 = span2.enter();
            span2.record("result", 42);
        });
    }
}
