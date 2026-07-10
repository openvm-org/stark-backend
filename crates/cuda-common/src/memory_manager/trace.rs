//! Optional allocation-event tracing for offline memory-planning analysis.
//!
//! When the `VPMM_TRACE` environment variable is set to a file path, every
//! `d_malloc`/`d_free` through the global memory manager appends one CSV line,
//! and every [`MemTracker`](super::MemTracker) scope appends phase markers:
//!
//! ```text
//! A,<ptr hex>,<requested size bytes>,<stream key>,<micros since start>,<call duration ns>
//! F,<ptr hex>,,<stream key>,<micros since start>,<call duration ns>
//! M,<start|end>:<label>,,,<micros since start>,
//! ```
//!
//! Sizes are the *requested* sizes (before page rounding). The call duration
//! covers lock acquisition plus the allocator work, i.e. what the calling
//! thread actually paid. Records are written under the memory-manager mutex,
//! so line order is the global allocation order. Traces are replayed against
//! the static allocator's packing pass (`benches/trace_replay.rs`,
//! `benches/trace_phases.rs`).

use std::{
    fs::File,
    io::{BufWriter, Write},
    time::Instant,
};

pub(super) struct AllocTracer {
    out: BufWriter<File>,
    start: Instant,
    /// Flush every N records so a crash or (ctor-static) missing drop loses
    /// at most one batch.
    pending: u32,
}

const FLUSH_EVERY: u32 = 64;

impl AllocTracer {
    /// Creates a tracer if `VPMM_TRACE` is set. Panics if the file cannot be
    /// created — a silently missing trace is worse than a loud failure.
    pub(super) fn from_env() -> Option<Self> {
        let path = std::env::var("VPMM_TRACE").ok()?;
        let file =
            File::create(&path).unwrap_or_else(|e| panic!("VPMM_TRACE: cannot create {path}: {e}"));
        tracing::info!("VPMM_TRACE: recording allocation trace to {path}");
        Some(Self {
            out: BufWriter::new(file),
            start: Instant::now(),
            pending: 0,
        })
    }

    pub(super) fn record_alloc(
        &mut self,
        ptr: *mut std::ffi::c_void,
        size: usize,
        stream: u64,
        dur_ns: u128,
    ) {
        let t_us = self.start.elapsed().as_micros();
        self.write_line(format_args!(
            "A,{:#x},{size},{stream},{t_us},{dur_ns}",
            ptr as usize
        ));
    }

    pub(super) fn record_free(&mut self, ptr: *mut std::ffi::c_void, stream: u64, dur_ns: u128) {
        let t_us = self.start.elapsed().as_micros();
        self.write_line(format_args!(
            "F,{:#x},,{stream},{t_us},{dur_ns}",
            ptr as usize
        ));
    }

    /// Phase marker; `edge` is "start" or "end".
    pub(super) fn record_marker(&mut self, edge: &str, label: &str) {
        let t_us = self.start.elapsed().as_micros();
        // Commas in labels would corrupt the CSV.
        let label = label.replace(',', ";");
        self.write_line(format_args!("M,{edge}:{label},,,{t_us},"));
    }

    fn write_line(&mut self, args: std::fmt::Arguments<'_>) {
        if writeln!(self.out, "{args}").is_err() {
            return;
        }
        self.pending += 1;
        if self.pending >= FLUSH_EVERY {
            let _ = self.out.flush();
            self.pending = 0;
        }
    }
}

impl Drop for AllocTracer {
    fn drop(&mut self) {
        let _ = self.out.flush();
    }
}
