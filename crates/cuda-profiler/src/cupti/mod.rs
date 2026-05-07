//! CUPTI sidecar.
//!
//! CUPTI is dynamically loaded with `dlopen("libcupti.so")` so a binary built
//! with the profiler feature still runs on a machine without CUPTI installed.
//! When the library is missing, `start()` returns an error and the rest of
//! the profiler keeps working (per-CTA telemetry is independent of CUPTI).
//!
//! The sidecar streams three things into the shared binary log:
//!
//! - **Activity API** — kernels, memcpys, NVTX ranges. (P1, this module.)
//! - **PC Sampling Continuous** — stall reasons. (P3, `pc_sampling.rs`.)
//! - **Range Profiler** — `sm__throughput` / `dram__throughput` / local-memory counters per NVTX
//!   range, no kernel replay. (P4, `range_profiler.rs`.)

pub mod activity;
pub mod pc_sampling;
pub mod range_profiler;
pub mod sys;

use std::sync::Arc;

use parking_lot::Mutex;
use thiserror::Error;

use crate::log_writer::LogWriter;

#[derive(Debug, Error)]
pub enum CuptiError {
    #[error("libcupti.so not found: {0}")]
    LibraryNotFound(String),
    #[error("symbol {sym} missing from libcupti.so")]
    SymbolMissing { sym: &'static str },
    #[error("CUPTI call {func} returned {code}")]
    Call { func: &'static str, code: i32 },
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Owns the CUPTI subscription/buffers for the lifetime of the profiler.
pub struct CuptiSidecar {
    activity: Option<activity::ActivityBackend>,
    pc_sampling: Option<pc_sampling::PcSamplingBackend>,
    range_profiler: Option<range_profiler::RangeProfilerBackend>,
}

impl CuptiSidecar {
    /// Start the sidecar. Activity API is always started if the env var
    /// allows; PC sampling and Range Profiler are off by default and gated by
    /// their own env vars (`SHADOW_PROFILER_PC_SAMPLING` and
    /// `SHADOW_PROFILER_RANGE_METRICS`).
    pub fn start(writer: Arc<Mutex<LogWriter>>) -> Result<Self, CuptiError> {
        // Load the CUPTI dynamic library once. If it can't be found, surface
        // the error so the caller can disable the sidecar without crashing.
        let lib = sys::Cupti::load()?;
        let lib = Arc::new(lib);

        let activity = match activity::ActivityBackend::start(lib.clone(), writer.clone()) {
            Ok(b) => Some(b),
            Err(e) => {
                tracing::warn!("cuda-profiler: CUPTI activity init failed: {e}");
                None
            }
        };

        let pc_sampling_enabled = std::env::var("SHADOW_PROFILER_PC_SAMPLING")
            .map(|v| v == "1")
            .unwrap_or(false);
        let pc_sampling = if pc_sampling_enabled {
            match pc_sampling::PcSamplingBackend::start(lib.clone(), writer.clone()) {
                Ok(b) => Some(b),
                Err(e) => {
                    tracing::warn!("cuda-profiler: CUPTI PC sampling init failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        let range_metrics_enabled = std::env::var("SHADOW_PROFILER_RANGE_METRICS")
            .map(|v| v == "1")
            .unwrap_or(false);
        let range_profiler = if range_metrics_enabled {
            match range_profiler::RangeProfilerBackend::start(lib.clone(), writer) {
                Ok(b) => Some(b),
                Err(e) => {
                    tracing::warn!("cuda-profiler: CUPTI Range Profiler init failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            activity,
            pc_sampling,
            range_profiler,
        })
    }

    /// Stop and flush. Idempotent in spirit; consumes the sidecar.
    pub fn stop(self) {
        if let Some(b) = self.range_profiler {
            b.stop();
        }
        if let Some(b) = self.pc_sampling {
            b.stop();
        }
        if let Some(b) = self.activity {
            b.stop();
        }
    }
}
