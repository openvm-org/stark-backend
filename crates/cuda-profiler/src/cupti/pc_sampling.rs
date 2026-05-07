//! CUPTI PC Sampling Continuous (P3).
//!
//! TODO: bind cuptiPCSamplingEnable / cuptiPCSamplingGetData and stream stall
//! reasons. The plan calls this out as ~1 week of work; the activity sidecar
//! gives us the much simpler "what kernel ran when" view first.
//!
//! For now this is a stub that always errors out, so callers fall through to
//! a profiler without PC sampling.

use std::sync::Arc;

use parking_lot::Mutex;

use super::{sys::Cupti, CuptiError};
use crate::log_writer::LogWriter;

pub struct PcSamplingBackend;

impl PcSamplingBackend {
    pub fn start(_cupti: Arc<Cupti>, _writer: Arc<Mutex<LogWriter>>) -> Result<Self, CuptiError> {
        Err(CuptiError::Call {
            func: "PcSamplingBackend::start",
            code: -1,
        })
    }

    pub fn stop(self) {}
}
