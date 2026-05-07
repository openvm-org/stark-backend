//! CUPTI Range Profiler API (P4).
//!
//! TODO: bind cuptiProfilerInitialize + cuptiProfilerBeginPass and pull the
//! metric set from the plan:
//!   - sm__throughput.avg.pct_of_peak_sustained_elapsed
//!   - dram__throughput.avg.pct_of_peak_sustained_elapsed
//!   - lts__t_sectors_aperture_device.sum
//!   - smsp__inst_executed.sum
//!   - l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum
//!   - l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum
//!
//! Stub for now; activation requires CUPTI 12.5+ and a writable
//! /dev/nvidia-counter-profile-control on the host.

use std::sync::Arc;

use parking_lot::Mutex;

use super::{sys::Cupti, CuptiError};
use crate::log_writer::LogWriter;

pub struct RangeProfilerBackend;

impl RangeProfilerBackend {
    pub fn start(_cupti: Arc<Cupti>, _writer: Arc<Mutex<LogWriter>>) -> Result<Self, CuptiError> {
        Err(CuptiError::Call {
            func: "RangeProfilerBackend::start",
            code: -1,
        })
    }

    pub fn stop(self) {}
}
