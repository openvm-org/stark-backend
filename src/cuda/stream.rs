use std::{borrow::Cow, ffi::c_void};

use crate::cuda::error::{check, CudaError};

#[link(name = "cudart")]
extern "C" {
    fn cudaStreamSynchronize(stream: cudaStream_t) -> i32;
    fn cudaEventCreate(event: *mut cudaEvent_t) -> i32;
    fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> i32;
    fn cudaEventSynchronize(event: cudaEvent_t) -> i32;
    fn cudaEventDestroy(event: cudaEvent_t) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> i32;
}

#[allow(non_camel_case_types)]
pub type cudaStream_t = *mut c_void;
#[allow(non_camel_case_types)]
pub type cudaEvent_t = *mut c_void;
#[allow(non_upper_case_globals)]
pub const cudaStreamPerThread: cudaStream_t = 0x02 as cudaStream_t;

pub fn stream_sync() -> Result<(), CudaError> {
    check(unsafe { cudaStreamSynchronize(cudaStreamPerThread) })
}

pub struct CudaEvent {
    event: cudaEvent_t,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    pub fn new() -> Result<Self, CudaError> {
        let mut event: cudaEvent_t = std::ptr::null_mut();
        check(unsafe { cudaEventCreate(&mut event) })?;
        Ok(Self { event })
    }
    /// # Safety
    /// The caller must ensure that `stream` is a valid stream.
    pub unsafe fn record_and_wait(&self, stream: cudaStream_t) -> Result<(), CudaError> {
        check(cudaEventRecord(self.event, stream))?;
        check(cudaEventSynchronize(self.event))
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe { cudaEventDestroy(self.event) };
    }
}

/// A GPU-aware span that collects a gauge metric using CUDA events.
pub fn gpu_metrics_span<R, F: FnOnce() -> R>(
    name: impl Into<Cow<'static, str>>,
    f: F,
) -> Result<R, CudaError> {
    let start = CudaEvent::new()?;
    let stop = CudaEvent::new()?;
    unsafe {
        check(cudaEventRecord(start.event, cudaStreamPerThread))?;
    }
    let res = f();
    unsafe { stop.record_and_wait(cudaStreamPerThread)? };

    let mut elapsed_ms = 0f32;
    unsafe {
        check(cudaEventElapsedTime(
            &mut elapsed_ms,
            start.event,
            stop.event,
        ))?
    };

    metrics::gauge!(name.into()).set(elapsed_ms as f64);
    Ok(res)
}
