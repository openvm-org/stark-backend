use std::{borrow::Cow, ffi::c_void};

use crate::cuda::error::{check, CudaError};

#[link(name = "cudart")]
extern "C" {
    fn cudaStreamCreate(stream: *mut cudaStream_t) -> i32;
    fn cudaStreamDestroy(stream: cudaStream_t) -> i32;
    fn cudaStreamSynchronize(stream: cudaStream_t) -> i32;
    fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: u32) -> i32;
    fn cudaEventCreate(event: *mut cudaEvent_t) -> i32;
    fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> i32;
    fn cudaEventSynchronize(event: cudaEvent_t) -> i32;
    fn cudaEventDestroy(event: cudaEvent_t) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> i32;
}

#[allow(non_camel_case_types)]
pub type cudaStream_t = *mut c_void;

pub struct CudaStream {
    stream: cudaStream_t,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// Creates a new non-blocking CUDA stream.
    pub fn new() -> Result<Self, CudaError> {
        let mut stream: cudaStream_t = std::ptr::null_mut();
        check(unsafe { cudaStreamCreate(&mut stream) })?;
        Ok(Self { stream })
    }

    /// Get the raw CUDA stream handle.
    #[inline]
    pub fn as_raw(&self) -> cudaStream_t {
        self.stream
    }

    /// Synchronize this stream.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        check(unsafe { cudaStreamSynchronize(self.stream) })
    }

    /// Wait for the given event.
    pub fn wait(&self, event: &CudaEvent) -> Result<(), CudaError> {
        check(unsafe { cudaStreamWaitEvent(self.stream, event.event, 0) })
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            self.synchronize().unwrap();
            let _ = unsafe { cudaStreamDestroy(self.stream) };
            self.stream = std::ptr::null_mut();
        }
    }
}

#[allow(non_camel_case_types)]
pub type cudaEvent_t = *mut c_void;
#[allow(non_upper_case_globals)]
pub const cudaStreamPerThread: cudaStream_t = 0x02 as cudaStream_t;

pub fn default_stream_sync() -> Result<(), CudaError> {
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
    pub unsafe fn record(&self, stream: cudaStream_t) -> Result<(), CudaError> {
        check(cudaEventRecord(self.event, stream))
    }

    /// # Safety
    /// The caller must ensure that `stream` is a valid stream.
    pub unsafe fn record_and_wait(&self, stream: cudaStream_t) -> Result<(), CudaError> {
        self.record(stream)?;
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
