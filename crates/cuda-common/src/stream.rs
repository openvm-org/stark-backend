use std::{
    borrow::Cow,
    ffi::c_void,
    ops::Deref,
    sync::{Arc, Mutex},
};

use crate::error::{check, CudaError};

#[link(name = "cudart")]
extern "C" {
    fn cudaDeviceSynchronize() -> i32;
    fn cudaStreamCreateWithFlags(stream: *mut cudaStream_t, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: cudaStream_t) -> i32;
    fn cudaStreamSynchronize(stream: cudaStream_t) -> i32;
    fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: u32) -> i32;
    fn cudaEventCreate(event: *mut cudaEvent_t) -> i32;
    fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> i32;
    fn cudaEventSynchronize(event: cudaEvent_t) -> i32;
    fn cudaEventQuery(event: cudaEvent_t) -> i32;
    fn cudaEventDestroy(event: cudaEvent_t) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> i32;
}

pub fn device_synchronize() -> Result<(), CudaError> {
    check(unsafe { cudaDeviceSynchronize() })
}

#[allow(non_camel_case_types)]
pub type cudaStream_t = *mut c_void;

pub struct CudaStream {
    stream: cudaStream_t,
    host_event: Mutex<CudaEvent>,
}

impl std::fmt::Debug for CudaStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaStream")
            .field("stream", &self.stream)
            .finish()
    }
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

/// `cudaStreamNonBlocking` flag: no implicit synchronization with the legacy
/// default stream (stream 0).
const CUDA_STREAM_NON_BLOCKING: u32 = 0x1;

impl CudaStream {
    /// Creates a new non-blocking CUDA stream using `cudaStreamCreateWithFlags`
    /// with `cudaStreamNonBlocking`. Non-blocking streams have no implicit
    /// synchronization with the legacy default stream (stream 0).
    pub fn new_non_blocking() -> Result<Self, CudaError> {
        let mut stream: cudaStream_t = std::ptr::null_mut();
        check(unsafe { cudaStreamCreateWithFlags(&mut stream, CUDA_STREAM_NON_BLOCKING) })?;
        let host_event = Mutex::new(CudaEvent::new()?);
        Ok(Self { stream, host_event })
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

    /// Record a per-stream event and synchronize to complete all pending D2H copies.
    /// Uses event-based sync rather than cudaStreamSynchronize because this waits
    /// only for work up to the event point, allowing future selective sync patterns
    /// (e.g., wait for a specific copy without draining the entire stream).
    pub fn to_host_sync(&self) -> Result<(), CudaError> {
        let event = self.host_event.lock().unwrap();
        unsafe { event.record(self.stream) }?;
        event.synchronize()
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            // Non-blocking: CUDA defers destruction until the stream is idle
            let err = unsafe { cudaStreamDestroy(self.stream) };
            debug_assert_eq!(err, 0, "cudaStreamDestroy failed with error code: {err}");
            self.stream = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// StreamGuard — keeps a CudaStream alive for allocation records
// ---------------------------------------------------------------------------

/// Keeps a `CudaStream` alive for the lifetime of an allocation record.
#[derive(Clone, Debug)]
pub struct StreamGuard(Arc<CudaStream>);

impl StreamGuard {
    pub fn new(stream: CudaStream) -> Self {
        Self(Arc::new(stream))
    }
}

impl PartialEq for StreamGuard {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for StreamGuard {}

impl Deref for StreamGuard {
    type Target = CudaStream;
    fn deref(&self) -> &CudaStream {
        &self.0
    }
}

// ---------------------------------------------------------------------------
// GpuDeviceCtx — bundles device ID with a stream
// ---------------------------------------------------------------------------

/// Thin context for all GPU operations in the explicit-stream path.
#[derive(Clone, Debug)]
pub struct GpuDeviceCtx {
    pub device_id: u32,
    pub stream: StreamGuard,
}

impl GpuDeviceCtx {
    /// Creates a new `GpuDeviceCtx` for the given device.
    ///
    /// NOTE: This calls `set_device_by_id` as a side effect, changing the
    /// current CUDA device for the calling thread.
    pub fn for_device(device_id: u32) -> Result<Self, CudaError> {
        crate::common::set_device_by_id(device_id as i32)?;
        Ok(Self {
            device_id,
            stream: StreamGuard::new(CudaStream::new_non_blocking()?),
        })
    }

    pub fn for_current_device() -> Result<Self, CudaError> {
        let device_id = crate::common::get_device()? as u32;
        Self::for_device(device_id)
    }
}

/// Synchronize the given explicit CUDA stream, blocking until all previously
/// enqueued work on `stream` has completed.
///
/// # Safety
/// The caller must ensure that `stream` is a valid CUDA stream handle.
pub unsafe fn sync_stream(stream: cudaStream_t) -> Result<(), CudaError> {
    check(cudaStreamSynchronize(stream))
}

// ---------------------------------------------------------------------------
// CudaEvent
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
pub type cudaEvent_t = *mut c_void;

#[derive(Debug)]
pub enum CudaEventStatus {
    Completed,
    NotReady,
    Error(CudaError),
}

impl PartialEq for CudaEventStatus {
    fn eq(&self, other: &Self) -> bool {
        use CudaEventStatus::*;
        matches!((self, other), (Completed, Completed) | (NotReady, NotReady))
    }
}

impl Eq for CudaEventStatus {}

impl PartialOrd for CudaEventStatus {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// Completed < NotReady < Error
impl Ord for CudaEventStatus {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        use CudaEventStatus::*;

        match (self, other) {
            (Completed, Completed) => Ordering::Equal,
            (Completed, _) => Ordering::Less,
            (_, Completed) => Ordering::Greater,
            (NotReady, NotReady) => Ordering::Equal,
            (NotReady, Error(_)) => Ordering::Less,
            (Error(_), NotReady) => Ordering::Greater,
            (Error(_), Error(_)) => Ordering::Equal,
        }
    }
}

#[derive(Debug)]
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

    /// Record this event on the given `CudaStream` (safe wrapper).
    pub fn record_on(&self, stream: &CudaStream) -> Result<(), CudaError> {
        check(unsafe { cudaEventRecord(self.event, stream.as_raw()) })
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        check(unsafe { cudaEventSynchronize(self.event) })
    }

    /// # Safety
    /// The caller must ensure that `stream` is a valid stream.
    pub unsafe fn record_and_wait(&self, stream: cudaStream_t) -> Result<(), CudaError> {
        self.record(stream)?;
        check(cudaEventSynchronize(self.event))
    }

    pub fn status(&self) -> CudaEventStatus {
        let status = unsafe { cudaEventQuery(self.event) };
        match status {
            0 => CudaEventStatus::Completed,  // CUDA_SUCCESS
            600 => CudaEventStatus::NotReady, // CUDA_ERROR_NOT_READY
            _ => CudaEventStatus::Error(CudaError::new(status)),
        }
    }

    pub fn completed(&self) -> bool {
        self.status() == CudaEventStatus::Completed
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        let err = unsafe { cudaEventDestroy(self.event) };
        debug_assert_eq!(err, 0, "cudaEventDestroy failed with error code: {err}");
    }
}

// ---------------------------------------------------------------------------
// GPU metrics spans
// ---------------------------------------------------------------------------

/// A GPU-aware span that collects a gauge metric using CUDA events on an explicit stream.
pub fn gpu_metrics_span_on<R, F: FnOnce() -> R>(
    name: impl Into<Cow<'static, str>>,
    stream: &CudaStream,
    f: F,
) -> Result<R, CudaError> {
    let start = CudaEvent::new()?;
    let stop = CudaEvent::new()?;
    start.record_on(stream)?;
    let res = f();
    stop.record_on(stream)?;
    stop.synchronize()?;

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
