use std::{
    collections::HashMap,
    ffi::c_void,
    ptr::NonNull,
    sync::{Mutex, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

use bytesize::ByteSize;

use crate::{
    error::{check, MemoryError},
    stream::{cudaStreamPerThread, cudaStream_t, current_stream_id},
};

mod cuda;
mod vm_pool;
use vm_pool::VirtualMemoryPool;

#[link(name = "cudart")]
extern "C" {
    fn cudaMallocAsync(dev_ptr: *mut *mut c_void, size: usize, stream: cudaStream_t) -> i32;
    fn cudaFreeAsync(dev_ptr: *mut c_void, stream: cudaStream_t) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
}

static MEMORY_MANAGER: OnceLock<Mutex<MemoryManager>> = OnceLock::new();

pub fn device_memory_used() -> usize {
    let mut free = 0usize;
    let mut total = 0usize;
    unsafe { cudaMemGetInfo(&mut free, &mut total) };
    total - free
}

#[ctor::ctor]
fn init() {
    let _ = MEMORY_MANAGER.set(Mutex::new(MemoryManager::new()));
    tracing::info!("Memory manager initialized at program start");
}

pub struct MemoryManager {
    pool: VirtualMemoryPool,
    allocated_ptrs: HashMap<NonNull<c_void>, usize>,
    current_size: usize,
    max_used_size: usize,
}

unsafe impl Send for MemoryManager {}
unsafe impl Sync for MemoryManager {}

impl MemoryManager {
    pub fn new() -> Self {
        // Create virtual memory pool
        let pool = VirtualMemoryPool::default();

        Self {
            pool,
            allocated_ptrs: HashMap::new(),
            current_size: 0,
            max_used_size: 0,
        }
    }

    fn d_malloc(&mut self, size: usize) -> Result<*mut c_void, MemoryError> {
        assert!(size != 0, "Requested size must be non-zero");

        let mut tracked_size = size;
        let ptr = if size < self.pool.page_size {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            check(unsafe { cudaMallocAsync(&mut ptr, size, cudaStreamPerThread) })?;
            self.allocated_ptrs
                .insert(NonNull::new(ptr).expect("cudaMalloc returned null"), size);
            Ok(ptr)
        } else {
            tracked_size = size.next_multiple_of(self.pool.page_size);
            self.pool
                .malloc_internal(tracked_size, current_stream_id()?)
        };

        self.current_size += tracked_size;
        if self.current_size > self.max_used_size {
            self.max_used_size = self.current_size;
        }
        ptr
    }

    /// # Safety
    /// The pointer `ptr` must be a valid, previously allocated device pointer.
    /// The caller must ensure that `ptr` is not used after this function is called.
    unsafe fn d_free(&mut self, ptr: *mut c_void) -> Result<(), MemoryError> {
        let nn = NonNull::new(ptr).ok_or(MemoryError::NullPointer)?;

        if let Some(size) = self.allocated_ptrs.remove(&nn) {
            self.current_size -= size;
            check(unsafe { cudaFreeAsync(ptr, cudaStreamPerThread) })?;
        } else {
            self.current_size -= self.pool.free_internal(ptr, current_stream_id()?)?;
        }

        Ok(())
    }
}

impl Drop for MemoryManager {
    fn drop(&mut self) {
        let ptrs: Vec<*mut c_void> = self.allocated_ptrs.keys().map(|nn| nn.as_ptr()).collect();
        for &ptr in &ptrs {
            unsafe { self.d_free(ptr).unwrap() };
        }
        if !self.allocated_ptrs.is_empty() {
            println!(
                "Error: {} allocations were automatically freed on MemoryManager drop",
                self.allocated_ptrs.len()
            );
        }
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

pub fn d_malloc(size: usize) -> Result<*mut c_void, MemoryError> {
    let manager = MEMORY_MANAGER.get().unwrap();
    let mut manager = manager.lock().map_err(|_| MemoryError::LockError)?;
    manager.d_malloc(size)
}

/// # Safety
/// The pointer `ptr` must be a valid, previously allocated device pointer.
/// The caller must ensure that `ptr` is not used after this function is called.
pub unsafe fn d_free(ptr: *mut c_void) -> Result<(), MemoryError> {
    let manager = MEMORY_MANAGER.get().unwrap();
    let mut manager = manager.lock().map_err(|_| MemoryError::LockError)?;
    manager.d_free(ptr)
}

#[derive(Debug, Clone)]
pub struct MemTracker {
    current: usize,
    label: &'static str,
}

impl MemTracker {
    pub fn start(label: &'static str) -> Self {
        let current = MEMORY_MANAGER
            .get()
            .and_then(|m| m.lock().ok())
            .map(|m| m.current_size)
            .unwrap_or_default();

        Self { current, label }
    }

    pub fn start_and_reset_peak(label: &'static str) -> Self {
        let mut mem = Self::start(label);
        mem.reset_peak();
        mem
    }

    pub fn emit_metrics(&self) {
        self.emit_metrics_with_label(self.label);
    }

    pub fn emit_metrics_with_label(&self, label: &'static str) {
        let Some(manager) = MEMORY_MANAGER.get().and_then(|m| m.lock().ok()) else {
            return;
        };

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
            * 1000.0;
        let current = manager.current_size;
        // local_peak is local maximum memory size, as observed by the manager, since the last
        // reset_peak call
        let local_peak = manager.max_used_size;
        let reserved = manager.pool.memory_usage();
        metrics::gauge!("gpu_mem.timestamp_ms", "module" => label).set(ts);
        metrics::gauge!("gpu_mem.current_bytes", "module" => label).set(current as f64);
        metrics::gauge!("gpu_mem.local_peak_bytes", "module" => label).set(local_peak as f64);
        metrics::gauge!("gpu_mem.reserved_bytes", "module" => label).set(reserved as f64);
    }

    #[inline]
    pub fn tracing_info(&self, msg: impl Into<Option<&'static str>>) {
        let Some(manager) = MEMORY_MANAGER.get().and_then(|m| m.lock().ok()) else {
            tracing::error!("Memory manager not available");
            return;
        };
        let current = manager.current_size;
        let peak = manager.max_used_size;
        let used = current as isize - self.current as isize;
        let sign = if used >= 0 { "+" } else { "-" };
        let pool_usage = manager.pool.memory_usage();
        tracing::info!(
            "GPU mem: used={}{}, current={}, peak={}, in pool={} ({})",
            sign,
            ByteSize::b(used.unsigned_abs() as u64),
            ByteSize::b(current as u64),
            ByteSize::b(peak as u64),
            ByteSize::b(pool_usage as u64),
            msg.into()
                .map_or(self.label.to_string(), |m| format!("{}:{}", self.label, m))
        );
    }

    pub fn reset_peak(&mut self) {
        if let Some(mut manager) = MEMORY_MANAGER.get().and_then(|m| m.lock().ok()) {
            manager.max_used_size = manager.current_size;
        }
    }
}

impl Drop for MemTracker {
    fn drop(&mut self) {
        self.tracing_info(None);
    }
}
