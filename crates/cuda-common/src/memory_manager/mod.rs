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
    stream::{cudaStream_t, device_synchronize, StreamGuard},
};

mod cuda;
mod vm_pool;
use vm_pool::VirtualMemoryPool;

#[cfg(test)]
mod tests;

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

/// Allocation record for the small-allocation path (`cudaMallocAsync`).
struct AllocRecord {
    size: usize,
    stream: StreamGuard,
}

pub struct MemoryManager {
    pool: VirtualMemoryPool,
    allocated_ptrs: HashMap<NonNull<c_void>, AllocRecord>,
    current_size: usize,
    max_used_size: usize,
}

/// # Safety
/// `MemoryManager` is not internally synchronized. These impls are safe because
/// the singleton instance is wrapped in `Mutex` via `MEMORY_MANAGER`.
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

    fn d_malloc_on(
        &mut self,
        size: usize,
        stream: &StreamGuard,
    ) -> Result<*mut c_void, MemoryError> {
        assert!(size != 0, "Requested size must be non-zero");

        let mut tracked_size = size;
        let ptr = if size < self.pool.page_size {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            check(unsafe { cudaMallocAsync(&mut ptr, size, stream.as_raw()) }).map_err(|e| {
                tracing::error!("cudaMallocAsync failed: size={}: {:?}", size, e);
                MemoryError::from(e)
            })?;
            self.allocated_ptrs.insert(
                NonNull::new(ptr).expect("BUG: cudaMallocAsync returned null"),
                AllocRecord {
                    size,
                    stream: stream.clone(),
                },
            );
            ptr
        } else {
            tracked_size = size.next_multiple_of(self.pool.page_size);
            self.pool.malloc_internal(tracked_size, stream)?
        };

        self.current_size += tracked_size;
        if self.current_size > self.max_used_size {
            self.max_used_size = self.current_size;
        }
        Ok(ptr)
    }

    /// Two-stage free: first resolves the record under the lock, returning the
    /// `StreamGuard` that must be dropped AFTER the lock is released.
    ///
    /// # Safety
    /// - The pointer `ptr` must be a valid, previously allocated device pointer.
    /// - The caller must ensure that `ptr` is not used after this function is called.
    /// - The caller must hold the `MEMORY_MANAGER` lock before calling this method.
    unsafe fn d_free_under_lock(&mut self, ptr: *mut c_void) -> Result<StreamGuard, MemoryError> {
        let nn = NonNull::new(ptr).ok_or(MemoryError::NullPointer)?;

        if let Some(record) = self.allocated_ptrs.remove(&nn) {
            let size = record.size;
            self.current_size -= size;
            check(unsafe { cudaFreeAsync(ptr, record.stream.as_raw()) }).map_err(|e| {
                tracing::error!("cudaFreeAsync failed: ptr={:p}: {:?}", ptr, e);
                MemoryError::from(e)
            })?;
            Ok(record.stream)
        } else {
            let (freed_size, guard) = self.pool.free_internal(ptr)?;
            self.current_size -= freed_size;
            Ok(guard)
        }
    }
}

impl Drop for MemoryManager {
    fn drop(&mut self) {
        device_synchronize().unwrap();
        let ptrs: Vec<*mut c_void> = self.allocated_ptrs.keys().map(|nn| nn.as_ptr()).collect();
        for &ptr in &ptrs {
            match unsafe { self.d_free_under_lock(ptr) } {
                Ok(guard) => drop(guard),
                Err(e) => tracing::error!("MemoryManager drop: failed to free {:p}: {:?}", ptr, e),
            }
        }
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

pub fn d_malloc_on(size: usize, stream: &StreamGuard) -> Result<*mut c_void, MemoryError> {
    let manager = MEMORY_MANAGER.get().unwrap();
    let mut manager = manager.lock().map_err(|_| MemoryError::LockError)?;
    manager.d_malloc_on(size, stream)
}

/// # Safety
/// The pointer `ptr` must be a valid, previously allocated device pointer.
/// The caller must ensure that `ptr` is not used after this function is called.
pub unsafe fn d_free(ptr: *mut c_void) -> Result<(), MemoryError> {
    let manager = MEMORY_MANAGER.get().unwrap();
    let mut manager = manager.lock().map_err(|_| MemoryError::LockError)?;
    let guard = manager.d_free_under_lock(ptr)?;
    drop(manager);
    drop(guard);
    Ok(())
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
