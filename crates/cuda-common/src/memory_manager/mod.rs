use std::{
    collections::HashMap,
    ffi::c_void,
    ptr::NonNull,
    sync::{Mutex, OnceLock},
};

use bytesize::ByteSize;

use crate::{
    error::{check, MemoryError},
    stream::{
        cudaStreamPerThread, cudaStream_t, current_stream_id, default_stream_sync, CudaStreamId,
    },
};

mod cuda;
mod vm_pool;
use vm_pool::VirtualMemoryPool;

#[link(name = "cudart")]
extern "C" {
    fn cudaMallocAsync(dev_ptr: *mut *mut c_void, size: usize, stream: cudaStream_t) -> i32;
    fn cudaFreeAsync(dev_ptr: *mut c_void, stream: cudaStream_t) -> i32;
}

static MM_MAP: OnceLock<Mutex<HashMap<CudaStreamId, MemoryManager>>> = OnceLock::new();

#[ctor::ctor]
fn init() {
    let _ = MM_MAP.set(Mutex::new(HashMap::new()));
}

#[inline]
fn stream_mm_mut(map: &mut HashMap<CudaStreamId, MemoryManager>) -> &mut MemoryManager {
    let key = current_stream_id().unwrap();
    map.entry(key).or_default()
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

    pub fn d_malloc(&mut self, size: usize) -> Result<*mut c_void, MemoryError> {
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
            self.pool.malloc_internal(tracked_size)
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
    pub unsafe fn d_free(&mut self, ptr: *mut c_void) -> Result<(), MemoryError> {
        let nn = NonNull::new(ptr).ok_or(MemoryError::NullPointer)?;

        if let Some(size) = self.allocated_ptrs.remove(&nn) {
            self.current_size -= size;
            check(unsafe { cudaFreeAsync(ptr, cudaStreamPerThread) })?;
        } else {
            self.current_size -= self.pool.free_internal(ptr)?;
        }

        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.allocated_ptrs.is_empty() && self.pool.is_empty()
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
    let mm_map = MM_MAP.get().ok_or(MemoryError::NotInitialized)?;
    let mut mm_map = mm_map.lock().map_err(|_| MemoryError::LockError)?;
    stream_mm_mut(&mut mm_map).d_malloc(size)
}

/// # Safety
/// The pointer `ptr` must be a valid, previously allocated device pointer.
/// The caller must ensure that `ptr` is not used after this function is called.
/// The caller must ensure that the pointer is allocated on the current stream.
pub unsafe fn d_free(ptr: *mut c_void) -> Result<(), MemoryError> {
    let mm_map = MM_MAP.get().ok_or(MemoryError::NotInitialized)?;
    let mut mm_map = mm_map.lock().map_err(|_| MemoryError::LockError)?;
    let stream_id = current_stream_id()?;

    if let Some(mm) = mm_map.get_mut(&stream_id) {
        mm.d_free(ptr)?;
        // Auto-cleanup pool if everything is freed
        if mm.is_empty() {
            default_stream_sync()?;
            tracing::info!(
                "GPU mem ({}): Auto-cleanup pool {}",
                stream_id,
                ByteSize::b(mm.pool.memory_usage() as u64)
            );
            mm.pool.clear();
        }
    } else {
        panic!(
            "Attempting to free ptr {:?} on stream {} which has no MemoryManager.",
            ptr, stream_id
        );
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct MemTracker {
    current: usize,
    label: &'static str,
}

impl MemTracker {
    pub fn start(label: &'static str) -> Self {
        let current = MM_MAP
            .get()
            .and_then(|m| m.lock().ok())
            .and_then(|mm_map| {
                let id = current_stream_id().unwrap();
                mm_map.get(&id).map(|mm| mm.current_size)
            })
            .unwrap_or(0);

        Self { current, label }
    }

    #[inline]
    pub fn tracing_info(&self, msg: impl Into<Option<&'static str>>) {
        let Some(mm_map_guard) = MM_MAP.get().and_then(|m| m.lock().ok()) else {
            tracing::error!("Memory manager not available");
            return;
        };
        let id = current_stream_id().unwrap();
        let (current, peak, pool_usage) = if let Some(mm) = mm_map_guard.get(&id) {
            (mm.current_size, mm.max_used_size, mm.pool.memory_usage())
        } else {
            (0, 0, 0)
        };
        let used = current as isize - self.current as isize;
        let sign = if used >= 0 { "+" } else { "-" };
        tracing::info!(
            "GPU mem ({}): used={}{}, current={}, peak={}, in pool={} ({})",
            id,
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
        if let Some(mut mm_map) = MM_MAP.get().and_then(|m| m.lock().ok()) {
            let id = current_stream_id().unwrap();
            if let Some(mm) = mm_map.get_mut(&id) {
                mm.max_used_size = mm.current_size;
            }
        }
    }
}

impl Drop for MemTracker {
    fn drop(&mut self) {
        self.tracing_info(None);
    }
}
