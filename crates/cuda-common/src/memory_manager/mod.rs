use std::{
    collections::HashMap,
    ffi::c_void,
    ptr::NonNull,
    sync::{Mutex, OnceLock},
};

use bytesize::ByteSize;

use crate::{error::MemoryError, stream::cudaStreamPerThread};

mod cuda;
mod vm_pool;
use vm_pool::VirtualMemoryPool;

static MEMORY_MANAGER: OnceLock<Mutex<MemoryManager>> = OnceLock::new();

#[ctor::ctor]
fn init() {
    let _ = MEMORY_MANAGER.set(Mutex::new(MemoryManager::new()));
    tracing::info!("Memory manager initialized at program start");
}

pub struct MemoryManager {
    pool: VirtualMemoryPool,

    // Tracking for compatibility with original interface
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
        // Try to allocate from pool
        match self.pool.malloc_internal(size) {
            Ok(ptr) => {
                let ptr = ptr as *mut c_void;
                self.allocated_ptrs
                    .insert(NonNull::new(ptr).expect("Null pointer from pool"), size);
                self.current_size += size;
                if self.current_size > self.max_used_size {
                    self.max_used_size = self.current_size;
                }
                Ok(ptr)
            }
            Err(e) => Err(e),
        }
    }

    pub unsafe fn d_free(&mut self, ptr: *mut c_void) -> Result<(), MemoryError> {
        let nn = NonNull::new(ptr).ok_or(MemoryError::NullPointer)?;

        if let Some(size) = self.allocated_ptrs.remove(&nn) {
            self.current_size -= size;
            // Free in pool (async with event tracking)
            self.pool.free_internal(ptr, cudaStreamPerThread)?;

            Ok(())
        } else {
            Err(MemoryError::UntrackedPointer)
        }
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

pub fn d_malloc(size: usize) -> Result<*mut c_void, MemoryError> {
    let manager = MEMORY_MANAGER.get().ok_or(MemoryError::NotInitialized)?;
    let mut manager = manager.lock().map_err(|_| MemoryError::LockError)?;
    manager.d_malloc(size)
}

pub unsafe fn d_free(ptr: *mut c_void) -> Result<(), MemoryError> {
    let manager = MEMORY_MANAGER.get().ok_or(MemoryError::NotInitialized)?;
    let mut manager = manager.lock().map_err(|_| MemoryError::LockError)?;
    manager.d_free(ptr)
}

fn peak_memory_usage() -> usize {
    MEMORY_MANAGER
        .get()
        .and_then(|m| m.lock().ok())
        .map(|m| m.max_used_size)
        .unwrap_or(0)
}

fn current_memory_usage() -> usize {
    MEMORY_MANAGER
        .get()
        .and_then(|m| m.lock().ok())
        .map(|m| m.current_size)
        .unwrap_or(0)
}

fn reset_peak_memory(new_value: usize) {
    if let Some(manager) = MEMORY_MANAGER.get() {
        if let Ok(mut manager) = manager.lock() {
            manager.max_used_size = new_value;
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemTracker {
    current: usize,
    label: &'static str,
}

impl MemTracker {
    pub fn start(label: &'static str) -> Self {
        Self {
            current: current_memory_usage(),
            label,
        }
    }

    #[inline]
    pub fn tracing_info(&self, msg: impl Into<Option<&'static str>>) {
        let current = current_memory_usage();
        let peak = peak_memory_usage();
        let used = current as isize - self.current as isize;
        let sign = if used >= 0 { "+" } else { "-" };
        tracing::info!(
            "GPU mem usage: used={}{}, current={}, peak={} ({})",
            sign,
            ByteSize::b(used.unsigned_abs() as u64),
            ByteSize::b(current as u64),
            ByteSize::b(peak as u64),
            msg.into()
                .map_or(self.label.to_string(), |m| format!("{}:{}", self.label, m))
        );
    }

    pub fn reset_peak(&mut self) {
        reset_peak_memory(self.current);
    }
}

impl Drop for MemTracker {
    fn drop(&mut self) {
        self.tracing_info(None);
    }
}
