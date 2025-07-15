use std::{collections::HashMap, ffi::c_void, ptr::NonNull, sync::Mutex};

use bytesize::ByteSize;
use lazy_static::lazy_static;

use crate::cuda::{
    error::{check, MemoryError},
    stream::{cudaStreamPerThread, cudaStream_t, stream_sync},
};

#[link(name = "cudart")]
extern "C" {
    fn cudaMallocAsync(dev_ptr: *mut *mut c_void, size: usize, stream: cudaStream_t) -> i32;
    fn cudaFreeAsync(dev_ptr: *mut c_void, stream: cudaStream_t) -> i32;
}

lazy_static! {
    static ref MEMORY_MANAGER: Mutex<MemoryManager> = Mutex::new(MemoryManager::default());
}

pub struct MemoryManager {
    allocated_ptrs: HashMap<NonNull<c_void>, usize>,
    current_size: usize,
    max_used_size: usize,
}

unsafe impl Send for MemoryManager {}
unsafe impl Sync for MemoryManager {}

impl MemoryManager {
    pub fn new() -> Self {
        Self {
            allocated_ptrs: HashMap::new(),
            current_size: 0,
            max_used_size: 0,
        }
    }

    pub fn d_malloc(&mut self, size: usize) -> Result<*mut c_void, MemoryError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        check(unsafe { cudaMallocAsync(&mut ptr, size, cudaStreamPerThread) })?;

        self.allocated_ptrs
            .insert(NonNull::new(ptr).expect("cudaMalloc returned null"), size);
        self.current_size += size;
        if self.current_size > self.max_used_size {
            self.max_used_size = self.current_size;
        }
        Ok(ptr)
    }

    /// # Safety
    /// The pointer `ptr` must be a valid, previously allocated device pointer.
    /// The caller must ensure that `ptr` is not used after this function is called.
    pub unsafe fn d_free(&mut self, ptr: *mut c_void) -> Result<(), MemoryError> {
        let nn = NonNull::new(ptr).ok_or(MemoryError::NullPointer)?;

        if let Some(size) = self.allocated_ptrs.remove(&nn) {
            self.current_size -= size;
        } else {
            return Err(MemoryError::UntrackedPointer);
        }

        check(unsafe { cudaFreeAsync(ptr, cudaStreamPerThread) })?;

        Ok(())
    }
}

impl Drop for MemoryManager {
    fn drop(&mut self) {
        for &nn in self.allocated_ptrs.keys() {
            unsafe { d_free(nn.as_ptr()).unwrap() };
        }
        stream_sync().unwrap();
        if !self.allocated_ptrs.is_empty() {
            println!(
                "Warning: {} allocations were automatically freed on MemoryManager drop",
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
    let mut manager = MEMORY_MANAGER.lock().map_err(|_| MemoryError::LockError)?;
    manager.d_malloc(size)
}

/// # Safety
/// The pointer `ptr` must be a valid, previously allocated device pointer.
/// The caller must ensure that `ptr` is not used after this function is called.
pub unsafe fn d_free(ptr: *mut c_void) -> Result<(), MemoryError> {
    let mut manager = MEMORY_MANAGER.lock().map_err(|_| MemoryError::LockError)?;
    manager.d_free(ptr)
}

fn peak_memory_usage() -> usize {
    let manager = MEMORY_MANAGER.lock().unwrap();
    manager.max_used_size
}

fn current_memory_usage() -> usize {
    let manager = MEMORY_MANAGER.lock().unwrap();
    manager.current_size
}

fn reset_peak_memory(new_value: usize) {
    let mut manager = MEMORY_MANAGER.lock().unwrap();
    manager.max_used_size = new_value;
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
