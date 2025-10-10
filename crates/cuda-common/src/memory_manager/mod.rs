use std::{
    collections::HashMap,
    ffi::c_void,
    ptr::NonNull,
    sync::{Arc, Mutex, OnceLock},
};

use bytesize::ByteSize;
use dashmap::DashMap;

use crate::{
    common::set_device,
    error::{check, MemoryError},
    stream::{
        cudaStreamPerThread, cudaStream_t, current_stream_id, current_stream_sync, CudaStreamId,
    },
};

mod cuda;
type SharedPages = Arc<Mutex<Vec<cuda::CUmemGenericAllocationHandle>>>;
mod vm_pool;
use vm_pool::*;

#[link(name = "cudart")]
extern "C" {
    fn cudaMallocAsync(dev_ptr: *mut *mut c_void, size: usize, stream: cudaStream_t) -> i32;
    fn cudaFreeAsync(dev_ptr: *mut c_void, stream: cudaStream_t) -> i32;
    fn cudaDeviceGetDefaultMemPool(memPool: *mut *mut c_void, device: i32) -> i32;
    fn cudaMemPoolTrimTo(pool: *mut c_void, minBytesToKeep: usize) -> i32;
}

struct GlobalMemoryManager {
    per_stream_memory_managers: DashMap<CudaStreamId, MemoryManager>,
    unused_pages: SharedPages,
    page_size: usize,
    device_id: i32,
}

impl GlobalMemoryManager {
    fn new() -> Self {
        let device_id = set_device().unwrap();
        let page_size = get_page_size(device_id).unwrap();
        let pages =
            create_new_pages(device_id, page_size, get_initial_num_pages(page_size)).unwrap();
        Self {
            per_stream_memory_managers: DashMap::new(),
            unused_pages: Arc::new(Mutex::new(pages)),
            page_size,
            device_id,
        }
    }
}

impl Drop for GlobalMemoryManager {
    fn drop(&mut self) {
        if !self.per_stream_memory_managers.is_empty() {
            tracing::warn!(
                "Dropping GlobalMemoryManager with {} active stream managers",
                self.per_stream_memory_managers.len()
            );
            self.per_stream_memory_managers.clear();
        }

        if let Ok(mut pages) = self.unused_pages.lock() {
            if !pages.is_empty() {
                tracing::debug!("GPU mem:Releasing {} pages", pages.len());

                release_pages(pages.drain(..).collect()).unwrap();
            }
        } else {
            tracing::error!("Failed to lock unused_pages during GlobalMemoryManager drop");
        }
    }
}

static GLOBAL_MM: OnceLock<GlobalMemoryManager> = OnceLock::new();

#[ctor::ctor]
fn init_global_mm() {
    let _ = GLOBAL_MM.set(GlobalMemoryManager::new());
}

struct MemoryManager {
    pool: VirtualMemoryPool,
    allocated_ptrs: HashMap<NonNull<c_void>, usize>,
    unused_pages: SharedPages,
    current_size: usize,
    max_used_size: usize,
}

unsafe impl Send for MemoryManager {}
unsafe impl Sync for MemoryManager {}

impl MemoryManager {
    fn new(device_id: i32, page_size: usize, unused_pages: SharedPages) -> Self {
        Self {
            pool: VirtualMemoryPool::new(device_id, page_size).unwrap(),
            allocated_ptrs: HashMap::new(),
            unused_pages,
            current_size: 0,
            max_used_size: 0,
        }
    }

    /// Allocates GPU memory on the current stream.
    /// Small allocations use cudaMallocAsync, large allocations use the virtual memory pool.
    /// If pool is out of pages, the pool will be extended by shared unused pages with allocating
    /// new pages if needed.
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
            let required_pages = tracked_size / self.pool.page_size;
            let available_in_pool = self.pool.available_pages();
            // Pool is out of pages, so we need to add more pages
            if available_in_pool < required_pages {
                let needed_pages = required_pages - available_in_pool;
                let pages_to_add = {
                    let mut unused_pages = self
                        .unused_pages
                        .lock()
                        .map_err(|_| MemoryError::LockError)?;
                    // If we don't have enough unused pages, we need to allocate more
                    if unused_pages.len() < needed_pages {
                        let to_allocate = needed_pages - unused_pages.len();
                        let new_pages = create_new_pages(
                            self.pool.device_id,
                            self.pool.page_size,
                            to_allocate,
                        )?;
                        unused_pages.extend(new_pages);
                    }
                    unused_pages.drain(..needed_pages).collect()
                };
                self.pool.add_pages(pages_to_add)?;
            }
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
    unsafe fn d_free(&mut self, ptr: *mut c_void) -> Result<(), MemoryError> {
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

    fn trim_async_pool(&self) {
        unsafe {
            let mut pool: *mut c_void = std::ptr::null_mut();
            if cudaDeviceGetDefaultMemPool(&mut pool, self.pool.device_id) == 0 {
                cudaMemPoolTrimTo(pool, 0);
            }
        }
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
        self.trim_async_pool();

        if let Ok(pages) = self.pool.reset() {
            if !pages.is_empty() {
                if let Ok(mut unused_pages) = self.unused_pages.lock() {
                    unused_pages.extend(pages);
                } else {
                    tracing::error!("Failed to lock unused_pages during MemoryManager drop");
                }
            }
        } else {
            tracing::error!("Failed to reset pool during MemoryManager drop");
        }
    }
}

pub fn d_malloc(size: usize) -> Result<*mut c_void, MemoryError> {
    let global_mm = GLOBAL_MM.get().unwrap();
    let mm_map = &global_mm.per_stream_memory_managers;
    let stream_id = current_stream_id()?;

    let mut mm = mm_map.entry(stream_id).or_insert_with(|| {
        MemoryManager::new(
            global_mm.device_id,
            global_mm.page_size,
            Arc::clone(&global_mm.unused_pages),
        )
    });
    mm.d_malloc(size)
}

/// # Safety
/// The pointer `ptr` must be a valid, previously allocated device pointer.
/// The caller must ensure that `ptr` is not used after this function is called.
/// The caller must ensure that the pointer is allocated on the current stream.
pub unsafe fn d_free(ptr: *mut c_void) -> Result<(), MemoryError> {
    let global_mm = GLOBAL_MM.get().unwrap();
    let mm_map = &global_mm.per_stream_memory_managers;
    let stream_id = current_stream_id()?;

    if let Some(mut mm) = mm_map.get_mut(&stream_id) {
        mm.d_free(ptr)?;
        // Auto-cleanup pool if everything is freed
        if mm.is_empty() {
            current_stream_sync()?;
            mm_map.remove(&stream_id);
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
        let current = GLOBAL_MM
            .get()
            .map(|gmm| &gmm.per_stream_memory_managers)
            .and_then(|mm_map| {
                let id = current_stream_id().unwrap();
                mm_map.get(&id).map(|mm| mm.current_size)
            })
            .unwrap_or(0);

        Self { current, label }
    }

    #[inline]
    pub fn tracing_info(&self, msg: impl Into<Option<&'static str>>) {
        let Some(mm_map) = GLOBAL_MM.get().map(|gmm| &gmm.per_stream_memory_managers) else {
            tracing::error!("Memory manager not available");
            return;
        };
        let id = current_stream_id().unwrap();
        let (current, peak, pool_usage) = if let Some(mm) = mm_map.get(&id) {
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
        if let Some(mut mm_map) = GLOBAL_MM.get().map(|gmm| &gmm.per_stream_memory_managers) {
            let id = current_stream_id().unwrap();
            if let Some(mut mm) = mm_map.get_mut(&id) {
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
