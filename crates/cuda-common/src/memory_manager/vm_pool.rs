#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]

use std::{
    cmp::Reverse,
    collections::{BTreeMap, HashMap},
    ffi::c_void,
    sync::{Arc, Mutex},
};

use bytesize::ByteSize;

use super::cuda::*;
use crate::{
    common::set_device,
    error::MemoryError,
    stream::{
        cudaStreamPerThread, cudaStream_t, current_stream_id, default_stream_wait, CudaEvent,
        CudaStreamId,
    },
};

#[link(name = "cudart")]
extern "C" {
    fn cudaMemGetInfo(free_bytes: *mut usize, total_bytes: *mut usize) -> i32;
    fn cudaLaunchHostFunc(
        stream: cudaStream_t,
        fn_: Option<unsafe extern "C" fn(*mut c_void)>,
        user_data: *mut c_void,
    ) -> i32;
}

#[derive(Debug, Clone)]
struct FreeRegion {
    size: usize,
    event: Arc<CudaEvent>,
    stream_id: CudaStreamId,
    id: usize,
}

const VIRTUAL_POOL_SIZE: usize = 1usize << 45; // 32 TB

/// Virtual memory pool implementation.
pub(super) struct VirtualMemoryPool {
    // Virtual address space root
    root: CUdeviceptr,

    // Current end of active address space
    curr_end: CUdeviceptr,

    // Map for all active pages
    active_pages: HashMap<CUdeviceptr, CUmemGenericAllocationHandle>,

    // Free regions in virtual space (sorted by address)
    free_regions: BTreeMap<CUdeviceptr, FreeRegion>,

    // Number of free calls (used to assign ids to free regions)
    free_num: usize,

    // Pending events to destroy after "wait for event" is finished
    // key: free_num as VMPool free state; value: non-completed events from other streams
    pending_events: Arc<Mutex<HashMap<usize, Vec<Arc<CudaEvent>>>>>,

    // Active allocations: (ptr, size)
    used_regions: HashMap<CUdeviceptr, usize>,

    // Granularity size: (page % 2MB must be 0)
    pub(super) page_size: usize,

    // Device ordinal
    pub(super) device_id: i32,
}

unsafe impl Send for VirtualMemoryPool {}
unsafe impl Sync for VirtualMemoryPool {}

impl VirtualMemoryPool {
    fn new() -> Self {
        let device_id = set_device().unwrap();
        let (root, page_size) = unsafe {
            match vpmm_check_support(device_id) {
                Ok(_) => {
                    let gran = vpmm_min_granularity(device_id).unwrap();
                    let page_size = match std::env::var("VPMM_PAGE_SIZE") {
                        Ok(val) => {
                            let custom_size: usize =
                                val.parse().expect("VPMM_PAGE_SIZE must be a valid number");
                            assert!(
                                custom_size > 0 && custom_size % gran == 0,
                                "VPMM_PAGE_SIZE must be > 0 and multiple of {}",
                                gran
                            );
                            custom_size
                        }
                        Err(_) => gran,
                    };
                    let va_base = vpmm_reserve(VIRTUAL_POOL_SIZE, page_size).unwrap();
                    (va_base, page_size)
                }
                Err(_) => {
                    tracing::warn!("VPMM not supported, falling back to cudaMallocAsync");
                    (0, usize::MAX)
                }
            }
        };

        Self {
            root,
            curr_end: root,
            active_pages: HashMap::new(),
            free_regions: BTreeMap::new(),
            free_num: 0,
            pending_events: Arc::new(Mutex::new(HashMap::new())),
            used_regions: HashMap::new(),
            page_size,
            device_id,
        }
    }

    /// Allocates memory from the pool's free regions.
    /// Attempts defragmentation if no suitable free region exists.
    /// Returns an error if there's insufficient memory even after defragmentation.
    pub(super) fn malloc_internal(
        &mut self,
        requested: usize,
        stream_id: CudaStreamId,
    ) -> Result<*mut c_void, MemoryError> {
        debug_assert!(
            requested != 0 && requested % self.page_size == 0,
            "Requested size must be a multiple of the page size"
        );
        // Phase 1: Zero-cost attempts
        let mut best_region = self.find_best_fit(requested, stream_id);

        if best_region.is_none() {
            // Phase 2: Defragmentation
            best_region = self.defragment_or_create_new_pages(requested, stream_id)?;
        }

        if let Some(ptr) = best_region {
            let region = self.free_regions.remove(&ptr).unwrap();
            debug_assert_eq!(region.stream_id, stream_id);

            // If region is larger, return remainder to free list
            if region.size > requested {
                self.free_regions.insert(
                    ptr + requested as u64,
                    FreeRegion {
                        size: region.size - requested,
                        event: region.event,
                        stream_id,
                        id: region.id,
                    },
                );
            }

            self.used_regions.insert(ptr, requested);
            return Ok(ptr as *mut c_void);
        }

        Err(MemoryError::OutOfMemory {
            requested,
            available: self.free_regions.values().map(|r| r.size).sum(),
        })
    }

    /// Phase 1: Try to find a suitable free region without defragmentation
    /// Returns address if found, None otherwise
    fn find_best_fit(&mut self, requested: usize, stream_id: CudaStreamId) -> Option<CUdeviceptr> {
        let mut candidates: Vec<(CUdeviceptr, &mut FreeRegion)> = self
            .free_regions
            .iter_mut()
            .filter(|(_, region)| region.size >= requested)
            .map(|(addr, region)| (*addr, region))
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // 1a. Prefer current stream (smallest fit)
        if let Some((addr, _)) = candidates
            .iter()
            .filter(|(_, region)| region.stream_id == stream_id)
            .min_by_key(|(_, region)| region.size)
        {
            return Some(*addr);
        }

        // 1b. Try completed from other streams (smallest completed fit)
        candidates
            .iter_mut()
            .filter(|(_, region)| region.event.completed())
            .min_by_key(|(_, region)| region.size)
            .map(|(addr, region)| {
                region.stream_id = stream_id;
                *addr
            })
    }

    /// Frees a pointer and returns the size of the freed memory.
    /// Coalesces adjacent free regions.
    pub(super) fn free_internal(
        &mut self,
        ptr: *mut c_void,
        stream_id: CudaStreamId,
    ) -> Result<usize, MemoryError> {
        debug_assert!(self.curr_end != self.root, "VM pool is empty");
        let ptr = ptr as CUdeviceptr;
        let size = self
            .used_regions
            .remove(&ptr)
            .ok_or(MemoryError::InvalidPointer)?;

        self.free_region_insert(ptr, size, stream_id);

        Ok(size)
    }

    fn free_region_insert(
        &mut self,
        mut ptr: CUdeviceptr,
        mut size: usize,
        stream_id: CudaStreamId,
    ) {
        // Potential merge with next neighbor
        if let Some((&next_ptr, next_region)) = self.free_regions.range(ptr + 1..).next() {
            if next_region.stream_id == stream_id && ptr + size as u64 == next_ptr {
                let next_region = self.free_regions.remove(&next_ptr).unwrap();
                size += next_region.size;
            }
        }
        // Potential merge with previous neighbor
        if let Some((&prev_ptr, prev_region)) = self.free_regions.range(..ptr).next_back() {
            if prev_region.stream_id == stream_id && prev_ptr + prev_region.size as u64 == ptr {
                let prev_region = self.free_regions.remove(&prev_ptr).unwrap();
                ptr = prev_ptr;
                size += prev_region.size;
            }
        }
        // Record a new event to capture the current point in the stream
        let event = Arc::new(CudaEvent::new().unwrap());
        event.record_on_this().unwrap();
        // Assign an id to the free region
        let id = self.free_num;
        self.free_num += 1;

        self.free_regions.insert(
            ptr,
            FreeRegion {
                size,
                event,
                stream_id,
                id,
            },
        );
    }

    /// Defragments the pool by consolidating free regions at the end of virtual address space.
    /// Remaps pages to create one large contiguous free region.
    fn defragment_or_create_new_pages(
        &mut self,
        requested: usize,
        stream_id: CudaStreamId,
    ) -> Result<Option<CUdeviceptr>, MemoryError> {
        if requested == 0 {
            return Ok(None);
        }

        tracing::debug!(
            "VPMM: Defragging or creating new pages: requested={}, stream_id={}",
            ByteSize::b(requested as u64),
            stream_id
        );

        let mut to_defrag = Vec::new();

        // 2.a. Defrag current stream (newest first)
        let mut current_stream_to_defrag: Vec<(CUdeviceptr, usize, usize)> = self
            .free_regions
            .iter()
            .filter(|(_, r)| r.stream_id == stream_id)
            .map(|(addr, r)| (*addr, r.size, r.id))
            .collect();

        // If last free region is at the virtual end, use it as base for defragmentation
        let (mut defrag_start, mut accumulated_size) = if current_stream_to_defrag
            .last()
            .is_some_and(|(addr, size, _)| *addr + *size as u64 == self.curr_end)
        {
            let (addr, size, _) = current_stream_to_defrag.pop().unwrap();
            (addr, size)
        } else {
            (self.curr_end, 0)
        };

        tracing::debug!(
            "VPMM: Current stream {} trying to defragment from {:?}",
            stream_id,
            current_stream_to_defrag
                .iter()
                .map(|(_, size, _)| ByteSize::b(*size as u64))
                .collect::<Vec<_>>()
        );

        let current_stream_to_defrag_size = current_stream_to_defrag
            .iter()
            .map(|(_, size, _)| size)
            .sum::<usize>();
        if current_stream_to_defrag_size + accumulated_size < requested {
            to_defrag.extend(current_stream_to_defrag.iter().map(|(ptr, _, _)| *ptr));
            accumulated_size += current_stream_to_defrag_size;
        } else {
            current_stream_to_defrag.sort_by_key(|(_, _, free_id)| Reverse(*free_id));

            for (ptr, size, _) in current_stream_to_defrag {
                to_defrag.push(ptr);
                accumulated_size += size;
                if accumulated_size >= requested {
                    self.remap_regions(to_defrag, stream_id)?;
                    return Ok(Some(defrag_start));
                }
            }
        }

        // 2.b. Defrag other streams (completed, oldest first)
        let mut other_streams_to_defrag: Vec<(CUdeviceptr, usize, Arc<CudaEvent>, usize)> = self
            .free_regions
            .iter()
            .filter(|(_, r)| r.stream_id != stream_id)
            .map(|(addr, r)| (*addr, r.size, r.event.clone(), r.id))
            .collect();

        // If last free region is at the virtual end, take it and use as base for defragmentation
        if accumulated_size == 0
            && other_streams_to_defrag
                .last()
                .is_some_and(|(addr, size, event, _)| {
                    event.completed() && *addr + *size as u64 == self.curr_end
                })
        {
            (defrag_start, accumulated_size, _, _) = other_streams_to_defrag.pop().unwrap();
            self.free_regions.get_mut(&defrag_start).unwrap().stream_id = stream_id;
        }

        tracing::debug!(
            "VPMM: Defragmented {} bytes from stream {}, other streams ready to borrow {:?}",
            ByteSize::b(accumulated_size as u64),
            stream_id,
            other_streams_to_defrag
                .iter()
                .map(|(_, size, event, id)| (ByteSize::b(*size as u64), event.status(), id))
                .collect::<Vec<_>>()
        );

        other_streams_to_defrag.sort_by_key(|(_, _, event, free_id)| (event.status(), *free_id));

        let mut events_to_wait = Vec::new();
        for (ptr, size, event, _) in other_streams_to_defrag {
            if !event.completed() {
                default_stream_wait(&event)?;
                events_to_wait.push(event);
            }
            to_defrag.push(ptr);
            accumulated_size += size;
            if accumulated_size >= requested {
                break;
            }
        }
        if !events_to_wait.is_empty() {
            // Destroy events only when "wait for event" is finished
            tracing::debug!(
                "VPMM: Async call {} events to destroy on stream {}",
                events_to_wait.len(),
                stream_id
            );
            self.pending_events
                .lock()
                .unwrap()
                .insert(self.free_num, events_to_wait);
            unsafe {
                cudaLaunchHostFunc(
                    cudaStreamPerThread,
                    Some(pending_events_destroy_callback),
                    Box::into_raw(Box::new((self.free_num, self.pending_events.clone())))
                        as *mut c_void,
                )
            };
        }

        self.remap_regions(to_defrag, stream_id)?;
        if accumulated_size >= requested {
            return Ok(Some(defrag_start));
        }

        // 2.c. Create new pages
        let allocate_start = self.curr_end;
        let mut allocated_size = 0;
        while accumulated_size < requested {
            let handle = unsafe {
                match vpmm_create_physical(self.device_id, self.page_size) {
                    Ok(handle) => handle,
                    Err(e) => {
                        if e.is_out_of_memory() {
                            return Err(MemoryError::OutOfMemory {
                                requested,
                                available: accumulated_size,
                            });
                        } else {
                            return Err(MemoryError::from(e));
                        }
                    }
                }
            };
            unsafe { vpmm_map(self.curr_end, self.page_size, handle)? };
            self.active_pages.insert(self.curr_end, handle);
            self.curr_end += self.page_size as u64;
            allocated_size += self.page_size;
            accumulated_size += self.page_size;
        }
        tracing::info!(
            "GPU mem: VPMM allocated {} bytes on stream {}. Total allocated: {}",
            ByteSize::b(allocated_size as u64),
            stream_id,
            ByteSize::b(self.memory_usage() as u64)
        );
        unsafe { vpmm_set_access(allocate_start, allocated_size, self.device_id)? };
        self.free_region_insert(allocate_start, allocated_size, stream_id);

        Ok(Some(defrag_start))
    }

    fn remap_regions(
        &mut self,
        regions: Vec<CUdeviceptr>,
        stream_id: CudaStreamId,
    ) -> Result<(), MemoryError> {
        if regions.is_empty() {
            return Ok(());
        }
        let new_region_start = self.curr_end;
        for region_addr in regions {
            let region = self
                .free_regions
                .remove(&region_addr)
                .ok_or(MemoryError::InvalidPointer)?;

            let num_pages = region.size / self.page_size;
            for i in 0..num_pages {
                let page = region_addr + (i * self.page_size) as u64;
                let handle = self
                    .active_pages
                    .remove(&page)
                    .ok_or(MemoryError::InvalidPointer)?;
                unsafe { vpmm_map(self.curr_end, self.page_size, handle)? };
                self.active_pages.insert(self.curr_end, handle);
                self.curr_end += self.page_size as u64;
            }
        }
        let new_region_size = (self.curr_end - new_region_start) as usize;

        unsafe { vpmm_set_access(new_region_start, new_region_size, self.device_id)? };
        self.free_region_insert(new_region_start, new_region_size, stream_id);
        Ok(())
    }

    /// Returns the total physical memory currently mapped in this pool (in bytes).
    pub(super) fn get_allocation_size(&self, ptr: u64) -> Option<usize> {
        self.used_regions.get(&ptr).copied()
    }

    pub(super) fn memory_usage(&self) -> usize {
        self.active_pages.len() * self.page_size
    }
}

impl Drop for VirtualMemoryPool {
    fn drop(&mut self) {
        if self.root != self.curr_end {
            unsafe {
                vpmm_unmap(self.root, (self.curr_end - self.root) as usize).unwrap();
                for (_, handle) in self.active_pages.drain() {
                    vpmm_release(handle).unwrap();
                }
            }
        }
        unsafe {
            vpmm_release_va(self.root, VIRTUAL_POOL_SIZE).unwrap();
        }
    }
}

impl Default for VirtualMemoryPool {
    fn default() -> Self {
        let mut pool = Self::new();

        // Skip allocation if vpmm not supported
        if pool.page_size == usize::MAX {
            return pool;
        }

        // Calculate initial number of pages to allocate
        let initial_pages = match std::env::var("VPMM_PAGES") {
            Ok(val) => {
                let pages: usize = val.parse().expect("VPMM_PAGES must be a valid number");
                pages
            }
            Err(_) => 0,
        };
        if let Err(e) = pool.defragment_or_create_new_pages(
            initial_pages * pool.page_size,
            current_stream_id().unwrap(),
        ) {
            // Check how much memory is available
            let mut free_mem = 0usize;
            let mut total_mem = 0usize;
            unsafe {
                cudaMemGetInfo(&mut free_mem, &mut total_mem);
            }
            panic!(
                "Error:{:?}\nPool: pages={}, page_size={}\nMemory: free_mem={}, total_mem={}",
                e, initial_pages, pool.page_size, free_mem, total_mem
            );
        }
        pool
    }
}

unsafe extern "C" fn pending_events_destroy_callback(data: *mut c_void) {
    let boxed =
        Box::from_raw(data as *mut (usize, Arc<Mutex<HashMap<usize, Vec<Arc<CudaEvent>>>>>));
    let (key, pending) = *boxed;
    pending.lock().unwrap().remove(&key);
}
