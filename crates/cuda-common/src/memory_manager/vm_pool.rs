#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]

use std::{
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
        current_stream_id, current_stream_sync, CudaEvent, CudaStreamId,
    },
};

#[link(name = "cudart")]
extern "C" {
    fn cudaMemGetInfo(free_bytes: *mut usize, total_bytes: *mut usize) -> i32;
}

#[derive(Debug, Clone)]
struct FreeRegion {
    size: usize,
    event: Arc<CudaEvent>,
    stream_id: CudaStreamId,
    id: usize,
}

const DEFAULT_VA_SIZE: usize = 8usize << 40; // 8 TB

/// Virtual memory pool implementation.
pub(super) struct VirtualMemoryPool {
    // Virtual address space roots
    roots: Vec<CUdeviceptr>,

    // Map for all active pages
    active_pages: HashMap<CUdeviceptr, CUmemGenericAllocationHandle>,

    // Free regions in virtual space (sorted by address)
    free_regions: BTreeMap<CUdeviceptr, FreeRegion>,

    // Active allocations: (ptr, size)
    malloc_regions: HashMap<CUdeviceptr, usize>,

    // Unmapped regions: (ptr, size)
    unmapped_regions: BTreeMap<CUdeviceptr, usize>,

    // Number of free calls (used to assign ids to free regions)
    free_num: usize,

    // Granularity size: (page % 2MB must be 0)
    pub(super) page_size: usize,

    // Reserved virtual address span in bytes
    va_size: usize,

    // Device ordinal
    pub(super) device_id: i32,
}

unsafe impl Send for VirtualMemoryPool {}
unsafe impl Sync for VirtualMemoryPool {}

impl VirtualMemoryPool {
    fn new() -> Self {
        let device_id = set_device().unwrap();
        let (root, page_size, va_size) = unsafe {
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
                    let va_size = match std::env::var("VPMM_VA_SIZE") {
                        Ok(val) => val.parse().expect("VPMM_VA_SIZE must be a valid number"),
                        Err(_) => DEFAULT_VA_SIZE,
                    };
                    assert!(
                        va_size > 0 && va_size % page_size == 0,
                        "Virtual pool size must be a multiple of page size"
                    );
                    let va_base = vpmm_reserve(va_size, page_size).unwrap();
                    tracing::info!(
                        "VPMM: Reserved virtual address space {} with page size {}",
                        ByteSize::b(va_size as u64),
                        ByteSize::b(page_size as u64)
                    );
                    (va_base, page_size, va_size)
                }
                Err(_) => {
                    tracing::warn!("VPMM not supported, falling back to cudaMallocAsync");
                    (0, usize::MAX, 0)
                }
            }
        };

        Self {
            roots: vec![root],
            active_pages: HashMap::new(),
            free_regions: BTreeMap::new(),
            malloc_regions: HashMap::new(),
            unmapped_regions: if va_size > 0 {
                BTreeMap::from_iter([(root, va_size)])
            } else {
                BTreeMap::new()
            },
            free_num: 0,
            page_size,
            va_size,
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
            best_region = self
                .defragment_or_create_new_pages(requested, stream_id)
                .expect("Failed to defragment or create new pages in virtual memory pool");
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

            self.malloc_regions.insert(ptr, requested);
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

        // 1b. Try the oldest from other streams
        candidates
            .iter_mut()
            .min_by_key(|(_, region)| region.id)
            .map(|(addr, region)| {
                region
                    .event
                    .synchronize()
                    .expect("Failed to synchronize CUDA event during allocation");
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
        let ptr = ptr as CUdeviceptr;
        let size = self
            .malloc_regions
            .remove(&ptr)
            .ok_or(MemoryError::InvalidPointer)
            .expect("Pointer not found in malloc_regions - invalid pointer freed");

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

    fn take_unmapped_region(&mut self, requested: usize) -> Result<CUdeviceptr, MemoryError> {
        debug_assert!(requested != 0);
        debug_assert_eq!(requested % self.page_size, 0);

        if let Some((&addr, &size)) = self
            .unmapped_regions
            .iter()
            .filter(|(_, region_size)| **region_size >= requested)
            .min_by_key(|(_, region_size)| *region_size)
        {
            self.unmapped_regions.remove(&addr);
            if size > requested {
                self.unmapped_regions
                    .insert(addr + requested as u64, size - requested);
            }
            return Ok(addr);
        }

        let addr = unsafe {
            vpmm_reserve(self.va_size, self.page_size).map_err(|_| MemoryError::ReserveFailed {
                size: self.va_size,
                page_size: self.page_size,
            })?
        };
        self.roots.push(addr);
        self.insert_unmapped_region(addr + requested as u64, self.va_size - requested);
        Ok(addr)
    }

    fn insert_unmapped_region(&mut self, mut addr: CUdeviceptr, mut size: usize) {
        if size == 0 {
            return;
        }

        if let Some((&prev_addr, &prev_size)) = self.unmapped_regions.range(..=addr).next_back() {
            if prev_addr + prev_size as u64 == addr {
                self.unmapped_regions.remove(&prev_addr);
                addr = prev_addr;
                size += prev_size;
            }
        }

        if let Some((&next_addr, &next_size)) = self.unmapped_regions.range(addr + 1..).next() {
            if addr + size as u64 == next_addr {
                self.unmapped_regions.remove(&next_addr);
                size += next_size;
            }
        }

        self.unmapped_regions.insert(addr, size);
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

        let total_free_size = self.free_regions.values().map(|r| r.size).sum::<usize>();
        tracing::debug!(
            "VPMM: Defragging or creating new pages: requested={}, free={} stream_id={}",
            ByteSize::b(requested as u64),
            ByteSize::b(total_free_size as u64),
            stream_id
        );

        // Find a best fit unmapped region
        let dst = self.take_unmapped_region(requested).expect("Failed to take unmapped region");
        
        // Allocate new pages if we don't have enough free regions
        let mut allocated_dst = dst;
        let allocate_size = if total_free_size < requested { requested - total_free_size } else { 0 };
        while allocated_dst < dst + allocate_size as u64 {
            let handle = unsafe {
                match vpmm_create_physical(self.device_id, self.page_size) {
                    Ok(handle) => handle,
                    Err(e) => {
                        if e.is_out_of_memory() {
                            return Err(MemoryError::OutOfMemory {
                                requested: allocate_size,
                                available: (allocated_dst - dst) as usize ,
                            });
                        } else {
                            return Err(MemoryError::from(e));
                        }
                    }
                }
            };
            unsafe {
                vpmm_map(allocated_dst, self.page_size, handle)
                    .expect("Failed to map physical memory page to virtual address");
            }
            self.active_pages.insert(  allocated_dst, handle);
            allocated_dst += self.page_size as u64;
        }
        if allocate_size > 0 {
            tracing::debug!(
                "VPMM: Allocated {} bytes on stream {}. Total allocated: {}",
                ByteSize::b(allocate_size as u64),
                stream_id,
                ByteSize::b(self.memory_usage() as u64)
            );
            unsafe {
                vpmm_set_access(dst, allocate_size, self.device_id).expect(
                    "Failed to set access permissions for newly allocated virtual memory region",
                );
            }
            self.free_region_insert(dst, allocate_size, stream_id);
        }
        let mut remaining = requested - allocate_size;
        if remaining == 0 {
            return Ok(Some(dst));
        }

        // Defragment free regions from oldest to newest
        let mut to_defrag: Vec<(CUdeviceptr, usize)> = Vec::new();
        let mut oldest_free_regions: Vec<_> = self.free_regions.iter().map(|(addr, region)| (region.id, *addr)).collect();
        oldest_free_regions.sort_by_key(|(id, _)| *id);
        for (_, addr) in oldest_free_regions {
            if remaining == 0 {
                break;
            }

            let region = self
                .free_regions
                .remove(&addr)
                .unwrap();
            region
                .event
                .synchronize()
                .expect("Failed to synchronize event");

            let take = remaining.min(region.size);
            to_defrag.push((addr, take));
            remaining -= take;

            if region.size > take {
                // stash the leftover right away
                let leftover_addr = addr + take as u64;
                let leftover_size = region.size - take;
                self.free_region_insert(leftover_addr, leftover_size, region.stream_id);
            }
        }
        self.remap_regions(to_defrag, allocated_dst, stream_id)?;

        Ok(Some(dst))
    }

    fn remap_regions(
        &mut self,
        regions: Vec<(CUdeviceptr, usize)>,
        dst: CUdeviceptr,
        stream_id: CudaStreamId,
    ) -> Result<(), MemoryError> {
        if regions.is_empty() {
            return Ok(());
        }

        let bytes_to_remap = regions.iter().map(|(_, size)| *size).sum::<usize>();
        tracing::debug!(
            "VPMM: Remapping {} regions. Total size = {}",
            regions.len(),
            ByteSize::b(bytes_to_remap as u64)
        );

        let mut curr_dst = dst;
        regions.into_iter().for_each(|(region_addr, region_size)| {
            // Unmap the region
            unsafe {
                vpmm_unmap(region_addr, region_size).expect("Failed to unmap region");
            }
            self.insert_unmapped_region(region_addr, region_size);

            // Remap the region
            let num_pages = region_size / self.page_size;
            for i in 0..num_pages {
                let page = region_addr + (i * self.page_size) as u64;
                let handle = self
                    .active_pages
                    .remove(&page)
                    .expect("Active page not found during remapping - page handle missing");
                unsafe {
                    vpmm_map(curr_dst, self.page_size, handle)
                        .expect("Failed to remap physical memory page to new virtual address");
                }
                self.active_pages.insert(curr_dst, handle);
                curr_dst += self.page_size as u64;
            }
        });

        debug_assert_eq!(curr_dst - dst, bytes_to_remap as u64);

        // Set access permissions for the remapped region
        unsafe {
            vpmm_set_access(dst, bytes_to_remap, self.device_id)
                .expect("Failed to set access permissions for remapped virtual memory region");
        }
        self.free_region_insert(dst, bytes_to_remap, stream_id);
        Ok(())
    }

    /// Returns the total physical memory currently mapped in this pool (in bytes).
    pub(super) fn memory_usage(&self) -> usize {
        self.active_pages.len() * self.page_size
    }
}

impl Drop for VirtualMemoryPool {
    fn drop(&mut self) {
        current_stream_sync().unwrap();
        for (ptr, handle) in self.active_pages.drain() {
            unsafe {
                vpmm_unmap(ptr, self.page_size).unwrap();
                vpmm_release(handle).unwrap();
            }
        }

        for root in self.roots.drain(..) {
            unsafe {
                vpmm_release_va(root, self.va_size).unwrap();
            }
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

#[allow(unused)]
unsafe extern "C" fn pending_events_destroy_callback(data: *mut c_void) {
    let boxed =
        Box::from_raw(data as *mut (usize, Arc<Mutex<HashMap<usize, Vec<Arc<CudaEvent>>>>>));
    let (key, pending) = *boxed;
    pending.lock().unwrap().remove(&key);
}
