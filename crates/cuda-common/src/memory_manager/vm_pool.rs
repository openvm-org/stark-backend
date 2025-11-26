#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]

use std::{
    collections::{BTreeMap, HashMap},
    ffi::c_void,
    sync::Arc,
};

use bytesize::ByteSize;

use super::cuda::*;
use crate::{
    common::set_device,
    error::MemoryError,
    stream::{current_stream_id, current_stream_sync, CudaEvent, CudaStreamId},
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

/// # Safety
/// `VirtualMemoryPool` is not internally synchronized. These impls are safe because
/// all access goes through `Mutex<MemoryManager>` in the parent module.
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
                    tracing::debug!(
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
            best_region = self.defragment_or_create_new_pages(requested, stream_id)?;
        }

        if let Some(ptr) = best_region {
            let region = self
                .free_regions
                .remove(&ptr)
                .expect("BUG: free region address not found after find_best_fit");
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
        if let Some((addr, region)) = candidates.iter_mut().min_by_key(|(_, region)| region.id) {
            if let Err(e) = region.event.synchronize() {
                tracing::error!("Event synchronize failed during find_best_fit: {:?}", e);
                return None;
            }
            region.stream_id = stream_id;
            Some(*addr)
        } else {
            None
        }
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
            .ok_or(MemoryError::InvalidPointer)?;

        let _ = self.free_region_insert(ptr, size, stream_id);

        Ok(size)
    }

    fn free_region_insert(
        &mut self,
        mut ptr: CUdeviceptr,
        mut size: usize,
        stream_id: CudaStreamId,
    ) -> CUdeviceptr {
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
        ptr
    }

    /// Return the base address of a virtual hole large enough for `requested` bytes.
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

    /// Insert a hole back into the unmapped set, coalescing with neighbors.
    fn insert_unmapped_region(&mut self, mut addr: CUdeviceptr, mut size: usize) {
        if size == 0 {
            return;
        }

        if let Some((&prev_addr, &prev_size)) = self.unmapped_regions.range(..addr).next_back() {
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

    /// Defragments the pool by reusing existing holes and, if needed, reserving more VA space.
    /// Moves just enough pages to satisfy `requested`, keeping the remainder in place.
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
        let dst = self.take_unmapped_region(requested)?;
        // Sentinel value until we have a valid free region pointer from allocation
        let mut allocated_ptr = CUdeviceptr::MAX;

        // Allocate new pages if we don't have enough free regions
        let mut allocated_dst = dst;
        let allocate_size = requested.saturating_sub(total_free_size);
        while allocated_dst < dst + allocate_size as u64 {
            let handle = unsafe {
                match vpmm_create_physical(self.device_id, self.page_size) {
                    Ok(handle) => handle,
                    Err(e) => {
                        tracing::error!(
                            "vpmm_create_physical failed: device={}, page_size={}: {:?}",
                            self.device_id, self.page_size, e
                        );
                        if e.is_out_of_memory() {
                            return Err(MemoryError::OutOfMemory {
                                requested: allocate_size,
                                available: (allocated_dst - dst) as usize,
                            });
                        } else {
                            return Err(MemoryError::from(e));
                        }
                    }
                }
            };
            unsafe {
                vpmm_map(allocated_dst, self.page_size, handle).map_err(|e| {
                    tracing::error!(
                        "vpmm_map failed: addr={:#x}, page_size={}, handle={}: {:?}",
                        allocated_dst, self.page_size, handle, e
                    );
                    MemoryError::from(e)
                })?;
            }
            self.active_pages.insert(allocated_dst, handle);
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
                vpmm_set_access(dst, allocate_size, self.device_id).map_err(|e| {
                    tracing::error!(
                        "vpmm_set_access failed: addr={:#x}, size={}, device={}: {:?}",
                        dst, allocate_size, self.device_id, e
                    );
                    MemoryError::from(e)
                })?;
            }
            allocated_ptr = self.free_region_insert(dst, allocate_size, stream_id);
        }
        let mut remaining = requested - allocate_size;
        if remaining == 0 {
            debug_assert_ne!(
                allocated_ptr,
                CUdeviceptr::MAX,
                "Allocation returned no valid free region"
            );
            return Ok(Some(allocated_ptr));
        }

        // Pull free regions (oldest first) until we've gathered enough pages.
        let mut to_defrag: Vec<(CUdeviceptr, usize)> = Vec::new();
        let mut oldest_free_regions: Vec<_> = self
            .free_regions
            .iter()
            .filter(|(&addr, _)| allocate_size == 0 || addr != allocated_ptr)
            .map(|(&addr, region)| (region.id, addr))
            .collect();
        oldest_free_regions.sort_by_key(|(id, _)| *id);
        for (_, addr) in oldest_free_regions {
            if remaining == 0 {
                break;
            }

            let region = self.free_regions.remove(&addr).expect("BUG: free region disappeared");
            region.event.synchronize().map_err(|e| {
                tracing::error!("Event synchronize failed during defrag: {:?}", e);
                MemoryError::from(e)
            })?;

            let take = remaining.min(region.size);
            to_defrag.push((addr, take));
            remaining -= take;

            if region.size > take {
                // Return the unused tail to the free list so it stays available.
                let leftover_addr = addr + take as u64;
                let leftover_size = region.size - take;
                let _ = self.free_region_insert(leftover_addr, leftover_size, region.stream_id);
            }
        }
        let remapped_ptr = self.remap_regions(to_defrag, allocated_dst, stream_id)?;
        let result = std::cmp::min(allocated_ptr, remapped_ptr);
        debug_assert_ne!(
            result,
            CUdeviceptr::MAX,
            "Both allocation and remapping returned no valid free region"
        );
        Ok(Some(result))
    }

    /// Remap a list of regions to a new base address.
    /// The regions already dropped from free regions
    fn remap_regions(
        &mut self,
        regions: Vec<(CUdeviceptr, usize)>,
        dst: CUdeviceptr,
        stream_id: CudaStreamId,
    ) -> Result<CUdeviceptr, MemoryError> {
        if regions.is_empty() {
            // Nothing to remap; return sentinel so caller's min() picks the other operand
            return Ok(CUdeviceptr::MAX);
        }

        let bytes_to_remap = regions.iter().map(|(_, size)| *size).sum::<usize>();
        tracing::debug!(
            "VPMM: Remapping {} regions. Total size = {}",
            regions.len(),
            ByteSize::b(bytes_to_remap as u64)
        );

        let mut curr_dst = dst;
        for (region_addr, region_size) in regions {
            // Unmap the region
            unsafe {
                vpmm_unmap(region_addr, region_size).map_err(|e| {
                    tracing::error!(
                        "vpmm_unmap failed: addr={:#x}, size={}: {:?}",
                        region_addr, region_size, e
                    );
                    MemoryError::from(e)
                })?;
            }
            self.insert_unmapped_region(region_addr, region_size);

            // Remap the region
            let num_pages = region_size / self.page_size;
            for i in 0..num_pages {
                let page = region_addr + (i * self.page_size) as u64;
                let handle = self
                    .active_pages
                    .remove(&page)
                    .expect("BUG: active page not found during remapping");
                unsafe {
                    vpmm_map(curr_dst, self.page_size, handle).map_err(|e| {
                        tracing::error!(
                            "vpmm_map (remap) failed: dst={:#x}, page_size={}, handle={}: {:?}",
                            curr_dst, self.page_size, handle, e
                        );
                        MemoryError::from(e)
                    })?;
                }
                self.active_pages.insert(curr_dst, handle);
                curr_dst += self.page_size as u64;
            }
        }

        debug_assert_eq!(curr_dst - dst, bytes_to_remap as u64);

        // Set access permissions for the remapped region
        unsafe {
            vpmm_set_access(dst, bytes_to_remap, self.device_id).map_err(|e| {
                tracing::error!(
                    "vpmm_set_access (remap) failed: addr={:#x}, size={}, device={}: {:?}",
                    dst, bytes_to_remap, self.device_id, e
                );
                MemoryError::from(e)
            })?;
        }
        Ok(self.free_region_insert(dst, bytes_to_remap, stream_id))
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
impl std::fmt::Debug for VirtualMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "VMPool (VA_SIZE={}, PAGE_SIZE={})",
            ByteSize::b(self.va_size as u64),
            ByteSize::b(self.page_size as u64)
        )?;

        let reserved = self.roots.len() * self.va_size;
        let allocated = self.memory_usage();
        let free_bytes: usize = self.free_regions.values().map(|r| r.size).sum();
        let malloc_bytes: usize = self.malloc_regions.values().sum();
        let unmapped_bytes: usize = self.unmapped_regions.values().sum();

        writeln!(
            f,
            "Total: reserved={}, allocated={}, free={}, malloc={}, unmapped={}",
            ByteSize::b(reserved as u64),
            ByteSize::b(allocated as u64),
            ByteSize::b(free_bytes as u64),
            ByteSize::b(malloc_bytes as u64),
            ByteSize::b(unmapped_bytes as u64)
        )?;

        let mut regions: Vec<(CUdeviceptr, usize, String)> = Vec::new();
        for (addr, region) in &self.free_regions {
            regions.push((*addr, region.size, format!("free (s {})", region.stream_id)));
        }
        for (addr, size) in &self.malloc_regions {
            regions.push((*addr, *size, "malloc".to_string()));
        }
        for (addr, size) in &self.unmapped_regions {
            regions.push((*addr, *size, "unmapped".to_string()));
        }
        regions.sort_by_key(|(addr, _, _)| *addr);

        write!(f, "Regions: ")?;
        for (_, size, label) in regions.iter() {
            write!(f, "[{} {}]", label, ByteSize::b(*size as u64))?;
        }
        Ok(())
    }
}
