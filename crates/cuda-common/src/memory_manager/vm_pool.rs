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
    stream::{device_synchronize, CudaEvent, CudaStream, StreamGuard},
};

#[link(name = "cudart")]
extern "C" {
    fn cudaMemGetInfo(free_bytes: *mut usize, total_bytes: *mut usize) -> i32;
}

// ============================================================================
// Configuration
// ============================================================================

const DEFAULT_VA_SIZE: usize = 8 << 40; // 8 TB

/// Configuration for the Virtual Memory Pool.
///
/// Use `VpmmConfig::from_env()` to load from environment variables,
/// or construct directly for testing.
#[derive(Debug, Clone)]
pub(super) struct VpmmConfig {
    /// Page size override. If `None`, uses CUDA's minimum granularity for the device.
    pub page_size: Option<usize>,
    /// Virtual address space size per reserved chunk (default: 8 TB).
    pub va_size: usize,
    /// Number of pages to preallocate at startup (default: 0).
    pub initial_pages: usize,
}

impl Default for VpmmConfig {
    fn default() -> Self {
        Self {
            page_size: None,
            va_size: DEFAULT_VA_SIZE,
            initial_pages: 0,
        }
    }
}

impl VpmmConfig {
    /// Load configuration from environment variables:
    /// - `VPMM_PAGE_SIZE`: Page size in bytes (must be multiple of CUDA granularity)
    /// - `VPMM_VA_SIZE`: Virtual address space size per chunk (default: 8 TB)
    /// - `VPMM_PAGES`: Number of pages to preallocate (default: 0)
    pub fn from_env() -> Self {
        let page_size = std::env::var("VPMM_PAGE_SIZE").ok().map(|val| {
            let size: usize = val.parse().expect("VPMM_PAGE_SIZE must be a valid number");
            assert!(size > 0, "VPMM_PAGE_SIZE must be > 0");
            size
        });

        let va_size = match std::env::var("VPMM_VA_SIZE") {
            Ok(val) => {
                let size: usize = val.parse().expect("VPMM_VA_SIZE must be a valid number");
                assert!(size > 0, "VPMM_VA_SIZE must be > 0");
                size
            }
            Err(_) => DEFAULT_VA_SIZE,
        };

        let initial_pages = match std::env::var("VPMM_PAGES") {
            Ok(val) => val.parse().expect("VPMM_PAGES must be a valid number"),
            Err(_) => 0,
        };

        Self {
            page_size,
            va_size,
            initial_pages,
        }
    }
}

// ============================================================================
// VPMM allocation records
// ============================================================================

/// Allocation record for the VPMM path.
pub(super) struct VpmmRecord {
    size: usize,
    stream: StreamGuard,
}

impl VpmmRecord {
    pub(super) fn size(&self) -> usize {
        self.size
    }
}

// ============================================================================
// Pool Implementation
// ============================================================================

/// Metadata for a free region in the virtual address space.
#[derive(Debug, Clone)]
struct FreeRegionMeta {
    size: usize,
    event: Arc<CudaEvent>,
    stream: StreamGuard,
    id: usize,
}

/// Remapped region that will be unmapped when the event completes.
#[derive(Debug)]
struct ZombieRegion {
    ptr: CUdeviceptr,
    size: usize,
    event: Arc<CudaEvent>,
}

/// Virtual memory pool implementation.
pub(super) struct VirtualMemoryPool {
    // Virtual address space roots
    pub(super) roots: Vec<CUdeviceptr>,

    // Map for all active pages
    active_pages: HashMap<CUdeviceptr, CUmemGenericAllocationHandle>,

    // Free regions in virtual space (sorted by address)
    free_regions: BTreeMap<CUdeviceptr, FreeRegionMeta>,

    // Active allocations: (ptr, record)
    malloc_regions: HashMap<CUdeviceptr, VpmmRecord>,

    // Unmapped regions: (ptr, size)
    unmapped_regions: BTreeMap<CUdeviceptr, usize>,

    // Zombie regions: (remapped, but not unmapped yet)
    zombie_regions: Vec<ZombieRegion>,

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
    pub(super) fn new(config: VpmmConfig) -> Self {
        let device_id = set_device().unwrap();

        // Check VPMM support and resolve page_size
        let (root, page_size, va_size) = unsafe {
            match vpmm_check_support(device_id) {
                Ok(_) => {
                    let granularity = vpmm_min_granularity(device_id).unwrap();

                    // Resolve page_size: use config override or device granularity
                    let page_size = match config.page_size {
                        Some(size) => {
                            assert!(
                                size > 0 && size % granularity == 0,
                                "VPMM_PAGE_SIZE must be > 0 and multiple of {}",
                                granularity
                            );
                            size
                        }
                        None => granularity,
                    };

                    // Validate va_size
                    let va_size = config.va_size;
                    assert!(
                        va_size > 0 && va_size.is_multiple_of(page_size),
                        "VPMM_VA_SIZE must be > 0 and multiple of page size ({})",
                        page_size
                    );

                    // Reserve initial VA chunk
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

        let mut pool = Self {
            roots: vec![root],
            active_pages: HashMap::new(),
            free_regions: BTreeMap::new(),
            malloc_regions: HashMap::new(),
            unmapped_regions: if va_size > 0 {
                BTreeMap::from_iter([(root, va_size)])
            } else {
                BTreeMap::new()
            },
            zombie_regions: Vec::new(),
            free_num: 0,
            page_size,
            va_size,
            device_id,
        };

        // Preallocate pages if requested (skip if VPMM not supported)
        if config.initial_pages > 0 && page_size != usize::MAX {
            let alloc_size = config.initial_pages * page_size;
            let init_stream = StreamGuard::new(CudaStream::new_non_blocking().unwrap());
            if let Err(e) = pool.defragment_or_create_new_pages(alloc_size, &init_stream) {
                let mut free_mem = 0usize;
                let mut total_mem = 0usize;
                unsafe {
                    cudaMemGetInfo(&mut free_mem, &mut total_mem);
                }
                panic!(
                    "VPMM preallocation failed: {:?}\n\
                     Config: pages={}, page_size={}\n\
                     GPU Memory: free={}, total={}",
                    e,
                    config.initial_pages,
                    ByteSize::b(page_size as u64),
                    ByteSize::b(free_mem as u64),
                    ByteSize::b(total_mem as u64)
                );
            }
            // Ensure the event recorded during pre-allocation is completed before
            // any caller uses the pool. Without this, find_best_fit's Phase 1b
            // may see the event as not-yet-completed (race with GPU processing)
            // and fall through to defragment_or_create_new_pages, remapping the
            // pre-allocated pages to a different VA location.
            init_stream.synchronize().expect("init_stream sync failed");
        }

        pool
    }

    // ========================================================================
    // Tracked (explicit-stream) path
    // ========================================================================

    /// Allocates memory from the pool's free regions.
    pub(super) fn malloc_internal(
        &mut self,
        requested: usize,
        stream: &StreamGuard,
    ) -> Result<*mut c_void, MemoryError> {
        debug_assert!(
            requested != 0 && requested.is_multiple_of(self.page_size),
            "Requested size must be a multiple of the page size"
        );

        self.cleanup_zombie_regions();

        let mut best_region = self.find_best_fit(requested, stream);

        if best_region.is_none() {
            best_region = self.defragment_or_create_new_pages(requested, stream)?;
        }

        if let Some(ptr) = best_region {
            let region = self
                .free_regions
                .remove(&ptr)
                .expect("BUG: free region address not found after find_best_fit");

            if region.size > requested {
                self.reinsert_split_free_region(
                    ptr + requested as u64,
                    region.size - requested,
                    region,
                );
            }

            self.malloc_regions.insert(
                ptr,
                VpmmRecord {
                    size: requested,
                    stream: stream.clone(),
                },
            );
            return Ok(ptr as *mut c_void);
        }

        Err(MemoryError::OutOfMemory {
            requested,
            available: self.free_regions.values().map(|r| r.size).sum(),
        })
    }

    /// Phase 1 best-fit: prefer same-stream regions via `Arc::ptr_eq`.
    fn find_best_fit(&mut self, requested: usize, stream: &StreamGuard) -> Option<CUdeviceptr> {
        let mut candidates: Vec<(CUdeviceptr, &mut FreeRegionMeta)> = self
            .free_regions
            .iter_mut()
            .filter(|(_, region)| region.size >= requested)
            .map(|(addr, region)| (*addr, region))
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // 1a. Prefer same stream (smallest fit) — could be not completed
        if let Some((addr, _)) = candidates
            .iter()
            .filter(|(_, region)| region.stream.is_same_stream(stream))
            .min_by_key(|(_, region)| region.size)
        {
            return Some(*addr);
        }

        // 1b. Other streams (smallest fit) — ONLY if already completed (no sync)
        candidates
            .iter_mut()
            .filter(|(_, region)| region.event.completed())
            .min_by_key(|(_, region)| region.size)
            .map(|(addr, region)| {
                region.stream = stream.clone();
                *addr
            })
    }

    /// Frees a pointer and returns `(size, StreamGuard)`.
    /// The `StreamGuard` must be dropped by the caller AFTER releasing the memory manager lock.
    pub(super) fn free_internal(
        &mut self,
        ptr: *mut c_void,
    ) -> Result<(usize, StreamGuard), MemoryError> {
        let ptr = ptr as CUdeviceptr;
        let record = self
            .malloc_regions
            .remove(&ptr)
            .ok_or(MemoryError::InvalidPointer)?;

        let size = record.size();
        self.free_region_insert(ptr, size, &record.stream);

        Ok((size, record.stream))
    }

    // ========================================================================
    // Shared helpers
    // ========================================================================

    /// Reclaims zombie regions whose events have completed.
    fn cleanup_zombie_regions(&mut self) {
        let mut i = 0;
        while i < self.zombie_regions.len() {
            if self.zombie_regions[i].event.completed() {
                let zombie = self.zombie_regions.swap_remove(i);
                if let Err(e) = unsafe { vpmm_unmap(zombie.ptr, zombie.size) } {
                    tracing::error!(
                        "vpmm_unmap (zombie) failed: addr={:#x}, size={}: {:?}",
                        zombie.ptr,
                        zombie.size,
                        e
                    );
                }
                self.insert_unmapped_region(zombie.ptr, zombie.size);
            } else {
                i += 1;
            }
        }
    }

    /// Inserts a new free region, possibly merging with adjacent same-stream regions.
    fn free_region_insert(
        &mut self,
        mut ptr: CUdeviceptr,
        mut size: usize,
        stream: &StreamGuard,
    ) -> (CUdeviceptr, usize) {
        // Potential merge with next neighbor
        if let Some((&next_ptr, next_region)) = self.free_regions.range(ptr + 1..).next() {
            if next_region.stream.is_same_stream(stream) && ptr + size as u64 == next_ptr {
                let next_region = self.free_regions.remove(&next_ptr).unwrap();
                size += next_region.size;
            }
        }
        // Potential merge with previous neighbor
        if let Some((&prev_ptr, prev_region)) = self.free_regions.range(..ptr).next_back() {
            if prev_region.stream.is_same_stream(stream)
                && prev_ptr + prev_region.size as u64 == ptr
            {
                let prev_region = self.free_regions.remove(&prev_ptr).unwrap();
                ptr = prev_ptr;
                size += prev_region.size;
            }
        }
        let event = Arc::new(CudaEvent::new().unwrap());
        event.record_on(stream).unwrap();
        let id = self.free_num;
        self.free_num += 1;

        self.free_regions.insert(
            ptr,
            FreeRegionMeta {
                size,
                event,
                stream: stream.clone(),
                id,
            },
        );
        (ptr, size)
    }

    /// Reinsert the untouched tail of a split free region without recording a new event.
    fn reinsert_split_free_region(
        &mut self,
        ptr: CUdeviceptr,
        size: usize,
        region: FreeRegionMeta,
    ) {
        debug_assert!(size > 0);
        self.free_regions
            .insert(ptr, FreeRegionMeta { size, ..region });
    }

    /// Return the base address of a virtual hole large enough for `requested` bytes.
    fn take_unmapped_region(&mut self, requested: usize) -> Result<CUdeviceptr, MemoryError> {
        debug_assert!(requested != 0);
        debug_assert_eq!(requested % self.page_size, 0);

        if requested > self.va_size {
            return Err(MemoryError::RequestedExceedsVaChunk {
                requested,
                va_size: self.va_size,
            });
        }

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

    /// Roll back partially allocated pages when allocation fails.
    fn rollback_new_pages(
        &mut self,
        reserved_ptr: CUdeviceptr,
        reserved_size: usize,
        allocated_pages: &[(CUdeviceptr, CUmemGenericAllocationHandle)],
    ) {
        for (addr, handle) in allocated_pages {
            if let Err(e) = unsafe { vpmm_unmap(*addr, self.page_size) } {
                tracing::error!(
                    "rollback: vpmm_unmap failed: addr={:#x}, size={}: {:?}",
                    addr,
                    self.page_size,
                    e
                );
            }
            self.active_pages.remove(addr);
            if let Err(e) = unsafe { vpmm_release(*handle) } {
                tracing::error!("rollback: vpmm_release failed: handle={}: {:?}", handle, e);
            }
        }
        self.insert_unmapped_region(reserved_ptr, reserved_size);
    }

    // ========================================================================
    // Defragmentation
    // ========================================================================

    fn defragment_or_create_new_pages(
        &mut self,
        requested: usize,
        stream: &StreamGuard,
    ) -> Result<Option<CUdeviceptr>, MemoryError> {
        debug_assert_eq!(requested % self.page_size, 0);
        if requested == 0 {
            return Ok(None);
        }

        let total_free_size = self.free_regions.values().map(|r| r.size).sum::<usize>();
        tracing::debug!(
            "VPMM: Defragging or creating new pages: requested={}, free={}",
            ByteSize::b(requested as u64),
            ByteSize::b(total_free_size as u64),
        );

        let dst = self.take_unmapped_region(requested)?;
        let mut allocated_ptr = CUdeviceptr::MAX;

        let mut allocated_dst = dst;
        let mut allocate_size = requested.saturating_sub(total_free_size);
        debug_assert_eq!(allocate_size % self.page_size, 0);
        let mut allocated_pages: Vec<(CUdeviceptr, CUmemGenericAllocationHandle)> = Vec::new();
        while allocated_dst < dst + allocate_size as u64 {
            let handle = unsafe {
                match vpmm_create_physical(self.device_id, self.page_size) {
                    Ok(handle) => handle,
                    Err(e) => {
                        tracing::error!(
                            "vpmm_create_physical failed: device={}, page_size={}: {:?}",
                            self.device_id,
                            self.page_size,
                            e
                        );
                        if e.is_out_of_memory() {
                            self.rollback_new_pages(dst, requested, &allocated_pages);
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
                        allocated_dst,
                        self.page_size,
                        handle,
                        e
                    );
                    MemoryError::from(e)
                })?;
            }
            self.active_pages.insert(allocated_dst, handle);
            allocated_pages.push((allocated_dst, handle));
            allocated_dst += self.page_size as u64;
        }
        debug_assert_eq!(allocated_dst, dst + allocate_size as u64);
        if allocate_size > 0 {
            tracing::debug!(
                "VPMM: Allocated {} bytes. Total allocated: {}",
                ByteSize::b(allocate_size as u64),
                ByteSize::b(self.memory_usage() as u64)
            );
            unsafe {
                vpmm_set_access(dst, allocate_size, self.device_id).map_err(|e| {
                    tracing::error!(
                        "vpmm_set_access failed: addr={:#x}, size={}, device={}: {:?}",
                        dst,
                        allocate_size,
                        self.device_id,
                        e
                    );
                    MemoryError::from(e)
                })?;
            }
            let (merged_ptr, merged_size) = self.free_region_insert(dst, allocate_size, stream);
            debug_assert!(merged_size >= allocate_size);
            allocated_ptr = merged_ptr;
            allocate_size = merged_size;
        }

        let mut remaining = requested.saturating_sub(allocate_size);
        if remaining == 0 {
            debug_assert_ne!(
                allocated_ptr,
                CUdeviceptr::MAX,
                "Allocation returned no valid free region"
            );
            return Ok(Some(allocated_ptr));
        }
        debug_assert!(allocate_size == 0 || allocated_ptr <= dst);

        // Pull free regions; prefer same stream, then oldest-first for other streams
        let mut to_defrag: Vec<(CUdeviceptr, usize)> = Vec::new();
        let mut ordered_free_regions: Vec<_> = self
            .free_regions
            .iter()
            .filter(|(&addr, _)| allocate_size == 0 || addr != allocated_ptr)
            .map(|(&addr, region)| (!region.stream.is_same_stream(stream), region.id, addr))
            .collect();
        ordered_free_regions.sort_by_key(|(is_other, id, _)| (*is_other, *id));
        for (other_stream, _, addr) in ordered_free_regions {
            if remaining == 0 {
                break;
            }

            let region = self
                .free_regions
                .remove(&addr)
                .expect("BUG: free region disappeared");

            if other_stream && !region.event.completed() {
                // Make the caller's stream wait on the cross-stream event
                stream.wait(&region.event)?;
            }

            let take = remaining.min(region.size);

            self.zombie_regions.push(ZombieRegion {
                ptr: addr,
                size: take,
                event: region.event.clone(),
            });

            to_defrag.push((addr, take));
            remaining -= take;

            if region.size > take {
                let leftover_addr = addr + take as u64;
                let leftover_size = region.size - take;
                self.reinsert_split_free_region(leftover_addr, leftover_size, region);
            }
        }
        let remapped_ptr = self.remap_regions(to_defrag, allocated_dst, stream)?;
        let result = std::cmp::min(allocated_ptr, remapped_ptr);
        debug_assert!(allocate_size == 0 || allocated_ptr == remapped_ptr);
        debug_assert_ne!(
            result,
            CUdeviceptr::MAX,
            "Both allocation and remapping returned no valid free region"
        );
        Ok(Some(result))
    }

    fn remap_regions(
        &mut self,
        regions: Vec<(CUdeviceptr, usize)>,
        dst: CUdeviceptr,
        stream: &StreamGuard,
    ) -> Result<CUdeviceptr, MemoryError> {
        if regions.is_empty() {
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
                            curr_dst,
                            self.page_size,
                            handle,
                            e
                        );
                        MemoryError::from(e)
                    })?;
                }
                self.active_pages.insert(curr_dst, handle);
                curr_dst += self.page_size as u64;
            }
        }

        debug_assert_eq!(curr_dst - dst, bytes_to_remap as u64);

        unsafe {
            vpmm_set_access(dst, bytes_to_remap, self.device_id).map_err(|e| {
                tracing::error!(
                    "vpmm_set_access (remap) failed: addr={:#x}, size={}, device={}: {:?}",
                    dst,
                    bytes_to_remap,
                    self.device_id,
                    e
                );
                MemoryError::from(e)
            })?;
        }
        let (remapped_ptr, _) = self.free_region_insert(dst, bytes_to_remap, stream);
        Ok(remapped_ptr)
    }

    /// Returns the total physical memory currently mapped in this pool (in bytes).
    pub(super) fn memory_usage(&self) -> usize {
        self.active_pages.len() * self.page_size
    }
}

impl Drop for VirtualMemoryPool {
    fn drop(&mut self) {
        device_synchronize().unwrap();

        // Unmap zombie regions first
        for zombie in self.zombie_regions.drain(..) {
            unsafe {
                vpmm_unmap(zombie.ptr, zombie.size).unwrap();
            }
        }

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
        Self::new(VpmmConfig::from_env())
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
        let malloc_bytes: usize = self.malloc_regions.values().map(|r| r.size()).sum();
        let unmapped_bytes: usize = self.unmapped_regions.values().sum();
        let zombies_bytes: usize = self.zombie_regions.iter().map(|r| r.size).sum();

        writeln!(
            f,
            "Total: reserved={}, allocated={}, free={}, malloc={}, unmapped={}, (zombies={})",
            ByteSize::b(reserved as u64),
            ByteSize::b(allocated as u64),
            ByteSize::b(free_bytes as u64),
            ByteSize::b(malloc_bytes as u64),
            ByteSize::b(unmapped_bytes as u64),
            ByteSize::b(zombies_bytes as u64),
        )?;

        let mut regions: Vec<(CUdeviceptr, usize, String)> = Vec::new();
        for (addr, region) in &self.free_regions {
            regions.push((
                *addr,
                region.size,
                format!("free ({:?})", region.stream.as_raw()),
            ));
        }
        for (addr, record) in &self.malloc_regions {
            regions.push((*addr, record.size(), "malloc".to_string()));
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{FreeRegionMeta, VirtualMemoryPool, VpmmConfig};
    use crate::stream::{CudaEvent, CudaStream, StreamGuard};

    fn test_stream() -> StreamGuard {
        StreamGuard::new(CudaStream::new_non_blocking().unwrap())
    }

    #[test]
    fn test_defrag_leftover_preserves_original_metadata() {
        let config = VpmmConfig {
            page_size: None,
            va_size: 1 << 30,
            initial_pages: 2,
        };
        let mut pool = VirtualMemoryPool::new(config);

        if pool.page_size == usize::MAX {
            println!("VPMM not supported, skipping test");
            return;
        }

        let page_size = pool.page_size;
        let stream = test_stream();
        let foreign_stream = test_stream();

        let ptr = pool.malloc_internal(2 * page_size, &stream).unwrap();
        pool.free_internal(ptr).unwrap();

        let (&free_addr, region) = pool.free_regions.iter_mut().next().unwrap();
        let original_event = Arc::new(CudaEvent::new().unwrap());
        original_event.record_on(&foreign_stream).unwrap();
        *region = FreeRegionMeta {
            size: region.size,
            event: original_event.clone(),
            stream: foreign_stream.clone(),
            id: 42,
        };

        let leftover_addr = free_addr + page_size as u64;
        pool.defragment_or_create_new_pages(page_size, &stream)
            .unwrap()
            .unwrap();

        let leftover = pool.free_regions.get(&leftover_addr).unwrap();
        assert_eq!(leftover.size, page_size);
        assert!(leftover.stream.is_same_stream(&foreign_stream));
        assert_eq!(leftover.id, 42);
        assert!(Arc::ptr_eq(&leftover.event, &original_event));
    }
}
