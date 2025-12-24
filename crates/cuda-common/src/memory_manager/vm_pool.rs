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
    stream::{
        current_stream_id, current_stream_sync, default_stream_wait, CudaEvent, CudaStreamId,
    },
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
// Pool Implementation
// ============================================================================

/// Metadata for a free region in the virtual address space.
#[derive(Debug, Clone)]
struct FreeRegionMeta {
    size: usize,
    event: Arc<CudaEvent>,
    stream_id: CudaStreamId,
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

    // Active allocations: (ptr, size)
    malloc_regions: HashMap<CUdeviceptr, usize>,

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
                        va_size > 0 && va_size % page_size == 0,
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
            if let Err(e) =
                pool.defragment_or_create_new_pages(alloc_size, current_stream_id().unwrap())
            {
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
        }

        pool
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

        // Phase 0: Cleanup zombie regions
        self.cleanup_zombie_regions();

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
                    FreeRegionMeta {
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

    /// Reclaims zombie regions whose events have completed.
    ///
    /// Zombie regions are old VAs that have been double-mapped to new locations during defrag.
    /// Once the associated event completes (meaning no GPU work is using the old VA),
    /// we can safely unmap the old VA and return it to `unmapped_regions` for reuse.
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

    /// Phase 1: Try to find a suitable free region without defragmentation
    /// Returns address if found, None otherwise
    fn find_best_fit(&mut self, requested: usize, stream_id: CudaStreamId) -> Option<CUdeviceptr> {
        let mut candidates: Vec<(CUdeviceptr, &mut FreeRegionMeta)> = self
            .free_regions
            .iter_mut()
            .filter(|(_, region)| region.size >= requested)
            .map(|(addr, region)| (*addr, region))
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // 1a. Prefer current stream (smallest fit) - could be not completed
        if let Some((addr, _)) = candidates
            .iter()
            .filter(|(_, region)| region.stream_id == stream_id)
            .min_by_key(|(_, region)| region.size)
        {
            return Some(*addr);
        }

        // 1b. Other streams (smallest fit) - ONLY if already completed (no sync)
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
    ///
    /// # Assumptions
    /// No CUDA streams will use the malloc region starting at `ptr` after the newly recorded event
    /// on `stream_id` at this point in time.
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

    /// Inserts a new free region for `(ptr, size)` into the free regions map, **possibly** merging
    /// with existing adjacent regions. Merges only occur if the regions are both adjacent in memory
    /// and on the same stream as `stream_id`. The new free region always records a new event on the
    /// stream.
    ///
    /// Returns the starting pointer and size of the (possibly merged) free region containing `(ptr,
    /// size)`.
    fn free_region_insert(
        &mut self,
        mut ptr: CUdeviceptr,
        mut size: usize,
        stream_id: CudaStreamId,
    ) -> (CUdeviceptr, usize) {
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
            FreeRegionMeta {
                size,
                event,
                stream_id,
                id,
            },
        );
        (ptr, size)
    }

    /// Return the base address of a virtual hole large enough for `requested` bytes.
    ///
    /// The returned region is not inside `free_region` or `unmapped_region` map and must be
    /// properly handled.
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
    ///
    /// The returned `pointer`, if not `None`, is guaranteed to exist as a key in the `free_regions`
    /// map. The corresponding free region will have size `>= requested`. Note the size _may_ be
    /// larger than requested.
    fn defragment_or_create_new_pages(
        &mut self,
        requested: usize,
        stream_id: CudaStreamId,
    ) -> Result<Option<CUdeviceptr>, MemoryError> {
        debug_assert_eq!(requested % self.page_size, 0);
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
        let mut allocate_size = requested.saturating_sub(total_free_size);
        debug_assert_eq!(allocate_size % self.page_size, 0);
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
            allocated_dst += self.page_size as u64;
        }
        debug_assert_eq!(allocated_dst, dst + allocate_size as u64);
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
                        dst,
                        allocate_size,
                        self.device_id,
                        e
                    );
                    MemoryError::from(e)
                })?;
            }
            let (merged_ptr, merged_size) = self.free_region_insert(dst, allocate_size, stream_id);
            debug_assert!(merged_size >= allocate_size);
            allocated_ptr = merged_ptr;
            allocate_size = merged_size;
        }
        // At this point, allocated_ptr is either
        // - CUdeviceptr::MAX if no allocations occurred
        // - or some pointer `<= dst` for the start of a free region (with VA-mapping) of at least
        //   `requested` bytes. The case `allocated_ptr < dst` happens if `free_region_insert`
        //   merged the allocated region with a previous free region. This only happens if the
        //   previous free region is on the same `stream_id`. In this case, we have
        //   [allocated_ptr..dst][dst..allocated_dst] which is all free and safe to use on the
        //   stream `stream_id` _without_ synchronization, because all events will be sequenced
        //   afterwards on the stream.
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

        // Pull free regions (current stream first, the other stream oldest first) until we've
        // gathered enough pages.
        let mut to_defrag: Vec<(CUdeviceptr, usize)> = Vec::new();
        let mut ordered_free_regions: Vec<_> = self
            .free_regions
            .iter()
            .filter(|(&addr, _)| allocate_size == 0 || addr != allocated_ptr)
            .map(|(&addr, region)| (region.stream_id != stream_id, region.id, addr))
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
                default_stream_wait(&region.event)?;
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
                // Return the unused tail to the free list so it stays available.
                let leftover_addr = addr + take as u64;
                let leftover_size = region.size - take;
                let _ = self.free_region_insert(leftover_addr, leftover_size, region.stream_id);
            }
        }
        let remapped_ptr = self.remap_regions(to_defrag, allocated_dst, stream_id)?;
        // Take the minimum in case allocated_ptr is CUdeviceptr::MAX when allocate_size = 0
        let result = std::cmp::min(allocated_ptr, remapped_ptr);
        debug_assert!(allocate_size == 0 || allocated_ptr == remapped_ptr);
        debug_assert_ne!(
            result,
            CUdeviceptr::MAX,
            "Both allocation and remapping returned no valid free region"
        );
        Ok(Some(result))
    }

    /// Remap a list of regions to a new base address via double-mapping.
    ///
    /// The regions are mapped consecutively starting at `dst`. The old VAs remain mapped
    /// (double-mapped) and are added to `zombie_regions` by the caller for async unmap.
    ///
    /// Returns the starting pointer of the new remapped free region or `CUdeviceptr::MAX` if no
    /// remapping is needed.
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
            // Double-map: map pages to new VA (old VA remains mapped, will be cleaned up as zombie)
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

        // Set access permissions for the remapped region
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
        let (remapped_ptr, _) = self.free_region_insert(dst, bytes_to_remap, stream_id);
        Ok(remapped_ptr)
    }

    /// Returns the total physical memory currently mapped in this pool (in bytes).
    pub(super) fn memory_usage(&self) -> usize {
        self.active_pages.len() * self.page_size
    }
}

impl Drop for VirtualMemoryPool {
    fn drop(&mut self) {
        current_stream_sync().unwrap();

        // Unmap zombie regions first
        for zombie in self.zombie_regions.drain(..) {
            unsafe {
                vpmm_unmap(zombie.ptr, zombie.size).ok();
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
