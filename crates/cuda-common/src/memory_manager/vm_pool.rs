#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]

use std::{
    cmp::{max, Reverse},
    collections::{BTreeMap, HashMap},
    ffi::c_void,
};

use bytesize::ByteSize;

use super::cuda::*;
use crate::{common::set_device, error::MemoryError};

#[link(name = "cudart")]
extern "C" {
    fn cudaMemGetInfo(free_bytes: *mut usize, total_bytes: *mut usize) -> i32;
}

const VIRTUAL_POOL_SIZE: usize = 128 << 30; // 128 GB

/// Virtual memory pool implementation.
pub(super) struct VirtualMemoryPool {
    // Virtual address space root
    root: CUdeviceptr,

    // Current end of active address space
    curr_end: CUdeviceptr,

    // Map for all active pages
    active_pages: HashMap<CUdeviceptr, CUmemGenericAllocationHandle>,

    // Free regions in virtual space (sorted by address)
    free_regions: BTreeMap<CUdeviceptr, usize>,

    // Active allocations
    used_regions: HashMap<CUdeviceptr, usize>,

    // Granularity size: (page % 2MB must be 0)
    pub(super) page_size: usize,

    // Device ordinal
    device_id: i32,
}

unsafe impl Send for VirtualMemoryPool {}
unsafe impl Sync for VirtualMemoryPool {}

impl VirtualMemoryPool {
    pub(super) fn new() -> Self {
        let device_id = set_device().unwrap();
        let (root, page_size) = unsafe {
            match vmm_check_support(device_id) {
                Ok(_) => {
                    let gran = vmm_min_granularity(device_id).unwrap();
                    let page_size = match std::env::var("VMM_PAGE_SIZE") {
                        Ok(val) => {
                            let custom_size: usize =
                                val.parse().expect("VMM_PAGE_SIZE must be a valid number");
                            assert!(
                                custom_size > 0 && custom_size % gran == 0,
                                "VMM_PAGE_SIZE must be > 0 and multiple of {}",
                                gran
                            );
                            custom_size
                        }
                        Err(_) => gran,
                    };
                    let va_base = vmm_reserve(VIRTUAL_POOL_SIZE, page_size).unwrap();
                    (va_base, page_size)
                }
                Err(_) => {
                    tracing::warn!("VMM not supported, falling back to cudaMallocAsync");
                    (0, usize::MAX)
                }
            }
        };

        Self {
            root,
            curr_end: root,
            active_pages: HashMap::new(),
            free_regions: BTreeMap::new(),
            used_regions: HashMap::new(),
            page_size,
            device_id,
        }
    }

    fn create_new_pages(&mut self, memory_size: usize) -> Result<(), MemoryError> {
        if memory_size == 0 {
            return Err(MemoryError::InvalidMemorySize { size: memory_size });
        }
        let pages = memory_size.div_ceil(self.page_size);
        let new_base = self.curr_end;

        for i in 0..pages {
            unsafe {
                // Create new physical memory handle
                let handle = vmm_create_physical(self.device_id, self.page_size)?;
                // Map it to virtual address space & set RW permissions
                vmm_map_and_set_access(
                    new_base + (i * self.page_size) as u64,
                    self.page_size,
                    handle,
                    self.device_id,
                )?;
                // Insert the new memory into the physical map
                self.active_pages
                    .insert(new_base + (i * self.page_size) as u64, handle);
            }
        }

        let total_size = pages * self.page_size;
        // Update virtual base size
        self.curr_end += total_size as u64;
        // Insert the new memory into the free regions
        self.free_region_insert(new_base, total_size);
        tracing::info!(
            "GPU mem: VM Pool allocated: {}, total: {}",
            ByteSize::b(total_size as u64),
            ByteSize::b((self.memory_usage()) as u64)
        );
        Ok(())
    }

    /// Allocate memory and return a pointer to the allocated memory
    pub(super) fn malloc_internal(&mut self, requested: usize) -> Result<*mut c_void, MemoryError> {
        if self.curr_end == self.root {
            return Err(MemoryError::NotInitialized);
        }
        assert!(
            requested != 0 && requested % self.page_size == 0,
            "Requested size must be a multiple of the page size"
        );
        // Try to find best fit free region
        let mut best_region = self
            .free_regions
            .iter()
            .filter(|(_, &size)| size >= requested)
            .min_by_key(|(_, &size)| size)
            .map(|(&ptr, _)| ptr);

        if best_region.is_none() {
            // Try to defragment or/and alloc more physical memory
            best_region = self.defragment_or_create_new_pages(requested)?;
        }

        if let Some(ptr) = best_region {
            let size = self.free_regions.remove(&ptr).unwrap();

            // If region is larger, return remainder to free list
            if size > requested {
                self.free_regions
                    .insert(ptr + requested as u64, size - requested);
            }

            self.used_regions.insert(ptr, requested);
            return Ok(ptr as *mut c_void);
        }

        Err(MemoryError::OutOfMemory {
            requested,
            available: self.free_regions.iter().map(|(_, &size)| size).sum(),
        })
    }

    /// Free a pointer and return the size of the freed memory
    pub(super) fn free_internal(&mut self, ptr: *mut c_void) -> Result<usize, MemoryError> {
        if self.curr_end == self.root {
            return Err(MemoryError::NotInitialized);
        }
        let ptr = ptr as CUdeviceptr;
        let size = self
            .used_regions
            .remove(&ptr)
            .ok_or(MemoryError::InvalidPointer)?;

        self.free_region_insert(ptr, size);

        Ok(size)
    }

    fn free_region_insert(&mut self, mut ptr: CUdeviceptr, mut size: usize) {
        // Potential merge with next neighbor
        if let Some((&next_ptr, &next_size)) = self.free_regions.range(ptr + 1..).next() {
            if ptr + size as u64 == next_ptr {
                self.free_regions.remove(&next_ptr);
                size += next_size;
            }
        }
        // Potential merge with previous neighbor
        if let Some((&prev_ptr, &prev_size)) = self.free_regions.range(..ptr).next_back() {
            if prev_ptr + prev_size as u64 == ptr {
                self.free_regions.remove(&prev_ptr);
                ptr = prev_ptr;
                size += prev_size;
            }
        }
        self.free_regions.insert(ptr, size);
    }

    fn defragment_or_create_new_pages(
        &mut self,
        requested: usize,
    ) -> Result<Option<CUdeviceptr>, MemoryError> {
        if self.free_regions.is_empty() {
            self.create_new_pages(requested)?;
            return Ok(self.free_regions.iter().next_back().map(|(&p, _)| p));
        }

        let mut to_defrag: Vec<(CUdeviceptr, usize)> =
            self.free_regions.iter().map(|(&p, &s)| (p, s)).collect();
        let mut sum = 0;
        // If last free region is at the virtual end, it will be used for defragmentation
        if to_defrag
            .last()
            .is_some_and(|(p, s)| p + *s as u64 == self.curr_end)
        {
            sum = to_defrag.pop().unwrap().1;
        }

        let mut new_start = self.curr_end;
        // Biggest first -> less blocks to defragment
        to_defrag.sort_by_key(|(_, size)| Reverse(*size));

        for (ptr, size) in to_defrag {
            if sum >= requested {
                break;
            }
            sum += size;
            self.free_regions.remove(&ptr);
            for i in 0..size / self.page_size {
                let page = ptr + (i * self.page_size) as u64;
                // delete this tyle from the physical map
                let handle = self
                    .active_pages
                    .remove(&page)
                    .ok_or(MemoryError::InvalidPointer)?;
                // map the tyle to the new start
                unsafe {
                    vmm_map_and_set_access(new_start, self.page_size, handle, self.device_id)?;
                }
                // add this tyle to the physical map
                self.active_pages.insert(new_start, handle);
                // move new start to the end of the tyle
                new_start += self.page_size as u64;
            }
            // create new free region
            self.free_region_insert(new_start - size as u64, size);
        }
        // update virtual base size
        self.curr_end = new_start;
        // if there is still memory left, create a new handle
        if sum < requested {
            self.create_new_pages(requested - sum)?;
        }

        Ok(self.free_regions.iter().next_back().map(|(&p, _)| p))
    }

    pub(super) fn memory_usage(&self) -> usize {
        self.active_pages.len() * self.page_size
    }
}

impl Drop for VirtualMemoryPool {
    fn drop(&mut self) {
        tracing::info!(
            "VirtualMemoryPool: GPU memory used total: {}",
            ByteSize::b((self.memory_usage()) as u64)
        );
        unsafe {
            // Unmap and release all pages
            for (&va, &handle) in &self.active_pages {
                vmm_unmap_release(va, self.page_size, handle).unwrap();
            }

            // Free the virtual address reservation
            vmm_release_va(self.root, VIRTUAL_POOL_SIZE).unwrap();
        }
    }
}

impl Default for VirtualMemoryPool {
    fn default() -> Self {
        let mut pool = Self::new();

        // Skip allocation if VMM not supported
        if pool.page_size == usize::MAX {
            return pool;
        }

        // Check how much memory is available
        let mut free_mem = 0usize;
        let mut total_mem = 0usize;
        unsafe {
            cudaMemGetInfo(&mut free_mem, &mut total_mem);
        }

        // Calculate initial number of pages to allocate
        let initial_pages = match std::env::var("VMM_PAGES") {
            Ok(val) => {
                let pages: usize = val.parse().expect("VMM_PAGES must be a valid number");
                assert!(pages > 0, "VMM_PAGES must be > 0");
                pages
            }
            Err(_) => {
                // Default: Use 80% of free memory divided by CPU count
                let cpu_count = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1);
                let per_process = (free_mem * 4 / 5) / cpu_count;
                // Convert to pages, minimum 256 pages (512MB at 2MB pages)
                max(256, per_process / pool.page_size)
            }
        };
        if let Err(e) = pool.create_new_pages(initial_pages * pool.page_size) {
            panic!(
                "Error:{:?}\nPool: pages={}, page_size={}\nMemory: free_mem={}, total_mem={}",
                e, initial_pages, pool.page_size, free_mem, total_mem
            );
        }
        pool
    }
}
