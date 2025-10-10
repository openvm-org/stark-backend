#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]

use std::{
    cmp::Reverse,
    collections::{BTreeMap, HashMap},
    ffi::c_void,
};

use bytesize::ByteSize;

use super::cuda::*;
use crate::{error::MemoryError, stream::current_stream_id};

const VIRTUAL_POOL_SIZE: usize = 1usize << 40; // 1 TB

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
    pub(super) device_id: i32,
}

unsafe impl Send for VirtualMemoryPool {}
unsafe impl Sync for VirtualMemoryPool {}

impl VirtualMemoryPool {
    pub(super) fn new(device_id: i32, page_size: usize) -> Result<Self, MemoryError> {
        let root = if page_size == usize::MAX {
            0
        } else {
            unsafe { vpmm_reserve(VIRTUAL_POOL_SIZE, page_size)? }
        };

        Ok(Self {
            root,
            curr_end: root,
            active_pages: HashMap::new(),
            free_regions: BTreeMap::new(),
            used_regions: HashMap::new(),
            page_size,
            device_id,
        })
    }

    /// Allocates memory from the pool's free regions.
    /// Attempts defragmentation if no suitable free region exists.
    /// Returns an error if there's insufficient memory even after defragmentation.
    pub(super) fn malloc_internal(&mut self, requested: usize) -> Result<*mut c_void, MemoryError> {
        if self.curr_end == self.root {
            return Err(MemoryError::OutOfMemory {
                requested,
                available: 0,
            });
        }
        debug_assert!(
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
            // Try to defragment memory
            best_region = self.defragment(requested)?;
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

    /// Frees a pointer and returns the size of the freed memory.
    /// Coalesces adjacent free regions.
    pub(super) fn free_internal(&mut self, ptr: *mut c_void) -> Result<usize, MemoryError> {
        debug_assert!(self.curr_end != self.root, "VM pool is empty");
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

    /// Defragments the pool by consolidating free regions at the end of virtual address space.
    /// Remaps pages to create one large contiguous free region.
    fn defragment(&mut self, requested: usize) -> Result<Option<CUdeviceptr>, MemoryError> {
        debug_assert!(
            !self.free_regions.is_empty(),
            "No free regions to defragment"
        );

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

        let mut new_end = self.curr_end;
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
                    vpmm_map_and_set_access(new_end, self.page_size, handle, self.device_id)?;
                }
                // add this tyle to the physical map
                self.active_pages.insert(new_end, handle);
                // move new start to the end of the tyle
                new_end += self.page_size as u64;
            }
            // create new free region
            self.free_region_insert(new_end - size as u64, size);
        }
        // update virtual base size
        self.curr_end = new_end;
        // if there is still memory left, return error
        if sum < requested {
            return Err(MemoryError::OutOfMemory {
                requested,
                available: sum,
            });
        }

        Ok(self.free_regions.iter().next_back().map(|(&p, _)| p))
    }

    /// Returns the total physical memory currently mapped in this pool (in bytes).
    pub(super) fn memory_usage(&self) -> usize {
        self.active_pages.len() * self.page_size
    }

    /// Returns true if no allocations are currently active in this pool.
    pub(super) fn is_empty(&self) -> bool {
        self.used_regions.is_empty()
    }

    /// Returns the number of available pages in this pool.
    pub(super) fn available_pages(&self) -> usize {
        self.free_regions.values().sum::<usize>() / self.page_size
    }

    /// Adds pre-allocated page handles to the pool by mapping them to virtual address space.
    /// Pages are added as one contiguous free region at the end of the current address space.
    pub(super) fn add_pages(
        &mut self,
        pages: Vec<CUmemGenericAllocationHandle>,
    ) -> Result<(), MemoryError> {
        if pages.is_empty() {
            return Ok(());
        }
        let new_base = self.curr_end;
        let num_pages = pages.len();

        for (i, handle) in pages.into_iter().enumerate() {
            let va = new_base + (i * self.page_size) as u64;
            unsafe {
                vpmm_map_and_set_access(va, self.page_size, handle, self.device_id)?;
            }
            self.active_pages.insert(va, handle);
        }
        let total_size = num_pages * self.page_size;
        self.curr_end += total_size as u64;
        self.free_region_insert(new_base, total_size);
        tracing::debug!(
            "GPU mem ({}): VM Pool added {} pages, total: {}",
            current_stream_id().unwrap(),
            num_pages,
            ByteSize::b((self.memory_usage()) as u64)
        );
        Ok(())
    }

    /// Resets the pool by unmapping all pages and returning their handles.
    /// The pool must be empty (no active allocations) before calling this.
    pub(super) fn reset(&mut self) -> Result<Vec<CUmemGenericAllocationHandle>, MemoryError> {
        if !self.used_regions.is_empty() {
            return Err(MemoryError::VirtualPoolInUse {
                used: self.used_regions.len(),
            });
        }

        if self.curr_end != self.root {
            unsafe {
                vpmm_unmap(self.root, (self.curr_end - self.root) as usize)?;
            }
        }

        let pages: Vec<_> = self
            .active_pages
            .drain()
            .map(|(_, handle)| handle)
            .collect();

        self.free_regions.clear();
        self.curr_end = self.root;

        tracing::debug!(
            "GPU mem ({}): VM pool reset, returned {} pages",
            current_stream_id().unwrap(),
            pages.len()
        );

        Ok(pages)
    }
}

impl Drop for VirtualMemoryPool {
    fn drop(&mut self) {
        if self.root != self.curr_end {
            tracing::warn!(
                "Dropping VM pool before reset: {} pages in pool",
                self.active_pages.len()
            );
            release_pages(self.reset().unwrap()).unwrap();
        }
        unsafe {
            vpmm_release_va(self.root, VIRTUAL_POOL_SIZE).unwrap();
        }
    }
}
/// Releases physical memory handles by calling cuMemRelease on each.
pub(super) fn release_pages(pages: Vec<CUmemGenericAllocationHandle>) -> Result<(), MemoryError> {
    for handle in pages {
        unsafe {
            vpmm_release(handle)?;
        }
    }
    Ok(())
}

/// Determines the page size for virtual memory management on the specified device.
/// Returns usize::MAX if VPMM is not supported, indicating fallback to cudaMallocAsync only.
pub(super) fn get_page_size(device_id: i32) -> Result<usize, MemoryError> {
    unsafe {
        match vpmm_check_support(device_id) {
            Ok(_) => {
                let gran = vpmm_min_granularity(device_id)?;
                match std::env::var("VPMM_PAGE_SIZE") {
                    Ok(val) => {
                        let custom_size: usize =
                            val.parse().expect("VPMM_PAGE_SIZE must be a valid number");
                        assert!(
                            custom_size > 0 && custom_size % gran == 0,
                            "VPMM_PAGE_SIZE must be > 0 and multiple of {}",
                            gran
                        );
                        Ok(custom_size)
                    }
                    Err(_) => Ok(gran),
                }
            }
            Err(_) => {
                tracing::warn!("VPMM not supported, falling back to cudaMallocAsync");
                Ok(usize::MAX)
            }
        }
    }
}

/// Returns the number of pages to pre-allocate based on VPMM_PAGES environment variable.
/// Returns 0 if not set or if VPMM is not supported.
pub(super) fn get_initial_num_pages(page_size: usize) -> usize {
    if page_size == usize::MAX {
        return 0;
    }
    match std::env::var("VPMM_PAGES") {
        Ok(val) => val.parse().expect("VPMM_PAGES must be a valid number"),
        Err(_) => 0, // Maybe we need some default value here like 50% of free memory
    }
}

/// Creates a new set of physical memory handles by calling cuMemCreate for each page.
/// Returns an empty vector if num_pages is 0 or VPMM is not supported.
pub(super) fn create_new_pages(
    device_id: i32,
    page_size: usize,
    num_pages: usize,
) -> Result<Vec<CUmemGenericAllocationHandle>, MemoryError> {
    if num_pages == 0 || page_size == usize::MAX {
        return Ok(Vec::new());
    }

    let mut pages = Vec::with_capacity(num_pages);
    for _ in 0..num_pages {
        let handle = unsafe { vpmm_create_physical(device_id, page_size)? };
        pages.push(handle);
    }

    tracing::info!(
        "GPU mem: Allocated {} pages ({})",
        num_pages,
        ByteSize::b((num_pages * page_size) as u64)
    );

    Ok(pages)
}
