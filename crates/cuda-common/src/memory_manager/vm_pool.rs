#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]

use std::{
    cmp::Reverse,
    collections::{BTreeMap, HashMap},
    ffi::c_void,
};

use bytesize::ByteSize;

use super::cuda::*;
use crate::{common::set_device, error::MemoryError};

const INITIAL_POOL_SIZE: usize = 1717986918 + (14 << 20); // 1.6GB 1 << 30; // 1 GB
const VIRTUAL_POOL_SIZE: usize = 128 << 30; // 128 GB

/// Virtual memory pool implementation.
pub(super) struct VirtualMemoryPool {
    // Virtual address space root
    #[allow(dead_code)]
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
    fn new() -> Self {
        let device_id = set_device().unwrap();
        unsafe {
            vmm_check_support(device_id).expect("VMM not supported");

            let gran = vmm_min_granularity(device_id).unwrap();

            let va_base = vmm_reserve(VIRTUAL_POOL_SIZE, gran).unwrap();

            Self {
                root: va_base,
                curr_end: va_base,
                active_pages: HashMap::new(),
                free_regions: BTreeMap::new(),
                used_regions: HashMap::new(),
                page_size: gran,
                device_id,
            }
        }
    }

    fn create_new_handle(&mut self, memory_size: usize) -> Result<(), MemoryError> {
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
            "VM Pool allocated {} GPU Memory. Total: {}",
            ByteSize::b(total_size as u64),
            ByteSize::b((self.active_pages.len() * self.page_size) as u64)
        );
        Ok(())
    }

    /// Allocate memory and return a pointer to the allocated memory
    pub(super) fn malloc_internal(&mut self, requested: usize) -> Result<*mut c_void, MemoryError> {
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
            best_region = self.defragment_or_create_new_handle(requested)?;
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

    fn defragment_or_create_new_handle(
        &mut self,
        requested: usize,
    ) -> Result<Option<CUdeviceptr>, MemoryError> {
        if self.free_regions.is_empty() {
            self.create_new_handle(requested)?;
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
            self.create_new_handle(requested - sum)?;
        }

        Ok(self.free_regions.iter().next_back().map(|(&p, _)| p))
    }
}

impl Drop for VirtualMemoryPool {
    fn drop(&mut self) {
        tracing::info!(
            "VirtualMemoryPool: GPU memory used total: {}",
            ByteSize::b((self.active_pages.len() * self.page_size) as u64)
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
        pool.create_new_handle(INITIAL_POOL_SIZE).unwrap();
        pool
    }
}
