#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]

use std::{
    cmp::{Ordering, Reverse},
    collections::{BTreeSet, HashMap},
    ffi::c_void,
};

use bytesize::ByteSize;

use super::cuda::*;
use crate::{common::set_device, error::MemoryError};

const INITIAL_POOL_SIZE: usize = 1717986918 + (14 << 20); // 1.6GB 1 << 30; // 1 GB
const VIRTUAL_POOL_SIZE: usize = 128 << 30; // 128 GB

#[derive(Clone, Debug)]
struct VirtualPtr {
    ptr: CUdeviceptr,
    size: usize,
}

impl VirtualPtr {
    fn end_ptr(&self) -> CUdeviceptr {
        self.ptr + self.size as u64
    }
}

impl PartialEq for VirtualPtr {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl Eq for VirtualPtr {}

impl PartialOrd for VirtualPtr {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VirtualPtr {
    fn cmp(&self, other: &Self) -> Ordering {
        self.ptr.cmp(&other.ptr)
    }
}

// Virtual memory pool implementation
pub(super) struct VirtualMemoryPool {
    // Virtual address space
    virtual_base: VirtualPtr,

    // Map for all active pages
    physical_map: HashMap<CUdeviceptr, CUmemGenericAllocationHandle>,

    // Free regions in virtual space (sorted by address)
    free_regions: BTreeSet<VirtualPtr>,

    // Active allocations
    allocations: HashMap<CUdeviceptr, usize>,

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
                virtual_base: VirtualPtr {
                    ptr: va_base,
                    size: 0,
                },
                physical_map: HashMap::new(),
                free_regions: BTreeSet::new(),
                allocations: HashMap::new(),
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
        let new_base = self.virtual_base.end_ptr();

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
                self.physical_map
                    .insert(new_base + (i * self.page_size) as u64, handle);
            }
        }

        // Update virtual base size
        self.virtual_base.size += pages * self.page_size;
        // Insert the new memory into the free regions
        self.free_region_insert(VirtualPtr {
            ptr: new_base,
            size: pages * self.page_size,
        });
        println!(
            "Virtual memory created: requested: {}, total: {}",
            ByteSize::b(memory_size as u64),
            ByteSize::b(self.virtual_base.size as u64)
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
            .filter(|vp| vp.size >= requested)
            .min_by_key(|vp| vp.size)
            .cloned();

        if best_region.is_none() {
            // Try to defragment or/and alloc more physical memory
            best_region = self.defragment_or_create_new_handle(requested)?;
        }

        if let Some(region) = best_region {
            self.free_regions.remove(&region);

            // If region is larger, return remainder to free list
            if region.size > requested {
                self.free_regions.insert(VirtualPtr {
                    ptr: region.ptr + requested as u64,
                    size: region.size - requested,
                });
            }

            self.allocations.insert(region.ptr, requested);
            return Ok(region.ptr as *mut c_void);
        }

        Err(MemoryError::OutOfMemory {
            requested,
            available: self.free_regions.iter().map(|vp| vp.size).sum(),
        })
    }

    /// Free a pointer and return the size of the freed memory
    pub(super) fn free_internal(&mut self, ptr: *mut c_void) -> Result<usize, MemoryError> {
        let ptr = ptr as CUdeviceptr;
        let size = self
            .allocations
            .remove(&ptr)
            .ok_or(MemoryError::InvalidPointer)?;

        self.free_region_insert(VirtualPtr { ptr, size });

        Ok(size)
    }

    fn free_region_insert(&mut self, region: VirtualPtr) {
        let mut region = region.clone();
        // Potential merge with next neighbor
        if let Some(next) = self.free_regions.range(region.clone()..).next().cloned() {
            if region.end_ptr() == next.ptr {
                self.free_regions.remove(&next);
                region.size += next.size;
            }
        }
        // Potential merge with previous neighbor
        if let Some(prev) = self
            .free_regions
            .range(..region.clone())
            .next_back()
            .cloned()
        {
            if prev.end_ptr() == region.ptr {
                self.free_regions.remove(&prev);
                region.ptr = prev.ptr;
                region.size += prev.size;
            }
        }
        self.free_regions.insert(region);
    }

    fn defragment_or_create_new_handle(
        &mut self,
        requested: usize,
    ) -> Result<Option<VirtualPtr>, MemoryError> {
        if self.free_regions.is_empty() {
            self.create_new_handle(requested)?;
            return Ok(self.free_regions.last().cloned());
        }

        let mut to_defrag: Vec<VirtualPtr> = self.free_regions.iter().cloned().collect();
        let mut sum = 0;
        // If last free region is at the end of the virtual address space, it will be used for
        // defragmentation
        if to_defrag
            .last()
            .is_some_and(|vp| vp.end_ptr() == self.virtual_base.end_ptr())
        {
            sum = to_defrag.pop().unwrap().size;
        }

        let mut new_start = self.virtual_base.end_ptr();
        // Biggest first -> less blocks to defragment
        to_defrag.sort_by_key(|vp| Reverse(vp.size));

        for vp in to_defrag {
            if sum >= requested {
                break;
            }
            sum += vp.size;
            self.free_regions.remove(&vp);
            for i in 0..vp.size / self.page_size {
                let page = vp.ptr + (i * self.page_size) as u64;
                // delete this tyle from the physical map
                let handle = self
                    .physical_map
                    .remove(&page)
                    .ok_or(MemoryError::InvalidPointer)?;
                // map the tyle to the new start
                unsafe {
                    vmm_map_and_set_access(new_start, self.page_size, handle, self.device_id)?;
                }
                // add this tyle to the physical map
                self.physical_map.insert(new_start, handle);
                // move new start to the end of the tyle
                new_start += self.page_size as u64;
            }
            // create new free region
            let new_free = VirtualPtr {
                ptr: new_start - vp.size as u64,
                size: vp.size,
            };
            self.free_region_insert(new_free.clone());
        }
        // update virtual base size
        self.virtual_base.size = (new_start - self.virtual_base.ptr) as usize;
        // if there is still memory left, create a new handle
        if sum < requested {
            self.create_new_handle(requested - sum)?;
        }

        Ok(self.free_regions.last().cloned())
    }
}

impl Drop for VirtualMemoryPool {
    fn drop(&mut self) {
        tracing::info!(
            "VirtualMemoryPool: GPU memory used total: {}",
            ByteSize::b((self.physical_map.len() * self.page_size) as u64)
        );
        self.free_regions.clear();
        self.physical_map.clear();
        self.allocations.clear();
        todo!();
    }
}

impl Default for VirtualMemoryPool {
    fn default() -> Self {
        let mut pool = Self::new();
        pool.create_new_handle(INITIAL_POOL_SIZE).unwrap();
        pool
    }
}
