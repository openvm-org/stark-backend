#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]

use std::{
    collections::{BTreeMap, HashMap},
    ffi::c_void,
};

use super::cuda::*;
use crate::{
    common::set_device,
    error::MemoryError,
    stream::{cudaStream_t, CudaEvent, CudaEventStatus},
};

const INITIAL_POOL_SIZE: usize = 256 << 20; // 256 MB

// Pending free tracking
struct PendingFree {
    ptr: CUdeviceptr,
    size: usize,
    event: CudaEvent,
}

unsafe impl Send for PendingFree {}
unsafe impl Sync for PendingFree {}

// Virtual memory pool implementation
pub(super) struct VirtualMemoryPool {
    // Virtual address space
    va_base: CUdeviceptr,
    va_size: usize,

    // Physical memory handles
    physical_handles: Vec<CUmemGenericAllocationHandle>,

    // Free regions in virtual space (sorted by address)
    free_regions: BTreeMap<CUdeviceptr, usize>,

    // Active allocations
    allocations: HashMap<CUdeviceptr, usize>,

    // Pending frees (async)
    pending_frees: Vec<PendingFree>,
}

unsafe impl Send for VirtualMemoryPool {}
unsafe impl Sync for VirtualMemoryPool {}

impl VirtualMemoryPool {
    pub(super) fn new(initial_size: usize) -> Result<Self, MemoryError> {
        unsafe {
            let device_id = set_device()?;
            vmm_check_support(device_id)?;

            let gran = vmm_min_granularity(device_id)?;

            let va_size = 48usize << 30;
            let va_base = vmm_reserve(va_size, gran)?;

            let initial = (initial_size + gran - 1) / gran * gran;
            let handle = vmm_create_physical(device_id, initial)?;
            vmm_map_set_access(va_base, initial, handle, device_id)?;

            let free_regions = {
                let mut regions = BTreeMap::new();
                regions.insert(va_base, initial_size);
                regions
            };
            println!("Free regions: {:?}", free_regions);

            Ok(Self {
                va_base,
                va_size,
                physical_handles: vec![handle],
                free_regions,
                allocations: HashMap::new(),
                pending_frees: Vec::new(),
            })
        }
    }

    pub(super) fn malloc_internal(&mut self, size: usize) -> Result<CUdeviceptr, MemoryError> {
        // Process any completed frees first
        self.process_pending_frees();

        // Align size to 256 bytes for better performance
        let aligned_size = (size + 255) & !255;

        // Find best fit free region
        let best_region = self
            .free_regions
            .iter()
            .filter(|(_, &size)| size >= aligned_size)
            .min_by_key(|(_, &size)| size)
            .map(|(&ptr, &size)| (ptr, size));

        if let Some((ptr, region_size)) = best_region {
            self.free_regions.remove(&ptr);

            // If region is larger, return remainder to free list
            if region_size > aligned_size {
                self.free_regions
                    .insert(ptr + aligned_size as u64, region_size - aligned_size);
            }

            self.allocations.insert(ptr, aligned_size);
            return Ok(ptr);
        }
        if !self.pending_frees.is_empty() {
            self.process_pending_frees();
            let best_region = self
                .free_regions
                .iter()
                .filter(|(_, &size)| size >= aligned_size)
                .min_by_key(|(_, &size)| size)
                .map(|(&ptr, &size)| (ptr, size));

            if let Some((ptr, region_size)) = best_region {
                self.free_regions.remove(&ptr);
                if region_size > aligned_size {
                    self.free_regions
                        .insert(ptr + aligned_size as u64, region_size - aligned_size);
                }
                self.allocations.insert(ptr, aligned_size);
                return Ok(ptr);
            }
        }

        Err(MemoryError::OutOfMemory {
            requested: size,
            available: self.free_regions.values().sum(),
        })
    }

    pub(super) fn free_internal(
        &mut self,
        ptr: *mut c_void,
        stream: cudaStream_t,
    ) -> Result<(), MemoryError> {
        let ptr = ptr as CUdeviceptr;
        let size = self
            .allocations
            .remove(&ptr)
            .ok_or(MemoryError::InvalidPointer)?;

        unsafe {
            // Create event to track when memory is safe to reuse
            let event = CudaEvent::new()?;
            event.record(stream)?;

            self.pending_frees.push(PendingFree { ptr, size, event });
        }

        Ok(())
    }

    fn process_pending_frees(&mut self) {
        let mut still_pending = Vec::new();

        for pending in self.pending_frees.drain(..) {
            match pending.event.status() {
                CudaEventStatus::Completed => {
                    // Add to free regions and coalesce if possible
                    self.free_regions.insert(pending.ptr, pending.size);
                    let next_ptr = pending.ptr + pending.size as u64;
                    if let Some(next_size) = self.free_regions.remove(&next_ptr) {
                        // Merge with next region
                        *self.free_regions.get_mut(&pending.ptr).unwrap() += next_size;
                    }
                }
                CudaEventStatus::NotReady => {
                    // Still pending
                    still_pending.push(pending);
                }
                CudaEventStatus::Error(e) => {
                    // Error - log and continue
                    tracing::error!("Event query failed: {}", e);
                }
            }
        }

        self.pending_frees = still_pending;
    }
}

impl Default for VirtualMemoryPool {
    fn default() -> Self {
        Self::new(INITIAL_POOL_SIZE).unwrap()
    }
}
