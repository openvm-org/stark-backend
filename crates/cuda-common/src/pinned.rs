//! Page-locked (pinned) host memory for low-latency device readbacks.
//!
//! Small per-round sumcheck readbacks with `cudaMemcpyAsync` from device to
//! *pageable* host memory go through an internal driver staging buffer, adding
//! noticeable latency to every transcript round trip. Copying into pinned
//! memory instead uses a direct DMA. See [`crate::copy::MemCopyD2H::to_pinned_on`].

use std::{ffi::c_void, marker::PhantomData};

use crate::error::{check, CudaError};

extern "C" {
    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaFreeHost(ptr: *mut c_void) -> i32;
}

/// A fixed-capacity page-locked host buffer.
///
/// `T` must be plain old data (any bit pattern valid), as reads may observe
/// device-written bytes.
pub struct PinnedBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

// SAFETY: PinnedBuffer owns its allocation; T: Send bounds inherited.
unsafe impl<T: Send> Send for PinnedBuffer<T> {}

impl<T: Copy> PinnedBuffer<T> {
    /// Allocates a pinned buffer with capacity for `len` elements of `T`.
    /// The contents are uninitialized (arbitrary bytes).
    pub fn with_capacity(len: usize) -> Result<Self, CudaError> {
        assert!(len > 0, "PinnedBuffer requires nonzero capacity");
        let mut ptr: *mut c_void = std::ptr::null_mut();
        check(unsafe { cudaHostAlloc(&mut ptr, len * std::mem::size_of::<T>(), 0) })?;
        Ok(Self {
            ptr: ptr as *mut T,
            len,
            _marker: PhantomData,
        })
    }

    /// Grows the buffer to at least `len` elements (contents are not preserved).
    pub fn ensure_capacity(&mut self, len: usize) -> Result<(), CudaError> {
        if self.len < len {
            *self = Self::with_capacity(len.next_power_of_two())?;
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Views the first `n` elements. Contents are whatever was last written
    /// (by the host or a synchronized device copy); `T` is plain old data so
    /// arbitrary bytes are valid values.
    pub fn as_slice(&self, n: usize) -> &[T] {
        assert!(n <= self.len);
        // SAFETY: allocation is live for `self.len` elements and T is POD.
        unsafe { std::slice::from_raw_parts(self.ptr, n) }
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            cudaFreeHost(self.ptr as *mut c_void);
        }
    }
}
