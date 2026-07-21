use std::{
    ffi::c_void,
    fmt::Debug,
    ops::{Bound, RangeBounds},
    ptr,
};

use crate::{
    copy::cuda_memcpy_on,
    error::{check, CudaError, MemCopyError},
    memory_manager::{d_free, d_malloc_on},
    stream::{cudaStream_t, GpuDeviceCtx},
};

#[link(name = "cudart")]
extern "C" {
    pub fn cudaMemsetAsync(dst: *mut c_void, value: i32, count: usize, stream: cudaStream_t)
        -> i32;
}

/// Struct that owns a buffer allocated on GPU device. The struct only holds the raw pointer and
/// length, but this struct has a `Drop` implementation which frees the associated device memory.
#[repr(C)]
pub struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
    #[cfg(feature = "debug-cuda-stream")]
    alloc_stream: cudaStream_t,
}

/// A struct that packs a pointer with a size in bytes to pass on CUDA.
/// It holds `*const c_void` for being a universal simple type that can be read by CUDA.
/// Since it is hard to enforce immutability preservation, it just holds `*const`,
/// but has two separate constructors for more robustness from the usage perspective.
/// This is essentially a [DeviceBuffer] but without owning the data.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DeviceBufferView {
    pub ptr: *const c_void,
    pub size: usize,
}

unsafe impl<T> Send for DeviceBuffer<T> {}
unsafe impl<T> Sync for DeviceBuffer<T> {}

impl<T> DeviceBuffer<T> {
    /// Creates an "empty" DeviceBuffer with a null pointer and zero length.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        DeviceBuffer {
            ptr: ptr::null_mut(),
            len: 0,
            #[cfg(feature = "debug-cuda-stream")]
            alloc_stream: std::ptr::null_mut(),
        }
    }

    /// # Safety
    /// - The caller must ensure that the pointer `ptr` is valid for `len` elements of type `T` in
    ///   device memory.
    /// - Dropping the constructed buffer will attempt to free the memory. As such, `ptr` must
    ///   either have been allocated by the internal memory manager (VPMM) or the caller must use
    ///   `ManuallyDrop` to prevent double-free.
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        DeviceBuffer {
            ptr,
            len,
            #[cfg(feature = "debug-cuda-stream")]
            alloc_stream: std::ptr::null_mut(),
        }
    }

    /// Allocate device memory for `len` elements of type `T` on an explicit stream.
    pub fn with_capacity_on(len: usize, device_ctx: &GpuDeviceCtx) -> Self {
        tracing::debug!(
            "Creating device buffer of size {} (sizeof type = {}) on stream {:?}",
            len,
            size_of::<T>(),
            device_ctx.stream
        );
        assert_ne!(len, 0, "Zero capacity request is wrong");
        let size_bytes = std::mem::size_of::<T>() * len;
        let raw_ptr = d_malloc_on(size_bytes, &device_ctx.stream).expect("GPU allocation failed");
        #[cfg(feature = "touchemall")]
        {
            unsafe {
                cudaMemsetAsync(raw_ptr, 0xff, size_bytes, device_ctx.stream.as_raw());
            }
        }
        let typed_ptr = raw_ptr as *mut T;

        DeviceBuffer {
            ptr: typed_ptr,
            len,
            #[cfg(feature = "debug-cuda-stream")]
            alloc_stream: device_ctx.stream.as_raw(),
        }
    }

    /// Fills the buffer with zeros on an explicit stream.
    ///
    /// The caller should use the same stream that allocated this buffer.
    /// `fill_zero` is async; same-stream guarantees ordering without explicit sync.
    pub fn fill_zero_on(&self, device_ctx: &GpuDeviceCtx) -> Result<(), CudaError> {
        if self.len == 0 {
            return Ok(());
        }
        #[cfg(feature = "debug-cuda-stream")]
        debug_assert_eq!(
            device_ctx.stream.as_raw(),
            self.alloc_stream,
            "fill_zero_on: stream mismatch"
        );
        let size_bytes = std::mem::size_of::<T>() * self.len;
        check(unsafe {
            cudaMemsetAsync(
                self.as_mut_raw_ptr(),
                0,
                size_bytes,
                device_ctx.stream.as_raw(),
            )
        })
    }

    /// Fills a suffix of the buffer with zeros on an explicit stream.
    pub fn fill_zero_suffix_on(
        &self,
        start_idx: usize,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<(), CudaError> {
        assert!(
            start_idx <= self.len,
            "start index has to be smaller than or equal to length"
        );
        if start_idx == self.len {
            return Ok(());
        }
        #[cfg(feature = "debug-cuda-stream")]
        debug_assert_eq!(
            device_ctx.stream.as_raw(),
            self.alloc_stream,
            "fill_zero_suffix_on: stream mismatch"
        );
        let size_bytes = std::mem::size_of::<T>() * (self.len - start_idx);
        check(unsafe {
            cudaMemsetAsync(
                self.as_mut_ptr().add(start_idx) as *mut c_void,
                0,
                size_bytes,
                device_ctx.stream.as_raw(),
            )
        })
    }

    /// Returns the number of elements in this buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the buffer is empty (null pointer or zero length).
    pub fn is_empty(&self) -> bool {
        self.len == 0 || self.ptr.is_null()
    }

    /// Returns a raw mutable pointer to the device data (typed).
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Returns a raw const pointer to the device data (typed).
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// Returns a `*mut c_void` (untyped) pointer.
    pub fn as_mut_raw_ptr(&self) -> *mut c_void {
        self.ptr as *mut c_void
    }

    /// Returns a `*const c_void` (untyped) pointer.
    pub fn as_raw_ptr(&self) -> *const c_void {
        self.ptr as *const c_void
    }

    /// Converts the buffer to a buffer of different type.
    /// `T` must be composable of `U`s.
    pub fn as_buffer<U>(mut self) -> DeviceBuffer<U> {
        assert_eq!(
            size_of::<T>() % size_of::<U>(),
            0,
            "the underlying type size must divide the former one"
        );
        assert_eq!(
            align_of::<T>() % align_of::<U>(),
            0,
            "the underlying type alignment must divide the former one"
        );
        let res = DeviceBuffer {
            ptr: self.ptr as *mut U,
            len: self.len * (size_of::<T>() / size_of::<U>()),
            #[cfg(feature = "debug-cuda-stream")]
            alloc_stream: self.alloc_stream,
        };
        self.ptr = ptr::null_mut(); // for safe drop
        self.len = 0;
        res
    }

    pub fn view(&self) -> DeviceBufferView {
        DeviceBufferView {
            ptr: self.ptr as *const c_void,
            size: self.len * size_of::<T>(),
        }
    }

    /// Returns a mutable device-side view over the elements indexed by `range`.
    ///
    /// Range semantics follow Rust standard slicing: an inclusive start / an
    /// exclusive end. Unbounded start defaults to `0`; unbounded end defaults
    /// to `self.len()`. Panics if the resulting `[lo, hi)` is out of bounds
    /// or has `lo > hi`.
    ///
    /// The returned [`DeviceBufferMutSlice`] borrows `self` mutably, so the
    /// borrow checker prevents overlapping mutable views for its lifetime.
    pub fn mut_slice<R>(&mut self, range: R) -> DeviceBufferMutSlice<'_, T>
    where
        R: RangeBounds<usize>,
    {
        let len = self.len;
        let lo = match range.start_bound() {
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i.checked_add(1).expect("mut_slice: range start overflow"),
            Bound::Unbounded => 0,
        };
        let hi = match range.end_bound() {
            Bound::Included(&i) => i.checked_add(1).expect("mut_slice: range end overflow"),
            Bound::Excluded(&i) => i,
            Bound::Unbounded => len,
        };
        assert!(lo <= hi, "mut_slice: lo ({}) > hi ({})", lo, hi);
        assert!(hi <= len, "mut_slice: hi ({}) > len ({})", hi, len);
        DeviceBufferMutSlice { buf: self, lo, hi }
    }
}

/// A mutable device-side range `[lo, hi)` inside a [`DeviceBuffer<T>`].
///
/// Obtained via [`DeviceBuffer::mut_slice`]. Holds a `&mut DeviceBuffer<T>`
/// so the borrow checker prevents overlapping mutable views of the same
/// buffer for the lifetime of the slice.
#[derive(Debug)]
pub struct DeviceBufferMutSlice<'a, T> {
    buf: &'a mut DeviceBuffer<T>,
    lo: usize,
    hi: usize,
}

impl<'a, T> DeviceBufferMutSlice<'a, T> {
    /// Number of elements in the slice.
    pub fn len(&self) -> usize {
        self.hi - self.lo
    }

    pub fn is_empty(&self) -> bool {
        self.hi == self.lo
    }

    /// Copies `src` (host memory) into the slice on `device_ctx`'s stream via
    /// `cudaMemcpyAsync`.
    ///
    /// Errors with [`MemCopyError::SizeMismatch`] if `src.len() != self.len()`.
    ///
    /// `src` must be pageable (ordinary heap) memory: `cudaMemcpyAsync` then
    /// stages the source before returning, so `src` may be freed immediately.
    /// Pinned host memory would make the copy truly asynchronous and require
    /// `src` to outlive a stream sync.
    pub fn copy_from_host(
        &mut self,
        src: &[T],
        device_ctx: &GpuDeviceCtx,
    ) -> Result<(), MemCopyError> {
        let count = self.hi - self.lo;
        if src.len() != count {
            return Err(MemCopyError::SizeMismatch {
                operation: "DeviceBufferMutSlice::copy_from_host",
                src_len: src.len(),
                dst_len: count,
            });
        }
        if count == 0 {
            return Ok(());
        }
        let dst_ptr = unsafe { self.buf.as_mut_ptr().add(self.lo) as *mut c_void };
        let src_ptr = src.as_ptr() as *const c_void;
        let size_bytes = std::mem::size_of::<T>() * count;
        unsafe {
            cuda_memcpy_on::<false, true>(dst_ptr, src_ptr, size_bytes, device_ctx)?;
        }
        Ok(())
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            tracing::debug!(
                "Freeing device buffer of size {} (sizeof type = {})",
                self.len,
                size_of::<T>()
            );
            // d_free enqueues cudaFreeAsync on the stream that originally
            // allocated this buffer (stored in the memory manager's record).
            // This is correct even when Drop runs on a different thread —
            // CUDA handles cross-thread stream operations safely.
            unsafe {
                d_free(self.ptr as *mut c_void).expect("GPU free failed");
            }
            self.ptr = ptr::null_mut();
            self.len = 0;
        }
    }
}

impl<T: Debug> Debug for DeviceBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DeviceBuffer(len = {}, ptr = {:?})",
            self.len(),
            self.ptr
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::copy::{MemCopyD2H, MemCopyH2D};

    fn test_ctx() -> GpuDeviceCtx {
        GpuDeviceCtx::for_current_device().unwrap()
    }

    #[test]
    fn test_device_buffer_float() {
        let device_ctx = test_ctx();
        // Create a DeviceBuffer of 10 floats
        let db = DeviceBuffer::<f32>::with_capacity_on(10, &device_ctx);

        assert_eq!(db.len(), 10);
        assert!(!db.as_ptr().is_null());
        assert!(!db.is_empty());

        // The buffer will be automatically freed at the end of this test
    }

    #[test]
    fn test_device_buffer_fill_zero() {
        let device_ctx = test_ctx();
        let v: Vec<u64> = (0..10).collect();
        let d_array = v.to_device_on(&device_ctx).unwrap();
        d_array.fill_zero_on(&device_ctx).unwrap();
        assert_eq!(d_array.to_host_on(&device_ctx).unwrap(), vec![0; v.len()]);
    }

    #[test]
    fn test_device_buffer_mut_slice_copy_from_host() {
        let device_ctx = test_ctx();
        let mut d_array = [0u64; 10].to_device_on(&device_ctx).unwrap();

        // Overwrite the middle range [3, 7) with new values.
        let patch: [u64; 4] = [10, 20, 30, 40];
        d_array
            .mut_slice(3..7)
            .copy_from_host(&patch, &device_ctx)
            .unwrap();

        let host = d_array.to_host_on(&device_ctx).unwrap();
        assert_eq!(host, vec![0, 0, 0, 10, 20, 30, 40, 0, 0, 0]);
    }

    #[test]
    fn test_device_buffer_mut_slice_size_mismatch() {
        let device_ctx = test_ctx();
        let mut d_array = DeviceBuffer::<u64>::with_capacity_on(4, &device_ctx);
        let src = [1u64, 2, 3];
        let err = d_array
            .mut_slice(0..4)
            .copy_from_host(&src, &device_ctx)
            .unwrap_err();
        assert!(matches!(err, MemCopyError::SizeMismatch { .. }));
    }
}
