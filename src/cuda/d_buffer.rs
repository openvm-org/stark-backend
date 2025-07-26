use std::{ffi::c_void, fmt::Debug, ptr};

use crate::cuda::{
    copy::MemCopyD2H,
    error::{check, CudaError},
    memory_manager::{d_free, d_malloc},
    stream::{cudaStreamPerThread, cudaStream_t},
};

#[link(name = "cudart")]
extern "C" {
    fn cudaMemsetAsync(dst: *mut c_void, value: i32, count: usize, stream: cudaStream_t) -> i32;
}

pub struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
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
        }
    }

    /// Allocate device memory for `len` elements of type `T`.
    pub fn with_capacity(len: usize) -> Self {
        assert_ne!(len, 0, "Zero capacity request is wrong");
        let size_bytes = std::mem::size_of::<T>() * len;
        let raw_ptr = d_malloc(size_bytes).expect("GPU allocation failed");
        #[cfg(feature = "touchemall")]
        {
            // 0xffffffff is `Fp::invalid()` and shouldn't occur in a trace
            unsafe {
                cudaMemsetAsync(raw_ptr, 0xff, size_bytes, cudaStreamPerThread);
            }
        }
        let typed_ptr = raw_ptr as *mut T;

        DeviceBuffer {
            ptr: typed_ptr,
            len,
        }
    }

    /// Fills the buffer with zeros.
    pub fn fill_zero(&self) -> Result<(), CudaError> {
        assert_ne!(self.len, 0, "Empty buffer");
        let size_bytes = std::mem::size_of::<T>() * self.len;
        check(unsafe { cudaMemsetAsync(self.as_mut_raw_ptr(), 0, size_bytes, cudaStreamPerThread) })
    }

    /// Fills a suffix of the buffer with zeros.
    /// The `start_idx` is the index in the buffer, in `T` elements.
    pub fn fill_zero_suffix(&self, start_idx: usize) -> Result<(), CudaError> {
        assert!(
            start_idx < self.len,
            "start index has to be smaller than length"
        );
        let size_bytes = std::mem::size_of::<T>() * (self.len - start_idx);
        check(unsafe {
            cudaMemsetAsync(
                self.as_mut_ptr().add(start_idx) as *mut c_void,
                0,
                size_bytes,
                cudaStreamPerThread,
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
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            tracing::debug!("Freeing device buffer of size {}", self.len);
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
        let host_vec = self.to_host().unwrap();
        write!(f, "DeviceBuffer (len = {}): {:?}", self.len(), host_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::copy::MemCopyH2D;

    #[test]
    fn test_device_buffer_float() {
        // Create a DeviceBuffer of 10 floats
        let db = DeviceBuffer::<f32>::with_capacity(10);

        assert_eq!(db.len(), 10);
        assert!(!db.as_ptr().is_null());
        assert!(!db.is_empty());

        // The buffer will be automatically freed at the end of this test
    }

    #[test]
    fn test_device_buffer_fill_zero() {
        let v: Vec<u64> = (0..10).collect();
        let d_array = v.to_device().unwrap();
        d_array.fill_zero().unwrap();
        assert_eq!(d_array.to_host().unwrap(), vec![0; v.len()]);
    }
}
