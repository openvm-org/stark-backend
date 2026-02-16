use std::{ffi::c_void, fmt::Debug, marker::PhantomData, mem, ptr, slice};

use metal::{Buffer as MetalRawBuffer, Device, MTLResourceOptions, NSUInteger};

use crate::device::get_context;

/// A typed wrapper around a Metal buffer allocated with `StorageModeShared`.
///
/// On Apple Silicon, `StorageModeShared` means the buffer resides in unified memory
/// and is directly accessible by both CPU and GPU without explicit copies.
/// The CPU can read/write via `contents()` pointer; the GPU accesses it during dispatch.
///
/// **Synchronization**: After GPU writes, you must wait for the command buffer
/// to complete before reading from the CPU side.
pub struct MetalBuffer<T> {
    buffer: MetalRawBuffer,
    len: usize,
    _marker: PhantomData<T>,
}

// Safety: Metal buffers are thread-safe. The Metal framework synchronizes
// access to buffer contents internally. Users must still ensure proper
// command buffer synchronization before CPU reads after GPU writes.
unsafe impl<T> Send for MetalBuffer<T> {}
unsafe impl<T> Sync for MetalBuffer<T> {}

impl<T> MetalBuffer<T> {
    /// Allocates a Metal buffer with capacity for `len` elements of type `T`.
    ///
    /// Uses `StorageModeShared` for unified CPU/GPU access.
    /// Panics if `len` is zero.
    pub fn with_capacity(len: usize) -> Self {
        assert_ne!(len, 0, "Zero capacity request");
        let ctx = get_context();
        Self::with_capacity_on_device(&ctx.device, len)
    }

    /// Allocates a Metal buffer on a specific device.
    pub fn with_capacity_on_device(device: &Device, len: usize) -> Self {
        assert_ne!(len, 0, "Zero capacity request");
        let size_bytes = mem::size_of::<T>() * len;
        tracing::debug!(
            "Allocating Metal buffer: {} elements, {} bytes",
            len,
            size_bytes
        );
        metrics::counter!("metal.buffer.alloc_bytes").increment(size_bytes as u64);

        let buffer = device.new_buffer(size_bytes as NSUInteger, MTLResourceOptions::StorageModeShared);
        MetalBuffer {
            buffer,
            len,
            _marker: PhantomData,
        }
    }

    /// Creates a Metal buffer by copying data from a host slice.
    pub fn from_slice(data: &[T]) -> Self {
        assert!(!data.is_empty(), "Cannot create buffer from empty slice");
        let ctx = get_context();
        Self::from_slice_on_device(&ctx.device, data)
    }

    /// Creates a Metal buffer on a specific device by copying data from a host slice.
    pub fn from_slice_on_device(device: &Device, data: &[T]) -> Self {
        assert!(!data.is_empty(), "Cannot create buffer from empty slice");
        let size_bytes = mem::size_of_val(data);
        metrics::counter!("metal.buffer.alloc_bytes").increment(size_bytes as u64);

        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            size_bytes as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );
        MetalBuffer {
            buffer,
            len: data.len(),
            _marker: PhantomData,
        }
    }

    /// Returns the number of elements in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the size of the buffer in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.len * mem::size_of::<T>()
    }

    /// Returns a const pointer to the buffer contents.
    ///
    /// The pointer is valid for `self.len()` elements of type `T`.
    /// For reads after GPU writes, ensure the command buffer has completed first.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.buffer.contents() as *const T
    }

    /// Returns a mutable pointer to the buffer contents.
    ///
    /// The pointer is valid for `self.len()` elements of type `T`.
    #[inline]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.buffer.contents() as *mut T
    }

    /// Returns a reference to the underlying `metal::Buffer` for use in kernel dispatch.
    #[inline]
    pub fn gpu_buffer(&self) -> &MetalRawBuffer {
        &self.buffer
    }

    /// Reads the buffer contents back to a `Vec<T>`.
    ///
    /// **Important**: Ensure any GPU work writing to this buffer has completed
    /// before calling this method (e.g., via `cmd_buffer.wait_until_completed()`).
    pub fn to_vec(&self) -> Vec<T> {
        if self.len == 0 {
            return Vec::new();
        }
        let src = self.as_ptr();
        let mut dst = Vec::with_capacity(self.len);
        unsafe {
            ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), self.len);
            dst.set_len(self.len);
        }
        dst
    }

    /// Returns a slice view of the buffer contents.
    ///
    /// # Safety
    /// The caller must ensure no GPU work is currently writing to this buffer.
    pub unsafe fn as_slice(&self) -> &[T] {
        slice::from_raw_parts(self.as_ptr(), self.len)
    }

    /// Returns a mutable slice view of the buffer contents.
    ///
    /// # Safety
    /// The caller must ensure no GPU work is currently accessing this buffer.
    pub unsafe fn as_mut_slice(&self) -> &mut [T] {
        slice::from_raw_parts_mut(self.as_mut_ptr(), self.len)
    }

    /// Copies data from a host slice into this buffer.
    ///
    /// Panics if the slice length exceeds the buffer capacity.
    pub fn copy_from_slice(&self, data: &[T]) {
        assert!(
            data.len() <= self.len,
            "Source slice length {} exceeds buffer capacity {}",
            data.len(),
            self.len
        );
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), self.as_mut_ptr(), data.len());
        }
    }

    /// Fills the entire buffer with zeros.
    pub fn fill_zero(&self) {
        let size_bytes = self.size_bytes();
        unsafe {
            ptr::write_bytes(self.as_mut_ptr() as *mut u8, 0, size_bytes);
        }
    }

    /// Fills a suffix of the buffer with zeros, starting at element index `start_idx`.
    pub fn fill_zero_suffix(&self, start_idx: usize) {
        assert!(
            start_idx < self.len,
            "start_idx {} must be less than buffer length {}",
            start_idx,
            self.len
        );
        let count = self.len - start_idx;
        let byte_count = count * mem::size_of::<T>();
        unsafe {
            ptr::write_bytes(self.as_mut_ptr().add(start_idx) as *mut u8, 0, byte_count);
        }
    }
}

impl<T: Debug> Debug for MetalBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.to_vec();
        write!(f, "MetalBuffer (len = {}): {:?}", self.len, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_buffer_alloc() {
        let buf = MetalBuffer::<f32>::with_capacity(10);
        assert_eq!(buf.len(), 10);
        assert!(!buf.is_empty());
        assert!(!buf.as_ptr().is_null());
    }

    #[test]
    fn test_metal_buffer_from_slice() {
        let data: Vec<u32> = (0..16).collect();
        let buf = MetalBuffer::from_slice(&data);
        assert_eq!(buf.len(), 16);
        assert_eq!(buf.to_vec(), data);
    }

    #[test]
    fn test_metal_buffer_copy_from_slice() {
        let buf = MetalBuffer::<u32>::with_capacity(8);
        let data: Vec<u32> = (100..108).collect();
        buf.copy_from_slice(&data);
        assert_eq!(buf.to_vec(), data);
    }

    #[test]
    fn test_metal_buffer_fill_zero() {
        let data: Vec<u32> = (0..10).collect();
        let buf = MetalBuffer::from_slice(&data);
        buf.fill_zero();
        assert_eq!(buf.to_vec(), vec![0u32; 10]);
    }

    #[test]
    fn test_metal_buffer_fill_zero_suffix() {
        let data: Vec<u32> = vec![1, 2, 3, 4, 5];
        let buf = MetalBuffer::from_slice(&data);
        buf.fill_zero_suffix(3);
        assert_eq!(buf.to_vec(), vec![1, 2, 3, 0, 0]);
    }
}
