use std::{
    any::{type_name, Any, TypeId},
    collections::HashMap,
    ffi::c_void,
    fmt::Debug,
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
    ptr, slice,
    sync::{Mutex, OnceLock},
};

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
    pub fn with_capacity(len: usize) -> Self {
        let ctx = get_context();
        Self::with_capacity_on_device(&ctx.device, len)
    }

    /// Allocates a Metal buffer on a specific device.
    pub fn with_capacity_on_device(device: &Device, len: usize) -> Self {
        let alloc_len = len.max(1);
        let size_bytes = mem::size_of::<T>() * alloc_len;
        tracing::debug!(
            "Allocating Metal buffer: {} elements, {} bytes, type {}",
            len,
            size_bytes,
            type_name::<T>(),
        );
        metrics::counter!("metal.buffer.alloc_bytes").increment(size_bytes as u64);

        let buffer = device.new_buffer(
            size_bytes as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );
        MetalBuffer {
            buffer,
            len,
            _marker: PhantomData,
        }
    }

    /// Creates a Metal buffer by copying data from a host slice.
    pub fn from_slice(data: &[T]) -> Self {
        if data.is_empty() {
            return Self::with_capacity(0);
        }
        let ctx = get_context();
        Self::from_slice_on_device(&ctx.device, data)
    }

    /// Creates a Metal buffer on a specific device by copying data from a host slice.
    pub fn from_slice_on_device(device: &Device, data: &[T]) -> Self {
        if data.is_empty() {
            return Self::with_capacity_on_device(device, 0);
        }
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

    /// Returns a const GPU virtual address pointer for this buffer.
    ///
    /// Use this pointer only for values passed to GPU kernels (e.g. pointer tables/context
    /// structs).
    #[inline]
    pub fn as_device_ptr(&self) -> *const T {
        self.buffer.gpu_address() as *const T
    }

    /// Returns a mutable pointer to the buffer contents.
    ///
    /// The pointer is valid for `self.len()` elements of type `T`.
    #[inline]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.buffer.contents() as *mut T
    }

    /// Returns a mutable GPU virtual address pointer for this buffer.
    ///
    /// Use this pointer only for values passed to GPU kernels (e.g. pointer tables/context
    /// structs).
    #[inline]
    pub fn as_device_mut_ptr(&self) -> *mut T {
        self.buffer.gpu_address() as *mut T
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

const DEFAULT_MAX_CACHED_PER_LEN: usize = 8;

type BufferPoolByLen = HashMap<usize, Vec<Box<dyn Any + Send>>>;

/// Global reusable scratch buffer pool keyed by `(TypeId, len)`.
///
/// Buffers are returned to the pool on [`PooledMetalBuffer`] drop. Reuse is exact-size to keep
/// existing length invariants unchanged in debug assertions.
pub struct SharedMetalBufferPool {
    buffers: Mutex<HashMap<TypeId, BufferPoolByLen>>,
    max_cached_per_len: usize,
}

impl Default for SharedMetalBufferPool {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_CACHED_PER_LEN)
    }
}

impl SharedMetalBufferPool {
    pub fn new(max_cached_per_len: usize) -> Self {
        Self {
            buffers: Mutex::new(HashMap::new()),
            max_cached_per_len: max_cached_per_len.max(1),
        }
    }

    pub fn checkout<'a, T: 'static + Send>(
        &'a self,
        len: usize,
        scope: &'static str,
    ) -> PooledMetalBuffer<'a, T> {
        let buffer = self.acquire::<T>(len, scope);
        PooledMetalBuffer {
            pool: self,
            buffer: Some(buffer),
            scope,
        }
    }

    fn acquire<T: 'static + Send>(&self, len: usize, scope: &'static str) -> MetalBuffer<T> {
        let mut all = self.buffers.lock().unwrap();
        if let Some(by_len) = all.get_mut(&TypeId::of::<T>()) {
            if let Some(cached) = by_len.get_mut(&len) {
                if let Some(buffer) = cached.pop() {
                    metrics::counter!("metal.buffer_pool.reuse").increment(1);
                    tracing::debug!(
                        scope,
                        len,
                        type_name = type_name::<T>(),
                        cached_remaining = cached.len(),
                        "metal_buffer_pool_checkout_reuse"
                    );
                    return *buffer.downcast::<MetalBuffer<T>>().unwrap_or_else(|_| {
                        panic!("buffer pool type mismatch for {}", type_name::<T>())
                    });
                }
            }
        }
        metrics::counter!("metal.buffer_pool.miss").increment(1);
        tracing::debug!(
            scope,
            len,
            type_name = type_name::<T>(),
            "metal_buffer_pool_checkout_alloc"
        );
        drop(all);
        MetalBuffer::with_capacity(len)
    }

    fn release<T: 'static + Send>(&self, buffer: MetalBuffer<T>, scope: &'static str) {
        let len = buffer.len();
        let mut all = self.buffers.lock().unwrap();
        let by_len = all.entry(TypeId::of::<T>()).or_default();
        let cached = by_len.entry(len).or_default();
        if cached.len() < self.max_cached_per_len {
            cached.push(Box::new(buffer));
            metrics::counter!("metal.buffer_pool.return").increment(1);
            tracing::debug!(
                scope,
                len,
                type_name = type_name::<T>(),
                cached_count = cached.len(),
                "metal_buffer_pool_release_cached"
            );
        } else {
            metrics::counter!("metal.buffer_pool.drop").increment(1);
            tracing::debug!(
                scope,
                len,
                type_name = type_name::<T>(),
                max_cached_per_len = self.max_cached_per_len,
                "metal_buffer_pool_release_drop"
            );
        }
    }
}

pub struct PooledMetalBuffer<'a, T: 'static + Send> {
    pool: &'a SharedMetalBufferPool,
    buffer: Option<MetalBuffer<T>>,
    scope: &'static str,
}

impl<T: 'static + Send> PooledMetalBuffer<'_, T> {
    pub fn into_inner(mut self) -> MetalBuffer<T> {
        self.buffer
            .take()
            .expect("pooled metal buffer already moved")
    }
}

impl<T: 'static + Send> Deref for PooledMetalBuffer<'_, T> {
    type Target = MetalBuffer<T>;

    fn deref(&self) -> &Self::Target {
        self.buffer
            .as_ref()
            .expect("pooled metal buffer already moved")
    }
}

impl<T: 'static + Send> DerefMut for PooledMetalBuffer<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer
            .as_mut()
            .expect("pooled metal buffer already moved")
    }
}

impl<T: 'static + Send> Drop for PooledMetalBuffer<'_, T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.release(buffer, self.scope);
        }
    }
}

static SHARED_METAL_BUFFER_POOL: OnceLock<SharedMetalBufferPool> = OnceLock::new();

pub fn global_metal_buffer_pool() -> &'static SharedMetalBufferPool {
    SHARED_METAL_BUFFER_POOL.get_or_init(SharedMetalBufferPool::default)
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

    #[test]
    fn test_buffer_pool_caches_returned_buffer() {
        let pool = SharedMetalBufferPool::new(2);
        {
            let _buf = pool.checkout::<u32>(8, "test.pool.cache");
        }
        let all = pool.buffers.lock().unwrap();
        let cached = all
            .get(&TypeId::of::<u32>())
            .and_then(|by_len| by_len.get(&8))
            .map(|v| v.len())
            .unwrap_or(0);
        assert_eq!(cached, 1);
    }

    #[test]
    fn test_buffer_pool_reuses_exact_len() {
        let pool = SharedMetalBufferPool::new(2);
        {
            let _buf = pool.checkout::<u32>(8, "test.pool.reuse.seed");
        }
        let _reuse = pool.checkout::<u32>(8, "test.pool.reuse.checkout");
        let all = pool.buffers.lock().unwrap();
        let cached = all
            .get(&TypeId::of::<u32>())
            .and_then(|by_len| by_len.get(&8))
            .map(|v| v.len())
            .unwrap_or(0);
        assert_eq!(cached, 0);
    }
}
