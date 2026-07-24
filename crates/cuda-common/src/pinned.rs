//! Page-locked (pinned) host memory utilities.
//!
//! Two independent facilities live here:
//!
//! 1. A pooled staging allocator for host-to-device copies ([`take`] / [`give_back`] /
//!    [`register_region`]).
//! 2. [`PinnedBuffer`], a fixed-capacity pinned buffer for low-latency device-to-host readbacks
//!    (see [`crate::copy::MemCopyD2H::to_pinned_on`]).
//!
//! # Staging pool
//!
//! Intended for large host buffers that are allocated fresh for every round
//! of work (e.g. per proving segment), partially written, copied to the
//! device, and dropped.
//!
//! Copies from pageable memory run at a fraction of PCIe bandwidth,
//! but page-locking a buffer is itself expensive (~1-2 GB/s), and buffers
//! are often provisioned at full capacity while only partially written,
//! so neither registration nor re-zeroing may sit on the caller's critical path.
//!
//! Dropped buffers are therefore handed via [give_back] to a background cleaner
//! thread which registers them once (cudaHostRegister), zeroes the prefix the
//! previous owner wrote, and only then returns them to the pool.
//! [take] hands out ready (registered, all-zero) buffers on a pool hit;
//! on a miss it falls back to a fresh pageable allocation.
//!
//! Lifetime hazard: `cudaMemcpyAsync` from *pageable* memory returns only
//! after the source has been staged, so code not using the pool may free a
//! source buffer right after enqueueing its copy. From *pinned* memory the
//! call returns immediately with the DMA still in flight, so a returned
//! buffer must not be zeroed or reused until previously enqueued work has
//! drained. The cleaner therefore calls `cudaDeviceSynchronize` (batched over
//! every buffer waiting in its queue) before touching buffer contents.
//!
//! # Readback buffers
//!
//! Small per-round sumcheck readbacks with `cudaMemcpyAsync` from device to
//! *pageable* host memory go through an internal driver staging buffer, adding
//! noticeable latency to every transcript round trip. Copying into pinned
//! memory instead uses a direct DMA.

use std::{
    collections::{BTreeMap, HashSet},
    ffi::c_void,
    marker::PhantomData,
    sync::{mpsc, Mutex, OnceLock},
};

use crate::{
    error::{check, CudaError},
    stream::device_synchronize,
};

#[link(name = "cudart")]
extern "C" {
    fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaHostUnregister(ptr: *mut c_void) -> i32;
    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaFreeHost(ptr: *mut c_void) -> i32;
}

/// A returned buffer together with its dirty-prefix length.
type ReturnedBuffer = (Vec<u8>, usize);

/// Page-locks `len` bytes at `ptr` in a single `cudaHostRegister` call.
/// Returns `false` (leaving the buffer pageable) if registration fails.
pub fn register_region(ptr: *mut u8, len: usize) -> bool {
    // SAFETY: [ptr, ptr+len) is a live allocation owned by the caller.
    let rc = unsafe { cudaHostRegister(ptr as *mut c_void, len, 0) };
    if rc != 0 {
        tracing::debug!(
            "cudaHostRegister failed: {}; buffer stays pageable",
            CudaError::new(rc)
        );
        return false;
    }
    true
}

/// Reverses a successful [`register_region`]. The caller must ensure no copy
/// from the region is still in flight.
pub fn unregister_region(ptr: *mut u8) {
    // SAFETY: mirrors a successful registration of the same base pointer.
    unsafe { cudaHostUnregister(ptr as *mut c_void) };
}

/// Registered, all-zero buffers ready for reuse, keyed by allocation size.
fn pool() -> &'static Mutex<BTreeMap<usize, Vec<Vec<u8>>>> {
    static POOL: OnceLock<Mutex<BTreeMap<usize, Vec<Vec<u8>>>>> = OnceLock::new();
    POOL.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Base pointers of buffers whose `cudaHostRegister` succeeded.
fn registered() -> &'static Mutex<HashSet<usize>> {
    static REGISTERED: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();
    REGISTERED.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Cleaner thread: registers (first cycle) and re-zeroes buffers off the
/// critical path, then makes them available to [`take`].
fn cleaner() -> &'static Mutex<mpsc::Sender<ReturnedBuffer>> {
    static TX: OnceLock<Mutex<mpsc::Sender<ReturnedBuffer>>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<ReturnedBuffer>();
        std::thread::Builder::new()
            .name("pinned-cleaner".into())
            .spawn(move || {
                while let Ok(first) = rx.recv() {
                    // Coalesce the burst of buffer returns behind one device sync.
                    let mut batch = vec![first];
                    while batch.len() < 64 {
                        match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                            Ok(next) => batch.push(next),
                            Err(_) => break,
                        }
                    }
                    // The H2D copies reading these buffers were enqueued
                    // before the owners gave them back; wait for them (and
                    // anything else in flight) before touching contents.
                    let _span = tracing::info_span!("pinned_cleaner_batch").entered();
                    if let Err(e) = device_synchronize() {
                        tracing::debug!(
                            "cudaDeviceSynchronize failed: {e}; dropping {} pooled buffers",
                            batch.len()
                        );
                        continue;
                    }
                    for (mut buffer, dirty_len) in batch {
                        let ptr = buffer.as_mut_ptr();
                        let is_new = !registered().lock().unwrap().contains(&(ptr as usize));
                        if is_new {
                            if !register_region(ptr, buffer.len()) {
                                continue;
                            }
                            registered().lock().unwrap().insert(ptr as usize);
                        }
                        let dirty_len = dirty_len.min(buffer.len());
                        buffer[..dirty_len].fill(0);
                        pool()
                            .lock()
                            .unwrap()
                            .entry(buffer.len())
                            .or_default()
                            .push(buffer);
                    }
                }
            })
            .expect("failed to spawn pinned-cleaner thread");
        Mutex::new(tx)
    })
}

/// Returns an all-zero buffer of at least `min_size` bytes (rounded up to the
/// next power of two), page-locked if it came from the pool.
pub fn take(min_size: usize) -> Vec<u8> {
    let size = min_size.next_power_of_two();
    if let Some(buffer) = pool()
        .lock()
        .unwrap()
        .get_mut(&size)
        .and_then(|bufs| bufs.pop())
    {
        debug_assert_eq!(buffer.len(), size);
        return buffer;
    }
    // Pool miss: when no recycled buffer of that size is available,
    // take returns a plain, unpinned allocation instead of pinning one on the spot.
    vec![0u8; size]
}

/// Hands `buffer` to the cleaner for registration, re-zeroing, and reuse.
/// `dirty_len` is an upper bound on the prefix of `buffer` that may have
/// been written since it left [`take`]; the rest must still be zero.
pub fn give_back(buffer: Vec<u8>, dirty_len: usize) {
    if buffer.is_empty() || !buffer.len().is_power_of_two() {
        return; // not a pool-shaped buffer; drop normally
    }
    // The cleaner owning the receiver never exits, so send only fails if
    // the buffer raced process teardown; dropping it then is fine.
    let _ = cleaner().lock().unwrap().send((buffer, dirty_len));
}

/// Unregisters and frees all pooled buffers (test hygiene; optional).
pub fn clear() {
    let mut pool = pool().lock().unwrap();
    let mut reg = registered().lock().unwrap();
    for (_, bufs) in pool.iter_mut() {
        for mut buf in bufs.drain(..) {
            reg.remove(&(buf.as_ptr() as usize));
            unsafe { cudaHostUnregister(buf.as_mut_ptr() as *mut c_void) };
        }
    }
    pool.clear();
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

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    /// Waits until the cleaner (batching window + device sync) has returned a
    /// buffer of exactly `size` bytes to the pool.
    fn wait_for_pooled(size: usize) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while pool()
            .lock()
            .unwrap()
            .get(&size)
            .is_none_or(|bufs| bufs.is_empty())
        {
            assert!(
                Instant::now() < deadline,
                "size-{size} buffer never came back to the pool"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn take_rounds_up_to_next_power_of_two_and_zero_fills() {
        for (min_size, expected) in [(1, 1), (3, 4), (1024, 1024), (1025, 2048)] {
            let buf = take(min_size);
            assert_eq!(buf.len(), expected, "take({min_size})");
            assert!(buf.iter().all(|&b| b == 0), "take({min_size}) not zeroed");
        }
    }

    // Tests share process-global pool state; each uses its own buffer size so
    // pool entries are never shared.
    #[test]
    fn round_trip_recycles_registered_rezeroed_buffer() {
        const SIZE: usize = 1 << 13;
        let mut buf = take(SIZE);
        let ptr = buf.as_ptr() as usize;
        buf.fill(0xAB);
        // Oversized dirty_len must be clamped, not panic.
        give_back(buf, usize::MAX);
        wait_for_pooled(SIZE);
        assert!(
            registered().lock().unwrap().contains(&ptr),
            "pooled buffer was not page-locked"
        );
        let buf = take(SIZE);
        assert_eq!(
            buf.as_ptr() as usize,
            ptr,
            "pool hit should reuse the allocation"
        );
        assert!(buf.iter().all(|&b| b == 0), "recycled buffer not re-zeroed");
    }
}
