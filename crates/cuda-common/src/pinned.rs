//! Page-locked (pinned) host-buffer pool for staging host-to-device copies.
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
//! [take] hands out [PinnedBuffer] guards: ready (registered, all-zero)
//! buffers on a pool hit; on a miss it falls back to a fresh pageable
//! allocation. Dropping the guard hands the buffer to a background cleaner
//! thread which registers it once (cudaHostRegister), zeroes the prefix the
//! previous owner wrote, and only then returns it to the pool.
//!
//! Lifetime hazard: `cudaMemcpyAsync` from *pageable* memory returns only
//! after the source has been staged, so code not using the pool may free a
//! source buffer right after enqueueing its copy. From *pinned* memory the
//! call returns immediately with the DMA still in flight, so a returned
//! buffer must not be zeroed or reused until previously enqueued work has
//! drained. The cleaner therefore calls `cudaDeviceSynchronize` (batched over
//! every buffer waiting in its queue) before touching buffer contents.

use std::{
    collections::BTreeMap,
    ffi::c_void,
    ops::{Deref, DerefMut},
    sync::{mpsc, Mutex, OnceLock},
};

use crate::{error::CudaError, stream::device_synchronize};

#[link(name = "cudart")]
extern "C" {
    fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaHostUnregister(ptr: *mut c_void) -> i32;
}

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

/// A host staging buffer owned by the pool, handed out by [take].
///
/// Dereferences to `[u8]`. The allocation can never grow or move, so the base
/// pointer a registration was made for stays valid for the buffer's whole
/// life, and dropping the guard is the only way to release the allocation —
/// it goes back through the pool's cleaner, never straight to the allocator
/// while page-locked.
pub struct PinnedBuffer {
    data: Vec<u8>,
    /// Upper bound on the prefix of `data` written since [take].
    dirty_len: usize,
    /// Whether `data` is currently page-locked.
    registered: bool,
}

impl PinnedBuffer {
    /// Whether the buffer is page-locked. False for pool-miss buffers, which
    /// stay pageable until their first trip through the cleaner.
    pub fn is_pinned(&self) -> bool {
        self.registered
    }

    /// Narrows the prefix the cleaner re-zeroes before pooling the buffer
    /// (by default the whole buffer). The caller asserts that bytes at
    /// `len..` have not been written since [take].
    pub fn set_dirty_len(&mut self, len: usize) {
        self.dirty_len = len;
    }
}

impl Deref for PinnedBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.data
    }
}

impl DerefMut for PinnedBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        let data = std::mem::take(&mut self.data);
        if data.is_empty() {
            return;
        }
        let returned = Returned {
            data,
            dirty_len: self.dirty_len,
            registered: self.registered,
        };
        // The cleaner owning the receiver never exits, so send only fails if
        // the buffer raced process teardown; clean up inline then.
        if let Err(mpsc::SendError(returned)) = cleaner().lock().unwrap().send(returned) {
            wait_and_release(returned);
        }
    }
}

/// Best-effort drain of the copies still reading `returned`, then [release].
/// For use outside the cleaner (whose batch loop amortizes the waiting).
fn wait_and_release(returned: Returned) {
    if let Err(e) = device_synchronize() {
        tracing::debug!("cudaDeviceSynchronize failed: {e}");
    }
    release(returned);
}

/// Frees `returned` without pooling it, unregistering first so the allocator
/// never gets back memory that is still page-locked. Called on failure paths
/// where in-flight copies could not be drained; a broken context is not
/// making DMA progress, so unregistering is the lesser evil there.
fn release(mut returned: Returned) {
    if returned.registered {
        unregister_region(returned.data.as_mut_ptr());
    }
}

/// A buffer on its way back to the pool.
struct Returned {
    data: Vec<u8>,
    dirty_len: usize,
    registered: bool,
}

/// Registered, all-zero buffers ready for reuse, keyed by allocation size.
/// Invariant: every pooled buffer is page-locked and quiescent.
fn pool() -> &'static Mutex<BTreeMap<usize, Vec<Vec<u8>>>> {
    static POOL: OnceLock<Mutex<BTreeMap<usize, Vec<Vec<u8>>>>> = OnceLock::new();
    POOL.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Cleaner thread: registers (first cycle) and re-zeroes buffers off the
/// critical path, then makes them available to [`take`].
fn cleaner() -> &'static Mutex<mpsc::Sender<Returned>> {
    static TX: OnceLock<Mutex<mpsc::Sender<Returned>>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Returned>();
        std::thread::Builder::new()
            .name("pinned-cleaner".into())
            .spawn(move || {
                let mut batch_idx = 0usize;
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
                    let _span =
                        tracing::info_span!("pinned_cleaner_batch", batch = batch_idx.to_string())
                            .entered();
                    batch_idx += 1;
                    if let Err(e) = device_synchronize() {
                        tracing::debug!(
                            "cudaDeviceSynchronize failed: {e}; dropping {} pooled buffers",
                            batch.len()
                        );
                        for returned in batch {
                            release(returned);
                        }
                        continue;
                    }
                    for returned in batch {
                        recycle(returned);
                    }
                }
            })
            .expect("failed to spawn pinned-cleaner thread");
        Mutex::new(tx)
    })
}

/// Registers `returned` if this is its first cycle, re-zeroes its dirty
/// prefix, and pools it. The buffer must be quiescent (no copies in flight).
fn recycle(mut returned: Returned) {
    if !returned.registered {
        if !register_region(returned.data.as_mut_ptr(), returned.data.len()) {
            return; // stays pageable; drop normally
        }
        returned.registered = true;
    }
    let dirty_len = returned.dirty_len.min(returned.data.len());
    returned.data[..dirty_len].fill(0);
    pool()
        .lock()
        .unwrap()
        .entry(returned.data.len())
        .or_default()
        .push(returned.data);
}

/// Returns an all-zero buffer of at least `min_size` bytes (rounded up to the
/// next power of two), page-locked if it came from the pool.
pub fn take(min_size: usize) -> PinnedBuffer {
    let size = min_size.next_power_of_two();
    if let Some(data) = pool()
        .lock()
        .unwrap()
        .get_mut(&size)
        .and_then(|bufs| bufs.pop())
    {
        debug_assert_eq!(data.len(), size);
        return PinnedBuffer {
            data,
            dirty_len: size,
            registered: true,
        };
    }
    // Pool miss: when no recycled buffer of that size is available,
    // take returns a plain, unpinned allocation instead of pinning one on the spot.
    PinnedBuffer {
        data: vec![0u8; size],
        dirty_len: size,
        registered: false,
    }
}

/// Unregisters and frees all pooled buffers (test hygiene; optional).
pub fn clear() {
    let mut pool = pool().lock().unwrap();
    for (_, bufs) in pool.iter_mut() {
        for mut buf in bufs.drain(..) {
            unregister_region(buf.as_mut_ptr());
        }
    }
    pool.clear();
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    /// Serializes tests: they share the process-global pool and cleaner, so
    /// distinct buffer sizes alone don't isolate them.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn lock_tests() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

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
        let _lock = lock_tests();
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
        let _lock = lock_tests();
        const SIZE: usize = 1 << 13;
        let mut buf = take(SIZE);
        let ptr = buf.as_ptr() as usize;
        buf.fill(0xAB);
        // Oversized dirty_len must be clamped, not panic.
        buf.set_dirty_len(usize::MAX);
        drop(buf);
        wait_for_pooled(SIZE);
        let buf = take(SIZE);
        assert!(buf.is_pinned(), "pool hit should be page-locked");
        assert_eq!(
            buf.as_ptr() as usize,
            ptr,
            "pool hit should reuse the allocation"
        );
        assert!(buf.iter().all(|&b| b == 0), "recycled buffer not re-zeroed");
    }
}
