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
//! buffer must not be zeroed or reused until the copies reading it have
//! drained. Callers should mark the last copy out of a buffer with
//! [PinnedBuffer::record_last_use]; the cleaner then waits on exactly those
//! events. Without a recorded event it falls back to synchronizing the device
//! that was current when the guard dropped.
//!
//! The pool retains at most [set_max_pooled_bytes] bytes (default 4 GiB);
//! past that, returning a buffer evicts pooled ones, largest first.

use std::{
    collections::BTreeMap,
    ffi::c_void,
    ops::{Deref, DerefMut},
    sync::{mpsc, Mutex, OnceLock},
};

use crate::{
    common::{get_device, set_device_by_id},
    error::CudaError,
    stream::{device_synchronize, CudaEvent, CudaStream},
};

#[link(name = "cudart")]
extern "C" {
    fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaHostUnregister(ptr: *mut c_void) -> i32;
}

/// Registered memory is pinned for all CUDA contexts, so copies issued from
/// any device get the fast path; without this flag, contexts other than the
/// registering one (the cleaner's) would treat the buffer as pageable.
const CUDA_HOST_REGISTER_PORTABLE: u32 = 0x1;

/// Page-locks `len` bytes at `ptr` in a single `cudaHostRegister` call, for
/// all CUDA contexts. Returns `false` (leaving the buffer pageable) if
/// registration fails.
pub fn register_region(ptr: *mut u8, len: usize) -> bool {
    // SAFETY: [ptr, ptr+len) is a live allocation owned by the caller.
    let rc = unsafe { cudaHostRegister(ptr as *mut c_void, len, CUDA_HOST_REGISTER_PORTABLE) };
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
    /// Events recorded after the last copies reading `data`; the cleaner
    /// waits on these before re-zeroing and reusing it.
    last_use: Vec<CudaEvent>,
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

    /// Records the point on `stream` after which the buffer is no longer
    /// read, i.e. right after enqueueing the last copy out of it there. The
    /// cleaner then waits on exactly the recorded events — instead of a
    /// whole-device synchronize — before reusing the buffer, which is also
    /// what makes reuse safe when several devices are active.
    ///
    /// Call from the thread that enqueued the copies, with the same device
    /// current (CUDA requires the event and stream to share a context). If
    /// several streams consume the buffer, record on each of them.
    pub fn record_last_use(&mut self, stream: &CudaStream) -> Result<(), CudaError> {
        let event = CudaEvent::new()?;
        event.record_on(stream)?;
        self.last_use.push(event);
        Ok(())
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
            last_use: std::mem::take(&mut self.last_use),
            // Fallback sync target when no event was recorded: per this
            // crate's convention (GpuDeviceCtx::for_device) the copies were
            // issued from the dropping thread, on its current device.
            device: get_device().unwrap_or(0),
        };
        send_to_cleaner(returned);
    }
}

/// Best-effort drain of the copies still reading `returned`, then [release].
/// For use outside the cleaner (whose batch loop amortizes the waiting).
fn wait_and_release(returned: Returned) {
    let result = if returned.last_use.is_empty() {
        set_device_by_id(returned.device).and_then(|_| device_synchronize())
    } else {
        returned
            .last_use
            .iter()
            .try_for_each(CudaEvent::synchronize)
    };
    if let Err(e) = result {
        tracing::debug!("draining copies from returned buffer failed: {e}");
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
    /// Events recorded after the last copies reading `data`.
    last_use: Vec<CudaEvent>,
    /// Device current when the guard dropped; whole-device sync fallback
    /// target when `last_use` is empty.
    device: i32,
}

/// Default value for [set_max_pooled_bytes].
const DEFAULT_MAX_POOLED_BYTES: usize = 4 << 30; // 4 GiB

/// Registered, all-zero buffers ready for reuse, keyed by allocation size.
/// Invariants: every pooled buffer is page-locked and quiescent, empty size
/// classes are removed, and `total_bytes` is the sum over all pooled buffers.
struct Pool {
    by_size: BTreeMap<usize, Vec<Vec<u8>>>,
    total_bytes: usize,
    max_bytes: usize,
}

fn pool() -> &'static Mutex<Pool> {
    static POOL: OnceLock<Mutex<Pool>> = OnceLock::new();
    POOL.get_or_init(|| {
        Mutex::new(Pool {
            by_size: BTreeMap::new(),
            total_bytes: 0,
            max_bytes: DEFAULT_MAX_POOLED_BYTES,
        })
    })
}

/// Caps the bytes of pinned memory the pool may retain (default 4 GiB).
/// A returned buffer that would push the total over the cap evicts pooled
/// buffers instead, largest size class first, and a buffer larger than the
/// whole cap is never pooled. Lowering the cap evicts immediately.
pub fn set_max_pooled_bytes(max_bytes: usize) {
    let evicted = {
        let mut pool = pool().lock().unwrap();
        pool.max_bytes = max_bytes;
        evict_to_fit(&mut pool, 0)
    };
    for mut buf in evicted {
        unregister_region(buf.as_mut_ptr());
    }
}

/// Pops pooled buffers, largest size class first, until a buffer of
/// `incoming` bytes fits under the cap (no-op if it never can). The caller
/// unregisters the evicted buffers, ideally outside the pool lock.
fn evict_to_fit(pool: &mut Pool, incoming: usize) -> Vec<Vec<u8>> {
    let mut evicted = Vec::new();
    if incoming > pool.max_bytes {
        return evicted;
    }
    while pool.total_bytes + incoming > pool.max_bytes {
        let Some((&class, bufs)) = pool.by_size.iter_mut().next_back() else {
            break;
        };
        evicted.push(bufs.pop().expect("empty size classes are removed"));
        if bufs.is_empty() {
            pool.by_size.remove(&class);
        }
        pool.total_bytes -= class;
    }
    evicted
}

/// A running cleaner thread and the channel feeding it.
struct Cleaner {
    tx: mpsc::Sender<Returned>,
    thread: std::thread::JoinHandle<()>,
}

/// Slot holding the live cleaner; empty until first use and after [shutdown].
fn cleaner_slot() -> &'static Mutex<Option<Cleaner>> {
    static SLOT: OnceLock<Mutex<Option<Cleaner>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// Queues `returned` for the cleaner, spawning it if none is running.
fn send_to_cleaner(returned: Returned) {
    let mut slot = cleaner_slot().lock().unwrap();
    let cleaner = slot.get_or_insert_with(spawn_cleaner);
    if let Err(mpsc::SendError(returned)) = cleaner.tx.send(returned) {
        // Only reachable if the cleaner thread died; clean up inline.
        tracing::debug!("pinned-cleaner thread is gone; releasing buffer inline");
        wait_and_release(returned);
    }
}

/// Spawns the cleaner thread: it drains in-flight copies, registers buffers
/// on their first cycle, and re-zeroes them off the critical path, then makes
/// them available to [`take`]. It exits once the feeding channel closes and
/// its queue is empty.
fn spawn_cleaner() -> Cleaner {
    let (tx, rx) = mpsc::channel::<Returned>();
    let thread = std::thread::Builder::new()
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
                let _span =
                    tracing::info_span!("pinned_cleaner_batch", batch = batch_idx.to_string())
                        .entered();
                batch_idx += 1;
                // The H2D copies reading each buffer were enqueued before
                // its owner gave it back; wait for them before touching
                // contents:
                //  - recorded last-use events cover exactly those copies;
                //  - a registered buffer without events falls back to a whole-device sync of the
                //    device recorded at drop (memoized per batch);
                //  - a never-registered buffer was pageable during use, and a pageable
                //    cudaMemcpyAsync returns only after the source is staged, so it is already
                //    quiescent.
                let mut synced_devices = BTreeMap::new();
                for returned in batch {
                    let drained = if !returned.last_use.is_empty() {
                        returned
                            .last_use
                            .iter()
                            .try_for_each(CudaEvent::synchronize)
                            .map_err(|e| tracing::debug!("cudaEventSynchronize failed: {e}"))
                            .is_ok()
                    } else if !returned.registered {
                        true
                    } else {
                        *synced_devices.entry(returned.device).or_insert_with(|| {
                            set_device_by_id(returned.device)
                                .and_then(|_| device_synchronize())
                                .map_err(|e| {
                                    tracing::debug!(
                                        "synchronizing device {} failed: {e}",
                                        returned.device
                                    )
                                })
                                .is_ok()
                        })
                    };
                    if drained {
                        recycle(returned);
                    } else {
                        release(returned);
                    }
                }
            }
        })
        .expect("failed to spawn pinned-cleaner thread");
    Cleaner { tx, thread }
}

/// Drains and joins the cleaner thread, then unregisters and frees every
/// pooled buffer. Call once outstanding buffers have been handed back, e.g.
/// before tearing down CUDA contexts or between tests. Not final: a
/// [PinnedBuffer] dropped afterwards respawns the cleaner (and may repopulate
/// the pool).
pub fn shutdown() {
    let cleaner = cleaner_slot().lock().unwrap().take();
    if let Some(Cleaner { tx, thread }) = cleaner {
        // Closing the channel lets the cleaner drain its queue, then exit.
        drop(tx);
        let _ = thread.join();
    }
    clear();
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
    let size = returned.data.len();
    let mut evicted;
    {
        let mut pool = pool().lock().unwrap();
        evicted = evict_to_fit(&mut pool, size);
        if pool.total_bytes + size <= pool.max_bytes {
            pool.total_bytes += size;
            pool.by_size.entry(size).or_default().push(returned.data);
        } else {
            // Larger than the whole cap: never pooled.
            evicted.push(returned.data);
        }
    }
    for mut buf in evicted {
        // Pooled buffers are quiescent, so unregistering evictees is safe.
        unregister_region(buf.as_mut_ptr());
    }
}

/// Returns an all-zero buffer of at least `min_size` bytes (rounded up to the
/// next power of two), page-locked if it came from the pool.
pub fn take(min_size: usize) -> PinnedBuffer {
    let size = min_size.next_power_of_two();
    {
        let mut pool = pool().lock().unwrap();
        if let Some(bufs) = pool.by_size.get_mut(&size) {
            let data = bufs.pop().expect("empty size classes are removed");
            if bufs.is_empty() {
                pool.by_size.remove(&size);
            }
            pool.total_bytes -= size;
            debug_assert_eq!(data.len(), size);
            return PinnedBuffer {
                data,
                dirty_len: size,
                registered: true,
                last_use: Vec::new(),
            };
        }
    }
    // Pool miss: when no recycled buffer of that size is available,
    // take returns a plain, unpinned allocation instead of pinning one on the spot.
    PinnedBuffer {
        data: vec![0u8; size],
        dirty_len: size,
        registered: false,
        last_use: Vec::new(),
    }
}

/// Unregisters and frees all pooled buffers. Does not touch buffers still
/// queued for the cleaner; [shutdown] drains those first.
pub fn clear() {
    let mut pool = pool().lock().unwrap();
    for (_, bufs) in pool.by_size.iter_mut() {
        for mut buf in bufs.drain(..) {
            unregister_region(buf.as_mut_ptr());
        }
    }
    pool.by_size.clear();
    pool.total_bytes = 0;
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
            .by_size
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

    #[test]
    fn shutdown_drains_joins_and_respawns_on_demand() {
        let _lock = lock_tests();
        const SIZE: usize = 1 << 15;
        let mut buf = take(SIZE);
        buf.fill(0xEF);
        drop(buf); // queued to the cleaner
        shutdown();
        assert!(
            pool().lock().unwrap().by_size.is_empty(),
            "shutdown left buffers pooled"
        );
        // The pool keeps working afterwards: the next drop respawns the cleaner.
        let buf = take(SIZE);
        assert!(!buf.is_pinned(), "pool should be empty after shutdown");
        drop(buf);
        wait_for_pooled(SIZE);
        assert!(take(SIZE).is_pinned());
    }

    #[test]
    fn byte_cap_evicts_largest_first() {
        let _lock = lock_tests();
        const SIZE: usize = 1 << 16;
        const MARKER: usize = 1 << 10;
        // Start from a drained, empty pool so the accounting is deterministic.
        shutdown();
        set_max_pooled_bytes(2 * SIZE + MARKER);
        let bufs: Vec<_> = (0..3).map(|_| take(SIZE)).collect();
        drop(bufs);
        // The channel is FIFO, so once this later-returned marker is pooled,
        // all three SIZE buffers have been processed.
        drop(take(MARKER));
        wait_for_pooled(MARKER);
        {
            let pool = pool().lock().unwrap();
            assert_eq!(
                pool.by_size.get(&SIZE).map(|bufs| bufs.len()),
                Some(2),
                "third buffer should have evicted one of the first two"
            );
            assert_eq!(pool.total_bytes, 2 * SIZE + MARKER);
        }
        set_max_pooled_bytes(DEFAULT_MAX_POOLED_BYTES);
    }

    #[test]
    fn recorded_last_use_event_gates_reuse() {
        let _lock = lock_tests();
        const SIZE: usize = 1 << 14;
        let stream = crate::stream::CudaStream::new_non_blocking().unwrap();
        let mut buf = take(SIZE);
        let ptr = buf.as_ptr() as usize;
        buf.fill(0xCD);
        buf.record_last_use(&stream).unwrap();
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
