//! Host-side ring-buffer management.
//!
//! We allocate a power-of-two ring of `CtaRecord` slots in pinned-mapped host
//! memory (`cudaHostAlloc(..., cudaHostAllocMapped)`), get the device pointer
//! via `cudaHostGetDevicePointer`, and pack both into the [`CtaProbeCtx`] that
//! every instrumented kernel launch reads through `shadow_cta_ctx()`.
//!
//! The head counter is allocated as a separate 8-byte pinned-mapped slot so
//! that GPU `atomicAdd` and host reads share an aligned 64-bit location.
//!
//! ## Producer/consumer protocol
//!
//! Producer (CTA_PROBE_END in `cta_probe.cuh`):
//!
//!   1. `slot = atomicAdd(head, 1)` — gpu-scope, gives a unique slot.
//!   2. Stores all non-tag fields of the record at `ring[slot & mask]`.
//!   3. `__threadfence_system()` — orders the stores before the next write.
//!   4. Stores `seq_tag = (slot + 1) & 0xFFFFFFFF` last.
//!
//! Consumer (this file's drain thread):
//!
//!   * Maintains `tail`: the next slot we expect to consume.
//!   * Reads the head with a relaxed volatile load to bound the work range.
//!   * For each slot `S` from `tail` up to `head_observed`, validates `record.seq_tag == ((S + 1) &
//!     0xFFFFFFFF)`.
//!     - If yes, the record is fully published; emit it and advance tail.
//!     - If no, the slot is either still being written or the producer has lapped us. Stop the
//!       sweep; we'll retry next cycle.
//!   * Overrun detection: if `head_observed - tail >= capacity` we have definitely lost records; we
//!     count and skip.
//!
//! This protocol is correct across ring wraps: the per-slot expected tag
//! is unique up to 2^32 records (~72 minutes at 1µs/CTA — see the comment
//! on `seq_tag` in `cta_probe.cuh`).
//!
//! ## Memory model on H100
//!
//! `cudaHostAllocMapped` is a UVA-backed allocation; the GPU writes to it
//! are write-combined and uncached on the device side, so the host sees
//! updates without an explicit memcpy. The `__threadfence_system()` in the
//! producer is what gives the host a coherent view of the writes-then-tag
//! ordering; without it, the seq_tag check would still occasionally accept a
//! torn record (stores reordered by the GPU memory subsystem).

use std::{
    ffi::c_void,
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use parking_lot::Mutex;

use crate::{
    ffi::{
        cudaFreeHost, cudaHostAlloc, cudaHostGetDevicePointer, CtaProbeCtx, CtaRecord,
        CUDA_HOST_ALLOC_MAPPED,
    },
    log_writer::LogWriter,
    record::Record,
};

/// Errors specific to ring allocation. We intentionally keep this small; the
/// caller wraps these into `ProfilerError::Cuda`.
#[derive(Debug)]
pub enum RingError {
    Alloc(i32),
    DevicePointer(i32),
    /// The requested capacity is too large; the mask must fit in u32.
    CapacityTooLarge(usize),
    /// Capacity must be at least 64 KB so the device sees a meaningful head
    /// counter even on tiny proofs.
    TooSmall(usize),
}

impl fmt::Display for RingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RingError::Alloc(rc) => write!(f, "cudaHostAlloc returned {rc}"),
            RingError::DevicePointer(rc) => write!(f, "cudaHostGetDevicePointer returned {rc}"),
            RingError::CapacityTooLarge(n) => {
                write!(
                    f,
                    "ring capacity {n} exceeds 2^31 records (mask must fit in u32)"
                )
            }
            RingError::TooSmall(n) => write!(f, "ring capacity {n} below minimum 65536"),
        }
    }
}

impl std::error::Error for RingError {}

/// Owns the pinned-mapped allocation. `Clone` produces a handle that *shares*
/// the same allocation (Arc inside); only the original owner frees on drop.
#[derive(Clone)]
pub struct HostRing {
    inner: Arc<HostRingInner>,
}

struct HostRingInner {
    /// Host-side address of the records ring. Same value as device-side
    /// pointer on UVA platforms (Linux x86_64 + H100), but obtained via
    /// `cudaHostGetDevicePointer` for correctness on non-UVA platforms.
    host_records: *mut CtaRecord,
    device_records: *mut CtaRecord,

    /// 8-byte head counter, separately allocated so the device atomicAdd hits
    /// a clean cache line and the host reads can use a relaxed atomic load.
    host_head: *mut u64,
    device_head: *mut u64,

    /// Number of records (power of two).
    capacity: u32,
    /// `capacity - 1`, used as the mask in the device-side `& mask` expression.
    mask: u32,
}

// SAFETY: pinned-mapped memory is process-global; the inner pointers are
// raw and we never dereference them in racy ways from Rust (the drain thread
// reads via raw pointer + Acquire load). Drop is single-threaded.
unsafe impl Send for HostRingInner {}
unsafe impl Sync for HostRingInner {}

impl HostRing {
    /// Allocates a ring of approximately `bytes` total bytes (rounded up to
    /// the next power-of-two record count). The minimum is 64 KB.
    pub fn allocate(bytes: usize) -> Result<Self, RingError> {
        const MIN_BYTES: usize = 64 * 1024;
        if bytes < MIN_BYTES {
            return Err(RingError::TooSmall(bytes));
        }

        let record_size = std::mem::size_of::<CtaRecord>();
        let raw_count = bytes.div_ceil(record_size);
        let capacity = raw_count.next_power_of_two();
        if capacity > (u32::MAX as usize / 2) + 1 {
            // mask = capacity - 1 must fit in u32; we keep capacity well below.
            return Err(RingError::CapacityTooLarge(capacity));
        }
        let capacity_bytes = capacity * record_size;

        // Records ring.
        let mut host_records: *mut c_void = std::ptr::null_mut();
        let rc =
            unsafe { cudaHostAlloc(&mut host_records, capacity_bytes, CUDA_HOST_ALLOC_MAPPED) };
        if rc != 0 {
            return Err(RingError::Alloc(rc));
        }
        let mut device_records: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { cudaHostGetDevicePointer(&mut device_records, host_records, 0) };
        if rc != 0 {
            unsafe { cudaFreeHost(host_records) };
            return Err(RingError::DevicePointer(rc));
        }

        // Zero the ring so that uninitialized slots don't look like real
        // records to the drain reader.
        unsafe {
            std::ptr::write_bytes(host_records as *mut u8, 0, capacity_bytes);
        }

        // Head counter. Allocated separately so the atomic hits its own line.
        let mut host_head: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { cudaHostAlloc(&mut host_head, 64, CUDA_HOST_ALLOC_MAPPED) };
        if rc != 0 {
            unsafe { cudaFreeHost(host_records) };
            return Err(RingError::Alloc(rc));
        }
        unsafe {
            std::ptr::write_bytes(host_head as *mut u8, 0, 64);
        }
        let mut device_head: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { cudaHostGetDevicePointer(&mut device_head, host_head, 0) };
        if rc != 0 {
            unsafe { cudaFreeHost(host_records) };
            unsafe { cudaFreeHost(host_head) };
            return Err(RingError::DevicePointer(rc));
        }

        Ok(Self {
            inner: Arc::new(HostRingInner {
                host_records: host_records as *mut CtaRecord,
                device_records: device_records as *mut CtaRecord,
                host_head: host_head as *mut u64,
                device_head: device_head as *mut u64,
                capacity: capacity as u32,
                mask: (capacity - 1) as u32,
            }),
        })
    }

    pub fn capacity(&self) -> u32 {
        self.inner.capacity
    }

    /// Returns the device-visible context for `shadow_set_cta_ctx`.
    pub fn ctx(&self) -> CtaProbeCtx {
        CtaProbeCtx {
            ring: self.inner.device_records,
            mask: self.inner.mask,
            _pad: 0,
            head: self.inner.device_head,
        }
    }

    /// Reads the current head, with Acquire ordering relative to the device.
    /// On the host, the device's writes via mapped memory aren't fully
    /// ordered against host loads without a CUDA fence, but for diagnostic
    /// sampling we accept best-effort visibility — the next drain catches up.
    fn read_head(&self) -> u64 {
        // We use a volatile read to defeat host CPU caching of the value;
        // pinned-mapped memory still goes through host caches on the CPU side.
        unsafe { std::ptr::read_volatile(self.inner.host_head) }
    }

    fn record_at(&self, slot: u32) -> CtaRecord {
        let idx = (slot & self.inner.mask) as isize;
        unsafe { std::ptr::read_volatile(self.inner.host_records.offset(idx)) }
    }
}

impl Drop for HostRingInner {
    fn drop(&mut self) {
        // Best-effort free; we ignore errors because the process is winding
        // down anyway.
        unsafe {
            if !self.host_records.is_null() {
                let _ = cudaFreeHost(self.host_records as *mut c_void);
            }
            if !self.host_head.is_null() {
                let _ = cudaFreeHost(self.host_head as *mut c_void);
            }
        }
    }
}

/// Spawn the drain thread. The thread holds an `Arc<HostRing>` keeping the
/// allocation alive, polls every `interval_ms`, and writes records via the
/// shared `LogWriter`. Stops on `stop` flag rising.
pub fn spawn_drain(
    ring: HostRing,
    writer: Arc<Mutex<LogWriter>>,
    stop: Arc<AtomicBool>,
    interval_ms: u64,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("cuda-profiler-drain".to_string())
        .spawn(move || drain_loop(ring, writer, stop, interval_ms))
        .expect("spawn drain thread")
}

fn drain_loop(
    ring: HostRing,
    writer: Arc<Mutex<LogWriter>>,
    stop: Arc<AtomicBool>,
    interval_ms: u64,
) {
    // `tail`: next slot we expect to consume. Lives in this thread.
    let tail = AtomicU64::new(0);
    let interval = Duration::from_millis(interval_ms);

    loop {
        let stopping = stop.load(Ordering::Acquire);
        drain_once(&ring, &writer, &tail);

        if stopping {
            // The shutdown sequence in lib.rs has already cleared the
            // device-visible ctx and synchronized the device, so no further
            // producers can advance head after this point. Do a final sweep
            // (which now reads the final stable head) and exit.
            drain_once(&ring, &writer, &tail);
            let _ = writer.lock().flush();
            return;
        }

        thread::sleep(interval);
    }
}

/// One drain sweep. Reads the current head, then walks forward from `*tail`
/// and consumes every slot whose `seq_tag` matches the expected value.
/// Stops at the first slot that hasn't been published yet, so a torn slot
/// never advances the tail.
fn drain_once(ring: &HostRing, writer: &Arc<Mutex<LogWriter>>, tail: &AtomicU64) {
    let cur_head = ring.read_head();
    let prev_tail = tail.load(Ordering::Relaxed);
    if cur_head <= prev_tail {
        return;
    }
    let cap = ring.capacity() as u64;

    // Overrun: the producer lapped us. Per `lane_hint` (now `seq_tag`)
    // ordering, the slots between `prev_tail` and `cur_head - cap` carry
    // tags from a future generation — they would still match a fresh-write
    // expected tag if the wrap-mod-2^32 happens to align, so we cannot
    // reliably consume them. Account for them as drops and skip ahead.
    let mut cursor = if cur_head.saturating_sub(prev_tail) > cap {
        let dropped = cur_head - prev_tail - cap;
        // Probe the slot at the new starting position to read its t_end —
        // gives the assembler an approximate "when did the overrun occur"
        // anchor instead of pinning the marker at t=0.
        let probe_slot = (cur_head - cap) & (ring.inner.mask as u64);
        let probe = ring.record_at(probe_slot as u32);
        let approx_t_ns = probe.t_end.max(probe.t_start);
        let _ = writer.lock().write_record(&Record::Drop {
            count: dropped,
            head_at_detection: cur_head,
            approx_t_ns,
        });
        cur_head - cap
    } else {
        prev_tail
    };

    let mut batch: Vec<CtaRecord> = Vec::with_capacity(4096);
    'sweep: while cursor < cur_head {
        let target_batch = std::cmp::min(cur_head - cursor, batch.capacity() as u64);
        let mut consumed_in_batch: u64 = 0;
        for i in 0..target_batch {
            let slot = cursor + i;
            let rec = ring.record_at((slot & (ring.inner.mask as u64)) as u32);
            let expected_tag = ((slot + 1) & 0xFFFFFFFF) as u32;
            if rec.seq_tag != expected_tag {
                // Slot not yet published. Don't advance past it on this
                // sweep; we'll retry next cycle. Stop the entire walk:
                // higher slots may already be published, but consuming them
                // out of order would lose this slot's record.
                break;
            }
            batch.push(rec);
            consumed_in_batch += 1;
        }
        if consumed_in_batch == 0 {
            break 'sweep;
        }
        {
            let mut w = writer.lock();
            for rec in &batch[..consumed_in_batch as usize] {
                let _ = w.write_record(&Record::Cta(*rec));
            }
        }
        cursor += consumed_in_batch;
        batch.clear();
        if consumed_in_batch < target_batch {
            // Hit a torn slot mid-batch.
            break 'sweep;
        }
    }

    tail.store(cursor, Ordering::Relaxed);
}
