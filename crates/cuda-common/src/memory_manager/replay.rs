//! Record-and-replay static allocation.
//!
//! When `VPMM_REPLAY` points at a `VPMM_TRACE` file recorded from an earlier
//! run of the same workload, the memory manager pre-plans all allocations of
//! at least `VPMM_REPLAY_MIN` bytes (default 16 MiB): a greedy packing pass
//! assigns each traced allocation a fixed offset in one workspace so that no
//! two allocations with overlapping traced lifetimes share memory. At run
//! time the k-th matching allocation is served at its planned offset — pointer
//! arithmetic instead of pool work — and frees of served pointers are pure
//! bookkeeping (no CUDA calls).
//!
//! Correctness does not depend on the run replaying the trace exactly:
//! - an allocation is served only if its size matches the next planned event AND its planned range
//!   overlaps no currently-live served allocation; otherwise it silently falls back to the dynamic
//!   allocator;
//! - replay assumes single-stream ordering (the recorded workloads run everything on one stream):
//!   if a second stream is observed, replay disables itself permanently and everything falls back.
//!
//! Divergence counters are logged when the plan is exhausted and on drop.

use std::collections::{BTreeMap, HashMap};

/// Minimum alignment for planned offsets; matches `cudaMalloc`'s guarantee.
const MIN_ALIGN: usize = 256;

struct PlannedEvent {
    size: usize,
    offset: usize,
}

pub(super) struct ReplayAllocator {
    /// Planned allocation events in traced order.
    seq: Vec<PlannedEvent>,
    next: usize,
    pub(super) min_size: usize,
    /// Total workspace bytes required by the plan.
    pub(super) total_size: usize,
    /// Base pointer of the workspace once allocated (lazily, on first serve).
    pub(super) workspace: Option<*mut std::ffi::c_void>,
    /// Live served allocations: ptr -> (offset, size).
    live: HashMap<usize, (usize, usize)>,
    /// Live planned intervals: offset -> end (for the aliasing safety check).
    live_intervals: BTreeMap<usize, usize>,
    /// The single stream replay is bound to (raw handle), set on first serve.
    stream: Option<u64>,
    disabled: bool,
    served: u64,
    fallback: u64,
    reported: bool,
}

impl ReplayAllocator {
    /// Builds the plan from `VPMM_REPLAY` if set. Panics on unreadable input:
    /// a silently absent plan would invalidate a replay benchmark.
    pub(super) fn from_env() -> Option<Self> {
        let path = std::env::var("VPMM_REPLAY").ok()?;
        let min_size = std::env::var("VPMM_REPLAY_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16 << 20);
        let data = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("VPMM_REPLAY: cannot read {path}: {e}"));
        Self::from_trace(&data, min_size, &path)
    }

    fn from_trace(data: &str, min_size: usize, path: &str) -> Option<Self> {
        // Parse plannable events: sizes + lifetime intervals in event order.
        let mut sizes: Vec<usize> = Vec::new();
        let mut births: Vec<u64> = Vec::new();
        let mut deaths: Vec<u64> = Vec::new();
        let mut live_ptr: HashMap<u64, usize> = HashMap::new();
        let mut clock: u64 = 0;
        for line in data.lines() {
            let mut f = line.split(',');
            let (op, ptr, size) = (
                f.next().unwrap_or(""),
                f.next().unwrap_or(""),
                f.next().unwrap_or(""),
            );
            match op {
                "A" => {
                    let ptr = u64::from_str_radix(ptr.trim_start_matches("0x"), 16)
                        .expect("VPMM_REPLAY: bad ptr");
                    let size: usize = size.parse().expect("VPMM_REPLAY: bad size");
                    if size >= min_size {
                        live_ptr.insert(ptr, sizes.len());
                        sizes.push(size);
                        births.push(clock);
                        deaths.push(u64::MAX);
                        clock += 1;
                    }
                }
                "F" => {
                    let ptr = u64::from_str_radix(ptr.trim_start_matches("0x"), 16)
                        .expect("VPMM_REPLAY: bad ptr");
                    if let Some(idx) = live_ptr.remove(&ptr) {
                        deaths[idx] = clock;
                        clock += 1;
                    }
                }
                _ => {}
            }
        }

        let (offsets, total_size) = plan_offsets(&sizes, &births, &deaths);
        if sizes.is_empty() || total_size == 0 {
            tracing::warn!(
                "VPMM_REPLAY: no allocations >= {min_size} bytes in {path}; replay disabled"
            );
            return None;
        }
        let seq = sizes
            .iter()
            .zip(&offsets)
            .map(|(&size, &offset)| PlannedEvent { size, offset })
            .collect::<Vec<_>>();
        tracing::info!(
            "VPMM_REPLAY: planned {} allocations (>= {} bytes) from {path} into a {} byte workspace",
            seq.len(),
            min_size,
            total_size,
        );
        Some(Self {
            seq,
            next: 0,
            min_size,
            total_size,
            workspace: None,
            live: HashMap::new(),
            live_intervals: BTreeMap::new(),
            stream: None,
            disabled: false,
            served: 0,
            fallback: 0,
            reported: false,
        })
    }

    /// Tries to serve an allocation from the plan. `None` means the caller
    /// must fall back to the dynamic allocator. The workspace must already be
    /// allocated by the caller.
    pub(super) fn try_alloc(&mut self, size: usize, stream: u64) -> Option<*mut std::ffi::c_void> {
        debug_assert!(size >= self.min_size);
        if self.disabled {
            return None;
        }
        match self.stream {
            None => self.stream = Some(stream),
            Some(s) if s != stream => {
                tracing::error!(
                    "VPMM_REPLAY: second stream observed; disabling replay (single-stream only)"
                );
                self.disabled = true;
                return None;
            }
            _ => {}
        }
        let base = self.workspace?;
        let ev = self.seq.get(self.next)?;
        if ev.size != size {
            self.note_fallback(size);
            return None;
        }
        let (offset, end) = (ev.offset, ev.offset + ev.size);
        // Aliasing safety: the planned range must not overlap any live served
        // allocation. Overlap here means the run diverged from the trace.
        if let Some((&o, &e)) = self.live_intervals.range(..end).next_back() {
            if e > offset && o < end {
                self.note_fallback(size);
                return None;
            }
        }
        self.next += 1;
        self.served += 1;
        let ptr = unsafe { base.byte_add(offset) };
        self.live.insert(ptr as usize, (offset, size));
        self.live_intervals.insert(offset, end);
        if self.next == self.seq.len() && !self.reported {
            self.reported = true;
            tracing::warn!(
                "VPMM_REPLAY: plan exhausted: served {} allocations, {} fell back to dynamic",
                self.served,
                self.fallback,
            );
        }
        Some(ptr)
    }

    /// Returns true if `ptr` was served from the workspace (bookkeeping-only free).
    pub(super) fn try_free(&mut self, ptr: *mut std::ffi::c_void) -> bool {
        if let Some((offset, _size)) = self.live.remove(&(ptr as usize)) {
            self.live_intervals.remove(&offset);
            true
        } else {
            false
        }
    }

    fn note_fallback(&mut self, size: usize) {
        self.fallback += 1;
        if self.fallback <= 10 {
            tracing::warn!(
                "VPMM_REPLAY: sequence divergence at planned event {} (requested {} bytes); \
                 serving dynamically",
                self.next,
                size,
            );
        }
    }
}

impl Drop for ReplayAllocator {
    fn drop(&mut self) {
        tracing::info!(
            "VPMM_REPLAY: final stats: served {}, fell back {}, plan position {}/{}",
            self.served,
            self.fallback,
            self.next,
            self.seq.len(),
        );
    }
}

/// Greedy best-fit-decreasing packing over the interval interference graph
/// (same heuristic as the static allocator's planner: place in decreasing
/// size order at the lowest 256-byte-aligned offset that does not overlap any
/// already-placed allocation with an overlapping lifetime).
fn plan_offsets(sizes: &[usize], births: &[u64], deaths: &[u64]) -> (Vec<usize>, usize) {
    let n = sizes.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(sizes[i]), births[i], i));

    let align_up = |x: usize| (x + MIN_ALIGN - 1) & !(MIN_ALIGN - 1);
    let mut offsets = vec![0usize; n];
    let mut placed: Vec<usize> = Vec::with_capacity(n);
    let mut total = 0usize;
    for &i in &order {
        let mut occupied: Vec<(usize, usize)> = placed
            .iter()
            .filter(|&&j| births[i] < deaths[j] && births[j] < deaths[i])
            .map(|&j| (offsets[j], offsets[j] + sizes[j]))
            .collect();
        occupied.sort_unstable();
        let mut cursor = 0usize;
        for (start, end) in occupied {
            if cursor + sizes[i] <= start {
                break;
            }
            cursor = cursor.max(align_up(end));
        }
        offsets[i] = cursor;
        placed.push(i);
        total = total.max(cursor + sizes[i]);
    }
    (offsets, total)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MI: usize = 1 << 20;

    /// A dummy device base pointer: replay only does pointer arithmetic on it.
    fn base() -> *mut std::ffi::c_void {
        0x1000_0000usize as *mut std::ffi::c_void
    }

    fn trace(events: &[(char, u64, usize)]) -> String {
        events
            .iter()
            .map(|&(op, ptr, size)| {
                if op == 'A' {
                    format!("A,{ptr:#x},{size},7,0,0\n")
                } else {
                    format!("F,{ptr:#x},,7,0,0\n")
                }
            })
            .collect()
    }

    fn replay_from(events: &[(char, u64, usize)]) -> ReplayAllocator {
        let mut r = ReplayAllocator::from_trace(&trace(events), 16 * MI, "test").unwrap();
        r.workspace = Some(base());
        r
    }

    #[test]
    fn test_convergent_run_serves_everything() {
        // A and B live concurrently, C reuses after both die.
        let ev = [
            ('A', 1, 32 * MI),
            ('A', 2, 32 * MI),
            ('F', 1, 0),
            ('F', 2, 0),
            ('A', 3, 32 * MI),
            ('F', 3, 0),
        ];
        let mut r = replay_from(&ev);
        assert_eq!(r.total_size, 64 * MI);

        let a = r.try_alloc(32 * MI, 7).unwrap();
        let b = r.try_alloc(32 * MI, 7).unwrap();
        assert_ne!(a, b, "concurrently live buffers must not alias");
        assert!(r.try_free(a));
        assert!(r.try_free(b));
        let c = r.try_alloc(32 * MI, 7).unwrap();
        assert!(r.try_free(c));
        assert_eq!(r.served, 3);
        assert_eq!(r.fallback, 0);
    }

    #[test]
    fn test_size_mismatch_falls_back_then_recovers() {
        let ev = [
            ('A', 1, 32 * MI),
            ('F', 1, 0),
            ('A', 2, 64 * MI),
            ('F', 2, 0),
        ];
        let mut r = replay_from(&ev);

        // Unexpected size: not served, cursor stays.
        assert!(r.try_alloc(48 * MI, 7).is_none());
        assert_eq!(r.fallback, 1);
        // The expected sequence still replays afterwards.
        let a = r.try_alloc(32 * MI, 7).unwrap();
        assert!(r.try_free(a));
        assert!(r.try_alloc(64 * MI, 7).is_some());
        assert_eq!(r.served, 2);
    }

    #[test]
    fn test_overlap_with_live_buffer_falls_back() {
        // In the trace, B reuses A's space after A dies. At run time A is
        // still held when B arrives: serving B would alias A, so it must
        // fall back even though the size matches.
        let ev = [
            ('A', 1, 32 * MI),
            ('F', 1, 0),
            ('A', 2, 32 * MI),
            ('F', 2, 0),
        ];
        let mut r = replay_from(&ev);
        assert_eq!(r.total_size, 32 * MI, "plan should reuse A's space for B");

        let a = r.try_alloc(32 * MI, 7).unwrap();
        assert!(r.try_alloc(32 * MI, 7).is_none(), "B would alias live A");
        assert_eq!(r.fallback, 1);
        assert!(r.try_free(a));
    }

    #[test]
    fn test_second_stream_disables_replay() {
        let ev = [
            ('A', 1, 32 * MI),
            ('F', 1, 0),
            ('A', 2, 32 * MI),
            ('F', 2, 0),
        ];
        let mut r = replay_from(&ev);
        assert!(r.try_alloc(32 * MI, 7).is_some());
        assert!(
            r.try_alloc(32 * MI, 8).is_none(),
            "other stream must not be served"
        );
        assert!(r.disabled);
        assert!(r.try_alloc(32 * MI, 7).is_none(), "replay stays disabled");
    }

    #[test]
    fn test_empty_plan_is_rejected() {
        assert!(ReplayAllocator::from_trace(&trace(&[('A', 1, MI)]), 16 * MI, "test").is_none());
        assert!(ReplayAllocator::from_trace("", 16 * MI, "test").is_none());
    }

    #[test]
    fn test_free_of_unknown_pointer_is_not_claimed() {
        let ev = [('A', 1, 32 * MI), ('F', 1, 0)];
        let mut r = replay_from(&ev);
        assert!(!r.try_free(base()), "never-served pointer is not replay's");
        let a = r.try_alloc(32 * MI, 7).unwrap();
        assert!(r.try_free(a));
        assert!(!r.try_free(a), "double free is not claimed twice");
    }
}
