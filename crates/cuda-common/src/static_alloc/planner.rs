//! Offline offset planning for the static allocator.
//!
//! Pure CPU logic with no CUDA dependencies: given the set of declared
//! allocations with their logical lifetime intervals, assign a byte offset
//! inside a single workspace to each allocation so that no two allocations
//! with interfering lifetimes overlap in memory, while keeping the total
//! workspace size close to the peak of concurrently live bytes.

use std::cmp::Reverse;

/// Minimum alignment (bytes) applied to every planned allocation. Matches the
/// 256-byte alignment guarantee of `cudaMalloc`, so kernel assumptions that
/// hold for individually `cudaMalloc`-ed buffers keep holding for planned
/// sub-buffers of the workspace.
pub const MIN_ALIGN: usize = 256;

/// Logical timestamp assigned by the [`AllocBuilder`](super::AllocBuilder)
/// clock. Each allocation and each drop gets a distinct tick.
pub(super) type Tick = u64;

/// A tick larger than any real event: the death of allocations still alive
/// when the plan is built.
pub(super) const IMMORTAL: Tick = Tick::MAX;

/// One declared allocation with its lifetime interval `[birth, death)`.
#[derive(Debug, Clone)]
pub(super) struct PlannedAlloc {
    /// Size in bytes (non-zero).
    pub size: usize,
    /// Required alignment in bytes (power of two); raised to [`MIN_ALIGN`]
    /// during placement.
    pub align: usize,
    pub birth: Tick,
    /// Exclusive end of the lifetime; [`IMMORTAL`] if alive at build time.
    pub death: Tick,
    /// Logical stream key. Allocations on different streams always interfere
    /// because reusing memory across streams would require extra
    /// synchronization that the planner does not insert.
    pub stream: u64,
    /// Human-readable name used in diagnostics.
    pub label: String,
}

impl PlannedAlloc {
    /// Whether two allocations may NOT share memory.
    pub(super) fn interferes(&self, other: &Self) -> bool {
        if self.stream != other.stream {
            return true;
        }
        self.birth < other.death && other.birth < self.death
    }
}

/// Result of packing: byte offsets parallel to the input allocations.
#[derive(Debug)]
pub(super) struct PlanLayout {
    pub offsets: Vec<usize>,
    /// Total workspace size in bytes.
    pub total_size: usize,
    /// Peak of concurrently live bytes over time — a lower bound for the
    /// workspace size of ANY layout (ignoring alignment and cross-stream
    /// constraints, so multi-stream plans may not be able to reach it).
    pub peak_live_size: usize,
}

#[inline]
fn align_up(x: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (x + align - 1) & !(align - 1)
}

/// Greedy best-fit-decreasing packing over the interference graph.
///
/// Allocations are placed in order of decreasing size (ties broken by birth
/// tick, then declaration order, so the result is fully deterministic). Each
/// allocation takes the lowest aligned offset that does not overlap any
/// already-placed allocation it interferes with. This is the classic
/// "greedy by size" heuristic for offline dynamic storage allocation; it is
/// not optimal (the problem is NP-hard) but stays close to the peak-live
/// lower bound on realistic skewed workloads.
pub(super) fn plan_offsets(allocs: &[PlannedAlloc]) -> PlanLayout {
    let mut order: Vec<usize> = (0..allocs.len()).collect();
    order.sort_by_key(|&i| (Reverse(allocs[i].size), allocs[i].birth, i));

    let mut offsets = vec![0usize; allocs.len()];
    let mut placed: Vec<usize> = Vec::with_capacity(allocs.len());
    let mut total_size = 0usize;

    for &i in &order {
        let alloc = &allocs[i];
        debug_assert_ne!(alloc.size, 0, "zero-size planned allocation");

        // Occupied ranges of already-placed interfering allocations.
        let mut occupied: Vec<(usize, usize)> = placed
            .iter()
            .filter(|&&j| alloc.interferes(&allocs[j]))
            .map(|&j| (offsets[j], offsets[j] + allocs[j].size))
            .collect();
        occupied.sort_unstable();

        // First fit: sweep the sorted ranges for the lowest aligned gap.
        let align = alloc.align.max(MIN_ALIGN);
        let mut cursor = 0usize;
        for (start, end) in occupied {
            if cursor + alloc.size <= start {
                break;
            }
            cursor = cursor.max(align_up(end, align));
        }

        offsets[i] = cursor;
        placed.push(i);
        total_size = total_size.max(cursor + alloc.size);
    }

    PlanLayout {
        offsets,
        total_size,
        peak_live_size: peak_live_size(allocs),
    }
}

/// Maximum over time of the sum of sizes of live allocations.
fn peak_live_size(allocs: &[PlannedAlloc]) -> usize {
    // (tick, delta); births and deaths have distinct ticks by construction.
    let mut events: Vec<(Tick, isize)> = Vec::with_capacity(allocs.len() * 2);
    for alloc in allocs {
        events.push((alloc.birth, alloc.size as isize));
        if alloc.death != IMMORTAL {
            events.push((alloc.death, -(alloc.size as isize)));
        }
    }
    events.sort_unstable();

    let mut live = 0isize;
    let mut peak = 0isize;
    for (_, delta) in events {
        live += delta;
        peak = peak.max(live);
    }
    peak as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alloc(size: usize, birth: Tick, death: Tick) -> PlannedAlloc {
        alloc_on_stream(size, birth, death, 0)
    }

    fn alloc_on_stream(size: usize, birth: Tick, death: Tick, stream: u64) -> PlannedAlloc {
        PlannedAlloc {
            size,
            align: MIN_ALIGN,
            birth,
            death,
            stream,
            label: format!("[{birth},{death})x{size}"),
        }
    }

    /// Checks that no two interfering allocations overlap in the layout and
    /// that all offsets are aligned.
    fn validate(allocs: &[PlannedAlloc], layout: &PlanLayout) {
        for (i, a) in allocs.iter().enumerate() {
            let a_range = layout.offsets[i]..layout.offsets[i] + a.size;
            assert_eq!(
                layout.offsets[i] % a.align.max(MIN_ALIGN),
                0,
                "misaligned offset for {}",
                a.label
            );
            assert!(a_range.end <= layout.total_size);
            for (j, b) in allocs.iter().enumerate().skip(i + 1) {
                if a.interferes(b) {
                    let b_range = layout.offsets[j]..layout.offsets[j] + b.size;
                    assert!(
                        a_range.end <= b_range.start || b_range.end <= a_range.start,
                        "interfering allocations '{}' and '{}' overlap: {:?} vs {:?}",
                        a.label,
                        b.label,
                        a_range,
                        b_range
                    );
                }
            }
        }
    }

    #[test]
    fn test_empty_plan() {
        let layout = plan_offsets(&[]);
        assert_eq!(layout.total_size, 0);
        assert_eq!(layout.peak_live_size, 0);
    }

    #[test]
    fn test_disjoint_lifetimes_share_space() {
        // A: [0,1), B: [2,3) — same stream, disjoint — must share offset 0.
        let allocs = vec![alloc(1 << 20, 0, 1), alloc(1 << 20, 2, 3)];
        let layout = plan_offsets(&allocs);
        validate(&allocs, &layout);
        assert_eq!(layout.offsets[0], layout.offsets[1]);
        assert_eq!(layout.total_size, 1 << 20);
    }

    #[test]
    fn test_overlapping_lifetimes_disjoint_space() {
        let allocs = vec![alloc(1 << 20, 0, 3), alloc(1 << 20, 1, 4)];
        let layout = plan_offsets(&allocs);
        validate(&allocs, &layout);
        assert_eq!(layout.total_size, 2 << 20);
    }

    #[test]
    fn test_cross_stream_always_interferes() {
        // Disjoint lifetimes but different streams — may not share memory.
        let allocs = vec![
            alloc_on_stream(1 << 20, 0, 1, 0),
            alloc_on_stream(1 << 20, 2, 3, 1),
        ];
        let layout = plan_offsets(&allocs);
        validate(&allocs, &layout);
        assert_ne!(layout.offsets[0], layout.offsets[1]);
        assert_eq!(layout.total_size, 2 << 20);
    }

    #[test]
    fn test_immortal_allocations_never_reused() {
        let allocs = vec![
            alloc(1 << 20, 0, IMMORTAL),
            alloc(1 << 20, 1, 2),
            alloc(1 << 20, 3, 4),
        ];
        let layout = plan_offsets(&allocs);
        validate(&allocs, &layout);
        // The two temps share space; the immortal buffer is separate.
        assert_eq!(layout.offsets[1], layout.offsets[2]);
        assert_ne!(layout.offsets[0], layout.offsets[1]);
        assert_eq!(layout.total_size, 2 << 20);
    }

    #[test]
    fn test_offsets_are_aligned_for_odd_sizes() {
        // Odd sizes force gaps: every offset must still be MIN_ALIGN-aligned.
        let allocs = vec![alloc(1000, 0, 5), alloc(999, 1, 5), alloc(1001, 2, 5)];
        let layout = plan_offsets(&allocs);
        validate(&allocs, &layout);
    }

    #[test]
    fn test_larger_alignment_respected() {
        let mut a = alloc(100, 0, 3);
        a.align = 1024;
        let allocs = vec![alloc(300, 0, 3), a];
        let layout = plan_offsets(&allocs);
        validate(&allocs, &layout);
        assert_eq!(layout.offsets[1] % 1024, 0);
    }

    #[test]
    fn test_deterministic() {
        let allocs: Vec<_> = (0u64..50)
            .map(|i| {
                alloc(
                    (((i * 7919) % 1000 + 1) << 10) as usize,
                    i,
                    i + ((i * 13) % 17) + 1,
                )
            })
            .collect();
        let l1 = plan_offsets(&allocs);
        let l2 = plan_offsets(&allocs);
        assert_eq!(l1.offsets, l2.offsets);
        assert_eq!(l1.total_size, l2.total_size);
    }

    #[test]
    fn test_phase_structured_reuse_hits_lower_bound() {
        // Two phases of equal-size temps + one long-lived output: the packer
        // should reach exactly the peak-live lower bound.
        let allocs = vec![
            alloc(4 << 20, 0, 4),        // phase 1 temp
            alloc(2 << 20, 1, 4),        // phase 1 temp
            alloc(1 << 20, 2, IMMORTAL), // output
            alloc(4 << 20, 5, 9),        // phase 2 temp (reuses phase 1 space)
            alloc(2 << 20, 6, 9),        // phase 2 temp
        ];
        let layout = plan_offsets(&allocs);
        validate(&allocs, &layout);
        assert_eq!(layout.total_size, layout.peak_live_size);
    }

    /// Deterministic pseudo-random stress: heavily skewed sizes, phase
    /// structure with interleaved lifetimes.
    #[test]
    fn test_fuzz_skewed_sizes() {
        // xorshift64* PRNG — deterministic, no external deps.
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut rng = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state = state.wrapping_mul(0x2545F4914F6CDD1D);
            state
        };

        for _round in 0..20 {
            let n = (rng() % 60 + 2) as usize;
            let mut clock: Tick = 0;
            let mut allocs = Vec::with_capacity(n);
            let mut live: Vec<usize> = Vec::new();
            for _ in 0..n {
                // Sizes skewed across 5 orders of magnitude.
                let size = 1usize << (rng() % 18 + 4);
                allocs.push(alloc(size, clock, IMMORTAL));
                live.push(allocs.len() - 1);
                clock += 1;
                // Randomly kill some live allocations.
                while !live.is_empty() && rng() % 3 == 0 {
                    let k = (rng() % live.len() as u64) as usize;
                    let idx = live.swap_remove(k);
                    allocs[idx].death = clock;
                    clock += 1;
                }
            }
            let layout = plan_offsets(&allocs);
            validate(&allocs, &layout);
            let sum: usize = allocs.iter().map(|a| a.size).sum();
            assert!(layout.total_size >= layout.peak_live_size);
            assert!(layout.total_size <= sum);
        }
    }
}
