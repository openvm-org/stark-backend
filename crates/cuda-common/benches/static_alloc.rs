//! Static vs dynamic allocation benchmark.
//!
//! Replays a STARK-prover-like allocation schedule (skewed sizes, phase
//! structure: commit -> perm -> quotient -> opening/FRI) twice:
//!   1. dynamically, through `DeviceBuffer::with_capacity`/drop (VPMM pool);
//!   2. statically, planning the same schedule with `AllocBuilder` and then serving every buffer
//!      from the packed workspace.
//!
//! Reports peak GPU memory, wall time per round, and cumulative CPU time
//! spent inside alloc/free (resp. get/drop) calls.
//!
//! Env knobs: `BENCH_NUM_AIRS` (default 40), `BENCH_LOG_MAX` (max log trace
//! height, default 19), `BENCH_ROUNDS` (measured rounds, default 3),
//! `BENCH_TOUCH` (memset every acquired buffer, default 1).
//!
//! Run: `cargo bench -p openvm-cuda-common --bench static_alloc`

use std::time::{Duration, Instant};

use bytesize::ByteSize;
use openvm_cuda_common::{
    d_buffer::DeviceBuffer,
    memory_manager::{memory_stats, MemTracker},
    static_alloc::{AllocBuilder, AllocIdx, StaticAllocator, StaticDeviceBuffer},
    stream::GpuDeviceCtx,
};

// ============================================================================
// Schedule: an abstract allocation trace with slot identities
// ============================================================================

enum Op {
    /// Allocate `len` u32 elements into `slot`.
    Alloc {
        slot: usize,
        len: usize,
        label: String,
    },
    /// Release `slot`.
    Free { slot: usize },
    /// Touch `slot` (async memset), standing in for kernel work.
    Touch { slot: usize },
}

struct Schedule {
    ops: Vec<Op>,
    num_slots: usize,
}

struct ScheduleBuilder {
    ops: Vec<Op>,
    next_slot: usize,
    touch: bool,
}

impl ScheduleBuilder {
    fn alloc(&mut self, len: usize, label: String) -> usize {
        let slot = self.next_slot;
        self.next_slot += 1;
        self.ops.push(Op::Alloc { slot, len, label });
        if self.touch {
            self.ops.push(Op::Touch { slot });
        }
        slot
    }

    fn free(&mut self, slot: usize) {
        self.ops.push(Op::Free { slot });
    }
}

/// xorshift64* — deterministic sizes without external deps.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0 = self.0.wrapping_mul(0x2545F4914F6CDD1D);
        self.0
    }

    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo + 1)
    }
}

/// Builds a prover-like schedule. Sizes are in u32 elements (4 bytes); the
/// extension-field factor (EF = 4 x F) is folded into element counts.
/// Lifetimes mirror the real pipeline: LDEs and digests live from commit
/// until the end of opening; per-phase temps die within their phase; the
/// quotient phase is serialized per AIR on purpose (as in the real prover).
fn build_schedule(num_airs: usize, log_max: u32, touch: bool) -> Schedule {
    let mut rng = Rng(0x5EED_5EED_5EED_5EED);
    let mut sb = ScheduleBuilder {
        ops: Vec::new(),
        next_slot: 0,
        touch,
    };

    // Skewed AIR shapes: mostly small, a few huge (heights 2^8..2^log_max).
    let mut airs: Vec<(u32, usize, usize)> = (0..num_airs) // (log_h, width, qd)
        .map(|_| {
            let p = rng.range(0, 99);
            let log_h = if p < 70 {
                rng.range(8, 13) as u32
            } else if p < 95 {
                rng.range(14, log_max as u64 - 2) as u32
            } else {
                rng.range(log_max as u64 - 1, log_max as u64) as u32
            };
            // Wide traces occur on small heights; cap width for huge ones.
            let width = if log_h >= 18 {
                rng.range(16, 128) as usize
            } else {
                rng.range(8, 400) as usize
            };
            let qd = 1usize << rng.range(0, 2);
            (log_h, width, qd)
        })
        .collect();
    airs.sort_by_key(|&(log_h, ..)| std::cmp::Reverse(log_h));

    const BLOWUP: usize = 2;
    let mut retained: Vec<usize> = Vec::new(); // slots that live until the end

    // Phase 1: commit — per AIR: trace (temp) -> LDE + digests (retained).
    for (i, &(log_h, width, _)) in airs.iter().enumerate() {
        let h = 1usize << log_h;
        let trace = sb.alloc(h * width, format!("trace{i}"));
        let lde = sb.alloc(BLOWUP * h * width, format!("lde{i}"));
        let digest = sb.alloc(2 * BLOWUP * h * 8, format!("digest{i}"));
        sb.free(trace);
        retained.push(lde);
        retained.push(digest);
    }

    // Phase 2: perm — every other AIR gets a perm trace, committed together.
    for (i, &(log_h, ..)) in airs.iter().enumerate().filter(|(i, _)| i % 2 == 0) {
        let h = 1usize << log_h;
        let perm_width = 4 * rng.range(2, 8) as usize;
        let cumsum = sb.alloc(h * 4, format!("cumsum{i}"));
        let perm_lde = sb.alloc(BLOWUP * h * perm_width, format!("perm_lde{i}"));
        sb.free(cumsum);
        retained.push(perm_lde);
    }

    // Phase 3: quotient — per AIR serialized temps, retained chunk LDEs.
    for (i, &(log_h, _, qd)) in airs.iter().enumerate() {
        let h = 1usize << log_h;
        let qsize = h * qd;
        let acc = sb.alloc(qsize * 4, format!("q_acc{i}"));
        let selectors = sb.alloc(4 * qsize, format!("q_sel{i}"));
        let chunks_lde = sb.alloc(BLOWUP * h * 4 * qd, format!("q_lde{i}"));
        sb.free(selectors);
        sb.free(acc);
        retained.push(chunks_lde);
    }

    // Phase 4: opening — inv denominators + reduced openings per distinct
    // height, then the FRI fold ladder from the max LDE height down.
    let mut heights: Vec<u32> = airs.iter().map(|&(log_h, ..)| log_h).collect();
    heights.dedup();
    let mut opening_temps = Vec::new();
    for &log_h in &heights {
        let lde_h = 1usize << (log_h + 1);
        let inv_denom = sb.alloc(lde_h * 4, format!("inv_denom{log_h}"));
        let reduced = sb.alloc(lde_h * 4, format!("reduced{log_h}"));
        opening_temps.push(inv_denom);
        opening_temps.push(reduced);
    }
    let max_lde = 1usize << (heights[0] + 1);
    let mut len = max_lde;
    let mut layer = 0;
    while len > 32 {
        let folded = sb.alloc(len / 2 * 8, format!("fri_folded{layer}"));
        let g_inv = sb.alloc(len / 2, format!("fri_ginv{layer}"));
        let d_result = sb.alloc(len / 2 * 4, format!("fri_dres{layer}"));
        sb.free(g_inv);
        sb.free(d_result);
        retained.push(folded);
        len /= 2;
        layer += 1;
    }
    for slot in opening_temps {
        sb.free(slot);
    }

    // Proof done: everything device-side is dropped.
    for slot in retained {
        sb.free(slot);
    }

    Schedule {
        num_slots: sb.next_slot,
        ops: sb.ops,
    }
}

// ============================================================================
// Runners
// ============================================================================

struct RunStats {
    wall: Duration,
    alloc_cpu: Duration,
    free_cpu: Duration,
}

fn run_dynamic(schedule: &Schedule, ctx: &GpuDeviceCtx) -> RunStats {
    let mut slots: Vec<Option<DeviceBuffer<u32>>> = (0..schedule.num_slots).map(|_| None).collect();
    let mut alloc_cpu = Duration::ZERO;
    let mut free_cpu = Duration::ZERO;
    ctx.stream.synchronize().unwrap();
    let start = Instant::now();
    for op in &schedule.ops {
        match op {
            Op::Alloc { slot, len, .. } => {
                let t = Instant::now();
                slots[*slot] = Some(DeviceBuffer::with_capacity_on(*len, ctx));
                alloc_cpu += t.elapsed();
            }
            Op::Free { slot } => {
                let t = Instant::now();
                slots[*slot] = None;
                free_cpu += t.elapsed();
            }
            Op::Touch { slot } => {
                slots[*slot].as_ref().unwrap().fill_zero_on(ctx).unwrap();
            }
        }
    }
    ctx.stream.synchronize().unwrap();
    RunStats {
        wall: start.elapsed(),
        alloc_cpu,
        free_cpu,
    }
}

fn plan_static(
    schedule: &Schedule,
    ctx: &GpuDeviceCtx,
) -> (StaticAllocator, Vec<AllocIdx<u32>>, Duration) {
    let start = Instant::now();
    let builder = AllocBuilder::new();
    let mut fakes: Vec<Option<openvm_cuda_common::static_alloc::DeviceBufFake<u32>>> =
        (0..schedule.num_slots).map(|_| None).collect();
    let mut idxs: Vec<Option<AllocIdx<u32>>> = vec![None; schedule.num_slots];
    for op in &schedule.ops {
        match op {
            Op::Alloc { slot, len, label } => {
                let fake = builder.alloc_fake_labeled::<u32>(*len, label);
                idxs[*slot] = Some(fake.idx());
                fakes[*slot] = Some(fake);
            }
            Op::Free { slot } => {
                fakes[*slot] = None;
            }
            Op::Touch { .. } => {}
        }
    }
    drop(fakes);
    let alloc = builder.build_on(ctx).unwrap();
    let elapsed = start.elapsed();
    (
        alloc,
        idxs.into_iter().map(Option::unwrap).collect(),
        elapsed,
    )
}

fn run_static(
    schedule: &Schedule,
    alloc: &StaticAllocator,
    idxs: &[AllocIdx<u32>],
    ctx: &GpuDeviceCtx,
) -> RunStats {
    let mut slots: Vec<Option<StaticDeviceBuffer<u32>>> =
        (0..schedule.num_slots).map(|_| None).collect();
    let mut alloc_cpu = Duration::ZERO;
    let mut free_cpu = Duration::ZERO;
    ctx.stream.synchronize().unwrap();
    let start = Instant::now();
    for op in &schedule.ops {
        match op {
            Op::Alloc { slot, .. } => {
                let t = Instant::now();
                slots[*slot] = Some(alloc.get(&idxs[*slot]).unwrap());
                alloc_cpu += t.elapsed();
            }
            Op::Free { slot } => {
                let t = Instant::now();
                slots[*slot] = None;
                free_cpu += t.elapsed();
            }
            Op::Touch { slot } => {
                slots[*slot].as_ref().unwrap().fill_zero_on(ctx).unwrap();
            }
        }
    }
    ctx.stream.synchronize().unwrap();
    RunStats {
        wall: start.elapsed(),
        alloc_cpu,
        free_cpu,
    }
}

// ============================================================================
// Micro-benchmark: raw alloc/free vs get/drop cost
// ============================================================================

fn micro_alloc_overhead(iters: usize, ctx: &GpuDeviceCtx) {
    let len = 1usize << 20; // 4 MB
    let t = Instant::now();
    for _ in 0..iters {
        let buf = DeviceBuffer::<u32>::with_capacity_on(len, ctx);
        std::hint::black_box(buf.as_ptr());
    }
    let dynamic = t.elapsed();

    let builder = AllocBuilder::new();
    let idx = builder.alloc_fake_labeled::<u32>(len, "micro").idx();
    let alloc = builder.build_on(ctx).unwrap();
    let t = Instant::now();
    for _ in 0..iters {
        let buf = alloc.get(&idx).unwrap();
        std::hint::black_box(buf.as_ptr());
    }
    let static_ = t.elapsed();

    println!(
        "micro ({iters} iters of 4 MB alloc+free): dynamic {:?}/op, static {:?}/op",
        dynamic / iters as u32,
        static_ / iters as u32
    );
}

// ============================================================================
// Main
// ============================================================================

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let num_airs = env_usize("BENCH_NUM_AIRS", 40);
    assert!(num_airs >= 1, "BENCH_NUM_AIRS must be >= 1");
    let log_max = env_usize("BENCH_LOG_MAX", 19) as u32;
    assert!(
        (17..=26).contains(&log_max),
        "BENCH_LOG_MAX must be in 17..=26"
    );
    let rounds = env_usize("BENCH_ROUNDS", 3);
    assert!(rounds >= 1, "BENCH_ROUNDS must be >= 1");
    let touch = env_usize("BENCH_TOUCH", 1) != 0;

    let ctx = GpuDeviceCtx::for_current_device().unwrap();
    let schedule = build_schedule(num_airs, log_max, touch);
    let num_allocs = schedule
        .ops
        .iter()
        .filter(|op| matches!(op, Op::Alloc { .. }))
        .count();
    println!(
        "schedule: {num_airs} AIRs, max height 2^{log_max}, {num_allocs} allocations, touch={touch}"
    );

    // --- Dynamic ---
    let mut tracker = MemTracker::start("bench-dynamic");
    tracker.reset_peak();
    let (_, base_peak) = memory_stats();
    let mut dyn_stats = Vec::new();
    for round in 0..=rounds {
        let stats = run_dynamic(&schedule, &ctx);
        if round > 0 {
            dyn_stats.push(stats); // round 0 = warm-up
        }
    }
    let (_, dyn_peak) = memory_stats();
    let dyn_peak = dyn_peak - base_peak.min(dyn_peak);

    // --- Static ---
    let (alloc, idxs, plan_time) = plan_static(&schedule, &ctx);
    println!(
        "static plan: workspace {} (peak live {}), planned in {:?}",
        ByteSize::b(alloc.workspace_size() as u64),
        ByteSize::b(alloc.peak_live_size() as u64),
        plan_time
    );
    let mut static_stats = Vec::new();
    for round in 0..=rounds {
        let stats = run_static(&schedule, &alloc, &idxs, &ctx);
        if round > 0 {
            static_stats.push(stats);
        }
    }

    // --- Report ---
    let avg = |stats: &[RunStats]| {
        let n = stats.len() as u32;
        (
            stats.iter().map(|s| s.wall).sum::<Duration>() / n,
            stats.iter().map(|s| s.alloc_cpu).sum::<Duration>() / n,
            stats.iter().map(|s| s.free_cpu).sum::<Duration>() / n,
        )
    };
    let (d_wall, d_alloc, d_free) = avg(&dyn_stats);
    let (s_wall, s_alloc, s_free) = avg(&static_stats);

    println!("\n=== results (avg over {rounds} rounds) ===");
    println!(
        "dynamic: wall {:?}, alloc cpu {:?}, free cpu {:?}, peak mem {}",
        d_wall,
        d_alloc,
        d_free,
        ByteSize::b(dyn_peak as u64)
    );
    println!(
        "static : wall {:?}, get cpu {:?}, drop cpu {:?}, workspace {}",
        s_wall,
        s_alloc,
        s_free,
        ByteSize::b(alloc.workspace_size() as u64)
    );
    if dyn_peak > 0 {
        println!(
            "workspace vs dynamic peak: {:.1}%",
            100.0 * alloc.workspace_size() as f64 / dyn_peak as f64
        );
    }
    println!(
        "packing efficiency (workspace / peak-live lower bound): {:.3}",
        alloc.workspace_size() as f64 / alloc.peak_live_size() as f64
    );

    micro_alloc_overhead(1000, &ctx);
}
