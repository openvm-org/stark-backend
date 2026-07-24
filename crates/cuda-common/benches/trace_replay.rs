//! Replays a `VPMM_TRACE` allocation trace against the static allocator's
//! packing pass and reports the workspace a static plan would need for the
//! same workload, next to what the dynamic allocator actually consumed.
//!
//! Usage:
//!   TRACE=trace.csv [MIN_SIZE=2097152] [PAGE=4194304] \
//!     cargo bench -p openvm-cuda-common --bench trace_replay
//!
//! - `TRACE`: CSV produced by running any workload with `VPMM_TRACE=<path>`.
//! - `MIN_SIZE` (default 2 MiB): allocations smaller than this are excluded from the packing pass
//!   (they take the `cudaMallocAsync` path in the dynamic allocator and are not the static
//!   planner's target), but they are still counted in the reported dynamic peaks.
//! - `PAGE` (default 4 MiB): page size used to reproduce the dynamic allocator's size rounding for
//!   allocations >= `MIN_SIZE`.
//!
//! Requires a CUDA device only because the crate initializes its memory
//! manager at program start; the replay itself never touches the GPU.

use std::collections::HashMap;

use bytesize::ByteSize;
use openvm_cuda_common::static_alloc::{AllocBuilder, DeviceBufFake};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let path = std::env::var("TRACE").expect("set TRACE=<trace.csv>");
    let min_size = env_usize("MIN_SIZE", 2 << 20);
    let page = env_usize("PAGE", 4 << 20);
    let data = std::fs::read_to_string(&path).expect("cannot read trace");

    let builder = AllocBuilder::new();
    // ptr -> (fake handle or None if below MIN_SIZE, tracked size)
    let mut live: HashMap<u64, (Option<DeviceBufFake<u8>>, usize)> = HashMap::new();
    let (mut n_alloc, mut n_small, mut n_unmatched_free) = (0usize, 0usize, 0usize);
    // Dynamic-allocator accounting: sizes rounded exactly as the memory
    // manager tracks them (page multiples for pool-path allocations).
    let (mut dyn_cur, mut dyn_peak) = (0usize, 0usize);
    let mut streams: Vec<u64> = Vec::new();

    for (lineno, line) in data.lines().enumerate() {
        let mut fields = line.split(',');
        let (op, ptr, size, stream) = (
            fields.next().unwrap_or(""),
            fields.next().unwrap_or(""),
            fields.next().unwrap_or(""),
            fields.next().unwrap_or(""),
        );
        if op != "A" && op != "F" {
            continue; // phase markers etc.
        }
        let ptr = u64::from_str_radix(ptr.trim_start_matches("0x"), 16)
            .unwrap_or_else(|_| panic!("bad ptr at line {lineno}"));
        match op {
            "A" => {
                let size: usize = size
                    .parse()
                    .unwrap_or_else(|_| panic!("bad size @{lineno}"));
                let stream: u64 = stream.parse().unwrap_or(0);
                if !streams.contains(&stream) {
                    streams.push(stream);
                }
                n_alloc += 1;
                let tracked = if size < page {
                    size
                } else {
                    size.next_multiple_of(page)
                };
                dyn_cur += tracked;
                dyn_peak = dyn_peak.max(dyn_cur);
                let fake = if size >= min_size {
                    Some(builder.alloc_fake_on_stream::<u8>(size, &format!("l{lineno}"), stream))
                } else {
                    n_small += 1;
                    None
                };
                // A ptr can only be live once; VPMM may reuse addresses after
                // a free, which removes the old entry below.
                live.insert(ptr, (fake, tracked));
            }
            "F" => match live.remove(&ptr) {
                Some((fake, tracked)) => {
                    dyn_cur -= tracked;
                    drop(fake); // ends the declared lifetime
                }
                None => n_unmatched_free += 1,
            },
            _ => {}
        }
    }
    let leaked = live.len();
    drop(live); // still-live allocations stay immortal in the plan

    let stats = builder.plan_stats();
    println!("trace: {path}");
    println!(
        "  events: {n_alloc} allocs ({n_small} below {} excluded from plan), \
         {n_unmatched_free} unmatched frees, {leaked} never freed, {} stream(s)",
        ByteSize::b(min_size as u64),
        streams.len(),
    );
    println!(
        "  dynamic (page {}): peak tracked {}",
        ByteSize::b(page as u64),
        ByteSize::b(dyn_peak as u64)
    );
    println!(
        "  static plan over {} allocs: workspace {}, peak live {}, sum of sizes {}",
        stats.num_allocs,
        ByteSize::b(stats.workspace_size as u64),
        ByteSize::b(stats.peak_live_size as u64),
        ByteSize::b(stats.sum_of_sizes as u64)
    );
    println!(
        "  packing efficiency (workspace / peak-live): {:.3}",
        stats.workspace_size as f64 / stats.peak_live_size.max(1) as f64
    );
    println!(
        "  static workspace vs dynamic peak: {:.1}%",
        100.0 * stats.workspace_size as f64 / dyn_peak.max(1) as f64
    );
}
