//! Per-phase static-vs-dynamic analysis of a `VPMM_TRACE` allocation trace.
//!
//! Produces three tables:
//!   1. totals — dynamic allocator vs a whole-trace static plan;
//!   2. per-phase (`MemTracker` scope label) — dynamic stats per proving phase: allocation
//!      counts/bytes, measured allocator CPU time, peak memory while the phase was open;
//!   3. per-stage — proofs are delimited by `UNIT_LABEL` scope starts and classified into stages by
//!      ordinal (`STAGE_COUNTS`); each proof window gets its own static packing pass, mimicking a
//!      per-proof plan.
//!
//! Usage:
//!   TRACE=trace.csv [PAGE=4194304] [MIN_SIZE=16777216] \
//!   [UNIT_LABEL=prover.stack_traces] [STAGE_COUNTS=app:150,leaf:150,internal:76,root:1] \
//!     cargo bench -p openvm-cuda-common --bench trace_phases

use std::collections::HashMap;

use bytesize::ByteSize;
use openvm_cuda_common::static_alloc::{AllocBuilder, DeviceBufFake};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[derive(Clone, Copy)]
enum Ev {
    /// (alloc id, requested size)
    Alloc(u32, usize),
    /// (alloc id)
    Free(u32),
    /// (label id) — scope start
    Start(u32),
    /// (label id) — scope end
    End(u32),
}

#[derive(Default, Clone)]
struct PhaseStat {
    instances: usize,
    allocs: usize,
    frees: usize,
    alloc_bytes: u128,
    alloc_ns: u128,
    free_ns: u128,
    /// max over instances of (max dynamic tracked bytes while the scope was open)
    peak_dyn: usize,
    /// max over instances of (max raw live bytes while the scope was open)
    peak_live: usize,
}

fn gib(x: impl TryInto<u64>) -> String {
    ByteSize::b(x.try_into().unwrap_or(0)).to_string()
}

fn ms(ns: u128) -> String {
    format!("{:.1}ms", ns as f64 / 1e6)
}

fn main() {
    let path = std::env::var("TRACE").expect("set TRACE=<trace.csv>");
    let page = env_usize("PAGE", 4 << 20);
    let min_size = env_usize("MIN_SIZE", 16 << 20);
    let unit_label = std::env::var("UNIT_LABEL").unwrap_or_else(|_| "prover.stack_traces".into());
    let stage_counts: Vec<(String, usize)> = std::env::var("STAGE_COUNTS")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            let (name, n) = s.split_once(':').expect("STAGE_COUNTS=name:count,...");
            (name.to_string(), n.parse().expect("bad stage count"))
        })
        .collect();
    let data = std::fs::read_to_string(&path).expect("cannot read trace");

    // ---- Pass 1: parse into a compact event stream ------------------------
    let mut events: Vec<Ev> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    let mut label_ids: HashMap<String, u32> = HashMap::new();
    // alloc id -> (size, birth event idx, death event idx or MAX)
    let mut allocs: Vec<(usize, usize, usize)> = Vec::new();
    let mut live_ptr: HashMap<u64, u32> = HashMap::new();
    // parallel per-event alloc/free durations (ns), 0 for markers
    let mut durs: Vec<u64> = Vec::new();

    for line in data.lines() {
        let mut f = line.split(',');
        let (op, a, b, _stream, _t, dur) = (
            f.next().unwrap_or(""),
            f.next().unwrap_or(""),
            f.next().unwrap_or(""),
            f.next().unwrap_or(""),
            f.next().unwrap_or(""),
            f.next().unwrap_or(""),
        );
        match op {
            "A" => {
                let ptr = u64::from_str_radix(a.trim_start_matches("0x"), 16).unwrap();
                let size: usize = b.parse().unwrap();
                let id = allocs.len() as u32;
                allocs.push((size, events.len(), usize::MAX));
                live_ptr.insert(ptr, id);
                durs.push(dur.parse().unwrap_or(0));
                events.push(Ev::Alloc(id, size));
            }
            "F" => {
                let ptr = u64::from_str_radix(a.trim_start_matches("0x"), 16).unwrap();
                if let Some(id) = live_ptr.remove(&ptr) {
                    allocs[id as usize].2 = events.len();
                    durs.push(dur.parse().unwrap_or(0));
                    events.push(Ev::Free(id));
                }
            }
            "M" => {
                let (edge, label) = a.split_once(':').unwrap_or(("", a));
                if edge != "start" && edge != "end" {
                    continue; // "mark" intermediate points not used here
                }
                let id = *label_ids.entry(label.to_string()).or_insert_with(|| {
                    labels.push(label.to_string());
                    labels.len() as u32 - 1
                });
                durs.push(0);
                events.push(if edge == "start" {
                    Ev::Start(id)
                } else {
                    Ev::End(id)
                });
            }
            _ => {}
        }
    }

    // ---- Pass 2: walk events, accumulate per-phase + per-stage stats ------
    let tracked = |size: usize| {
        if size < page {
            size
        } else {
            size.next_multiple_of(page)
        }
    };
    let mut stats: Vec<PhaseStat> = vec![PhaseStat::default(); labels.len()];
    // open scope stack: (label id, peak_dyn so far, peak_live so far)
    let mut open: Vec<(u32, usize, usize)> = Vec::new();
    let mut dyn_cur = 0usize;
    let mut live_cur = 0usize;
    let (mut dyn_peak, mut live_peak) = (0usize, 0usize);
    let (mut total_alloc_ns, mut total_free_ns) = (0u128, 0u128);

    // proof-unit windows (event index ranges)
    let unit_id = label_ids.get(unit_label.as_str()).copied();
    let mut unit_starts: Vec<usize> = Vec::new();

    for (i, ev) in events.iter().enumerate() {
        match *ev {
            Ev::Alloc(_, size) => {
                dyn_cur += tracked(size);
                live_cur += size;
                dyn_peak = dyn_peak.max(dyn_cur);
                live_peak = live_peak.max(live_cur);
                total_alloc_ns += durs[i] as u128;
                if let Some(&mut (top, ..)) = open.last_mut() {
                    let s = &mut stats[top as usize];
                    s.allocs += 1;
                    s.alloc_bytes += size as u128;
                    s.alloc_ns += durs[i] as u128;
                }
                for o in open.iter_mut() {
                    o.1 = o.1.max(dyn_cur);
                    o.2 = o.2.max(live_cur);
                }
            }
            Ev::Free(id) => {
                let size = allocs[id as usize].0;
                dyn_cur -= tracked(size);
                live_cur -= size;
                total_free_ns += durs[i] as u128;
                if let Some(&mut (top, ..)) = open.last_mut() {
                    let s = &mut stats[top as usize];
                    s.frees += 1;
                    s.free_ns += durs[i] as u128;
                }
            }
            Ev::Start(l) => {
                if Some(l) == unit_id {
                    unit_starts.push(i);
                }
                open.push((l, dyn_cur, live_cur));
            }
            Ev::End(l) => {
                // scopes may close out of order: pop the most recent match
                if let Some(pos) = open.iter().rposition(|&(ol, ..)| ol == l) {
                    let (_, pd, pl) = open.remove(pos);
                    let s = &mut stats[l as usize];
                    s.instances += 1;
                    s.peak_dyn = s.peak_dyn.max(pd);
                    s.peak_live = s.peak_live.max(pl);
                }
            }
        }
    }

    // ---- Whole-trace static plan ------------------------------------------
    let global_plan = plan_window(&allocs, &events, 0, events.len(), min_size);

    println!(
        "trace: {path}  ({} events, {} allocs)",
        events.len(),
        allocs.len()
    );
    println!("\n=== totals: dynamic default vs static plan ===");
    println!(
        "  dynamic: peak tracked {} (page {}), raw peak-live {}",
        gib(dyn_peak),
        gib(page),
        gib(live_peak)
    );
    println!(
        "  dynamic: allocator CPU {} alloc + {} free over {} allocs",
        ms(total_alloc_ns),
        ms(total_free_ns),
        allocs.len()
    );
    println!(
        "  static : workspace {} = {:.1}% of dynamic peak (plan over {} allocs >= {}; efficiency {:.3})",
        gib(global_plan.0),
        100.0 * global_plan.0 as f64 / dyn_peak.max(1) as f64,
        global_plan.2,
        gib(min_size),
        global_plan.0 as f64 / global_plan.1.max(1) as f64,
    );

    // ---- Per-phase table ----------------------------------------------------
    println!("\n=== per-phase (MemTracker scopes; events attributed to innermost open scope) ===");
    println!(
        "{:<38} {:>5} {:>8} {:>10} {:>10} {:>10} {:>11} {:>11}",
        "phase", "inst", "allocs", "allocGiB", "alloc cpu", "free cpu", "peak dyn", "peak live"
    );
    let mut order: Vec<usize> = (0..labels.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(stats[i].alloc_bytes));
    for i in order {
        let s = &stats[i];
        if s.instances == 0 && s.allocs == 0 {
            continue;
        }
        println!(
            "{:<38} {:>5} {:>8} {:>10.2} {:>10} {:>10} {:>11} {:>11}",
            labels[i],
            s.instances,
            s.allocs,
            s.alloc_bytes as f64 / (1u64 << 30) as f64,
            ms(s.alloc_ns),
            ms(s.free_ns),
            gib(s.peak_dyn),
            gib(s.peak_live),
        );
    }

    // ---- Per-stage windows --------------------------------------------------
    if !unit_starts.is_empty() && !stage_counts.is_empty() {
        println!(
            "\n=== per-stage (windows of '{}' scopes; per-proof static plan) ===",
            unit_label
        );
        println!(
            "{:<10} {:>6} {:>10} {:>12} {:>12} {:>12} {:>10}",
            "stage", "proofs", "allocs", "alloc cpu", "max dyn", "max static", "ws/dyn"
        );
        let mut idx = 0usize;
        for (stage, count) in &stage_counts {
            let lo = idx.min(unit_starts.len());
            let hi = (idx + count).min(unit_starts.len());
            idx += count;
            if lo >= hi {
                continue;
            }
            let mut st_allocs = 0usize;
            let mut st_ns = 0u128;
            let (mut st_dyn, mut st_ws) = (0usize, 0usize);
            for w in lo..hi {
                let s = unit_starts[w];
                let e = if w + 1 < unit_starts.len() {
                    unit_starts[w + 1]
                } else {
                    events.len()
                };
                let mut dcur = live_at_tracked(&allocs, s, page);
                let mut dmax = dcur;
                for i in s..e {
                    match events[i] {
                        Ev::Alloc(_, size) => {
                            dcur += tracked(size);
                            dmax = dmax.max(dcur);
                            st_allocs += 1;
                            st_ns += durs[i] as u128;
                        }
                        Ev::Free(id) => dcur -= tracked(allocs[id as usize].0),
                        _ => {}
                    }
                }
                st_dyn = st_dyn.max(dmax);
                let (ws, ..) = plan_window(&allocs, &events, s, e, min_size);
                st_ws = st_ws.max(ws);
            }
            println!(
                "{:<10} {:>6} {:>10} {:>12} {:>12} {:>12} {:>9.1}%",
                stage,
                hi - lo,
                st_allocs,
                ms(st_ns),
                gib(st_dyn),
                gib(st_ws),
                100.0 * st_ws as f64 / st_dyn.max(1) as f64,
            );
        }
        if idx < unit_starts.len() {
            println!(
                "  (warning: {} proof windows beyond STAGE_COUNTS)",
                unit_starts.len() - idx
            );
        }
    } else {
        println!(
            "\n(per-stage table skipped: UNIT_LABEL '{}' seen {} times, STAGE_COUNTS {:?})",
            unit_label,
            unit_starts.len(),
            stage_counts
        );
    }
}

/// Tracked bytes of allocations alive at event index `s` (born before, not yet freed).
fn live_at_tracked(allocs: &[(usize, usize, usize)], s: usize, page: usize) -> usize {
    allocs
        .iter()
        .filter(|&&(_, b, d)| b < s && d >= s)
        .map(|&(size, ..)| {
            if size < page {
                size
            } else {
                size.next_multiple_of(page)
            }
        })
        .sum()
}

/// Packs the window `[s, e)` as its own static plan: carried-over live buffers
/// become window-start allocations, buffers still live at the end stay
/// immortal. Returns (workspace, peak_live, planned alloc count).
fn plan_window(
    allocs: &[(usize, usize, usize)],
    events: &[Ev],
    s: usize,
    e: usize,
    min_size: usize,
) -> (usize, usize, usize) {
    let builder = AllocBuilder::new();
    let mut fakes: HashMap<u32, DeviceBufFake<u8>> = HashMap::new();
    // Carried-over buffers first: all alive together at window start.
    for (id, &(size, b, d)) in allocs.iter().enumerate() {
        if size >= min_size && b < s && d >= s {
            fakes.insert(id as u32, builder.alloc_fake::<u8>(size));
        }
    }
    for ev in &events[s..e] {
        match *ev {
            Ev::Alloc(id, size) => {
                if size >= min_size {
                    fakes.insert(id, builder.alloc_fake::<u8>(size));
                }
            }
            Ev::Free(id) => {
                fakes.remove(&id);
            }
            _ => {}
        }
    }
    // Buffers still live at the window end die here, after every in-window
    // birth — equivalent to immortal for interference purposes.
    drop(fakes);
    let stats = builder.plan_stats();
    (stats.workspace_size, stats.peak_live_size, stats.num_allocs)
}
