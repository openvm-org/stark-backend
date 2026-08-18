//! Micro-benchmark for `list_v2`'s per-step primitives (`cost`,
//! `put_on`, `state.clone()`).
//!
//! Runs a greedy walk (no beam search) on a dumped
//! [`AbstractTimingGraph`], sampling each op millions of times, then
//! prints per-op averages and totals.
//!
//! Usage:
//!
//! ```
//! cargo run --release -p crypto-compiler --features planner \
//!     --example bench_list_v2_ops -- <graph.bin> <timings.json>
//! ```
//!
//! Optional env vars:
//! - `V2_OPS_STREAMS` — num_streams for the walk (default `8`)
//! - `V2_OPS_MAX_MEM_GIB` — memory bound in GiB (default `256`)

use std::path::PathBuf;

use crypto_compiler::planner::{
    list_v2::{bench_ops, OpBench},
    load_abstract_timing_graph,
};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let graph_path: PathBuf = args
        .next()
        .expect("usage: bench_list_v2_ops <graph.bin> <timings.json>")
        .into();
    let timing_path: PathBuf = args
        .next()
        .expect("usage: bench_list_v2_ops <graph.bin> <timings.json>")
        .into();

    let streams = env_usize("V2_OPS_STREAMS", 8);
    let max_mem_gib = env_usize("V2_OPS_MAX_MEM_GIB", 256);
    let max_memory_bound: usize = max_mem_gib * 1024 * 1024 * 1024;
    let w_m: f64 = 1.0;
    let w_t: f64 = 1.0;
    let w_c: f64 = -1.0;

    println!(
        "loading {} + {}",
        graph_path.display(),
        timing_path.display()
    );
    let atg = load_abstract_timing_graph(&graph_path, &timing_path).expect("load");
    let sum_ms: f64 = atg.node_times.iter().sum();
    println!(
        "graph:  num_nodes={}  bufs={}  inputs={}  outputs={}  Σnode_times={:.2} ms",
        atg.num_nodes,
        atg.buf_info.len(),
        atg.inputs.len(),
        atg.outputs.len(),
        sum_ms,
    );
    println!();
    println!(
        "--- list_v2 op timing (greedy walk, streams={}, max_mem={} GiB) ---",
        streams, max_mem_gib
    );

    let OpBench {
        nodes_scheduled,
        cost_calls,
        put_on_calls,
        clone_calls,
        cost_ns_avg,
        put_on_ns_avg,
        clone_ns_avg,
        total_ns,
    } = bench_ops(&atg, streams, max_memory_bound, w_m, w_t, w_c);

    println!(
        "  nodes_scheduled={}  cost_calls={}  put_on_calls={}  clone_calls={}",
        nodes_scheduled, cost_calls, put_on_calls, clone_calls
    );
    println!(
        "  cost_ns_avg  = {:>9.1} ns  ({:>7.2} ms total)",
        cost_ns_avg,
        (cost_ns_avg * cost_calls as f64) / 1e6
    );
    println!(
        "  put_on_ns_avg= {:>9.1} ns  ({:>7.2} ms total)",
        put_on_ns_avg,
        (put_on_ns_avg * put_on_calls as f64) / 1e6
    );
    println!(
        "  clone_ns_avg = {:>9.1} ns  ({:>7.2} ms total)",
        clone_ns_avg,
        (clone_ns_avg * clone_calls as f64) / 1e6
    );
    println!("  total wall   = {:>9.2} ms", total_ns as f64 / 1e6);
}
