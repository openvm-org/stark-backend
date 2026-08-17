//! Benchmark harness for candidate scheduling algorithms operating on
//! a dumped `AbstractTimingGraph`.
//!
//! Usage:
//!
//! ```
//! cargo run --release -p crypto-compiler --features planner \
//!     --example bench_abstract_planners -- <graph.bin> <timings.json>
//! ```
//!
//! Runs the heuristic (single-stream) packer plus `list_v1` at
//! `max_concurrency ∈ {1, 2, 4, 8}` over the loaded graph and prints
//! each planner's estimated wall-clock time (perfect-parallel stream
//! model) and packed-pool peak bytes.

use std::{
    io::Write,
    path::PathBuf,
    time::Instant,
};

use crypto_compiler::planner::{
    load_abstract_timing_graph, perf_est, plan_heuristic_v2, plan_list_v1_v2, AbstractTimingGraph,
    ListSchedulerV1, PerfEst, PlanFn, StreamMemoryPlan,
};

fn fmt_bytes(b: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * KIB;
    const GIB: f64 = MIB * KIB;
    let f = b as f64;
    if f >= GIB {
        format!("{:>7.2} GiB", f / GIB)
    } else if f >= MIB {
        format!("{:>7.2} MiB", f / MIB)
    } else if f >= KIB {
        format!("{:>7.2} KiB", f / KIB)
    } else {
        format!("{b:>7} B")
    }
}

fn run<F>(name: &str, atg: &AbstractTimingGraph, planner: F)
where
    F: FnOnce(&AbstractTimingGraph) -> Result<StreamMemoryPlan, crypto_compiler::planner::PlanError>,
{
    print!("{name:<18}  running... ");
    std::io::stdout().flush().ok();
    let t0 = Instant::now();
    let plan = match planner(atg) {
        Ok(p) => p,
        Err(e) => {
            println!("FAILED: {e:?}");
            return;
        }
    };
    let planner_ms = t0.elapsed().as_secs_f64() * 1e3;
    let PerfEst { time, peak_bytes } = perf_est(atg, &plan);
    println!(
        "streams={:>2}  events={:>5}  time={:>8.3} ms  peak={}  planner_cost={:>9.2} ms",
        plan.num_streams,
        plan.num_events,
        time,
        fmt_bytes(peak_bytes),
        planner_ms,
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let graph_path: PathBuf = args
        .next()
        .expect("usage: bench_abstract_planners <graph.bin> <timings.json>")
        .into();
    let timing_path: PathBuf = args
        .next()
        .expect("usage: bench_abstract_planners <graph.bin> <timings.json>")
        .into();

    println!("loading {} + {}", graph_path.display(), timing_path.display());
    let atg = load_abstract_timing_graph(&graph_path, &timing_path).expect("load");
    let sum_ms: f64 = atg.node_times.iter().sum();
    let n_edges: usize = (0..atg.num_nodes)
        .map(|v| atg.node_requires[v].len())
        .sum();
    println!(
        "graph:  num_nodes={}  bufs={}  edges={}  inputs={}  outputs={}  ready={}  Σnode_times={:.2} ms",
        atg.num_nodes,
        atg.buf_info.len(),
        n_edges,
        atg.inputs.len(),
        atg.outputs.len(),
        atg.inital_ready_nodes.len(),
        sum_ms,
    );
    println!();

    // List scheduler at streams=1,2,4,8.
    for max_conc in [1u32, 2, 4, 8] {
        let params = ListSchedulerV1 {
            max_concurrency: max_conc,
            ..ListSchedulerV1::default()
        };
        run(&format!("list_v1 s={max_conc}"), &atg, |g| {
            plan_list_v1_v2(g, params.clone())
        });
    }

    // Regular (heuristic) scheduler — single stream. On large graphs the
    // MAX_PASSES=50 hill-climb repack cost can dominate; set
    // `SKIP_HEURISTIC=1` to skip it.
    if std::env::var_os("SKIP_HEURISTIC").is_some() {
        println!("heuristic           skipped (SKIP_HEURISTIC set)");
    } else {
        run("heuristic", &atg, plan_heuristic_v2 as PlanFn);
    }
}
