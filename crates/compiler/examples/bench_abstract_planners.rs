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
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

use crypto_compiler::planner::{
    load_abstract_timing_graph, perf_est, plan_heuristic_v2, plan_list_v1_v2, validate_plan,
    AbstractTimingGraph, ListSchedulerV1, PerfEst, PlanFn, StreamInstr, StreamMemoryPlan,
};

/// Report the first few validation errors for `plan` on `atg`.
/// Called by the two `run*` helpers after every successful plan so a
/// scheduler that produces an unsound schedule fails loudly.
fn check_valid(name: &str, atg: &AbstractTimingGraph, plan: &StreamMemoryPlan) {
    let errs = validate_plan(atg, plan);
    if errs.is_empty() {
        return;
    }
    println!("  ⚠ {name}: {} validation error(s):", errs.len());
    for e in errs.iter().take(3) {
        println!("      - {e}");
    }
    if errs.len() > 3 {
        println!("      … {} more", errs.len() - 3);
    }
}

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

fn run<F>(name: &str, atg: &AbstractTimingGraph, planner: F) -> Option<StreamMemoryPlan>
where
    F: FnOnce(
        &AbstractTimingGraph,
    ) -> Result<StreamMemoryPlan, crypto_compiler::planner::PlanError>,
{
    print!("{name:<18}  running... ");
    std::io::stdout().flush().ok();
    let t0 = Instant::now();
    let plan = match planner(atg) {
        Ok(p) => p,
        Err(e) => {
            println!("FAILED: {e:?}");
            return None;
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
    check_valid(name, atg, &plan);
    Some(plan)
}

/// Same as [`run`] but bounds wall-clock at `timeout` via a worker
/// thread + `mpsc::recv_timeout`. If the planner doesn't return in
/// time we print "TIMED OUT" and continue. The worker thread is
/// leaked (Rust has no cooperative-thread-cancel), so its `Arc<atg>`
/// keeps the graph alive on the abandoned thread until it exits on
/// its own.
fn run_with_timeout<F>(
    name: &str,
    atg: &Arc<AbstractTimingGraph>,
    timeout: Duration,
    planner: F,
) -> Option<StreamMemoryPlan>
where
    F: FnOnce(
            &AbstractTimingGraph,
        ) -> Result<StreamMemoryPlan, crypto_compiler::planner::PlanError>
        + Send
        + 'static,
{
    print!("{name:<22}  running... ");
    std::io::stdout().flush().ok();
    let t0 = Instant::now();
    let (tx, rx) = mpsc::channel();
    let atg_thread = Arc::clone(atg);
    thread::spawn(move || {
        let res = planner(&*atg_thread);
        let _ = tx.send(res);
    });
    match rx.recv_timeout(timeout) {
        Ok(Ok(plan)) => {
            let planner_ms = t0.elapsed().as_secs_f64() * 1e3;
            let PerfEst { time, peak_bytes } = perf_est(&*atg, &plan);
            println!(
                "streams={:>2}  events={:>5}  time={:>8.3} ms  peak={}  planner_cost={:>9.2} ms",
                plan.num_streams,
                plan.num_events,
                time,
                fmt_bytes(peak_bytes),
                planner_ms,
            );
            check_valid(name, &*atg, &plan);
            Some(plan)
        }
        Ok(Err(e)) => {
            println!("FAILED: {e:?}");
            None
        }
        Err(_) => {
            println!("TIMED OUT (>{:.1}s)", timeout.as_secs_f64());
            None
        }
    }
}

/// Simulate `plan` and return `(start_time[node], stream_wait[stream])`
/// under the same perfect-parallel model as `perf_est` — each stream
/// keeps an independent clock; a `WaitOn` bumps the clock to the event
/// time if it's ahead.
fn simulate_plan(atg: &AbstractTimingGraph, plan: &StreamMemoryPlan) -> (Vec<f64>, Vec<f64>) {
    let n_streams = plan.num_streams as usize;
    let n_events = plan.num_events as usize;
    let mut stream_time = vec![0.0f64; n_streams.max(1)];
    let mut stream_wait = vec![0.0f64; n_streams.max(1)];
    let mut event_time = vec![0.0f64; n_events];
    let mut start = vec![f64::NAN; atg.num_nodes];
    for instr in &plan.instructions {
        match *instr {
            StreamInstr::Node(v) => {
                let s = plan.stream[v] as usize;
                start[v] = stream_time[s];
                stream_time[s] += atg.node_times[v];
                if let Some(e) = plan.record_event[v] {
                    event_time[e as usize] = stream_time[s];
                }
            }
            StreamInstr::WaitOn(s, e) => {
                if let Some(t) = event_time.get(e).copied() {
                    if t > stream_time[s] {
                        stream_wait[s] += t - stream_time[s];
                        stream_time[s] = t;
                    }
                }
            }
        }
    }
    (start, stream_wait)
}

/// Bottom-level per node: longest weighted path from `v` to any sink
/// (including `v`'s own duration). Classic critical-path priority for
/// list scheduling.
fn bottom_levels(atg: &AbstractTimingGraph) -> Vec<f64> {
    let n = atg.num_nodes;
    // Successor list from buf_producers: v's successors are the nodes
    // that consume any buf v produces.
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for v in 0..n {
        for &bid in &atg.node_produces[v] {
            if let Some(users) = atg.buf_users.get(&bid) {
                for &u in users {
                    if u != v {
                        succ[v].push(u);
                    }
                }
            }
        }
        succ[v].sort_unstable();
        succ[v].dedup();
    }
    // Reverse-topo via Kahn's on the successor graph.
    let mut indeg = vec![0usize; n];
    for sv in &succ {
        for &u in sv {
            indeg[u] += 1;
        }
    }
    let mut ready: Vec<usize> = (0..n).filter(|&v| indeg[v] == 0).collect();
    let mut fwd = Vec::with_capacity(n);
    let mut cur = 0;
    while cur < ready.len() {
        let v = ready[cur];
        cur += 1;
        fwd.push(v);
        for &u in &succ[v] {
            indeg[u] -= 1;
            if indeg[u] == 0 {
                ready.push(u);
            }
        }
    }
    let mut bl = vec![0.0f64; n];
    for &v in fwd.iter().rev() {
        let mut m = 0.0f64;
        for &u in &succ[v] {
            if bl[u] > m {
                m = bl[u];
            }
        }
        bl[v] = atg.node_times[v] + m;
    }
    bl
}

fn analyze(atg: &AbstractTimingGraph, v1: &StreamMemoryPlan, v2: &StreamMemoryPlan) {
    let (start_v1, wait_v1) = simulate_plan(atg, v1);
    let (start_v2, wait_v2) = simulate_plan(atg, v2);
    let bl = bottom_levels(atg);

    // Per-stream sum of node_times and wait.
    let n = atg.num_nodes;
    let mut work_v1 = vec![0.0f64; v1.num_streams as usize];
    let mut work_v2 = vec![0.0f64; v2.num_streams as usize];
    for v in 0..n {
        if !start_v1[v].is_nan() {
            work_v1[v1.stream[v] as usize] += atg.node_times[v];
        }
        if !start_v2[v].is_nan() {
            work_v2[v2.stream[v] as usize] += atg.node_times[v];
        }
    }

    let critical_path = bl.iter().copied().fold(0.0f64, f64::max);
    let time_v1 = start_v1
        .iter()
        .enumerate()
        .filter(|(_, t)| !t.is_nan())
        .map(|(v, t)| t + atg.node_times[v])
        .fold(0.0f64, f64::max);
    let time_v2 = start_v2
        .iter()
        .enumerate()
        .filter(|(_, t)| !t.is_nan())
        .map(|(v, t)| t + atg.node_times[v])
        .fold(0.0f64, f64::max);
    println!();
    println!("=== list_v1 vs list_v2 diagnostics (streams=8) ===");
    println!("critical_path (lower bound): {:>8.3} ms", critical_path);
    println!(
        "makespan v1: {:>8.3} ms  |  v2: {:>8.3} ms  |  Δ: {:>+7.3} ms",
        time_v1,
        time_v2,
        time_v2 - time_v1
    );

    println!();
    println!("per-stream work (ms):");
    for s in 0..work_v1.len().max(work_v2.len()) {
        let w1 = work_v1.get(s).copied().unwrap_or(0.0);
        let w2 = work_v2.get(s).copied().unwrap_or(0.0);
        let iw1 = wait_v1.get(s).copied().unwrap_or(0.0);
        let iw2 = wait_v2.get(s).copied().unwrap_or(0.0);
        println!(
            "  s={s}: v1 work={:>7.3} wait={:>7.3}  |  v2 work={:>7.3} wait={:>7.3}",
            w1, iw1, w2, iw2
        );
    }

    // Top nodes where v2 starts later than v1.
    let mut deltas: Vec<(usize, f64)> = (0..n)
        .filter(|&v| !start_v1[v].is_nan() && !start_v2[v].is_nan())
        .map(|v| (v, start_v2[v] - start_v1[v]))
        .collect();
    deltas.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!();
    println!("top 15 nodes where v2 starts later than v1 (start Δ = v2 - v1):");
    println!(
        "  {:>5}  {:>6}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {}",
        "node", "Δ_ms", "rt_ms", "bl_ms", "s_v1", "s_v2", "st_v1", "st_v2"
    );
    for &(v, d) in deltas.iter().take(15) {
        println!(
            "  {:>5}  {:>+6.2}  {:>7.3}  {:>7.3}  {:>7}  {:>7}  {:>7.3}  {:>7.3}",
            v, d, atg.node_times[v], bl[v], v1.stream[v], v2.stream[v], start_v1[v], start_v2[v],
        );
    }

    // Nodes on the critical path (bl within 1% of critical path length).
    let cp_thresh = critical_path * 0.99;
    let mut cp_delayed: Vec<(usize, f64)> = (0..n)
        .filter(|&v| bl[v] >= cp_thresh && !start_v1[v].is_nan() && !start_v2[v].is_nan())
        .map(|v| (v, start_v2[v] - start_v1[v]))
        .collect();
    cp_delayed.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!();
    println!(
        "critical-path nodes (bl ≥ 99% of {:.3} ms): {} nodes, top 10 v2 delays:",
        critical_path,
        cp_delayed.len()
    );
    for &(v, d) in cp_delayed.iter().take(10) {
        println!(
            "  node {:>5}  Δ={:>+6.2} ms  bl={:>7.3}  s_v1={} s_v2={}  st_v1={:>7.3} st_v2={:>7.3}",
            v, d, bl[v], v1.stream[v], v2.stream[v], start_v1[v], start_v2[v]
        );
    }
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

    println!(
        "loading {} + {}",
        graph_path.display(),
        timing_path.display()
    );
    let atg = load_abstract_timing_graph(&graph_path, &timing_path).expect("load");
    let sum_ms: f64 = atg.node_times.iter().sum();
    // Cross-node dataflow edges = per-node consume set size, filtered to bufs
    // that some other node produces (external inputs contribute nothing).
    let n_edges: usize = (0..atg.num_nodes)
        .map(|v| {
            atg.node_consumes[v]
                .iter()
                .filter(|bid| {
                    atg.buf_producers
                        .get(bid)
                        .map(|ps| ps.iter().any(|&p| p != v))
                        .unwrap_or(false)
                })
                .count()
        })
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

    let max_memory_bound: usize = 2 * 1024 * 1024 * 1024; // 256 GiB
    let w_m: f64 = 1.0;
    let w_t: f64 = 1.0;
    let w_c: f64 = -2.0;

    let mut v1_s8: Option<StreamMemoryPlan> = None;
    let mut v2_s8: Option<StreamMemoryPlan> = None;

    // List scheduler at streams=1,2,4,8. Cap peak pool at 1 GiB to
    // match list_v2's memory footprint for a fair time comparison.
    //
    // Tunings (over the default) to survive the tight budget:
    // - `lookahead_k = 1`: rollout state.clone() is O(n_bufs) and
    //   dominates under memory pressure (each pick triggers many).
    // - `mem_target_frac = 0.0`: without this, the memory-pressure
    //   penalty only kicks in at 90% of `max_memory`; by then the
    //   greedy commit has already filled the pool with big writes
    //   whose readers can't be scheduled next (their writes wouldn't
    //   fit), and the loop deadlocks into Infeasible.
    // - `w_mem` raised so mem_pen (bytes over target) has the same
    //   magnitude as `w_cp * (bl + est_finish)` in ms — for a 1 GiB
    //   budget, bytes over 0 target dominate node timings unless
    //   w_mem is around ~1e-8.
    // The online scheduler needs headroom to make progress on this
    // graph; `offline_repack` packs the final peak down afterwards.
    let v1_max_memory: u64 = 2 * 1024 * 1024 * 1024; // 2 GiB (online budget)
    for max_conc in [8, 12] {
        let params = ListSchedulerV1 {
            max_concurrency: max_conc,
            max_memory: v1_max_memory,
            lookahead_k: 1,
            beam: 1,
            mem_target_frac: 0.0,
            w_cp: 0.0,
            w_mem: 1.0,
            ..ListSchedulerV1::default()
        };
        let plan = run(&format!("list_v1 s={max_conc}"), &atg, |g| {
            plan_list_v1_v2(g, params.clone())
        });
        if max_conc == 8 {
            v1_s8 = plan;
        }
    }

    let plan_timeout = Duration::from_secs(60);
    let frontier_cap: usize = 32;
    let atg_arc = Arc::new(atg);
    for &(num_beams, beam_depth) in &[(4, 1), (1, 1), (2, 2)] {
        println!();
        println!("--- list_v2 (num_beams={num_beams}, beam_depth={beam_depth}, frontier_cap={frontier_cap}) ---");
        for streams in [8, 16] {
            let name = format!("list_v2 s={streams} b={num_beams} d={beam_depth}");
            let plan = run_with_timeout(&name, &atg_arc, plan_timeout, move |g| {
                crypto_compiler::planner::list_v2::plan_v2(
                    g,
                    streams,
                    max_memory_bound,
                    num_beams,
                    beam_depth,
                    frontier_cap,
                    w_m,
                    w_t,
                    w_c,
                )
            });
            if streams == 8 && num_beams == 4 && beam_depth == 1 {
                v2_s8 = plan;
            }
        }
    }
    let atg = Arc::try_unwrap(atg_arc).unwrap_or_else(|arc| (*arc).clone());

    // Regular (heuristic) scheduler — single stream. On large graphs the
    // MAX_PASSES=50 hill-climb repack cost can dominate; set
    // `SKIP_HEURISTIC=1` to skip it.
    if std::env::var_os("SKIP_HEURISTIC").is_some() {
        println!("heuristic           skipped (SKIP_HEURISTIC set)");
    } else {
        run("heuristic", &atg, plan_heuristic_v2 as PlanFn);
    }

    if let (Some(v1), Some(v2)) = (v1_s8, v2_s8) {
        analyze(&atg, &v1, &v2);
    }
}
