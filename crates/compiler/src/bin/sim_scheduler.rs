//! `sim_scheduler` — offline scheduler simulator.
//!
//! Given
//!   1. a compiler-config TOML (see [`GraphCompilerConfig`]),
//!   2. a bincode-serialized `SerializableGraphBuilder` snapshot, and
//!   3. a JSON-serialized [`GraphInfo`] with per-node runtime timings,
//!
//! run the memory + stream planner with the config's scheduler (feeding
//! the timings in as `node_times`) and print stats:
//!   - graph inputs / outputs (shape, size, device),
//!   - peak pool bytes,
//!   - predicted latency (makespan) from replaying the plan with the timings,
//!   - per-stream busy / wait / idle breakdown and utilization.
//!
//! Usage:
//!   sim_scheduler <config.toml> <graph.bin> <timing.json>
//!
//! The graph snapshot must be a **post-fuse+dce** dump: this tool never
//! runs fusion. Post-pass node indices align with the timing JSON's
//! `nodes[]` (that's the contract of
//! [`crate::graph_exe::GraphExe::collect_graph_info`]) so the timings
//! feed straight into the scheduler as `node_times`.

use std::{collections::BTreeMap, path::PathBuf, process::ExitCode, time::Instant};

use crypto_compiler::{
    graph_compiler_config::GraphCompilerConfig,
    graph_info::GraphInfo,
    graph_ir::{BufId, DeviceType, GraphBuilder},
    graph_serializer::SerializableGraphBuilder,
    planner::{
        access_from_node, perf_est, plan_raw, validate_plan, AbstractTimingGraph, PerfEst,
        SchedulerMode, StreamInstr, StreamMemoryPlan, ValidationError,
    },
};

/// Aggregates over a `validate_plan` error list, splitting the
/// `MissingCrossStreamSync` bucket into "producer_idx > consumer_idx"
/// (a hallmark of the ATG's WAR-vs-RAW confusion for multi-writer
/// carried_outputs buffers) vs "real" RAW mismatches, and further
/// splitting the "real" bucket by whether the shared buffer between
/// producer and consumer has multiple writers (i.e. also an aliasing
/// artifact) or is SSA-clean (a true missing sync).
struct ValidationSummary {
    total: usize,
    missing_sync_war_alias: usize,
    missing_sync_real: usize,
    /// Of `missing_sync_real`, the number whose (producer, consumer)
    /// pair is only linked through a multi-writer BufId. These are
    /// still aliasing artifacts even though the producer index is
    /// lower than the consumer's.
    missing_sync_real_alias: usize,
    /// Of `missing_sync_real`, the number whose (producer, consumer)
    /// share an SSA-clean (single-writer) BufId — a genuine list_v1
    /// missing-sync bug candidate.
    missing_sync_real_ssa: usize,
    data_dep_races: usize,
    pool_overlaps: usize,
    other: usize,
    multi_writer_bufs: usize,
    max_writers_per_buf: usize,
}

fn summarize_validation(errors: &[ValidationError], atg: &AbstractTimingGraph) -> ValidationSummary {
    let mut s = ValidationSummary {
        total: errors.len(),
        missing_sync_war_alias: 0,
        missing_sync_real: 0,
        missing_sync_real_alias: 0,
        missing_sync_real_ssa: 0,
        data_dep_races: 0,
        pool_overlaps: 0,
        other: 0,
        multi_writer_bufs: 0,
        max_writers_per_buf: 0,
    };
    for e in errors {
        match e {
            ValidationError::MissingCrossStreamSync {
                producer, consumer, ..
            } => {
                // In insertion-order-versioned RAW dep model, a real
                // producer must have a strictly lower node index than
                // its consumer. ATG's unversioned `buf_producers` also
                // reports the WAR case (a later writer to a buffer
                // this consumer reads), which is not a required
                // predecessor.
                if producer > consumer {
                    s.missing_sync_war_alias += 1;
                } else {
                    s.missing_sync_real += 1;
                    if pair_only_shares_multi_writer_buf(atg, *producer, *consumer) {
                        s.missing_sync_real_alias += 1;
                    } else {
                        s.missing_sync_real_ssa += 1;
                    }
                }
            }
            ValidationError::DataDepRace { .. } => s.data_dep_races += 1,
            ValidationError::PoolLifetimeOverlap { .. } => s.pool_overlaps += 1,
            _ => s.other += 1,
        }
    }
    for producers in atg.buf_producers.values() {
        if producers.len() > 1 {
            s.multi_writer_bufs += 1;
        }
        if producers.len() > s.max_writers_per_buf {
            s.max_writers_per_buf = producers.len();
        }
    }
    s
}

/// Returns `true` iff every BufId written by `producer` and read by
/// `consumer` has more than one writer in the ATG. When `true`, the
/// (producer, consumer) pair only shares carry-chain / WAR-aliased
/// buffers — the reported sync miss is another aliasing artifact
/// rather than a real RAW missing sync.
fn pair_only_shares_multi_writer_buf(
    atg: &AbstractTimingGraph,
    producer: usize,
    consumer: usize,
) -> bool {
    let writes: std::collections::HashSet<BufId> =
        atg.node_produces[producer].iter().copied().collect();
    let mut any_shared = false;
    for b in &atg.node_consumes[consumer] {
        if !writes.contains(b) {
            continue;
        }
        any_shared = true;
        let n_writers = atg.buf_producers.get(b).map_or(0, |v| v.len());
        if n_writers <= 1 {
            return false; // SSA-clean shared buffer -> real RAW pair
        }
    }
    // If there's no shared buf at all, buf_producers alone can't
    // explain the pair — call it "not multi-writer aliased" so it
    // surfaces as an SSA-real case.
    any_shared
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("sim_scheduler: {e}");
            ExitCode::from(1)
        }
    }
}

struct Args {
    config: PathBuf,
    graph: PathBuf,
    timings: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().collect();
    if raw.len() != 4 {
        return Err(format!(
            "usage: {} <config.toml> <graph.bin> <timing.json>",
            raw.first().map(String::as_str).unwrap_or("sim_scheduler"),
        ));
    }
    Ok(Args {
        config: PathBuf::from(&raw[1]),
        graph: PathBuf::from(&raw[2]),
        timings: PathBuf::from(&raw[3]),
    })
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;

    // ---- Load inputs ----
    let config_text = std::fs::read_to_string(&args.config)
        .map_err(|e| format!("read {}: {e}", args.config.display()))?;
    let config: GraphCompilerConfig = toml::from_str(&config_text)
        .map_err(|e| format!("parse {}: {e}", args.config.display()))?;

    let graph_bytes =
        std::fs::read(&args.graph).map_err(|e| format!("read {}: {e}", args.graph.display()))?;
    let snapshot: SerializableGraphBuilder = bincode::deserialize(&graph_bytes)
        .map_err(|e| format!("bincode {}: {e}", args.graph.display()))?;
    let snapshot_original_hash = *snapshot.original_hash();
    let g: GraphBuilder = snapshot.into_graph_builder_offline();

    let timing_text = std::fs::read_to_string(&args.timings)
        .map_err(|e| format!("read {}: {e}", args.timings.display()))?;
    let timings: GraphInfo = serde_json::from_str(&timing_text)
        .map_err(|e| format!("parse {}: {e}", args.timings.display()))?;

    if snapshot_original_hash != timings.graph_hash {
        eprintln!(
            "warning: graph original_hash {} != timings.graph_hash {} — the timing JSON \
             may have been collected on a different source graph",
            hex_of(&snapshot_original_hash),
            hex_of(&timings.graph_hash),
        );
    }

    // Never fuse here — sim_scheduler operates on the post-fuse+dce
    // graph the timing JSON was collected against. If the two shapes
    // disagree the user handed us the wrong snapshot.
    if g.content_hash() == snapshot_original_hash {
        eprintln!(
            "warning: snapshot's content_hash matches its original_hash — this looks like a \
             pre-pass graph dump. sim_scheduler does not run fusion; feed the post-fuse+dce \
             snapshot (see load_or_compile_and_dump_with_hook's `fused_snapshot_path`) if the \
             node counts below don't line up."
        );
    }

    if g.nodes.len() != timings.nodes.len() {
        return Err(format!(
            "graph has {} nodes, timing JSON has {}. sim_scheduler does not fuse; \
             feed a post-fuse+dce snapshot whose node indices align with the timing JSON \
             (see GraphExe::collect_graph_info docs).",
            g.nodes.len(),
            timings.nodes.len(),
        )
        .into());
    }

    let node_times: Vec<f64> = timings.nodes.iter().map(|n| n.mean_ms).collect();

    // ---- Build scheduler with the timings injected as node_times ----
    let mut scheduler: SchedulerMode = config.scheduler.clone().into();
    match &mut scheduler {
        SchedulerMode::ListV1 { params } => params.node_times = node_times.clone(),
        SchedulerMode::ListV2 { params } => params.node_times = node_times.clone(),
        #[cfg(feature = "planner-ortools")]
        SchedulerMode::CpSat { .. } => {
            // CP-SAT joint-schedule doesn't consume per-node runtime timings;
            // the plan_raw contract is happy with an empty vec.
        }
    }

    // ---- Plan ----
    let device: DeviceType = config.device.clone().into();
    let env: BTreeMap<_, _> = BTreeMap::new();
    let pin: Vec<BufId> = g
        .input_bufs()
        .iter()
        .chain(g.output_bufs().iter())
        .copied()
        .collect();
    let t_access = Instant::now();
    let node_accesses: Vec<_> = g.nodes.iter().map(access_from_node).collect();
    let access_secs = t_access.elapsed().as_secs_f64();

    let t_plan = Instant::now();
    let plan = plan_raw(
        &g.bufs,
        &node_accesses,
        &env,
        device,
        &pin,
        &g.aliases,
        &scheduler,
    )
    .map_err(|e| format!("plan_raw: {e}"))?;
    let plan_secs = t_plan.elapsed().as_secs_f64();

    // ---- Validation + library-side perf estimate ----
    let t_atg = Instant::now();
    let atg = AbstractTimingGraph::from_graph_and_info(&g, &timings);
    let atg_secs = t_atg.elapsed().as_secs_f64();

    let t_validate = Instant::now();
    let validation_errors = validate_plan(&atg, &plan);
    let validate_secs = t_validate.elapsed().as_secs_f64();

    let t_perf = Instant::now();
    let perf = perf_est(&atg, &plan);
    let perf_secs = t_perf.elapsed().as_secs_f64();

    // ---- Report ----
    print_header(&args, &g, &timings);
    print_shapes(&g);
    print_plan_summary(&scheduler, &plan);
    print_planner_timing(access_secs, plan_secs, atg_secs, validate_secs, perf_secs);
    print_validation(&validation_errors, &atg);
    let sim = simulate(&plan, &node_times);
    print_latency(&sim, &perf);
    print_streams(&plan, &sim);

    if !validation_errors.is_empty() {
        return Err(format!(
            "plan validation returned {} error(s); see report above",
            validation_errors.len()
        )
        .into());
    }
    Ok(())
}

fn hex_of(h: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in h {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn print_header(args: &Args, g: &GraphBuilder, timings: &GraphInfo) {
    println!("=== sim_scheduler ===");
    println!("  config:  {}", args.config.display());
    println!("  graph:   {}", args.graph.display());
    println!("  timings: {}", args.timings.display());
    println!("  graph nodes: {}", g.nodes.len());
    println!(
        "  timings: {} iters (warmup {}), total per iter {:.3} ± {:.3} ms",
        timings.num_iters, timings.num_warmup, timings.total_ms_mean, timings.total_ms_std,
    );
}

fn print_shapes(g: &GraphBuilder) {
    fn print_row(idx: usize, buf_id: BufId, g: &GraphBuilder) {
        let info = g.buf_info(buf_id);
        let name = info.name.as_deref().unwrap_or("<unnamed>");
        let bytes = info.concrete_size;
        let elems = if info.elem_size == 0 {
            0
        } else {
            bytes / info.elem_size
        };
        println!(
            "  [{idx:>3}] buf={:<4} {name:<40}  {elems:>10} elems  x {:>2} B = {:>12} B  ({:?})",
            buf_id.0, info.elem_size, bytes, info.device_type,
        );
    }

    println!("\n--- inputs ({}) ---", g.input_bufs().len());
    for (i, &b) in g.input_bufs().iter().enumerate() {
        print_row(i, b, g);
    }

    println!("\n--- outputs ({}) ---", g.output_bufs().len());
    for (i, &b) in g.output_bufs().iter().enumerate() {
        print_row(i, b, g);
    }
}

fn scheduler_label(s: &SchedulerMode) -> &'static str {
    match s {
        #[cfg(feature = "planner-ortools")]
        SchedulerMode::CpSat { .. } => "cp_sat",
        SchedulerMode::ListV1 { .. } => "list_v1",
        SchedulerMode::ListV2 { .. } => "list_v2",
    }
}

fn print_plan_summary(sched: &SchedulerMode, plan: &StreamMemoryPlan) {
    println!("\n--- schedule ---");
    println!("  scheduler:     {}", scheduler_label(sched));
    println!("  streams:       {}", plan.num_streams);
    println!("  cross-stream events: {}", plan.num_events);
    let mib = plan.peak_bytes as f64 / (1024.0 * 1024.0);
    println!(
        "  peak pool:     {:>12} B  ({:.2} MiB)",
        plan.peak_bytes, mib,
    );
    let (n_launches, n_waits) =
        plan.instructions
            .iter()
            .fold((0usize, 0usize), |(n, w), i| match i {
                StreamInstr::Node(_) => (n + 1, w),
                StreamInstr::WaitOn(_, _) => (n, w + 1),
            });
    println!(
        "  instructions:  {} node launches, {} cross-stream waits",
        n_launches, n_waits,
    );
}

struct SimResult {
    /// Total wall-clock predicted latency (max over per-stream end times).
    makespan: f64,
    /// Sum of all node times as if run serially — an upper bound the
    /// scheduler is trying to beat.
    sum_serial: f64,
    /// Per-stream busy time (Σ node_times of nodes placed on that stream).
    busy: Vec<f64>,
    /// Per-stream stall time contributed by `WaitOn` instructions (only
    /// the delta between the current stream time and the event's ready
    /// time — a "free" WaitOn on an already-recorded event is 0).
    wait: Vec<f64>,
    /// Per-stream number of `Node(_)` instructions.
    nodes: Vec<usize>,
    /// Per-stream count of `WaitOn(this_stream, _)` instructions.
    wait_count: Vec<usize>,
}

fn simulate(plan: &StreamMemoryPlan, node_times: &[f64]) -> SimResult {
    let n_streams = plan.num_streams as usize;
    let n_events = plan.num_events as usize;
    let mut current_time = vec![0.0_f64; n_streams];
    let mut event_ready = vec![0.0_f64; n_events];
    let mut busy = vec![0.0_f64; n_streams];
    let mut wait = vec![0.0_f64; n_streams];
    let mut nodes = vec![0usize; n_streams];
    let mut wait_count = vec![0usize; n_streams];
    let mut sum_serial = 0.0_f64;

    for instr in &plan.instructions {
        match *instr {
            StreamInstr::Node(v) => {
                let s = plan.stream[v] as usize;
                let cost = node_times[v].max(0.0);
                current_time[s] += cost;
                busy[s] += cost;
                nodes[s] += 1;
                sum_serial += cost;
                if let Some(e) = plan.record_event[v] {
                    event_ready[e as usize] = current_time[s];
                }
            }
            StreamInstr::WaitOn(s_u, e_u) => {
                let ready = event_ready[e_u];
                if ready > current_time[s_u] {
                    wait[s_u] += ready - current_time[s_u];
                    current_time[s_u] = ready;
                }
                wait_count[s_u] += 1;
            }
        }
    }

    let makespan = current_time.iter().copied().fold(0.0_f64, f64::max);
    SimResult {
        makespan,
        sum_serial,
        busy,
        wait,
        nodes,
        wait_count,
    }
}

fn print_planner_timing(
    access_secs: f64,
    plan_secs: f64,
    atg_secs: f64,
    validate_secs: f64,
    perf_secs: f64,
) {
    println!("\n--- planner cost ---");
    println!("  access_from_node:    {:>10.3} ms", access_secs * 1e3);
    println!("  plan_raw:            {:>10.3} ms", plan_secs * 1e3);
    println!("  ATG build:           {:>10.3} ms", atg_secs * 1e3);
    println!("  validate_plan:       {:>10.3} ms", validate_secs * 1e3);
    println!("  perf_est:            {:>10.3} ms", perf_secs * 1e3);
    let total = access_secs + plan_secs + atg_secs + validate_secs + perf_secs;
    println!("  total:               {:>10.3} ms", total * 1e3);
}

fn print_validation(errors: &[ValidationError], atg: &AbstractTimingGraph) {
    println!("\n--- validation ---");
    if errors.is_empty() {
        println!("  OK — validate_plan found no inconsistencies");
        return;
    }
    let s = summarize_validation(errors, atg);
    println!(
        "  FAILED — {} inconsistenc{} found:",
        s.total,
        if s.total == 1 { "y" } else { "ies" },
    );
    println!(
        "    MissingCrossStreamSync (producer_idx > consumer_idx, WAR-alias): {}",
        s.missing_sync_war_alias,
    );
    println!(
        "    MissingCrossStreamSync (producer_idx <= consumer_idx, \"real\"):   {}",
        s.missing_sync_real,
    );
    println!(
        "      of which only share multi-writer bufs (also aliasing):        {}",
        s.missing_sync_real_alias,
    );
    println!(
        "      of which share an SSA-clean single-writer buf:                {}",
        s.missing_sync_real_ssa,
    );
    println!("    DataDepRace:                                                    {}", s.data_dep_races);
    println!("    PoolLifetimeOverlap:                                            {}", s.pool_overlaps);
    println!("    other:                                                          {}", s.other);
    println!(
        "  ATG buffer producers: {} bufs with >1 writer (max writers/buf = {})",
        s.multi_writer_bufs, s.max_writers_per_buf,
    );
    if s.missing_sync_war_alias > 0 && s.missing_sync_war_alias == s.total {
        println!(
            "  diagnosis: every error is a MissingCrossStreamSync where the reported \n\
             \"producer\" has a higher node index than the consumer. Those are WAR aliases \n\
             (later writers of a buffer the consumer already read) surfacing because \n\
             `AbstractTimingGraph::from_graph_and_info` lists all writers per BufId without \n\
             insertion-order versioning, and `validate_plan` treats every writer as a required \n\
             RAW predecessor. The plan itself is not necessarily broken — `PlanCtx::edges` (used \n\
             by list_v1) does version writers correctly."
        );
    }

    // Cap the printed list so a broken plan doesn't drown the report.
    const MAX_PRINTED: usize = 20;
    println!("  first {MAX_PRINTED} errors:");
    for (i, e) in errors.iter().take(MAX_PRINTED).enumerate() {
        println!("  [{i}] {e}");
    }
    if errors.len() > MAX_PRINTED {
        println!("  ... {} more", errors.len() - MAX_PRINTED);
    }

    if s.missing_sync_real > 0 {
        println!("  first {MAX_PRINTED} \"real\" MissingCrossStreamSync (producer_idx <= consumer_idx):");
        let mut shown = 0;
        for e in errors {
            if let ValidationError::MissingCrossStreamSync {
                producer, consumer, ..
            } = e
            {
                if producer <= consumer {
                    println!("    [{shown}] {e}");
                    shown += 1;
                    if shown == MAX_PRINTED {
                        break;
                    }
                }
            }
        }
    }
}

fn print_latency(sim: &SimResult, perf: &PerfEst) {
    println!("\n--- predicted latency ---");
    println!("  perf_est.time:       {:>10.3} ms", perf.time);
    println!("  local sim makespan:  {:>10.3} ms", sim.makespan);
    println!("  sum(node_times):     {:>10.3} ms", sim.sum_serial);
    let speedup = if perf.time > 0.0 {
        sim.sum_serial / perf.time
    } else {
        0.0
    };
    println!("  speedup vs serial:   {:>10.2}x", speedup);
    if (perf.time - sim.makespan).abs() > 1e-6 {
        println!(
            "  note: perf_est and local sim disagree by {:.3e} ms (should be within fp noise)",
            perf.time - sim.makespan,
        );
    }
}

fn print_streams(plan: &StreamMemoryPlan, sim: &SimResult) {
    println!("\n--- per-stream ---");
    println!(
        "  {:>4}  {:>6}  {:>10}  {:>10}  {:>10}  {:>6}  {:>7}",
        "sid", "nodes", "busy_ms", "wait_ms", "idle_ms", "util%", "waits",
    );
    for s in 0..plan.num_streams as usize {
        let busy = sim.busy[s];
        let wait = sim.wait[s];
        let idle = (sim.makespan - busy - wait).max(0.0);
        let util = if sim.makespan > 0.0 {
            busy / sim.makespan * 100.0
        } else {
            0.0
        };
        println!(
            "  {s:>4}  {:>6}  {busy:>10.3}  {wait:>10.3}  {idle:>10.3}  {util:>5.1}%  {:>7}",
            sim.nodes[s], sim.wait_count[s],
        );
    }
}
