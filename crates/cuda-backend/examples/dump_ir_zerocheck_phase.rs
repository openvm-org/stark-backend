//! Builds the **whole logup-zerocheck phase** as a graph-IR graph and
//! compiles it, dumping the HIR / KIR / CUDA sources the compiler emits.
//!
//! This is the acceptance evidence for INT-8917 ("capture the entirety of
//! `logup_zerocheck_gpu` in a graph"): stages C (univariate round 0),
//! D (the `n_max` MLE rounds, fully unrolled) and E (column openings) of
//! `logup_zerocheck::prove_zerocheck_and_logup_gpu` are emitted onto one
//! `GraphBuilder` by `logup_zerocheck::zerocheck_ir::logup_zerocheck_gpu_ir`,
//! and `GraphCompiler::compile` turns them into a `GraphExe`.
//!
//! Usage:
//!   cargo run -p openvm-cuda-backend --release --example dump_ir_zerocheck_phase --features
//! graph-ir
//!
//! By default writes to `target/ir-dumps/zerocheck_phase`. Set `DUMP_DIR` to
//! override, or `NUM_TRACES` / `L_SKIP` / `N_MAX` to pick a different phase
//! shape (defaults: 3 traces, `l_skip = 2`, `n_max = 4`).
//!
//! The graph is built with **zeroed** trace inputs
//! (`TraceBufs::alloc_zeroed`), so it is not run here — running it needs a
//! real proving key and trace. Building and compiling it is the point.

use std::path::PathBuf;

use crypto_compiler::{
    graph_exe::GraphCompiler,
    graph_ir::{DeviceType, GraphBuilder},
    planner::SchedulerMode,
    runtime::Verbosity,
};
use openvm_cuda_backend::{
    logup_zerocheck::zerocheck_ir::{logup_zerocheck_gpu_ir, synthetic_plan, TraceBufs},
    sponge_graph_ir::DuplexSpongeGpuIR,
};

fn parse_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn main() {
    let num_traces = parse_usize("NUM_TRACES", 3);
    let l_skip = parse_usize("L_SKIP", 2);
    let n_max = parse_usize("N_MAX", 4);
    let dump_dir: PathBuf = std::env::var_os("DUMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../target/ir-dumps/zerocheck_phase"
            ))
        });

    println!(
        "building logup-zerocheck phase graph: num_traces={num_traces} l_skip={l_skip} n_max={n_max}"
    );
    println!("dump dir: {}", dump_dir.display());

    let device = DeviceType::Cuda(0);
    let plan = synthetic_plan(num_traces, l_skip, n_max);

    let mut g = GraphBuilder::new();
    // A fresh transcript. A caller wiring this into a real prove would use
    // `DuplexSpongeGpuIR::from_live(&mut g, device, &live.snapshot())` so the
    // phase graph chains onto the Fiat-Shamir stream stages A/B left behind.
    let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);

    let bufs: Vec<TraceBufs> = (0..plan.num_traces())
        .map(|t| TraceBufs::alloc_zeroed(&mut g, device, &plan, t))
        .collect();

    let proof = logup_zerocheck_gpu_ir(&mut g, &mut transcript, &plan, &bufs, device);
    println!(
        "graph built: {} round-0 zerocheck evals, {} round-0 logup evals, {} MLE rounds, {} traces of openings",
        proof.round0_zc_evals.len(),
        proof.round0_logup_evals.len(),
        proof.round_evals.len(),
        proof.column_openings.len(),
    );

    // The graph's own node listing is the primary artifact: it is the phase,
    // node by node, in the order the builder emitted it.
    let graph_txt = g.print();
    let node_lines = graph_txt.lines().count();
    if let Err(e) = std::fs::create_dir_all(&dump_dir) {
        eprintln!("could not create dump dir: {e}");
        std::process::exit(1);
    }
    let graph_path = dump_dir.join("zerocheck_phase.graph.txt");
    if let Err(e) = std::fs::write(&graph_path, &graph_txt) {
        eprintln!("could not write {}: {e}", graph_path.display());
        std::process::exit(1);
    }
    println!("wrote {} ({node_lines} lines)", graph_path.display());

    let compiler = GraphCompiler::new()
        .device(device)
        .scheduler(SchedulerMode::Heuristic)
        .dump_dir(dump_dir.clone())
        .verbosity(Verbosity::Verbose);
    let exe = match compiler.compile(g) {
        Ok(exe) => exe,
        Err(e) => {
            eprintln!("compile failed: {e}");
            std::process::exit(1);
        }
    };
    println!("compile succeeded: {} graph outputs", exe.num_outputs());

    // Per-module HIR / KIR / CUDA dumps are written by the module compiler
    // only for modules it compiles from source; a warm kernel cache means
    // nothing new lands here.
    if let Ok(entries) = std::fs::read_dir(&dump_dir) {
        let mut names: Vec<_> = entries
            .filter_map(|e| e.ok().and_then(|e| e.file_name().into_string().ok()))
            .collect();
        names.sort();
        println!("--- files in {} ---", dump_dir.display());
        for n in names {
            println!("  {n}");
        }
    }
}
