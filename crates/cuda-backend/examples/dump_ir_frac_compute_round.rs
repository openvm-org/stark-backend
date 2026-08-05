//! Dumps HIR / KIR / CUDA sources for the `frac_compute_round` DSL module,
//! covering all of the passes the parallel-reduce rewrite affects.
//!
//! Usage:
//!   cargo run -p openvm-cuda-backend --release --example dump_ir_frac_compute_round --features
//! graph-ir
//!
//! By default writes to `target/ir-dumps/frac_compute_round`. Set
//! `DUMP_DIR=/some/path` to override, or `NUM_X` / `EQ_LOW_CAP` to pick a
//! different problem size (defaults: `num_x=2^14`, `eq_low_cap=2^7`, i.e.
//! `K = num_x/2 = 2^13 = 8192` — solidly above the tree-lowering
//! threshold of 64, so the parallel_reduce_rewrite pass fires).

use std::path::PathBuf;

use crypto_compiler::{
    compile_and_load,
    runtime::{CompileOptions, Verbosity},
};
use openvm_cuda_backend::logup_zerocheck::fractional_ir_dsl::build_frac_compute_round_module;

fn parse_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn main() {
    let num_x = parse_usize("NUM_X", 1 << 14);
    let dump_dir: PathBuf = std::env::var_os("DUMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../target/ir-dumps/frac_compute_round"
            ))
        });

    println!(
        "dumping frac_compute_round DSL module: num_x={num_x} (K={}), eq caps symbolic",
        num_x / 2
    );
    println!("dump dir: {}", dump_dir.display());

    let module = build_frac_compute_round_module(num_x);

    let options = CompileOptions {
        dump_ir: Some(dump_dir.clone()),
        verbosity: Verbosity::Verbose,
        ..CompileOptions::default()
    };

    // Compile the module end-to-end. We drop the loaded artifact — the dump
    // files are the point of the example.
    match compile_and_load(&module, &options) {
        Ok(_) => println!("compile succeeded"),
        Err(e) => {
            eprintln!("compile failed: {e}");
            std::process::exit(1);
        }
    }

    // Enumerate what got written so callers can grep it.
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
