//! Dump HIR/KIR/CUDA for `frac_precompute_m_eval_round` (small-K case).
use std::path::PathBuf;

use crypto_compiler::{
    compile_and_load,
    runtime::{CompileOptions, Verbosity},
};
use openvm_cuda_backend::logup_zerocheck::fractional_ir_dsl::build_frac_precompute_m_eval_round_module;

fn main() {
    let w = std::env::var("W")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4usize);
    let t = std::env::var("T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2usize);
    let dump_dir: PathBuf = std::env::var_os("DUMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../target/ir-dumps/eval_round"
            ))
        });

    println!("dumping eval_round: w={w} t={t}");
    println!("dump dir: {}", dump_dir.display());

    let module = build_frac_precompute_m_eval_round_module(w, t);
    let options = CompileOptions {
        dump_ir: Some(dump_dir.clone()),
        verbosity: Verbosity::Verbose,
        ..CompileOptions::default()
    };
    match compile_and_load(&module, &options) {
        Ok(_) => println!("compile ok"),
        Err(e) => {
            eprintln!("compile failed: {e}");
            std::process::exit(1);
        }
    }
}
