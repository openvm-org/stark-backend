//! Dumps every stage of the fold_ef_frac_columns → frac_compute_round fusion
//! chain that turns nvcc-uncompilable at log_n=20.
//!
//! Writes per-bare-module HIR files plus one HIR per fusion round to
//! `$DUMP_DIR` (defaults to `target/frac_fusion_chain/`):
//!
//!   000.frac_compute_round_dsl_n32_c16.hir
//!   001.fold_ef_frac_columns_dsl_128.hir
//!   ...
//!   005.fold_ef_frac_columns_dsl_2048.hir
//!   fusion/000.frac_compute_round_dsl_n32_c16__f__fold_ef_frac_columns_dsl_128.hir
//!   fusion/001.frac_compute_round_dsl_n32_c16__f__..._128__f__..._256.hir
//!   ...
//!
//! Run:
//!   cargo run -p openvm-cuda-backend --release --example \
//!     dump_fold_frac_fusion_chain --features graph-ir

use std::{path::PathBuf, sync::Arc};

use crypto_compiler::{
    dump::dump_hir,
    graph_ir::{BufId, BufInfo, DeviceType, GraphBuilder},
    ir::Module,
    passes::fusion::{fuse_graph, FusionOptions},
    quast::Quast,
};
use openvm_cuda_backend::logup_zerocheck::fractional_ir_dsl::{
    build_fold_ef_frac_columns_module, build_frac_compute_round_module,
};

const FRAC_EF_BYTES: usize = 32;
const EF_BYTES: usize = 16;
const BB_BYTES: usize = 4;
const D_EF: usize = 4;

fn add_buf(g: &mut GraphBuilder, name: &str, bytes: usize) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: DeviceType::Cuda(0),
        size: Quast::cst(bytes as i64),
        elem_size: bytes,
    })
}

fn add_frac_ef_buf(g: &mut GraphBuilder, name: &str, n_frac: usize) -> BufId {
    add_buf(g, name, n_frac * FRAC_EF_BYTES)
}

fn write_module(dir: &std::path::Path, idx: usize, m: &Module) {
    let path = dir.join(format!("{idx:03}.{}.hir", m.name));
    std::fs::write(&path, dump_hir(m)).expect("write bare hir");
    println!("  wrote {}", path.display());
}

fn main() {
    let dump_dir = std::env::var_os("DUMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../target/frac_fusion_chain"
            ))
        });
    std::fs::create_dir_all(&dump_dir).expect("create dump dir");
    let fusion_dir = dump_dir.join("fusion");
    // Wipe prior fusion trace so numbering restarts clean.
    let _ = std::fs::remove_dir_all(&fusion_dir);
    std::fs::create_dir_all(&fusion_dir).expect("create fusion dir");

    // The exact chain that showed up in the timed-out kernel name:
    //   input → fold_2048 → fold_1024 → fold_512 → fold_256 → fold_128 → frac_compute_round_n32_c16
    // ("_n32_c16" corresponds to num_x=32, eq_low_cap=16; the fold sizes are
    // the `size` arg to `build_fold_ef_frac_columns_module`.)
    let fold_sizes = [2048usize, 1024, 512, 256, 128];
    // num_x=32 means frac_compute_round reads pq[64, 2] — same as fold_128's output.
    let cr_num_x: usize = 32;
    let cr_eq_low_cap: usize = 16;

    // --- Stage 1: build every bare module and dump its HIR ---
    println!("=== bare module HIRs ===");
    let cr_module = Arc::new(build_frac_compute_round_module(cr_num_x, cr_eq_low_cap));
    write_module(&dump_dir, 0, &cr_module);
    let fold_modules: Vec<Arc<Module>> = fold_sizes
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &sz)| {
            let m = Arc::new(build_fold_ef_frac_columns_module(sz));
            write_module(&dump_dir, i + 1, &m);
            m
        })
        .collect();
    // fold_modules is now in [fold_128, fold_256, fold_512, fold_1024, fold_2048] order.

    // --- Stage 2: assemble the chain into a graph ---
    println!("\n=== graph assembly ===");
    let mut g = GraphBuilder::new();
    // Buffers along the chain: input pq[2048], then fold outputs halving down.
    let mut chain_buf = add_frac_ef_buf(&mut g, "src_pq", fold_sizes[0]);
    g.register_input(chain_buf);
    let mut sizes = fold_sizes.to_vec();
    sizes.push(cr_num_x * 2);
    // Insert fold_{2048,1024,512,256,128} one after the other.
    for (i, &size_in) in fold_sizes.iter().enumerate() {
        let out_len = size_in / 2;
        let out = add_frac_ef_buf(&mut g, &format!("fold_out_{size_in}"), out_len);
        // Every fold takes a scalar challenge `r`. The DSL declares it as
        // `[D_EF]` BabyBear, so allocate one dedicated buf per stage.
        let r = add_buf(&mut g, &format!("r_{size_in}"), D_EF * BB_BYTES);
        g.register_input(r);
        let m = &fold_modules[fold_modules.len() - 1 - i]; // fold_2048 first
        g.insert_kernel((**m).clone(), [chain_buf, r], [out]);
        chain_buf = out;
    }
    // frac_compute_round: reads (eq_low[16], eq_high[1], pq[64,2], lambda[4]).
    let eq_low = add_frac_ef_buf(&mut g, "eq_low", cr_eq_low_cap);
    let eq_high = add_frac_ef_buf(&mut g, "eq_high", cr_num_x / cr_eq_low_cap);
    let lambda = add_buf(&mut g, "lambda", D_EF * BB_BYTES);
    g.register_input(eq_low);
    g.register_input(eq_high);
    g.register_input(lambda);
    let out = add_buf(&mut g, "s_out", 2 * EF_BYTES);
    g.register_output(out);
    g.insert_kernel(
        (*cr_module).clone(),
        [eq_low, eq_high, chain_buf, lambda],
        [out],
    );
    println!(
        "  built graph: {} kernel nodes, {} buffers, chain {} -> frac_compute_round",
        g.nodes.len(),
        g.bufs.len(),
        fold_sizes
            .iter()
            .map(|s| format!("fold_{s}"))
            .collect::<Vec<_>>()
            .join(" -> "),
    );

    // --- Stage 3: run fuse_graph with FUSION_DUMP_STEPS = fusion_dir ---
    // apply_fusion writes every merged module's HIR before the shared-Arc dedup swap,
    // in the order they were successfully produced. See fusion.rs.
    println!(
        "\n=== fusion (FUSION_DUMP_STEPS -> {}) ===",
        fusion_dir.display()
    );
    // SAFETY: single-threaded main; we own the env for the duration.
    std::env::set_var("FUSION_DUMP_STEPS", &fusion_dir);
    let opts = FusionOptions {
        verbose: true,
        max_iterations: 20,
        ..FusionOptions::default()
    };
    let report = fuse_graph(&mut g, &opts);
    std::env::remove_var("FUSION_DUMP_STEPS");
    println!(
        "  fusion summary: {} -> {} nodes, {} fusions in {} rounds, {} unique modules deduped",
        report.nodes_before,
        report.nodes_after,
        report.fused.len(),
        report.rounds,
        report.deduped,
    );

    // --- Enumerate what got written ---
    let mut files: Vec<_> = std::fs::read_dir(&fusion_dir)
        .expect("read fusion dir")
        .filter_map(|e| e.ok().map(|e| e.file_name().into_string().ok()).flatten())
        .collect();
    files.sort();
    println!("\n=== fusion/ contents ({} files) ===", files.len());
    for f in files {
        println!("  {f}");
    }
}
