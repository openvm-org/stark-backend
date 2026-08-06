//! Micro-benchmark: DSL-based `*_ir_dsl` ports vs eager blackbox
//! `*_ir_bufid` / `*_ir` kernels.
//!
//! For each of the 8 kernel ports, builds two independent graphs (one
//! containing only the blackbox variant, one containing only the DSL
//! variant), each solving the same synthetic input. Each graph is compiled
//! once; then we time repeated `exe.run(...)` calls with
//! `stream.synchronize()` between them and print a Markdown table.
//!
//! Requires the `graph-ir` feature. Run with:
//!   cargo run -p openvm-cuda-backend --release --features graph-ir \
//!       --example bench_ir_dsl_ports

use std::time::Instant;

use crypto_compiler::{
    graph_exe::GraphCompiler,
    graph_ir::{BufId, BufInfo, ConstBuf, DeviceType, GraphBuilder},
    planner::SchedulerMode,
};
use openvm_cuda_backend::{
    logup_zerocheck::{
        fractional_ir::{
            fold_ef_frac_columns_ir_bufid, frac_build_tree_layer_ir, frac_build_tree_two_layers_ir,
            frac_compute_round_and_fold_ir_bufid, frac_compute_round_and_revert_ir_bufid,
            frac_compute_round_ir_bufid, frac_multifold_ir, frac_precompute_m_eval_round_ir,
            SqrtEqLayersIR,
        },
        fractional_ir_dsl::{
            fold_ef_frac_columns_ir_dsl, frac_build_tree_layer_revert_ir_dsl,
            frac_build_tree_two_layers_ir_dsl, frac_compute_round_and_fold_ir_dsl,
            frac_compute_round_and_revert_ir_dsl, frac_compute_round_ir_dsl, frac_multifold_ir_dsl,
            frac_precompute_m_eval_round_ir_dsl,
        },
    },
    prelude::EF,
};
use openvm_cuda_common::{
    common::get_device,
    stream::{CudaStream, GpuDeviceCtx, StreamGuard},
};
use openvm_stark_backend::prover::fractional_sumcheck_gkr::Frac;
use rand::{rngs::StdRng, Rng, SeedableRng};

// The `cuda` module is private in openvm-cuda-backend, so we redeclare the
// symbol here. It is provided by the linked-in nvcc-compiled object file.
extern "C" {
    fn _frac_compute_round_temp_buffer_size(stride: u32) -> u32;
}
fn frac_compute_round_temp_len(num_x: usize) -> usize {
    unsafe { _frac_compute_round_temp_buffer_size(num_x as u32) as usize }
}

// ---------------------------------------------------------------------------
// Byte-size constants + local buffer-allocation helpers (mirroring the
// pub(crate) helpers in `fractional_ir.rs` — reproduced here because they're
// not part of the public API).

const D_EF: usize = 4;
const EF_BYTES: usize = std::mem::size_of::<EF>();
const FRAC_EF_BYTES: usize = std::mem::size_of::<Frac<EF>>();

fn add_ef_buf(g: &mut GraphBuilder, name: &str, n: usize) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: DeviceType::Cuda(0),
        size: crypto_compiler::quast::Quast::cst((n * EF_BYTES) as i64),
        elem_size: EF_BYTES,
    })
}

fn add_frac_ef_buf(g: &mut GraphBuilder, name: &str, n: usize) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: DeviceType::Cuda(0),
        size: crypto_compiler::quast::Quast::cst((n * FRAC_EF_BYTES) as i64),
        elem_size: FRAC_EF_BYTES,
    })
}

fn add_ext_scalar_buf(g: &mut GraphBuilder, name: &str) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: DeviceType::Cuda(0),
        size: crypto_compiler::quast::Quast::cst((D_EF as i64) * 4),
        elem_size: D_EF * 4,
    })
}

fn ef_const_ext_scalar_buf(g: &mut GraphBuilder, name: &str, value: EF) -> BufId {
    let buf = add_ext_scalar_buf(g, name);
    let bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(&value as *const EF as *const u8, std::mem::size_of::<EF>())
            .to_vec()
    };
    g.insert_const(buf, ConstBuf::HostBuf(bytes));
    buf
}

fn ef_slice_const_buf(g: &mut GraphBuilder, name: &str, xs: &[EF]) -> BufId {
    let buf = add_ef_buf(g, name, xs.len());
    let bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs)).to_vec()
    };
    g.insert_const(buf, ConstBuf::HostBuf(bytes));
    buf
}

fn frac_bytes(leaves: &[Frac<EF>]) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(leaves.as_ptr() as *const u8, std::mem::size_of_val(leaves))
            .to_vec()
    }
}

fn frac_const_buf(g: &mut GraphBuilder, name: &str, leaves: &[Frac<EF>]) -> BufId {
    let buf = add_frac_ef_buf(g, name, leaves.len());
    g.insert_const(buf, ConstBuf::HostBuf(frac_bytes(leaves)));
    buf
}

// ---------------------------------------------------------------------------
// Randomization helpers.

fn make_host_leaves(rng: &mut StdRng, len: usize) -> Vec<Frac<EF>> {
    (0..len)
        .map(|_| Frac {
            p: rng.random::<EF>(),
            q: rng.random::<EF>(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Env parsing.

fn parse_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|x| x.parse::<usize>().ok())
        .unwrap_or(default)
}

// ---------------------------------------------------------------------------
// eq_low_cap convention: sqrt(num_x) rounded to a power of two (matches the
// natural SqrtEqLayers split — low_n = n/2, high_n = n - n/2, where
// num_x = 2^(n+1)).
fn eq_low_cap_for(num_x: usize) -> usize {
    // num_x = 2^(n+1)  →  n = trailing_zeros(num_x) - 1
    let n = (num_x.trailing_zeros() as usize).saturating_sub(1);
    let low_n = n / 2;
    1usize << low_n
}

/// Build a `SqrtEqLayersIR` from host EF `xi` values staged as `[D_EF]`
/// challenge consts.
fn ir_eq_layers(g: &mut GraphBuilder, xi: &[EF]) -> SqrtEqLayersIR {
    let xi_bufs: Vec<BufId> = xi
        .iter()
        .enumerate()
        .map(|(j, v)| ef_const_ext_scalar_buf(g, &format!("xi_{j}"), *v))
        .collect();
    SqrtEqLayersIR::from_xi(g, &xi_bufs, DeviceType::Cuda(0))
}

// ---------------------------------------------------------------------------
// Bench primitive: compile a graph, warmup, and time N calls.

struct BenchStats {
    median_ms: f64,
    q25_ms: f64,
    q75_ms: f64,
}

fn bench_graph(
    g: GraphBuilder,
    ctx: &GpuDeviceCtx,
    warmup: usize,
    iters: usize,
) -> Result<BenchStats, Box<dyn std::error::Error>> {
    let mut exe = GraphCompiler::new()
        .device(DeviceType::Cuda(0))
        .scheduler(SchedulerMode::Heuristic)
        .compile(g)?;

    // Warmup.
    for _ in 0..warmup {
        exe.run(ctx)?;
    }
    ctx.stream.synchronize()?;

    // Timed.
    let mut samples: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        exe.run(ctx)?;
        ctx.stream.synchronize()?;
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| -> f64 {
        let idx = ((samples.len() as f64 - 1.0) * p) as usize;
        samples[idx.min(samples.len() - 1)]
    };
    Ok(BenchStats {
        median_ms: pct(0.5),
        q25_ms: pct(0.25),
        q75_ms: pct(0.75),
    })
}

// ---------------------------------------------------------------------------
// Row emitter.

struct Row {
    kernel: &'static str,
    size_label: String,
    bb: Option<BenchStats>,
    dsl: Option<BenchStats>,
    notes: String,
}

impl Row {
    fn print_md(&self) {
        let bb = self
            .bb
            .as_ref()
            .map(|s| format!("{:.4} ({:.4}/{:.4})", s.median_ms, s.q25_ms, s.q75_ms))
            .unwrap_or_else(|| "ERR".to_string());
        let dsl = self
            .dsl
            .as_ref()
            .map(|s| format!("{:.4} ({:.4}/{:.4})", s.median_ms, s.q25_ms, s.q75_ms))
            .unwrap_or_else(|| "ERR".to_string());
        let ratio = match (self.bb.as_ref(), self.dsl.as_ref()) {
            (Some(bb), Some(dsl)) => format!("{:.3}", dsl.median_ms / bb.median_ms),
            _ => "-".to_string(),
        };
        println!(
            "| {} | {} | {} | {} | {} | {} |",
            self.kernel, self.size_label, bb, dsl, ratio, self.notes
        );
    }
    fn ratio(&self) -> Option<f64> {
        match (self.bb.as_ref(), self.dsl.as_ref()) {
            (Some(bb), Some(dsl)) => Some(dsl.median_ms / bb.median_ms),
            _ => None,
        }
    }
}

fn print_header() {
    println!("| kernel | size | blackbox median ms (q25/q75) | DSL median ms (q25/q75) | ratio (DSL/BB) | notes |");
    println!("|---|---|---|---|---|---|");
}

// ---------------------------------------------------------------------------
// Kernel 1: fold_ef_frac_columns.

fn bench_fold_ef_frac_columns(
    ctx: &GpuDeviceCtx,
    rng: &mut StdRng,
    size: usize,
    warmup: usize,
    iters: usize,
) -> Row {
    let src = make_host_leaves(rng, size);
    let r: EF = rng.random();
    let alpha: EF = rng.random();

    let mut notes = String::new();
    let bb = {
        let mut g = GraphBuilder::new();
        let src_buf = frac_const_buf(&mut g, "src", &src);
        let r_buf = ef_const_ext_scalar_buf(&mut g, "r", r);
        let dst_buf = add_frac_ef_buf(&mut g, "dst", size / 2);
        fold_ef_frac_columns_ir_bufid(&mut g, src_buf, dst_buf, size, size, size, r_buf, alpha);
        // Add a memcpy target to force `dst_buf` to be materialized.
        let out = add_frac_ef_buf(&mut g, "dst_out", size / 2);
        g.insert_memcpy(dst_buf, out);
        g.register_output(out);
        bench_graph(g, ctx, warmup, iters)
            .map_err(|e| notes.push_str(&format!("BB err: {e}; ")))
            .ok()
    };
    let dsl = {
        let mut g = GraphBuilder::new();
        let src_buf = frac_const_buf(&mut g, "src", &src);
        let r_buf = ef_const_ext_scalar_buf(&mut g, "r", r);
        let dst_buf = add_frac_ef_buf(&mut g, "dst", size / 2);
        fold_ef_frac_columns_ir_dsl(&mut g, src_buf, dst_buf, size, r_buf);
        let out = add_frac_ef_buf(&mut g, "dst_out", size / 2);
        g.insert_memcpy(dst_buf, out);
        g.register_output(out);
        bench_graph(g, ctx, warmup, iters)
            .map_err(|e| notes.push_str(&format!("DSL err: {e}; ")))
            .ok()
    };
    Row {
        kernel: "fold_ef_frac_columns",
        size_label: format!("size=2^{}", (size as f64).log2().round() as u32),
        bb,
        dsl,
        notes,
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: frac_compute_round.

fn bench_frac_compute_round(
    ctx: &GpuDeviceCtx,
    rng: &mut StdRng,
    num_x: usize,
    warmup: usize,
    iters: usize,
) -> Row {
    // num_x = 2^(n+1); split half/half between low/high.
    // We need to have enough xi to give num_x = 2^(n+1), so n = log2(num_x) - 1.
    let n = (num_x.trailing_zeros() as usize).saturating_sub(1);
    let xi: Vec<EF> = (0..n).map(|_| rng.random()).collect();
    let lambda: EF = rng.random();
    let pq_size = 2 * num_x;
    let pq = make_host_leaves(rng, pq_size);
    let eq_low_cap = eq_low_cap_for(num_x);
    let tmp_len = frac_compute_round_temp_len(num_x);

    let mut notes = String::new();
    let bb = {
        let mut g = GraphBuilder::new();
        let eq_xi = ir_eq_layers(&mut g, &xi);
        let pq_buf = frac_const_buf(&mut g, "pq", &pq);
        let lambda_buf = ef_const_ext_scalar_buf(&mut g, "lambda", lambda);
        let out_buf = add_ef_buf(&mut g, "out", 2);
        let tmp_buf = add_ef_buf(&mut g, "tmp", tmp_len);
        frac_compute_round_ir_bufid(&mut g, &eq_xi, pq_buf, num_x, lambda_buf, out_buf, tmp_buf);
        let out_out = add_ef_buf(&mut g, "out_out", 2);
        g.insert_memcpy(out_buf, out_out);
        g.register_output(out_out);
        bench_graph(g, ctx, warmup, iters)
            .map_err(|e| notes.push_str(&format!("BB err: {e}; ")))
            .ok()
    };
    let dsl = {
        let mut g = GraphBuilder::new();
        // Materialize the eq_low / eq_high buffers as consts. Easiest: build the
        // IR eq layers and read the top-level BufIds directly (they're graph
        // consts already).
        let eq_xi = ir_eq_layers(&mut g, &xi);
        let low_n = eq_xi.low_n();
        let high_n = eq_xi.high_n();
        let eq_low_buf = eq_xi.low.get(low_n);
        let eq_high_buf = eq_xi.high.get(high_n);
        let pq_buf = frac_const_buf(&mut g, "pq", &pq);
        let lambda_buf = ef_const_ext_scalar_buf(&mut g, "lambda", lambda);
        let out_buf = add_ef_buf(&mut g, "out", 2);
        frac_compute_round_ir_dsl(
            &mut g,
            eq_low_buf,
            eq_high_buf,
            pq_buf,
            lambda_buf,
            out_buf,
            num_x,
            eq_low_cap,
        );
        let out_out = add_ef_buf(&mut g, "out_out", 2);
        g.insert_memcpy(out_buf, out_out);
        g.register_output(out_out);
        bench_graph(g, ctx, warmup, iters)
            .map_err(|e| notes.push_str(&format!("DSL err: {e}; ")))
            .ok()
    };
    Row {
        kernel: "frac_compute_round",
        size_label: format!("num_x=2^{}", (num_x as f64).log2().round() as u32),
        bb,
        dsl,
        notes,
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: frac_compute_round_and_revert.

fn bench_frac_compute_round_and_revert(
    ctx: &GpuDeviceCtx,
    rng: &mut StdRng,
    num_x: usize,
    warmup: usize,
    iters: usize,
) -> Row {
    let n = (num_x.trailing_zeros() as usize).saturating_sub(1);
    let xi: Vec<EF> = (0..n).map(|_| rng.random()).collect();
    let lambda: EF = rng.random();
    // alpha=0 for both — dense apply_alpha=false.
    let alpha: EF = <EF as p3_field::PrimeCharacteristicRing>::ZERO;
    let layer_size = 2 * num_x;
    let leaves = make_host_leaves(rng, layer_size);
    let eq_low_cap = eq_low_cap_for(num_x);
    let tmp_len = frac_compute_round_temp_len(num_x);

    let mut notes = String::new();
    let bb = {
        let mut g = GraphBuilder::new();
        let eq_xi = ir_eq_layers(&mut g, &xi);
        // Need a mutable layer buffer: stage as const and memcpy into a fresh buf.
        let layer_init = frac_const_buf(&mut g, "layer_init", &leaves);
        let layer = add_frac_ef_buf(&mut g, "layer", layer_size);
        g.insert_memcpy(layer_init, layer);
        let lambda_buf = ef_const_ext_scalar_buf(&mut g, "lambda", lambda);
        let out_buf = add_ef_buf(&mut g, "out", 2);
        let tmp_buf = add_ef_buf(&mut g, "tmp", tmp_len);
        frac_compute_round_and_revert_ir_bufid(
            &mut g, &eq_xi, layer, layer_size, num_x, layer_size, lambda_buf, alpha, out_buf,
            tmp_buf,
        );
        let out_out = add_ef_buf(&mut g, "out_out", 2);
        g.insert_memcpy(out_buf, out_out);
        g.register_output(out_out);
        bench_graph(g, ctx, warmup, iters)
            .map_err(|e| notes.push_str(&format!("BB err: {e}; ")))
            .ok()
    };
    let dsl = {
        let mut g = GraphBuilder::new();
        let eq_xi = ir_eq_layers(&mut g, &xi);
        let low_n = eq_xi.low_n();
        let high_n = eq_xi.high_n();
        let eq_low_buf = eq_xi.low.get(low_n);
        let eq_high_buf = eq_xi.high.get(high_n);
        let layer_in = frac_const_buf(&mut g, "layer_in", &leaves);
        let layer_post = add_frac_ef_buf(&mut g, "layer_post", layer_size);
        let lambda_buf = ef_const_ext_scalar_buf(&mut g, "lambda", lambda);
        let out_buf = add_ef_buf(&mut g, "out", 2);
        frac_compute_round_and_revert_ir_dsl(
            &mut g,
            eq_low_buf,
            eq_high_buf,
            layer_in,
            layer_post,
            lambda_buf,
            out_buf,
            layer_size,
            eq_low_cap,
        );
        let out_out = add_ef_buf(&mut g, "out_out", 2);
        g.insert_memcpy(out_buf, out_out);
        g.register_output(out_out);
        bench_graph(g, ctx, warmup, iters)
            .map_err(|e| notes.push_str(&format!("DSL err: {e}; ")))
            .ok()
    };
    Row {
        kernel: "frac_compute_round_and_revert",
        size_label: format!("num_x=2^{}", (num_x as f64).log2().round() as u32),
        bb,
        dsl,
        notes,
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: frac_compute_round_and_fold.

fn bench_frac_compute_round_and_fold(
    ctx: &GpuDeviceCtx,
    rng: &mut StdRng,
    src_pq_size: usize,
    warmup: usize,
    iters: usize,
) -> Row {
    // Post-fold num_x = src_pq_size / 4; eq_xi built from n = log2(num_x) - 1
    // challenges.
    let num_x = src_pq_size >> 2;
    let n = (num_x.trailing_zeros() as usize).saturating_sub(1);
    let xi: Vec<EF> = (0..n).map(|_| rng.random()).collect();
    let lambda: EF = rng.random();
    let r_prev: EF = rng.random();
    let alpha: EF = rng.random();
    let src_pq = make_host_leaves(rng, src_pq_size);
    let eq_low_cap = eq_low_cap_for(num_x);
    let tmp_len = frac_compute_round_temp_len(num_x);

    let mut notes = String::new();
    let bb = {
        let mut g = GraphBuilder::new();
        let eq_xi = ir_eq_layers(&mut g, &xi);
        let src_buf = frac_const_buf(&mut g, "src_pq", &src_pq);
        let dst_buf = add_frac_ef_buf(&mut g, "dst_pq", src_pq_size / 2);
        let lambda_buf = ef_const_ext_scalar_buf(&mut g, "lambda", lambda);
        let r_prev_buf = ef_const_ext_scalar_buf(&mut g, "r_prev", r_prev);
        let out_buf = add_ef_buf(&mut g, "out", 2);
        let tmp_buf = add_ef_buf(&mut g, "tmp", tmp_len);
        frac_compute_round_and_fold_ir_bufid(
            &mut g,
            &eq_xi,
            src_buf,
            dst_buf,
            src_pq_size,
            src_pq_size,
            src_pq_size,
            lambda_buf,
            r_prev_buf,
            alpha,
            out_buf,
            tmp_buf,
        );
        let out_out = add_ef_buf(&mut g, "out_out", 2);
        g.insert_memcpy(out_buf, out_out);
        g.register_output(out_out);
        bench_graph(g, ctx, warmup, iters)
            .map_err(|e| notes.push_str(&format!("BB err: {e}; ")))
            .ok()
    };
    let dsl = {
        let mut g = GraphBuilder::new();
        let eq_xi = ir_eq_layers(&mut g, &xi);
        let low_n = eq_xi.low_n();
        let high_n = eq_xi.high_n();
        let eq_low_buf = eq_xi.low.get(low_n);
        let eq_high_buf = eq_xi.high.get(high_n);
        let src_buf = frac_const_buf(&mut g, "src_pq", &src_pq);
        let dst_buf = add_frac_ef_buf(&mut g, "dst_pq", src_pq_size / 2);
        let lambda_buf = ef_const_ext_scalar_buf(&mut g, "lambda", lambda);
        let r_prev_buf = ef_const_ext_scalar_buf(&mut g, "r_prev", r_prev);
        let out_buf = add_ef_buf(&mut g, "out", 2);
        frac_compute_round_and_fold_ir_dsl(
            &mut g,
            eq_low_buf,
            eq_high_buf,
            src_buf,
            dst_buf,
            lambda_buf,
            r_prev_buf,
            out_buf,
            src_pq_size,
            eq_low_cap,
        );
        let out_out = add_ef_buf(&mut g, "out_out", 2);
        g.insert_memcpy(out_buf, out_out);
        g.register_output(out_out);
        bench_graph(g, ctx, warmup, iters)
            .map_err(|e| notes.push_str(&format!("DSL err: {e}; ")))
            .ok()
    };
    Row {
        kernel: "frac_compute_round_and_fold",
        size_label: format!(
            "src_pq_size=2^{}",
            (src_pq_size as f64).log2().round() as u32
        ),
        bb,
        dsl,
        notes,
    }
}

// ---------------------------------------------------------------------------
// Kernel 5: frac_precompute_m_eval_round.

fn bench_frac_precompute_m_eval_round(
    ctx: &GpuDeviceCtx,
    rng: &mut StdRng,
    w: usize,
    t: usize,
    warmup: usize,
    iters: usize,
) -> Row {
    let m = 1usize << w;
    let prefix_size = 1usize << t;
    let suffix_size = 1usize << (w - t - 1);
    let m_total: Vec<EF> = (0..m * m).map(|_| rng.random()).collect();
    let eq_r_prefix: Vec<EF> = (0..prefix_size).map(|_| rng.random()).collect();
    let eq_suffix: Vec<EF> = (0..suffix_size).map(|_| rng.random()).collect();

    let mut notes = String::new();
    let bb = {
        let mut g = GraphBuilder::new();
        let m_buf = ef_slice_const_buf(&mut g, "m_total", &m_total);
        let ep_buf = ef_slice_const_buf(&mut g, "eq_r_prefix", &eq_r_prefix);
        let es_buf = ef_slice_const_buf(&mut g, "eq_suffix", &eq_suffix);
        let out_buf = add_ef_buf(&mut g, "out", 2);
        frac_precompute_m_eval_round_ir(&mut g, m_buf, ep_buf, es_buf, out_buf, w, t);
        let out_out = add_ef_buf(&mut g, "out_out", 2);
        g.insert_memcpy(out_buf, out_out);
        g.register_output(out_out);
        bench_graph(g, ctx, warmup, iters)
            .map_err(|e| notes.push_str(&format!("BB err: {e}; ")))
            .ok()
    };
    let dsl = {
        let mut g = GraphBuilder::new();
        let m_buf = ef_slice_const_buf(&mut g, "m_total", &m_total);
        let ep_buf = ef_slice_const_buf(&mut g, "eq_r_prefix", &eq_r_prefix);
        let es_buf = ef_slice_const_buf(&mut g, "eq_suffix", &eq_suffix);
        let out_buf = add_ef_buf(&mut g, "out", 2);
        frac_precompute_m_eval_round_ir_dsl(&mut g, m_buf, ep_buf, es_buf, out_buf, w, t);
        let out_out = add_ef_buf(&mut g, "out_out", 2);
        g.insert_memcpy(out_buf, out_out);
        g.register_output(out_out);
        bench_graph(g, ctx, warmup, iters)
            .map_err(|e| notes.push_str(&format!("DSL err: {e}; ")))
            .ok()
    };
    Row {
        kernel: "frac_precompute_m_eval_round",
        size_label: format!("(w={w}, t={t})"),
        bb,
        dsl,
        notes,
    }
}

// ---------------------------------------------------------------------------
// Kernel 6: frac_multifold.

fn bench_frac_multifold(
    ctx: &GpuDeviceCtx,
    rng: &mut StdRng,
    tail_size: usize,
    w: usize,
    warmup: usize,
    iters: usize,
) -> Row {
    let beta_size = 1usize << w;
    let poly_stride = tail_size * beta_size;
    let pre_size = 2 * poly_stride;
    let src = make_host_leaves(rng, pre_size);
    let eq_r_window: Vec<EF> = (0..beta_size).map(|_| rng.random()).collect();
    let alpha: EF = rng.random();
    let dst_len = 2 * tail_size;
    // rem_n = log2(pre_size / 2) = log2(poly_stride).
    let rem_n = (poly_stride as f64).log2().round() as usize;

    let mut notes = String::new();
    let bb = {
        let mut g = GraphBuilder::new();
        let src_buf = frac_const_buf(&mut g, "src", &src);
        let eq_buf = ef_slice_const_buf(&mut g, "eq_r_window", &eq_r_window);
        let dst_buf = add_frac_ef_buf(&mut g, "dst", dst_len);
        frac_multifold_ir(
            &mut g, src_buf, dst_buf, eq_buf, pre_size, pre_size, rem_n, w, alpha,
        );
        let dst_out = add_frac_ef_buf(&mut g, "dst_out", dst_len);
        g.insert_memcpy(dst_buf, dst_out);
        g.register_output(dst_out);
        bench_graph(g, ctx, warmup, iters)
            .map_err(|e| notes.push_str(&format!("BB err: {e}; ")))
            .ok()
    };
    let dsl = {
        let mut g = GraphBuilder::new();
        let src_buf = frac_const_buf(&mut g, "src", &src);
        let eq_buf = ef_slice_const_buf(&mut g, "eq_r_window", &eq_r_window);
        let dst_buf = add_frac_ef_buf(&mut g, "dst", dst_len);
        frac_multifold_ir_dsl(&mut g, src_buf, dst_buf, eq_buf, tail_size, w);
        let dst_out = add_frac_ef_buf(&mut g, "dst_out", dst_len);
        g.insert_memcpy(dst_buf, dst_out);
        g.register_output(dst_out);
        bench_graph(g, ctx, warmup, iters)
            .map_err(|e| notes.push_str(&format!("DSL err: {e}; ")))
            .ok()
    };
    Row {
        kernel: "frac_multifold",
        size_label: format!(
            "(tail=2^{}, w={w})",
            (tail_size as f64).log2().round() as u32
        ),
        bb,
        dsl,
        notes,
    }
}

// ---------------------------------------------------------------------------
// Kernel 7: frac_build_tree_layer (revert, apply_alpha=false).

fn bench_frac_build_tree_layer_revert(
    ctx: &GpuDeviceCtx,
    rng: &mut StdRng,
    layer_size: usize,
    warmup: usize,
    iters: usize,
) -> Row {
    let leaves = make_host_leaves(rng, layer_size);
    let alpha: EF = <EF as p3_field::PrimeCharacteristicRing>::ZERO;

    let mut notes = String::new();
    let bb = {
        let mut g = GraphBuilder::new();
        // Blackbox needs a mutable layer: const-init + memcpy into a mutable buf.
        let layer_init = frac_const_buf(&mut g, "layer_init", &leaves);
        let layer = add_frac_ef_buf(&mut g, "layer", layer_size);
        g.insert_memcpy(layer_init, layer);
        frac_build_tree_layer_ir(
            &mut g, layer, layer_size, layer_size, layer_size, true, alpha, false,
        );
        let layer_out = add_frac_ef_buf(&mut g, "layer_out", layer_size);
        g.insert_memcpy(layer, layer_out);
        g.register_output(layer_out);
        bench_graph(g, ctx, warmup, iters)
            .map_err(|e| notes.push_str(&format!("BB err: {e}; ")))
            .ok()
    };
    let dsl = {
        let mut g = GraphBuilder::new();
        let src = frac_const_buf(&mut g, "layer_in", &leaves);
        let dst = add_frac_ef_buf(&mut g, "layer_out", layer_size);
        frac_build_tree_layer_revert_ir_dsl(&mut g, src, dst, layer_size);
        let dst_out = add_frac_ef_buf(&mut g, "dst_out", layer_size);
        g.insert_memcpy(dst, dst_out);
        g.register_output(dst_out);
        bench_graph(g, ctx, warmup, iters)
            .map_err(|e| notes.push_str(&format!("DSL err: {e}; ")))
            .ok()
    };
    Row {
        kernel: "frac_build_tree_layer (revert)",
        size_label: format!("layer_size=2^{}", (layer_size as f64).log2().round() as u32),
        bb,
        dsl,
        notes,
    }
}

// ---------------------------------------------------------------------------
// Kernel 8: frac_build_tree_two_layers.

fn bench_frac_build_tree_two_layers(
    ctx: &GpuDeviceCtx,
    rng: &mut StdRng,
    half_i1: usize,
    warmup: usize,
    iters: usize,
) -> Row {
    let layer_size = 4 * half_i1;
    let leaves = make_host_leaves(rng, layer_size);
    let alpha: EF = rng.random();

    let mut notes = String::new();
    let bb = {
        let mut g = GraphBuilder::new();
        let layer_init = frac_const_buf(&mut g, "layer_init", &leaves);
        let layer = add_frac_ef_buf(&mut g, "layer", layer_size);
        g.insert_memcpy(layer_init, layer);
        frac_build_tree_two_layers_ir(&mut g, layer, layer_size, half_i1, layer_size, alpha);
        let layer_out = add_frac_ef_buf(&mut g, "layer_out", layer_size);
        g.insert_memcpy(layer, layer_out);
        g.register_output(layer_out);
        bench_graph(g, ctx, warmup, iters)
            .map_err(|e| notes.push_str(&format!("BB err: {e}; ")))
            .ok()
    };
    let dsl = {
        let mut g = GraphBuilder::new();
        let src = frac_const_buf(&mut g, "layer_in", &leaves);
        let dst = add_frac_ef_buf(&mut g, "layer_out", layer_size);
        frac_build_tree_two_layers_ir_dsl(&mut g, src, dst, half_i1);
        let dst_out = add_frac_ef_buf(&mut g, "dst_out", layer_size);
        g.insert_memcpy(dst, dst_out);
        g.register_output(dst_out);
        bench_graph(g, ctx, warmup, iters)
            .map_err(|e| notes.push_str(&format!("DSL err: {e}; ")))
            .ok()
    };
    Row {
        kernel: "frac_build_tree_two_layers",
        size_label: format!("half_i1=2^{}", (half_i1 as f64).log2().round() as u32),
        bb,
        dsl,
        notes,
    }
}

// ---------------------------------------------------------------------------
// Main.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let warmup = parse_usize("BENCH_WARMUP", 5);
    let iters = parse_usize("BENCH_ITERS", 50);
    let seed = parse_usize("BENCH_SEED", 0xB01D_FACE) as u64;

    eprintln!("bench_ir_dsl_ports: warmup={warmup} iters={iters} seed=0x{seed:x}",);
    let bench_t0 = Instant::now();

    let ctx = GpuDeviceCtx {
        device_id: get_device()? as u32,
        stream: StreamGuard::new(CudaStream::new_non_blocking()?),
    };
    let mut rng = StdRng::seed_from_u64(seed);

    let mut rows: Vec<Row> = Vec::new();

    // Sizing (from task spec, adjusted slightly for numerics constraints).

    // 1. fold_ef_frac_columns: size = number of Fracs.
    for size in [1 << 12, 1 << 16, 1 << 20] {
        rows.push(bench_fold_ef_frac_columns(
            &ctx, &mut rng, size, warmup, iters,
        ));
    }

    // 2. frac_compute_round: num_x >= 4 (n >= 1).
    for num_x in [1 << 10, 1 << 14, 1 << 18] {
        rows.push(bench_frac_compute_round(
            &ctx, &mut rng, num_x, warmup, iters,
        ));
    }

    // 3. frac_compute_round_and_revert.
    for num_x in [1 << 10, 1 << 14, 1 << 18] {
        rows.push(bench_frac_compute_round_and_revert(
            &ctx, &mut rng, num_x, warmup, iters,
        ));
    }

    // 4. frac_compute_round_and_fold: needs src_pq_size / 4 = num_x, and num_x = 2^(n+1) with n >=
    //    1 → src_pq_size >= 16.
    for src_pq_size in [1 << 12, 1 << 16, 1 << 20] {
        rows.push(bench_frac_compute_round_and_fold(
            &ctx,
            &mut rng,
            src_pq_size,
            warmup,
            iters,
        ));
    }

    // 5. frac_precompute_m_eval_round: (w, t) with t < w.
    for (w, t) in [(3usize, 1usize), (4, 2), (5, 3)] {
        rows.push(bench_frac_precompute_m_eval_round(
            &ctx, &mut rng, w, t, warmup, iters,
        ));
    }

    // 6. frac_multifold: w in [2..=5].
    for (tail_size, w) in [(1usize << 8, 2usize), (1 << 12, 3), (1 << 16, 4)] {
        rows.push(bench_frac_multifold(
            &ctx, &mut rng, tail_size, w, warmup, iters,
        ));
    }

    // 7. frac_build_tree_layer (revert).
    for layer_size in [1 << 10, 1 << 16, 1 << 20] {
        rows.push(bench_frac_build_tree_layer_revert(
            &ctx, &mut rng, layer_size, warmup, iters,
        ));
    }

    // 8. frac_build_tree_two_layers.
    for half_i1 in [1 << 8, 1 << 12, 1 << 18] {
        rows.push(bench_frac_build_tree_two_layers(
            &ctx, &mut rng, half_i1, warmup, iters,
        ));
    }

    // Print the Markdown table.
    println!();
    print_header();
    for row in &rows {
        row.print_md();
    }

    // Summary: geomean ratio over rows that have a ratio.
    let ratios: Vec<f64> = rows.iter().filter_map(|r| r.ratio()).collect();
    if !ratios.is_empty() {
        let ln_sum: f64 = ratios.iter().map(|r| r.ln()).sum();
        let geomean = (ln_sum / ratios.len() as f64).exp();
        println!();
        println!(
            "geomean ratio: {:.3} across {} rows (DSL / blackbox; <1.0 = DSL faster)",
            geomean,
            ratios.len(),
        );
    }
    let n_err = rows.len() - ratios.len();
    if n_err > 0 {
        println!("errored rows: {n_err} (see 'notes' column)");
    }
    eprintln!(
        "bench total wall time: {:.2} s",
        bench_t0.elapsed().as_secs_f64()
    );

    Ok(())
}
