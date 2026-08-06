//! BabyBear NTT throughput: the crypto-compiler JIT kernels (flat, shared-
//! memory, register-tiled) vs the production supra NTT
//! (`openvm_cuda_backend::ntt::batch_ntt`). Requires a CUDA GPU.
//!
//! Run with: `cargo bench -p crypto-compiler --bench ntt`

use std::time::Instant;

use crypto_compiler::{
    graph_exe::{GraphCompiler, GraphExe},
    graph_ir::GraphModule,
    ir::Module,
    kernels::{ntt_module, ntt_reg_module, ntt_shared_module, ntt_twiddles},
    test_utils::{from_monty, to_monty},
};
use openvm_cuda_backend::{ntt::batch_ntt, prelude::F};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::GpuDeviceCtx,
};
use p3_field::PrimeField32;

const P: u64 = 2013265921;
const LOG_SIZES: &[usize] = &[12, 14, 16, 18, 20, 22, 24];

/// Deterministic pseudo-random canonical BabyBear elements (splitmix64).
fn pseudo_field_elems(n: usize, seed: u64) -> Vec<u32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            (z % P) as u32
        })
        .collect()
}

/// Warmup + timed loop around `f`, bracketed by stream syncs; returns the
/// average time per iteration in milliseconds.
fn measure(ctx: &GpuDeviceCtx, warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    ctx.stream.synchronize().expect("warmup sync");
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    ctx.stream.synchronize().expect("bench sync");
    start.elapsed().as_secs_f64() * 1e3 / iters as f64
}

/// JIT-compiles `module` and binds the NTT inputs/outputs; returns the
/// compiled [`GraphExe`] (single-kernel — drive it via
/// [`GraphExe::kernel_program`]) and the nvcc compile time in seconds.
fn setup_jit(
    module: Module,
    d_in: &DeviceBuffer<u32>,
    d_tw: &DeviceBuffer<u32>,
    d_out: &DeviceBuffer<u32>,
) -> (GraphExe, f64) {
    let t0 = Instant::now();
    let gm = GraphModule::from_ir(module, &[]).unwrap();
    let mut exe = GraphCompiler::new()
        .compile(gm.into_builder())
        .expect("JIT compile");
    let compile_s = t0.elapsed().as_secs_f64();
    let km = exe.kernel_program(0);
    km.set_input(0, d_in).unwrap();
    km.set_input(1, d_tw).unwrap();
    km.set_output(0, d_out).unwrap();
    (exe, compile_s)
}

fn bench_size(ctx: &GpuDeviceCtx, log_n: usize) {
    let n = 1usize << log_n;
    let input = pseudo_field_elems(n, 1);

    // JIT NTTs: Montgomery-form u32 in/out, out-of-place, own twiddle
    // input. The DSL kernel operates on Montgomery-encoded BabyBear
    // throughout, so callers of the raw `KernelProgram` (as this bench
    // does, bypassing `ModuleRunner`) must encode/decode themselves.
    let input_mont: Vec<u32> = input.iter().map(|&x| to_monty(x)).collect();
    let twiddles_mont: Vec<u32> = ntt_twiddles(log_n).iter().map(|&x| to_monty(x)).collect();
    let d_in = input_mont.as_slice().to_device_on(ctx).unwrap();
    let d_tw = twiddles_mont.as_slice().to_device_on(ctx).unwrap();
    let d_out_flat = DeviceBuffer::<u32>::with_capacity_on(n, ctx);
    let d_out_sh = DeviceBuffer::<u32>::with_capacity_on(n, ctx);
    let d_out_reg = DeviceBuffer::<u32>::with_capacity_on(n, ctx);
    let (mut exe_flat, _) = setup_jit(ntt_module(log_n), &d_in, &d_tw, &d_out_flat);
    let (mut exe_sh, sh_compile_s) = setup_jit(ntt_shared_module(log_n), &d_in, &d_tw, &d_out_sh);
    let (mut exe_reg, reg_compile_s) = setup_jit(ntt_reg_module(log_n), &d_in, &d_tw, &d_out_reg);

    // Supra NTT: Montgomery-form BabyBear, in-place, natural-order input.
    let input_f: Vec<F> = input.iter().map(|&x| F::new(x)).collect();
    let d_f = input_f.as_slice().to_device_on(ctx).unwrap();

    // One-time cross-check: all four must produce the same NTT.
    exe_flat
        .kernel_program(0)
        .run(&ctx.stream)
        .expect("flat JIT NTT run");
    exe_sh
        .kernel_program(0)
        .run(&ctx.stream)
        .expect("shared JIT NTT run");
    exe_reg
        .kernel_program(0)
        .run(&ctx.stream)
        .expect("reg JIT NTT run");
    // JIT outputs are Montgomery-encoded; decode before comparing against
    // supra's (already-canonicalized-via-`as_canonical_u32`) output below.
    let got_flat: Vec<u32> = d_out_flat
        .to_host_on(ctx)
        .unwrap()
        .into_iter()
        .map(from_monty)
        .collect();
    let got_sh: Vec<u32> = d_out_sh
        .to_host_on(ctx)
        .unwrap()
        .into_iter()
        .map(from_monty)
        .collect();
    let got_reg: Vec<u32> = d_out_reg
        .to_host_on(ctx)
        .unwrap()
        .into_iter()
        .map(from_monty)
        .collect();
    batch_ntt(&d_f, log_n as u32, 0, 1, true, false, ctx);
    let got_supra: Vec<u32> = d_f
        .to_host_on(ctx)
        .unwrap()
        .iter()
        .map(|x| x.as_canonical_u32())
        .collect();
    assert_eq!(
        got_flat, got_supra,
        "flat JIT vs supra NTT mismatch at n=2^{log_n}"
    );
    assert_eq!(
        got_sh, got_supra,
        "shared JIT vs supra NTT mismatch at n=2^{log_n}"
    );
    assert_eq!(
        got_reg, got_supra,
        "reg JIT vs supra NTT mismatch at n=2^{log_n}"
    );

    let iters = ((1usize << 28) / n).clamp(10, 400);
    let warmup = (iters / 10).max(3);

    let flat_ms = measure(ctx, warmup, iters, || {
        exe_flat
            .kernel_program(0)
            .run(&ctx.stream)
            .expect("flat JIT NTT run");
    });
    let sh_ms = measure(ctx, warmup, iters, || {
        exe_sh
            .kernel_program(0)
            .run(&ctx.stream)
            .expect("shared JIT NTT run");
    });
    let reg_ms = measure(ctx, warmup, iters, || {
        exe_reg
            .kernel_program(0)
            .run(&ctx.stream)
            .expect("reg JIT NTT run");
    });
    // Repeated in-place transforms of (field-valued) garbage: identical work.
    let supra_ms = measure(ctx, warmup, iters, || {
        batch_ntt(&d_f, log_n as u32, 0, 1, true, false, ctx);
    });

    let gelems = |ms: f64| n as f64 / (ms * 1e-3) / 1e9;
    let nvcc_s = sh_compile_s + reg_compile_s;
    println!(
        "| 2^{log_n:<2} | {flat_ms:>9.3} | {sh_ms:>10.3} | {reg_ms:>7.3} | {supra_ms:>10.3} | {:>9.2} | {:>10.2} | {:>10.2} | {:>7.2}x | {:>7.2}x | {nvcc_s:>7.1} |",
        gelems(sh_ms),
        gelems(reg_ms),
        gelems(supra_ms),
        sh_ms / reg_ms,
        reg_ms / supra_ms,
    );
}

fn main() {
    let ctx = GpuDeviceCtx::for_current_device().expect("CUDA context");
    println!("BabyBear forward NTT, natural-order input and output, single column");
    println!("| n     | flat (ms) | shared (ms) | reg (ms) | supra (ms) | sh Gelem/s | reg Gelem/s | su Gelem/s | sh/reg | reg/supra | nvcc (s) |");
    println!("|-------|-----------|-------------|----------|------------|------------|-------------|------------|--------|-----------|----------|");
    for &log_n in LOG_SIZES {
        bench_size(&ctx, log_n);
    }
}
