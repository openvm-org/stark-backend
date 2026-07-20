//! BabyBear NTT throughput: the crypto-compiler JIT kernel vs the production
//! supra NTT (`openvm_cuda_backend::ntt::batch_ntt`). Requires a CUDA GPU.
//!
//! Run with: `cargo bench -p crypto-compiler --bench ntt`

use std::time::Instant;

use crypto_compiler::{
    compile_and_load,
    kernels::{ntt_module, ntt_twiddles},
    runtime::CompileOptions,
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

fn bench_size(ctx: &GpuDeviceCtx, log_n: usize) {
    let n = 1usize << log_n;
    let input = pseudo_field_elems(n, 1);

    // JIT-compiled NTT: canonical u32 in/out, out-of-place, own twiddle input.
    let t0 = Instant::now();
    let mut km =
        compile_and_load(ntt_module(log_n), &CompileOptions::default()).expect("JIT compile");
    let jit_compile_s = t0.elapsed().as_secs_f64();
    let d_in = input.as_slice().to_device_on(ctx).unwrap();
    let d_tw = ntt_twiddles(log_n).as_slice().to_device_on(ctx).unwrap();
    let d_out = DeviceBuffer::<u32>::with_capacity_on(n, ctx);
    km.set_input(0, &d_in).unwrap();
    km.set_input(1, &d_tw).unwrap();
    km.set_output(0, &d_out).unwrap();
    km.ensure_scratch(ctx);

    // Supra NTT: Montgomery-form BabyBear, in-place, natural-order input.
    let input_f: Vec<F> = input.iter().map(|&x| F::new(x)).collect();
    let d_f = input_f.as_slice().to_device_on(ctx).unwrap();

    // One-time cross-check: both must produce the same natural-order NTT.
    km.run(&ctx.stream).expect("JIT NTT run");
    let got_jit: Vec<u32> = d_out.to_host_on(ctx).unwrap();
    batch_ntt(&d_f, log_n as u32, 0, 1, true, false, ctx);
    let got_supra: Vec<u32> = d_f
        .to_host_on(ctx)
        .unwrap()
        .iter()
        .map(|x| x.as_canonical_u32())
        .collect();
    assert_eq!(
        got_jit, got_supra,
        "JIT vs supra NTT mismatch at n=2^{log_n}"
    );

    let iters = ((1usize << 28) / n).clamp(10, 400);
    let warmup = (iters / 10).max(3);

    let jit_ms = measure(ctx, warmup, iters, || {
        km.run(&ctx.stream).expect("JIT NTT run");
    });
    // Repeated in-place transforms of (field-valued) garbage: identical work.
    let supra_ms = measure(ctx, warmup, iters, || {
        batch_ntt(&d_f, log_n as u32, 0, 1, true, false, ctx);
    });

    let gelems = |ms: f64| n as f64 / (ms * 1e-3) / 1e9;
    println!(
        "| 2^{log_n:<2} | {jit_ms:>9.3} | {:>8.2} | {supra_ms:>9.3} | {:>8.2} | {:>6.1}x | {jit_compile_s:>7.1} |",
        gelems(jit_ms),
        gelems(supra_ms),
        jit_ms / supra_ms,
    );
}

fn main() {
    let ctx = GpuDeviceCtx::for_current_device().expect("CUDA context");
    println!("BabyBear forward NTT, natural-order input and output, single column");
    println!("| n     |  jit (ms) | Gelem/s | supra (ms) | Gelem/s | jit/supra | nvcc (s) |");
    println!("|-------|-----------|---------|-----------|---------|-----------|----------|");
    for &log_n in LOG_SIZES {
        bench_size(&ctx, log_n);
    }
}
