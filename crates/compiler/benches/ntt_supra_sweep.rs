//! Sweeps `ntt_supra_module(log_n, nthreads, z_count, coalesced=false)`
//! across a matrix of block-size × per-thread-batching configurations
//! and compares each to the production supra NTT
//! (`openvm_cuda_backend::ntt::batch_ntt`) at the same domain sizes.
//!
//! Every config's group width is `radix = 1 + log2(nthreads) +
//! log2(z_count)`. The sweep picks configs that fit inside the smallest
//! `LOG_SIZES` entry so every cell of the table is populated.
//!
//! Run with: `cargo bench -p crypto-compiler --bench ntt_supra_sweep`

use std::time::Instant;

use crypto_compiler::{
    compile_and_load,
    ir::Module,
    kernels::{ntt_partial_twiddles, ntt_supra_module},
    runtime::{CompileOptions, KernelModule},
};
use openvm_cuda_backend::{ntt::batch_ntt, prelude::F};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::GpuDeviceCtx,
};
use p3_field::PrimeField32;

const P: u64 = 2013265921;

/// Domain sizes to time — same as the main NTT bench so results are
/// directly comparable.
const LOG_SIZES: &[usize] = &[12, 14, 16, 18, 20, 22, 24];

/// `(nthreads, z_count)` pairs. `radix = 1 + log2(nthreads) +
/// log2(z_count)` must be ≤ min(LOG_SIZES) = 12 so the register-tiled
/// group fits inside every configured domain size. The sweep spans:
///
/// - Fixed `z_count = 1`, growing `nthreads`, to isolate the effect of block size on the "no
///   per-thread batching" case supra actually runs when it takes the `z_count=1` branch.
/// - `nthreads * z_count` matched across (128, 4), (256, 2), (512, 1) and again at (256, 4), (512,
///   2), (1024, 1) so per-thread batching and block-size scaling can be compared at fixed tile
///   width (`radix` constant).
const CONFIGS: &[(usize, usize)] = &[
    (128, 1),
    (256, 1),
    (512, 1),
    (1024, 1),
    (128, 4),
    (256, 2),
    (256, 4),
    (512, 2),
];

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

fn setup_jit(
    module: Module,
    ctx: &GpuDeviceCtx,
    d_in: &DeviceBuffer<u32>,
    d_tw: &DeviceBuffer<u32>,
    d_out: &DeviceBuffer<u32>,
) -> KernelModule {
    let mut km = compile_and_load(&module, &CompileOptions::default()).expect("JIT compile");
    km.set_input(0, d_in).unwrap();
    km.set_input(1, d_tw).unwrap();
    km.set_output(0, d_out).unwrap();
    km.ensure_scratch(ctx);
    km
}

/// One row of the sweep — for a given `log_n`, time supra plus every
/// entry of `CONFIGS`. Every value in the row is Gelem/s (higher is
/// better). The supra column is timed once, then each config's kernel
/// is JIT-compiled and cross-checked against supra before timing.
fn bench_row(ctx: &GpuDeviceCtx, log_n: usize) -> (f64, Vec<f64>) {
    let n = 1usize << log_n;
    let input = pseudo_field_elems(n, 1);

    let d_in = input.as_slice().to_device_on(ctx).unwrap();
    let d_tw = ntt_partial_twiddles(log_n)
        .as_slice()
        .to_device_on(ctx)
        .unwrap();

    // Supra baseline (in-place, Montgomery form).
    let input_f: Vec<F> = input.iter().map(|&x| F::new(x)).collect();
    let d_f = input_f.as_slice().to_device_on(ctx).unwrap();
    // Warmup + reference output.
    batch_ntt(&d_f, log_n as u32, 0, 1, true, false, ctx);
    let got_supra: Vec<u32> = d_f
        .to_host_on(ctx)
        .unwrap()
        .iter()
        .map(|x| x.as_canonical_u32())
        .collect();

    let iters = ((1usize << 28) / n).clamp(10, 400);
    let warmup = (iters / 10).max(3);
    let supra_ms = measure(ctx, warmup, iters, || {
        batch_ntt(&d_f, log_n as u32, 0, 1, true, false, ctx);
    });
    let gelems = |ms: f64| n as f64 / (ms * 1e-3) / 1e9;
    let supra_gelem_s = gelems(supra_ms);

    // Each supra-module config in turn.
    let mut row = Vec::with_capacity(CONFIGS.len());
    for &(nthreads, z_count) in CONFIGS {
        let radix = 1 + nthreads.trailing_zeros() as usize + z_count.trailing_zeros() as usize;
        if radix > log_n {
            // Config's group is wider than the domain — the register
            // tile can't fit, so there's nothing meaningful to time.
            row.push(f64::NAN);
            continue;
        }
        let d_out = DeviceBuffer::<u32>::with_capacity_on(n, ctx);
        let km = setup_jit(
            ntt_supra_module(log_n, nthreads, z_count, false),
            ctx,
            &d_in,
            &d_tw,
            &d_out,
        );
        km.run(&ctx.stream).expect("supra JIT NTT run");
        let got: Vec<u32> = d_out.to_host_on(ctx).unwrap();
        assert_eq!(
            got, got_supra,
            "supra-module ({nthreads}, {z_count}) mismatch at n=2^{log_n}"
        );
        let ms = measure(ctx, warmup, iters, || {
            km.run(&ctx.stream).expect("supra JIT NTT run");
        });
        row.push(gelems(ms));
    }
    (supra_gelem_s, row)
}

fn main() {
    let ctx = GpuDeviceCtx::for_current_device().expect("CUDA context");
    println!(
        "BabyBear forward NTT, natural-order input/output, single column. Cells are Gelem/s (higher is better)."
    );
    println!(
        "Configs: (nthreads, z_count). radix = 1 + log2(nthreads) + log2(z_count). Cells marked `-` didn't fit."
    );

    // Header
    print!("| n     | supra   |");
    for (nthreads, z_count) in CONFIGS {
        print!(" t{nthreads:<4}/z{z_count} |");
    }
    println!();
    print!("|-------|---------|");
    for _ in CONFIGS {
        print!("----------|");
    }
    println!();

    for &log_n in LOG_SIZES {
        let (supra_gs, row) = bench_row(&ctx, log_n);
        print!("| 2^{log_n:<2} | {supra_gs:>7.2} |");
        for cell in row {
            if cell.is_nan() {
                print!("        - |");
            } else {
                print!(" {cell:>8.2} |");
            }
        }
        println!();
    }

    // Second table: ratios vs supra so the winning cell is obvious.
    println!();
    println!("Speedup vs supra (× — higher is better; supra column = 1.00×):");
    print!("| n     | supra |");
    for (nthreads, z_count) in CONFIGS {
        print!(" t{nthreads:<4}/z{z_count} |");
    }
    println!();
    print!("|-------|-------|");
    for _ in CONFIGS {
        print!("----------|");
    }
    println!();
    for &log_n in LOG_SIZES {
        let (supra_gs, row) = bench_row(&ctx, log_n);
        print!("| 2^{log_n:<2} | 1.00× |");
        for cell in row {
            if cell.is_nan() {
                print!("        - |");
            } else {
                print!(" {:>7.3}× |", cell / supra_gs);
            }
        }
        println!();
    }
}
