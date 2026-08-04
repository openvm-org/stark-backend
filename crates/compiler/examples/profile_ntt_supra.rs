//! Compares the DSL `ntt_supra_module` (nthreads=128, z_count=1) against
//! supra's `batch_ntt` at `log_n = 24`. The DSL config is the sweep winner
//! from `benches/ntt_supra_sweep.rs` at 2^24.
//!
//! Default: benchmark both kernels and print per-launch median/q25/q75 in ms.
//!
//! With `NCU_ENABLED=1`: run each kernel exactly once, after warmup, inside
//! a single NVTX range called `NCU_PROFILE` — so
//!   `ncu --nvtx --nvtx-include "NCU_PROFILE/"` (or `--range-filter`)
//! captures only those launches. Combine with `CUDA_LINEINFO=1` — that adds
//! `-lineinfo` to both the JIT nvcc invocation (see `runtime.rs`) and the
//! AOT-built supra kernels (see `openvm-cuda-builder`), so
//! `ncu --set full --import-source yes` can attach SASS to source.
//!
//! Example ncu invocation:
//!   CUDA_LINEINFO=1 NCU_ENABLED=1 \
//!     ncu --set full --import-source yes \
//!         --nvtx --nvtx-include "NCU_PROFILE/" \
//!         -f -o ntt_supra_report \
//!         target/release/examples/profile_ntt_supra

// Parked pending scatter-inverse migration: `ntt_supra_module` is
// `#[cfg(any())]`-gated in kernels.rs (refactor-plan.md).
#[cfg(any())]
use std::time::Instant;

#[cfg(any())]
use crypto_compiler::{
    kernels::{ntt_partial_twiddles, ntt_supra_module},
    runner::ModuleRunner,
    runtime::CompileOptions,
};
#[cfg(any())]
use openvm_cuda_backend::{ntt::batch_ntt, prelude::F};
#[cfg(any())]
use openvm_cuda_common::{copy::MemCopyH2D, stream::GpuDeviceCtx};

#[cfg(any())]
const P: u64 = 2013265921;

#[cfg(any())]
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

#[cfg(any())]
#[derive(Clone, Copy)]
struct Stats {
    median: f64,
    q25: f64,
    q75: f64,
}

#[cfg(any())]
fn quantiles(mut samples: Vec<f64>) -> Stats {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    let pick = |q: f64| samples[((n as f64 * q) as usize).min(n - 1)];
    Stats {
        median: pick(0.5),
        q25: pick(0.25),
        q75: pick(0.75),
    }
}

fn main() {
    eprintln!("profile_ntt_supra is disabled: ntt_supra_module is parked pending scatter-inverse migration");
}

#[cfg(any())]
fn main_disabled() {
    let log_n: usize = 24;
    let n = 1usize << log_n;
    let ctx = GpuDeviceCtx::for_current_device().expect("CUDA context");
    let coeffs = pseudo_field_elems(n, 1);
    let twiddles = ntt_partial_twiddles(log_n);

    // ModuleRunner Mont-encodes the BabyBear inputs at H2D and decodes
    // outputs on read — matching the emitted kernel's Montgomery domain.
    let mut runner = ModuleRunner::new(
        &ntt_supra_module(log_n, 128, 1, false),
        &CompileOptions::default(),
    )
    .expect("JIT compile");
    runner.set_inputs(&[coeffs.clone(), twiddles]);

    // Supra baseline: build the device input once, reuse across warmup /
    // profile / bench. `batch_ntt` is in-place, but the arithmetic is
    // structurally identical across launches, so wall-clock timing is fair.
    let input_f: Vec<F> = coeffs.iter().map(|&x| F::new(x)).collect();
    let d_f = input_f.as_slice().to_device_on(&ctx).unwrap();

    // Warm up both paths so first-touch faults and lazy driver init don't
    // pollute the measurement. Each path uses its own stream (the runner
    // owns one, `ctx` holds the one supra targets), so both must be synced.
    runner.run();
    batch_ntt(&d_f, log_n as u32, 0, 1, true, false, &ctx);
    runner.sync();
    ctx.stream.synchronize().expect("warmup sync");

    if std::env::var_os("NCU_ENABLED").is_some() {
        eprintln!("[ncu] one launch each inside NCU_PROFILE @ log_n={log_n}");
        nvtx::range_push!("NCU_PROFILE");
        runner.run();
        batch_ntt(&d_f, log_n as u32, 0, 1, true, false, &ctx);
        runner.sync();
        ctx.stream.synchronize().expect("ncu sync");
        nvtx::range_pop!();
        return;
    }

    let iters: usize = std::env::var("BENCH_KERNEL_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let mut dsl_samples: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        runner.run();
        runner.sync();
        dsl_samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let dsl = quantiles(dsl_samples);

    let mut supra_samples: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        batch_ntt(&d_f, log_n as u32, 0, 1, true, false, &ctx);
        ctx.stream.synchronize().expect("supra sync");
        supra_samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let supra = quantiles(supra_samples);

    println!(
        "log_n={log_n}, iters={iters}\n\
         [DSL   ntt_supra(128,1)] median={:.4} ms  q25={:.4} ms  q75={:.4} ms\n\
         [supra batch_ntt       ] median={:.4} ms  q25={:.4} ms  q75={:.4} ms\n\
         DSL / supra = {:.2}x  (>1 means DSL is slower)",
        dsl.median,
        dsl.q25,
        dsl.q75,
        supra.median,
        supra.q25,
        supra.q75,
        dsl.median / supra.median,
    );
}
