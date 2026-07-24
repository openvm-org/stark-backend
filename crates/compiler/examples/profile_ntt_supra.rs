//! Runs the DSL `ntt_supra_module` (nthreads=128, z_count=1) and supra's
//! `batch_ntt` once each at `log_n = 24`, sharing one process so both
//! kernels land in the same Nsight Compute profile. The DSL config
//! picked here is the winner of `benches/ntt_supra_sweep.rs` at 2^24.

use crypto_compiler::{
    compile_and_load,
    kernels::{ntt_partial_twiddles, ntt_supra_module},
    runner::to_monty,
    runtime::CompileOptions,
};
use openvm_cuda_backend::{ntt::batch_ntt, prelude::F};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, stream::GpuDeviceCtx};

const P: u64 = 2013265921;

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

fn main() {
    let log_n: usize = 24;
    let n = 1usize << log_n;
    let ctx = GpuDeviceCtx::for_current_device().expect("CUDA context");
    let input = pseudo_field_elems(n, 1);

    // DSL: best config at 2^24 is (nthreads=128, z_count=1). Both the
    // coefficient input and the windowed twiddle table go to device in
    // Montgomery form to match the emitted kernel's data representation.
    let input_mont: Vec<u32> = input.iter().map(|&x| to_monty(x)).collect();
    let twiddles_mont: Vec<u32> = ntt_partial_twiddles(log_n)
        .iter()
        .map(|&x| to_monty(x))
        .collect();
    let d_in = input_mont.as_slice().to_device_on(&ctx).unwrap();
    let d_tw = twiddles_mont.as_slice().to_device_on(&ctx).unwrap();
    let d_out = DeviceBuffer::<u32>::with_capacity_on(n, &ctx);
    let mut km = compile_and_load(
        &ntt_supra_module(log_n, 128, 1, false),
        &CompileOptions::default(),
    )
    .expect("JIT compile");
    km.set_input(0, &d_in).unwrap();
    km.set_input(1, &d_tw).unwrap();
    km.set_output(0, &d_out).unwrap();
    km.ensure_scratch(&ctx);

    // Warmup (compile fusion, first-touch pages) — nvtx-range these so
    // the profile can be filtered to just the timed launches if desired.
    km.run(&ctx.stream).expect("DSL NTT warmup");
    ctx.stream.synchronize().expect("warmup sync");

    // Supra baseline setup.
    let input_f: Vec<F> = input.iter().map(|&x| F::new(x)).collect();
    let d_f = input_f.as_slice().to_device_on(&ctx).unwrap();
    batch_ntt(&d_f, log_n as u32, 0, 1, true, false, &ctx);
    ctx.stream.synchronize().expect("supra warmup sync");

    // The two profiled launches. One of each — ncu can filter by
    // kernel name to compare metrics side by side.
    eprintln!("=== DSL ntt_supra (128, 1) @ log_n=24 ===");
    km.run(&ctx.stream).expect("DSL NTT profile run");
    ctx.stream.synchronize().expect("dsl sync");

    eprintln!("=== supra batch_ntt @ log_n=24 ===");
    batch_ntt(&d_f, log_n as u32, 0, 1, true, false, &ctx);
    ctx.stream.synchronize().expect("supra sync");
}
