//! Poseidon2-16 permutation throughput: crypto-compiler JIT kernels wrapped
//! around the transcript's on-perm shapes, serial vs warp-parallel. Requires
//! a CUDA GPU.
//!
//! Each row builds the same wrapper kernel twice — once using the reference
//! [`poseidon2_permutation`] from `crypto_compiler::kernels`, once using the
//! warp-parallel [`poseidon2_permute_par`] from
//! `crypto_compiler::poseidon2_parallel` — then measures average launch time
//! over many iterations (10 warmup + 100 timed by default). One perm per
//! launch; the numbers are meant to reflect what a transcript emitting a
//! single perm at a time actually experiences.
//!
//! Run with: `cargo bench -p crypto-compiler --bench poseidon2`

use std::time::Instant;

use crypto_compiler::{
    compile_and_load,
    ir::{IRBuilder, Module, ScalarType},
    kernel,
    kernels::{poseidon2_permutation, Poseidon2Constants},
    poseidon2_parallel::{poseidon2_permute_par, WIDTH},
    runtime::{CompileOptions, KernelModule},
};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::GpuDeviceCtx,
};

const P: u64 = 2_013_265_921;
const CHUNK: usize = 8;
const D_EF: usize = 4;

fn splitmix(n: usize, seed: u64) -> Vec<u32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z % P) as u32
        })
        .collect()
}

/// Warmup + timed loop around `f`, bracketed by stream syncs; returns the
/// average time per iteration in microseconds.
fn measure_us(ctx: &GpuDeviceCtx, warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    ctx.stream.synchronize().expect("warmup sync");
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    ctx.stream.synchronize().expect("bench sync");
    start.elapsed().as_secs_f64() * 1e6 / iters as f64
}

// ---------------------------------------------------------------------------
// Kernel shape builders. Each shape has a `serial` and `parallel` version;
// they take the same inputs and produce the same outputs.
// ---------------------------------------------------------------------------

const ABSORB_IDX_7: usize = CHUNK - 1;

/// `sponge_observe_perm_at_7` — the observe that fills slot 7 and permutes.
fn observe_perm_at_7_serial(c: &Poseidon2Constants) -> Module {
    let mut b = IRBuilder::new();
    let state = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let value = b.input("value", ScalarType::BabyBear, vec![1]);
    let body = b.compute(1, |b, _i| {
        let mut s = [b.const_u32(0); WIDTH];
        for (j, slot) in s.iter_mut().enumerate() {
            *slot = kernel!(b, state[0, #j]);
        }
        s[ABSORB_IDX_7] = kernel!(b, value[0]);
        poseidon2_permutation(b, &mut s, c);
        b.pack(&s)
    });
    b.finish("sponge_observe_perm_at_7_serial", body)
}

fn observe_perm_at_7_parallel(c: &Poseidon2Constants) -> Module {
    let consts = c.clone();
    let mut b = IRBuilder::new();
    let state = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let value = b.input("value", ScalarType::BabyBear, vec![1]);
    let body = b.compute_with(1, None, None, Some(WIDTH), move |b, _outer| {
        let par = b.par_map(|th, _s, _c| th.clone());
        let gather = b.compute_with(WIDTH, None, Some(par), None, |b, j| {
            kernel!(b,
                let orig = state[0, j];
                let v = value[0];
                if j == #(ABSORB_IDX_7 as u32) then v else orig
            )
        });
        b.bind(gather, move |b, s| {
            poseidon2_permute_par(b, s, &consts, |b, v| {
                let par = b.par_map(|th, _s, _c| th.clone());
                b.compute_with(WIDTH, None, Some(par), None, |b, j| kernel!(b, v[j]))
            })
        })
    });
    b.finish("sponge_observe_perm_at_7_parallel", body)
}

/// `sponge_sample_perm` — permute, then read state[CHUNK - 1] into `sample`.
fn sample_perm_serial(c: &Poseidon2Constants) -> Module {
    let mut b = IRBuilder::new();
    let state = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let permuted = b.compute(1, |b, _i| {
        let mut s = [b.const_u32(0); WIDTH];
        for (j, slot) in s.iter_mut().enumerate() {
            *slot = kernel!(b, state[0, #j]);
        }
        poseidon2_permutation(b, &mut s, c);
        b.pack(&s)
    });
    let permuted = b.let_bound(permuted);
    let read_idx = CHUNK - 1;
    let sample = b.compute(1, |b, _i| kernel!(b, permuted[0, #read_idx]));
    let out = b.tuple(&[permuted, sample]);
    b.finish("sponge_sample_perm_serial", out)
}

fn sample_perm_parallel(c: &Poseidon2Constants) -> Module {
    let consts = c.clone();
    let mut b = IRBuilder::new();
    let state = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let permuted = b.compute_with(1, None, None, Some(WIDTH), move |b, _outer| {
        let par = b.par_map(|th, _s, _c| th.clone());
        let gather = b.compute_with(WIDTH, None, Some(par), None, |b, j| kernel!(b, state[0, j]));
        b.bind(gather, move |b, s| {
            poseidon2_permute_par(b, s, &consts, |b, v| {
                let par = b.par_map(|th, _s, _c| th.clone());
                b.compute_with(WIDTH, None, Some(par), None, |b, j| kernel!(b, v[j]))
            })
        })
    });
    let permuted = b.let_bound(permuted);
    let read_idx = CHUNK - 1;
    let sample = b.compute(1, |b, _i| kernel!(b, permuted[0, #read_idx]));
    let out = b.tuple(&[permuted, sample]);
    b.finish("sponge_sample_perm_parallel", out)
}

// ---------------------------------------------------------------------------
// Setup + benchmark driver.
// ---------------------------------------------------------------------------

/// JIT-compiles a module, binds each input to `input_bufs[i]` (in order),
/// each output to a fresh device buffer sized from the module's declared
/// output size, and preallocates scratch. Returns the loaded kernel plus
/// the output device buffers.
fn setup_jit(
    ctx: &GpuDeviceCtx,
    module: Module,
    input_bufs: &[&DeviceBuffer<u32>],
) -> (KernelModule, Vec<DeviceBuffer<u32>>) {
    let mut km = compile_and_load(&module, &CompileOptions::default()).expect("JIT compile");
    for (i, buf) in input_bufs.iter().enumerate() {
        km.set_input(i, *buf).unwrap();
    }
    let outs: Vec<DeviceBuffer<u32>> = (0..km.num_outputs())
        .map(|i| DeviceBuffer::with_capacity_on(km.output_size(i) / 4, ctx))
        .collect();
    for (i, buf) in outs.iter().enumerate() {
        km.set_output(i, buf).unwrap();
    }
    km.ensure_scratch(ctx);
    (km, outs)
}

fn bench_shape(
    ctx: &GpuDeviceCtx,
    label: &str,
    serial: Module,
    parallel: Module,
    inputs: &[&DeviceBuffer<u32>],
    warmup: usize,
    iters: usize,
) {
    let (km_s, out_s) = setup_jit(ctx, serial, inputs);
    let (km_p, out_p) = setup_jit(ctx, parallel, inputs);

    // Correctness check: single run of each, compare each output vector.
    km_s.run(&ctx.stream).unwrap();
    km_p.run(&ctx.stream).unwrap();
    for (idx, (s, p)) in out_s.iter().zip(&out_p).enumerate() {
        let sv: Vec<u32> = s.to_host_on(ctx).unwrap();
        let pv: Vec<u32> = p.to_host_on(ctx).unwrap();
        assert_eq!(
            sv, pv,
            "shape {label} output {idx} differs between serial and parallel"
        );
    }

    let serial_us = measure_us(ctx, warmup, iters, || {
        km_s.run(&ctx.stream).expect("serial run");
    });
    let parallel_us = measure_us(ctx, warmup, iters, || {
        km_p.run(&ctx.stream).expect("parallel run");
    });
    let speedup = serial_us / parallel_us;
    println!("| {label:<40} | {serial_us:>10.3} | {parallel_us:>12.3} | {speedup:>7.2}x |");
}

fn main() {
    let ctx = GpuDeviceCtx::for_current_device().expect("CUDA context");
    let consts = Poseidon2Constants::p3_default();

    // Representative inputs.
    let state_vec = splitmix(WIDTH, 1);
    let value_vec = vec![splitmix(1, 2)[0]];
    let ext_val_vec = splitmix(D_EF, 3);
    let d_state: DeviceBuffer<u32> = state_vec.as_slice().to_device_on(&ctx).unwrap();
    let d_value: DeviceBuffer<u32> = value_vec.as_slice().to_device_on(&ctx).unwrap();
    let _ = &d_state;
    let _ = &d_value;
    let _ = ext_val_vec;

    let warmup = 10usize;
    let iters = 100usize;
    println!(
        "Poseidon2-16 permutation kernels: serial vs warp-parallel (µs/launch, 1 perm per launch)"
    );
    println!(
        "| {:<40} | {:>10} | {:>12} | {:>8} |",
        "shape", "serial", "parallel", "speedup"
    );
    println!("|{:-<42}|{:->12}|{:->14}|{:->10}|", "", "", "", "");

    bench_shape(
        &ctx,
        "sponge_observe_perm_at_7",
        observe_perm_at_7_serial(&consts),
        observe_perm_at_7_parallel(&consts),
        &[&d_state, &d_value],
        warmup,
        iters,
    );

    bench_shape(
        &ctx,
        "sponge_sample_perm",
        sample_perm_serial(&consts),
        sample_perm_parallel(&consts),
        &[&d_state],
        warmup,
        iters,
    );
}

/// Silence unused-import warnings from unused helpers when only a subset of
/// shapes is enabled — keep helpers loaded for future expansion.
#[allow(dead_code)]
fn _hold_refs(_a: fn(usize, u64) -> Vec<u32>) {}
