//! Poseidon2-16 permutation and non-perm transcript ops: crypto-compiler JIT
//! kernels wrapped around the transcript's shapes, serial vs warp-parallel.
//! Requires a CUDA GPU.
//!
//! Each row builds the same wrapper kernel twice — once using the reference
//! [`poseidon2_permutation`] from `crypto_compiler::kernels`, once using the
//! warp-parallel [`poseidon2_permute_par`] from
//! `crypto_compiler::poseidon2_parallel` — then measures average launch time
//! over many iterations (10 warmup + 100 timed by default), both direct
//! (`GraphExe::run`, dominated by host-side launch overhead) and captured
//! (`GraphExe::launch_graph`, one `cudaGraphLaunch` per iter — closer to
//! per-node cost inside a real transcript-heavy graph).
//!
//! Also covers non-perm transcript ops (observe, observe_ext, sample_ext_pack)
//! where the current serial shapes launch a `compute(1)` single-thread kernel
//! and the parallel variants use a full-warp `compute_with(WIDTH, par=id)`.
//! For non-perm ops the "work" is a handful of selects/writes per launch, so
//! the win (if any) is only visible in captured mode where launch overhead
//! is a small constant.
//!
//! Run with: `cargo bench -p crypto-compiler --bench poseidon2`

use std::time::Instant;

use crypto_compiler::{
    graph_exe::{GraphCompiler, GraphExe},
    graph_ir::GraphModule,
    ir::{IRBuilder, Module, ScalarType},
    kernel,
    kernels::{poseidon2_permutation, Poseidon2Constants},
    poseidon2_parallel::{poseidon2_permute_par, WIDTH},
    test_utils::to_monty,
};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, stream::GpuDeviceCtx};

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
                if j == #ABSORB_IDX_7 then v else orig
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
// Non-perm transcript ops. The serial variants match the shapes currently
// used in `openvm_cuda_backend::sponge_graph_ir` (single-thread `compute(1)`
// that walks all 16 state slots); the parallel variants distribute the
// same slot updates across a 16-lane warp under `par=id`.
// ---------------------------------------------------------------------------

const ABSORB_IDX_0: usize = 0;

/// `sponge_observe` at position 0: slot 0 takes `value[0]`, others pass through.
/// Serial shape mirrors `sponge_graph_ir::build_observe_module` at absorb_idx 0.
fn observe_at_0_serial() -> Module {
    let mut b = IRBuilder::new();
    let state = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let value = b.input("value", ScalarType::BabyBear, vec![1]);
    let body = b.compute(1, |b, _i| {
        let mut s = [b.const_u32(0); WIDTH];
        for (j, slot) in s.iter_mut().enumerate() {
            *slot = kernel!(b, state[0, #j]);
        }
        s[ABSORB_IDX_0] = kernel!(b, value[0]);
        b.pack(&s)
    });
    b.finish("sponge_observe_at_0_serial", body)
}

/// Parallel gather: 16 lanes under `par=id`, each lane produces its own
/// output slot with a single select.
fn observe_at_0_parallel() -> Module {
    let mut b = IRBuilder::new();
    let state = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let value = b.input("value", ScalarType::BabyBear, vec![1]);
    let body = b.compute_with(1, None, None, Some(WIDTH), move |b, _outer| {
        let par = b.par_map(|th, _s, _c| th.clone());
        b.compute_with(WIDTH, None, Some(par), None, |b, j| {
            kernel!(b,
                let orig = state[0, j];
                let v = value[0];
                if j == #ABSORB_IDX_0 then v else orig
            )
        })
    });
    b.finish("sponge_observe_at_0_parallel", body)
}

/// `sponge_observe_ext` at position 0: slots `0..D_EF` take `value[0..D_EF]`,
/// others pass through. Serial shape mirrors
/// `sponge_graph_ir::build_observe_ext_module_serial` at absorb_idx 0.
fn observe_ext_at_0_serial() -> Module {
    let mut b = IRBuilder::new();
    let state = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let value = b.input("value", ScalarType::BabyBear, vec![D_EF]);
    let body = b.compute(1, |b, _i| {
        let mut s = [b.const_u32(0); WIDTH];
        for (j, slot) in s.iter_mut().enumerate() {
            *slot = kernel!(b, state[0, #j]);
        }
        for k in 0..D_EF {
            s[k] = kernel!(b, value[#k]);
        }
        b.pack(&s)
    });
    b.finish("sponge_observe_ext_at_0_serial", body)
}

fn observe_ext_at_0_parallel() -> Module {
    let mut b = IRBuilder::new();
    let state = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let value = b.input("value", ScalarType::BabyBear, vec![D_EF]);
    let body = b.compute_with(1, None, None, Some(WIDTH), move |b, _outer| {
        let par = b.par_map(|th, _s, _c| th.clone());
        b.compute_with(WIDTH, None, Some(par), None, |b, j| {
            kernel!(b,
                let orig = state[0, j];
                let d = #D_EF;
                if j < d then value[j] else orig
            )
        })
    });
    b.finish("sponge_observe_ext_at_0_parallel", body)
}

/// `sample_ext_pack` non-perm path: `p = D_EF`, so every one of the 4
/// output slots picks a pre-perm read. Serial variant walks the 4-slot
/// output in `compute(D_EF)`; parallel splits it across 4 lanes.
fn sample_ext_pack_at_p_serial() -> Module {
    let mut b = IRBuilder::new();
    let pre = b.input("pre", ScalarType::BabyBear, vec![1, WIDTH]);
    let body = b.compute(D_EF, |b, k| {
        // No-perm: samples[k] = pre[0, p - 1 - k], p = D_EF.
        let p = b.const_u32(D_EF as u32);
        let one = b.const_u32(1);
        let zero = b.const_u32(0);
        let idx = kernel!(b, p - one - k);
        b.index(pre, &[zero, idx])
    });
    b.finish("sample_ext_pack_serial", body)
}

fn sample_ext_pack_at_p_parallel() -> Module {
    let mut b = IRBuilder::new();
    let pre = b.input("pre", ScalarType::BabyBear, vec![1, WIDTH]);
    // With D_EF lanes there's no meaningful warp; par=id still yields
    // one thread per output slot, which is the shape we want to test.
    let body = b.compute_with(1, None, None, Some(D_EF), move |b, _outer| {
        let par = b.par_map(|th, _s, _c| th.clone());
        b.compute_with(D_EF, None, Some(par), None, |b, k| {
            let p = b.const_u32(D_EF as u32);
            let one = b.const_u32(1);
            let zero = b.const_u32(0);
            let idx = kernel!(b, p - one - k);
            b.index(pre, &[zero, idx])
        })
    });
    b.finish("sample_ext_pack_parallel", body)
}

// ---------------------------------------------------------------------------
// Setup + benchmark driver.
// ---------------------------------------------------------------------------

/// JIT-compiles a module and binds each input to `input_bufs[i]` on the
/// `GraphExe` (so both `run` and `capture_graph`/`launch_graph` are
/// callable). Outputs live inside the exe's device pool — read them
/// back with [`GraphExe::get_output`].
fn setup_jit(ctx: &GpuDeviceCtx, module: Module, input_bufs: &[&DeviceBuffer<u8>]) -> GraphExe {
    let gm = GraphModule::from_ir(module, &[]).unwrap();
    let mut exe = GraphCompiler::new()
        .compile(gm.into_builder())
        .expect("JIT compile");
    for (i, buf) in input_bufs.iter().enumerate() {
        exe.set_input(ctx, i, buf).unwrap();
    }
    exe
}

/// Copy a `[u32]` slice into a freshly-allocated `DeviceBuffer<u8>` — the
/// byte view every `GraphExe::set_input` call expects.
fn u8_device_buf(ctx: &GpuDeviceCtx, data: &[u32]) -> DeviceBuffer<u8> {
    // SAFETY: `[u32]` is contiguously laid out; we only reinterpret its
    // bytes for a device-side H2D copy.
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    bytes.to_device_on(ctx).unwrap()
}

fn bench_shape(
    ctx: &GpuDeviceCtx,
    label: &str,
    serial: Module,
    parallel: Module,
    inputs: &[&DeviceBuffer<u8>],
    warmup: usize,
    iters: usize,
) {
    let mut exe_s = setup_jit(ctx, serial, inputs);
    let mut exe_p = setup_jit(ctx, parallel, inputs);

    // Correctness check: single run of each, compare each output vector.
    exe_s.run(ctx).unwrap();
    exe_p.run(ctx).unwrap();
    let num_outputs = exe_s.num_outputs();
    for i in 0..num_outputs {
        let sv = exe_s.get_output(i).to_host_on(ctx).unwrap();
        let pv = exe_p.get_output(i).to_host_on(ctx).unwrap();
        assert_eq!(
            sv, pv,
            "shape {label} output {i} differs between serial and parallel"
        );
    }

    // Direct-launch: `GraphExe::run` — one host-side per-node dispatch per
    // kernel. Dominated by launch overhead for tiny kernels like these.
    let serial_direct = measure_us(ctx, warmup, iters, || exe_s.run(ctx).unwrap());
    let parallel_direct = measure_us(ctx, warmup, iters, || exe_p.run(ctx).unwrap());
    let direct_speedup = serial_direct / parallel_direct;

    // Captured-graph: one `cudaGraphLaunch` per iter. Removes most of the
    // host-dispatch cost, exposing the actual per-kernel GPU time.
    exe_s.capture_graph(ctx).expect("capture serial");
    exe_p.capture_graph(ctx).expect("capture parallel");
    let serial_captured = measure_us(ctx, warmup, iters, || exe_s.launch_graph(ctx).unwrap());
    let parallel_captured = measure_us(ctx, warmup, iters, || exe_p.launch_graph(ctx).unwrap());
    let captured_speedup = serial_captured / parallel_captured;

    println!(
        "| {label:<40} | {serial_direct:>7.3} | {parallel_direct:>9.3} | {direct_speedup:>5.2}x \
         | {serial_captured:>7.3} | {parallel_captured:>9.3} | {captured_speedup:>5.2}x |"
    );
}

fn main() {
    let ctx = GpuDeviceCtx::for_current_device().expect("CUDA context");
    let consts = Poseidon2Constants::p3_default();

    // Representative inputs. DSL kernels expect Montgomery-encoded BabyBear
    // on device; this bench uses the raw `KernelProgram` (bypassing
    // `ModuleRunner`), so we Mont-encode here. The bench only checks
    // serial-vs-parallel *equivalence*, so the exact input value doesn't
    // matter; encoding preserves that equivalence.
    let state_vec: Vec<u32> = splitmix(WIDTH, 1).into_iter().map(to_monty).collect();
    let value_vec: Vec<u32> = vec![to_monty(splitmix(1, 2)[0])];
    let ext_val_vec: Vec<u32> = splitmix(D_EF, 3).into_iter().map(to_monty).collect();
    let d_state = u8_device_buf(&ctx, &state_vec);
    let d_value = u8_device_buf(&ctx, &value_vec);
    let d_ext = u8_device_buf(&ctx, &ext_val_vec);

    let warmup = 10usize;
    let iters = 100usize;
    println!(
        "Poseidon2-16 transcript kernels: serial vs warp-parallel (µs/launch, 1 op per launch)"
    );
    println!(
        "| {:<40} | {:>7} | {:>9} | {:>6} | {:>7} | {:>9} | {:>6} |",
        "shape", "s.dir", "p.dir", "d.spd", "s.cap", "p.cap", "c.spd"
    );
    println!(
        "|{:-<42}|{:->9}|{:->11}|{:->8}|{:->9}|{:->11}|{:->8}|",
        "", "", "", "", "", "", ""
    );

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

    bench_shape(
        &ctx,
        "sponge_observe_at_0 (non-perm)",
        observe_at_0_serial(),
        observe_at_0_parallel(),
        &[&d_state, &d_value],
        warmup,
        iters,
    );

    bench_shape(
        &ctx,
        "sponge_observe_ext_at_0 (non-perm)",
        observe_ext_at_0_serial(),
        observe_ext_at_0_parallel(),
        &[&d_state, &d_ext],
        warmup,
        iters,
    );

    bench_shape(
        &ctx,
        "sponge_sample_ext_pack (no perm)",
        sample_ext_pack_at_p_serial(),
        sample_ext_pack_at_p_parallel(),
        &[&d_state],
        warmup,
        iters,
    );
}

/// Silence unused-import warnings from unused helpers when only a subset of
/// shapes is enabled — keep helpers loaded for future expansion.
#[allow(dead_code)]
fn _hold_refs(_a: fn(usize, u64) -> Vec<u32>) {}
