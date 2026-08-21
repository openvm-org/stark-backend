//! Scratch: nsys target for per-kernel time distribution of the three
//! JIT NTT variants at 2^24. Setup happens before cudaProfilerStart; each
//! variant runs `ITERS` times inside its own NVTX range.
//!
//!   nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node \
//!     --gpu-metrics-devices=visible -f true -o /tmp/ntt_kdist \
//!     target/debug/examples/tmp_ntt_kdist
//!   nsys stats --report cuda_gpu_kern_sum /tmp/ntt_kdist.nsys-rep

use crypto_compiler::{
    graph_compiler::GraphCompiler, graph_exe::GraphExe,
    graph_ir::GraphModule,
    ir::Module,
    kernels::{ntt_module, ntt_reg_module, ntt_shared_module, ntt_twiddles},
    test_utils::to_monty,
};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, stream::GpuDeviceCtx};

extern "C" {
    fn cudaProfilerStart() -> i32;
    fn cudaProfilerStop() -> i32;
}

const P: u64 = 2013265921;
const ITERS: usize = 20;

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

fn to_device_bytes(ctx: &GpuDeviceCtx, xs: &[u32]) -> DeviceBuffer<u8> {
    let bytes: Vec<u8> = xs.iter().flat_map(|x| x.to_le_bytes()).collect();
    bytes.as_slice().to_device_on(ctx).unwrap()
}

fn setup(
    ctx: &GpuDeviceCtx,
    module: Module,
    d_in: &DeviceBuffer<u8>,
    d_tw: &DeviceBuffer<u8>,
) -> GraphExe {
    let gm = GraphModule::from_ir(module, &[]).unwrap();
    let mut exe = GraphCompiler::new()
        .compile(gm.into_builder())
        .expect("JIT compile");
    exe.set_input(ctx, 0, d_in).unwrap();
    exe.set_input(ctx, 1, d_tw).unwrap();
    exe
}

fn main() {
    let log_n = 24usize;
    let n = 1usize << log_n;
    let ctx = GpuDeviceCtx::for_current_device().expect("CUDA context");
    let input_mont: Vec<u32> = pseudo_field_elems(n, 1)
        .iter()
        .map(|&x| to_monty(x))
        .collect();
    let twiddles_mont: Vec<u32> = ntt_twiddles(log_n).iter().map(|&x| to_monty(x)).collect();
    let d_in = to_device_bytes(&ctx, &input_mont);
    let d_tw = to_device_bytes(&ctx, &twiddles_mont);

    let mut exe_flat = setup(&ctx, ntt_module(log_n), &d_in, &d_tw);
    let mut exe_sh = setup(&ctx, ntt_shared_module(log_n), &d_in, &d_tw);
    let mut exe_reg = setup(&ctx, ntt_reg_module(log_n), &d_in, &d_tw);

    // Warmup outside the profiled region.
    exe_flat.run(&ctx).expect("flat warmup");
    exe_sh.run(&ctx).expect("shared warmup");
    exe_reg.run(&ctx).expect("reg warmup");
    ctx.stream.synchronize().expect("warmup sync");

    unsafe { cudaProfilerStart() };

    nvtx::range_push!("ntt_flat_2^24");
    for _ in 0..ITERS {
        exe_flat.run(&ctx).expect("flat run");
    }
    ctx.stream.synchronize().expect("flat sync");
    nvtx::range_pop!();

    nvtx::range_push!("ntt_shared_2^24");
    for _ in 0..ITERS {
        exe_sh.run(&ctx).expect("shared run");
    }
    ctx.stream.synchronize().expect("shared sync");
    nvtx::range_pop!();

    nvtx::range_push!("ntt_reg_2^24");
    for _ in 0..ITERS {
        exe_reg.run(&ctx).expect("reg run");
    }
    ctx.stream.synchronize().expect("reg sync");
    nvtx::range_pop!();

    unsafe { cudaProfilerStop() };
    println!("done: {ITERS} iters each of flat, shared, reg NTT at n=2^{log_n}");
}
