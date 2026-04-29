use std::{env, process};

use openvm_cuda_backend::{
    logup_zerocheck::{fractional_sumcheck_gpu, make_synthetic_leaves},
    prelude::EF,
    sponge::DuplexSpongeGpu,
};
use openvm_cuda_common::{
    common::get_device,
    copy::MemCopyD2D,
    stream::{CudaStream, GpuDeviceCtx, StreamGuard},
};
use p3_field::PrimeCharacteristicRing;

fn parse_usize(var: &str, default: usize) -> usize {
    env::var(var)
        .ok()
        .and_then(|x| x.parse::<usize>().ok())
        .unwrap_or(default)
}

/// All benchmarks output the same CSV format: `n,run_idx,is_warmup,elapsed_ms`
fn bench_fractional_sumcheck() -> Result<(), Box<dyn std::error::Error>> {
    let n = parse_usize("SWIRL_BENCH_N", 24);
    let repeats = parse_usize("SWIRL_BENCH_REPEATS", 3);
    let warmups = parse_usize("SWIRL_BENCH_WARMUPS", 1);

    let device_ctx = GpuDeviceCtx {
        device_id: get_device()? as u32,
        stream: StreamGuard::new(CudaStream::new_non_blocking()?),
    };
    let template = make_synthetic_leaves(n, &device_ctx)?;

    println!("run_idx,is_warmup,elapsed_ms");

    for run_idx in 0..(warmups + repeats) {
        let is_warmup = run_idx < warmups;
        let leaves = template
            .device_copy_on(&device_ctx)
            .expect("device copy leaves");

        let mut transcript = DuplexSpongeGpu::default();
        let mut mem = openvm_cuda_common::memory_manager::MemTracker::start("bench.fractional");

        device_ctx.stream.synchronize().expect("sync before timing");
        let t0 = std::time::Instant::now();
        let _ = fractional_sumcheck_gpu(
            &mut transcript,
            leaves,
            1usize << n,
            1usize << n,
            EF::ZERO,
            false,
            &mut mem,
            &device_ctx,
        )?;
        device_ctx.stream.synchronize().expect("sync after timing");
        let ms = t0.elapsed().as_secs_f64() * 1000.0;

        println!("{run_idx},{},{:.4}", is_warmup as u8, ms);
    }

    Ok(())
}

#[allow(clippy::type_complexity)]
const BENCHMARKS: &[(&str, fn() -> Result<(), Box<dyn std::error::Error>>)] =
    &[("fractional-sumcheck", bench_fractional_sumcheck)];

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench <benchmark>");
        eprintln!();
        eprintln!("Available benchmarks:");
        for (name, _) in BENCHMARKS {
            eprintln!("  {name}");
        }
        process::exit(1);
    }

    let name = &args[1];
    for (bench_name, bench_fn) in BENCHMARKS {
        if name == bench_name {
            if let Err(e) = bench_fn() {
                eprintln!("Error: {e}");
                process::exit(1);
            }
            return;
        }
    }

    eprintln!("Unknown benchmark: {name}");
    eprintln!();
    eprintln!("Available benchmarks:");
    for (name, _) in BENCHMARKS {
        eprintln!("  {name}");
    }
    process::exit(1);
}
