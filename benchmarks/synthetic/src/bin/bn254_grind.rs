use std::time::Instant;

use clap::Parser;
use openvm_cuda_backend::{
    bn254_sponge::MultiFieldTranscriptGpu, sponge::GpuFiatShamirTranscript, GpuDevice,
};
use openvm_cuda_common::{common::get_device, stream::GpuDeviceCtx};

#[derive(Parser)]
#[command(about = "Sampled synthetic benchmark runner (cuda-backend / GPU)")]
struct Args {
    /// number of bits in PoW to zero
    #[arg(long, default_value_t = 18)]
    bits: usize,
}

fn main() {
    let Args { bits } = Args::parse();
    let mut ctx = GpuDeviceCtx::for_device(get_device().unwrap() as u32).unwrap();
    let mut dev = MultiFieldTranscriptGpu::default();
    dev.sync_h2d(&ctx).unwrap();
    let t0 = Instant::now();
    dev.grind_gpu(bits, &ctx).unwrap();

    println!(
        "bn254_grind bits {bits}, elapsed {} ms",
        t0.elapsed().as_millis()
    );
}
