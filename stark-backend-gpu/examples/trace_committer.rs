use std::sync::Arc;

use clap::Parser;
use openvm_stark_backend::prover::{
    cpu::CpuDevice,
    hal::{DeviceDataTransporter, TraceCommitter},
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{config_from_perm, default_perm, BabyBearPoseidon2Config},
        fri_params::SecurityParameters,
        FriParameters,
    },
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::Field;
use p3_matrix::dense::DenseMatrix;
use rand::Rng;
use stark_backend_gpu::gpu_device::{GpuConfig, GpuDevice};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "2")]
    log_blowup: usize,

    #[arg(long, default_value = "16")]
    log_height: u32,

    #[arg(long, default_value = "3")]
    width: u32,

    #[arg(long, default_value = "2")]
    matrix_num: u32,
}

fn main() {
    let args = Args::parse();

    let log_blowup = args.log_blowup;
    let height = 1 << args.log_height;
    let width = args.width;
    let matrix_num = args.matrix_num;
    let elements = height * width;
    let shift = BabyBear::GENERATOR;

    println!("log_blowup: {}", log_blowup);
    println!("height: {} (log = {})", height, args.log_height);
    println!("width: {}", width);
    println!("matrix_num: {}", matrix_num);
    println!(
        "input memory size: {} MB",
        elements * matrix_num * 4 * (1 << log_blowup) / 1024 / 1024
    );

    let mut rng = create_seeded_rng();
    let trace = (0..elements)
        .map(|_| BabyBear::new(rng.gen()))
        .collect::<Vec<_>>();
    let matrix = Arc::new(DenseMatrix::new(trace.clone(), width.try_into().unwrap()));
    let trace_vec = vec![matrix.clone(); matrix_num.try_into().unwrap()];

    // --- CPU commit ---
    #[allow(clippy::arc_with_non_send_sync)]
    let config = Arc::new(config_from_perm(
        &default_perm(),
        SecurityParameters::standard_100_bits_with_fri_log_blowup(log_blowup),
    ));
    let cpu_device = CpuDevice::<BabyBearPoseidon2Config>::new(config, log_blowup);
    let cpu_time = std::time::Instant::now();
    let (root, _pcs_data) = cpu_device.commit(&trace_vec);
    let cpu_time = cpu_time.elapsed();
    println!("CPU root: {:?}", root.as_ref());

    // --- GPU commit ---
    let gpu_time = std::time::Instant::now();
    let gpu_device = GpuDevice::new(
        GpuConfig::new(
            FriParameters::standard_with_100_bits_conjectured_security(log_blowup),
            shift,
        ),
        None,
    );
    let gpu_trace_vec = trace_vec
        .iter()
        .map(|matrix| gpu_device.transport_matrix_to_device(matrix))
        .collect::<Vec<_>>();
    let (gpu_root, _gpu_pcs_data) = gpu_device.commit(&gpu_trace_vec);
    let gpu_time = gpu_time.elapsed();
    println!("GPU root: {:?}", gpu_root.as_ref());

    assert_eq!(root, gpu_root);
    println!("------------------------");
    println!(
        "CPU time = {:.2?}, GPU time = {:.2?}, Speedup = x{:?}",
        cpu_time,
        gpu_time,
        cpu_time.as_micros() as f64 / gpu_time.as_micros() as f64
    );
    println!("------------------------");
}
