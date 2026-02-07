use openvm_benchmarks_fields::{run_all_benchmarks, BenchConfig};

fn main() {
    // Initialize CUDA context
    openvm_cuda_common::common::set_device().expect("Failed to initialize CUDA device");

    run_all_benchmarks(&BenchConfig::default());
}
