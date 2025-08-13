use std::{env, process::Command};

// Detect optimal NVCC parallel jobs
fn nvcc_parallel_jobs() -> String {
    // Try to detect CPU count from std
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    // Allow override from NVCC_THREADS env var
    let threads = std::env::var("NVCC_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(threads);

    format!("-t{}", threads)
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-changed=src/cuda");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_OPT_LEVEL");
    println!("cargo:rerun-if-env-changed=CUDA_DEBUG");

    // Get CUDA_ARCH from environment or detect it
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| {
        // Run nvidia-smi command to get arch
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
            .expect("Failed to execute nvidia-smi");

        let full_output = String::from_utf8(output.stdout).unwrap();
        let arch = full_output
            .lines()
            .next()
            .expect("`nvidia-smi --query-gpu=compute_cap` failed to return any output")
            .trim()
            .replace('.', ""); // Convert "7.5" to "75"

        // Set environment variable for future builds
        println!("cargo:rustc-env=CUDA_ARCH={}", arch);
        arch
    });

    // CUDA_DEBUG shortcut
    if env::var("CUDA_DEBUG").map(|v| v == "1").unwrap_or(false) {
        env::set_var("CUDA_OPT_LEVEL", "0");
        env::set_var("CUDA_LAUNCH_BLOCKING", "1");
        env::set_var("CUDA_MEMCHECK", "1");
        env::set_var("RUST_BACKTRACE", "1");
        println!("cargo:warning=CUDA_DEBUG=1 → forcing CUDA_OPT_LEVEL=0, CUDA_LAUNCH_BLOCKING=1, CUDA_MEMCHECK=1, RUST_BACKTRACE=1");
    }

    // Get CUDA_OPT_LEVEL from environment or use default value
    // 0 → No optimization (fast compile, debug-friendly)
    // 1 → Minimal optimization
    // 2 → Balanced optimization (often same as -O3 for some kernels)
    // 3 → Maximum optimization (usually default for release builds)
    let cuda_opt_level = env::var("CUDA_OPT_LEVEL").unwrap_or("3".to_string());

    let mut common = cc::Build::new();
    common
        .cuda(true)
        // CUDA specific flags
        .flag("--std=c++17")
        .flag("--expt-relaxed-constexpr")
        .flag("--device-link")
        // Compute capability
        .flag("-gencode")
        .flag(format!("arch=compute_{},code=sm_{}", cuda_arch, cuda_arch))
        .flag(nvcc_parallel_jobs());

    if cuda_opt_level == "0" {
        common.debug(true);
        common.flag("-O0");
    } else {
        common.debug(false);
        common.flag(format!("--ptxas-options=-O{}", cuda_opt_level));
    }

    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        common.include(format!("{}/include", cuda_path));
    }

    let mut builder = common.clone();
    builder
        .include("cuda/include")
        .file("cuda/src/matrix.cu")
        .file("cuda/src/lde.cu")
        .file("cuda/src/poseidon2.cu")
        .file("cuda/src/eltwise.cu")
        .file("cuda/src/quotient.cu")
        .file("cuda/src/permute.cu")
        .file("cuda/src/prefix.cu")
        .file("cuda/src/fri.cu")
        .compile("stark_backend_gpu");

    let mut ntt_builder = common.clone();
    ntt_builder
        .cpp(true)
        .include("cuda/include")
        .define("FEATURE_BABY_BEAR", None)
        .include("cuda/supra")
        .file("cuda/src/supra_ntt_api.cu")
        .compile("supra_ntt");

    // Make sure CUDA and our utilities are linked
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
}
