use std::{env, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_OPT_LEVEL");

    // Get CUDA_ARCH from environment or detect it
    let cuda_arch = match env::var("CUDA_ARCH") {
        Ok(arch) => arch,
        Err(_) => {
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
        }
    };
    println!("cargo:rerun-if-changed=build.rs");

    // Get CUDA_OPT_LEVEL from environment or use default value
    // 0 → No optimization (fast compile, debug-friendly)
    // 1 → Minimal optimization
    // 2 → Balanced optimization (often same as -O3 for some kernels)
    // 3 → Maximum optimization (usually default for release builds)
    let cuda_opt_level = env::var("CUDA_OPT_LEVEL").unwrap_or("3".to_string());
    let sppark_root = env::var("DEP_SPPARK_ROOT").expect("sppark dependency not found");

    let mut builder = cc::Build::new();
    builder
        .cuda(true)
        // Include paths
        .include("cuda/include")
        // CUDA specific flags
        .flag("--std=c++17")
        .flag("--expt-relaxed-constexpr")
        .flag("--device-link")
        // Compute capability for T4
        .flag("-gencode")
        .flag(format!("arch=compute_{},code=sm_{}", cuda_arch, cuda_arch));

    if cuda_opt_level == "0" {
        builder.debug(true);
        builder.flag("-O0");
    } else {
        builder.debug(false);
        builder.flag(format!("--ptxas-options=-O{}", cuda_opt_level));
    }

    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        builder.include(format!("{}/include", cuda_path));
    }

    builder
        .file("cuda/src/matrix.cu")
        .file("cuda/src/lde.cu")
        .file("cuda/src/poseidon2.cu")
        .file("cuda/src/eltwise.cu")
        .file("cuda/src/quotient.cu")
        .file("cuda/src/permute.cu")
        .file("cuda/src/prefix.cu")
        .file("cuda/src/fri.cu")
        .compile("stark_backend_gpu");

    println!(
        "SPPARK_ROOT = {}",
        env::var("DEP_SPPARK_ROOT").unwrap_or("NOT FOUND".to_string())
    );

    let mut ntt_builder = cc::Build::new();
    ntt_builder
        .cuda(true)
        .cpp(true)
        .define("FEATURE_BABY_BEAR", None)
        .file(format!("{}/util/all_gpus.cpp", sppark_root))
        .file("cuda/src/supra_ntt_api.cu")
        .include(&sppark_root)
        .include(format!("{}/util", sppark_root))
        .include(format!("{}/ntt", sppark_root))
        .flag("--std=c++17")
        .flag(format!(
            "-gencode=arch=compute_{},code=sm_{}",
            cuda_arch, cuda_arch
        ));

    if cuda_opt_level == "0" {
        ntt_builder.debug(true);
        ntt_builder.flag("-O0");
    } else {
        ntt_builder.debug(false);
        ntt_builder.flag(format!("--ptxas-options=-O{}", cuda_opt_level));
    }
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        ntt_builder.include(format!("{}/include", cuda_path));
    }

    ntt_builder.compile("supra_ntt");

    // Make sure CUDA and our utilities are linked
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
}
