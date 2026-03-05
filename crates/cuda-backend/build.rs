use std::process::exit;

use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    if !cuda_available() {
        eprintln!("cargo:warning=CUDA is not available");
        exit(1);
    }

    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_BABY_BEAR_BN254_POSEIDON2");

    let common = CudaBuilder::new()
        .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
        .include("cuda/include");

    common.emit_link_directives();

    let mut builder = common
        .clone()
        .library_name("cuda-backend")
        .watch("cuda")
        .include("cuda/include");

    // Collect .cu files, excluding bn254_poseidon2.cu unless the feature is enabled.
    let bn254_enabled = std::env::var("CARGO_FEATURE_BABY_BEAR_BN254_POSEIDON2").is_ok();
    for entry in glob::glob("cuda/src/**/*.cu").expect("failed to glob cuda/src/**/*.cu") {
        let path = entry.expect("glob error");
        if !bn254_enabled
            && path
                .file_name()
                .is_some_and(|f| f == "bn254_poseidon2.cu")
        {
            continue;
        }
        builder = builder.file(path);
    }

    builder.build();

    common
        .clone()
        .library_name("supra_ntt")
        .include("cuda/supra/include")
        .files_from_glob("cuda/supra/*.cu")
        .build();
}
