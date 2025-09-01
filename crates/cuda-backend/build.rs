use std::process::exit;

use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    if !cuda_available() {
        eprintln!("cargo:warning=CUDA is not available");
        exit(1);
    }

    let common = CudaBuilder::new().include_from_dep("DEP_CUDA_COMMON_INCLUDE");

    common
        .clone()
        .library_name("stark_backend_gpu")
        .include("cuda/include")
        .files_from_glob("cuda/src/*.cu")
        .build();

    common
        .clone()
        .library_name("supra_ntt")
        .flag("--device-link")
        .include("cuda/supra/include")
        .file("cuda/supra/ntt_api.cu")
        .build();

    common.emit_link_directives();
}
