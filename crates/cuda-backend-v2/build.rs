use std::process::exit;

use openvm_cuda_builder::{CudaBuilder, cuda_available};

fn main() {
    if !cuda_available() {
        eprintln!("cargo:warning=CUDA is not available");
        exit(1);
    }

    let common = CudaBuilder::new()
        .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
        .include("cuda/include");

    common.emit_link_directives();

    common
        .clone()
        .library_name("cuda-backend-v2")
        .files_from_glob("cuda/src/*.cu")
        .build();
}
