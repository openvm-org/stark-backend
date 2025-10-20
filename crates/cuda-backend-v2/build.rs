use std::process::exit;

use openvm_cuda_builder::{CudaBuilder, cuda_available};

fn main() {
    if !cuda_available() {
        eprintln!("cargo:warning=CUDA is not available");
        exit(1);
    }

    let common = CudaBuilder::new().include_from_dep("DEP_CUDA_COMMON_INCLUDE");

    common.emit_link_directives();

    // common
    //     .clone()
    //     .library_name("stark_backend_gpu")
    //     .include("cuda/include")
    //     .files_from_glob("cuda/src/*.cu")
    //     .build();
}
