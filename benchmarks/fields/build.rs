use std::process::exit;

use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    if !cuda_available() {
        eprintln!("cargo:warning=CUDA is not available");
        exit(1);
    }
    
    let builder = CudaBuilder::new()
        .library_name("openvm_benchmarks_fields")
        .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
        .include("cuda/include")
        .files_from_glob("cuda/src/*.cu");

    builder.emit_link_directives();
    builder.build();
}
