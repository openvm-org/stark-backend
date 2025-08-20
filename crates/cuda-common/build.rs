use std::{env, path::PathBuf, process::exit};

use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    if cuda_available() {
        CudaBuilder::new().emit_link_directives();

        let include_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("include");
        println!("cargo:include={}", include_path.display()); // -> DEP_CUDA_COMMON_INCLUDE
    } else {
        eprintln!("cargo:warning=CUDA is not available");
        exit(1);
    }
}
