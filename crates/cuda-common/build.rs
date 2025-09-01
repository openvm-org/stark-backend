use std::{env, path::PathBuf, process::exit};

use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    if cuda_available() {
        println!("cargo:rerun-if-changed=cuda");

        let builder = CudaBuilder::new()
            .library_name("vmm_shim")
            .flag("-Xcompiler=-fPIC")
            .file("cuda/src/vmm_shim.cu");

        builder.emit_link_directives();
        builder.build();

        let include_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("include");
        println!("cargo:include={}", include_path.display()); // -> DEP_CUDA_COMMON_INCLUDE
    } else {
        eprintln!("cargo:warning=CUDA is not available");
        exit(1);
    }
}
