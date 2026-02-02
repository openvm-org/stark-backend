use std::{env, path::PathBuf, process::exit};

use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    if cuda_available() {
        println!("cargo:rerun-if-changed=cuda");
        println!("cargo:rerun-if-changed=include");

        let include_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("include");

        // Build vmm_shim library
        let builder = CudaBuilder::new()
            .library_name("vmm_shim")
            .flag("-Xcompiler=-fPIC")
            .file("cuda/src/vpmm_shim.cu");

        builder.clone().build();
        builder.emit_link_directives();

        // Build extension field benchmark library
        CudaBuilder::new()
            .library_name("ext_field_bench")
            .include(&include_path)
            .flag("-Xcompiler=-fPIC")
            .file("cuda/benchmark/ext_field_bench.cu")
            .build();

        println!("cargo:include={}", include_path.display()); // -> DEP_CUDA_COMMON_INCLUDE
    } else {
        eprintln!("cargo:warning=CUDA is not available");
        exit(1);
    }
}
