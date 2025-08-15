use std::{env, path::PathBuf};

use openvm_cuda_builder::{cuda_available, emit_cuda_cfg_if_available};

fn main() {
    emit_cuda_cfg_if_available();

    if cuda_available() {
        let include_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("include");
        println!("cargo:include={}", include_path.display()); // -> DEP_CUDA_COMMON_INCLUDE
        println!("cargo:rerun-if-changed={}", include_path.display());
    }
}
