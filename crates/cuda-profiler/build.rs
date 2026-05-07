use std::process::exit;

use openvm_cuda_builder::{cuda_available, CudaBuilder};

// Compiles the small CUDA host glue file that defines `shadow_cta_ctx()`.
// cuda-backend's instrumented kernels emit calls to this symbol when
// SHADOW_CTA_PROFILE is defined; both static libraries link into the same
// final Rust binary, so the symbol resolves there.
//
// We do not need SHADOW_CTA_PROFILE defined here: cta_probe.cuh's struct
// definitions are unconditional; only the BEGIN/END probe macros in
// launcher.cuh are gated, and shadow_ctx.cu doesn't use those macros.
fn main() {
    if !cuda_available() {
        eprintln!("cargo:warning=CUDA is not available");
        exit(1);
    }

    let builder = CudaBuilder::new()
        .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
        .library_name("openvm_cuda_profiler_glue")
        .flag("-Xcompiler=-fPIC")
        .file("cuda/src/shadow_ctx.cu")
        .file("cuda/src/test_kernel.cu");

    builder.clone().build();
    builder.emit_link_directives();

    println!("cargo:rerun-if-changed=cuda/src/shadow_ctx.cu");
    println!("cargo:rerun-if-changed=cuda/src/test_kernel.cu");
}
