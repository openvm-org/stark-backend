use openvm_cuda_builder::{cuda_available, emit_cuda_cfg_if_available, CudaBuilder};

fn main() {
    emit_cuda_cfg_if_available();
    if !cuda_available() {
        eprintln!("cargo:warning=CUDA is not available");
        return;
    }

    let common = CudaBuilder::new()
        .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
        .watch("cuda")
        .watch("src/cuda")
        .flag("--device-link");

    common
        .clone()
        .library_name("stark_backend_gpu")
        .files([
            "cuda/src/matrix.cu",
            "cuda/src/lde.cu",
            "cuda/src/poseidon2.cu",
            "cuda/src/quotient.cu",
            "cuda/src/permute.cu",
            "cuda/src/prefix.cu",
            "cuda/src/fri.cu",
        ])
        .build();

    common
        .clone()
        .library_name("supra_ntt")
        .include("cuda/include/supra")
        .file("cuda/src/supra_ntt_api.cu")
        .build();

    common.emit_link_directives();
}
