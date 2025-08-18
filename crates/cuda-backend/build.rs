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
        .watch("src/cuda");

    common
        .clone()
        .library_name("stark_backend_gpu")
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
