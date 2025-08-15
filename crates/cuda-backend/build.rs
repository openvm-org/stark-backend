use openvm_cuda_builder::CudaBuilder;

fn main() {
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
}
