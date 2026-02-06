/**
 * Poseidon2 Benchmark Kernels
 *
 * Benchmarks for poseidon2_mix permutation on GPU.
 * Each thread operates on one 16-element state.
 */

#include "poseidon2.cuh"
#include "koala_bear/poseidon2_kb.cuh"

// ============================================================================
// Launch Configuration
// ============================================================================

constexpr int P2_BLOCK_SIZE = 512;

inline dim3 get_p2_launch_config(size_t n, int& grid_size) {
    grid_size = (n + P2_BLOCK_SIZE - 1) / P2_BLOCK_SIZE;
    return dim3(P2_BLOCK_SIZE);
}

// ============================================================================
// Templated Kernels
// ============================================================================

template<typename T>
__global__ void poseidon2_init_kernel(T* out, const uint32_t* raw_data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    size_t base = idx * 16;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        out[base + i] = T(raw_data[base + i]);
    }
}

template<typename T, void (*poseidon2_mix_fn)(T*)>
__global__ void poseidon2_bench_kernel(T* states, size_t n, int reps) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    T cells[16];
    size_t base = idx * 16;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        cells[i] = states[base + i];
    }

    #pragma unroll 1
    for (int r = 0; r < reps; r++) {
        poseidon2_mix_fn(cells);
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        states[base + i] = cells[i];
    }
}

// ============================================================================
// Extern "C" Wrappers
// ============================================================================

extern "C" int init_poseidon2_bb(void* out, const uint32_t* raw_data, size_t n) {
    int grid_size;
    dim3 block = get_p2_launch_config(n, grid_size);
    poseidon2_init_kernel<Fp><<<grid_size, block>>>(static_cast<Fp*>(out), raw_data, n);
    return cudaGetLastError();
}

extern "C" int run_poseidon2_bb(void* states, size_t n, int reps) {
    int grid_size;
    dim3 block = get_p2_launch_config(n, grid_size);
    poseidon2_bench_kernel<Fp, poseidon2::poseidon2_mix><<<grid_size, block>>>(static_cast<Fp*>(states), n, reps);
    return cudaGetLastError();
}

extern "C" int init_poseidon2_kb(void* out, const uint32_t* raw_data, size_t n) {
    int grid_size;
    dim3 block = get_p2_launch_config(n, grid_size);
    poseidon2_init_kernel<Kb><<<grid_size, block>>>(static_cast<Kb*>(out), raw_data, n);
    return cudaGetLastError();
}

extern "C" int run_poseidon2_kb(void* states, size_t n, int reps) {
    int grid_size;
    dim3 block = get_p2_launch_config(n, grid_size);
    poseidon2_bench_kernel<Kb, kb_poseidon2::poseidon2_mix><<<grid_size, block>>>(static_cast<Kb*>(states), n, reps);
    return cudaGetLastError();
}
