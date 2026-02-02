/**
 * Extension Field Benchmark Kernels
 * 
 * Provides templated kernels for benchmarking field arithmetic operations.
 * Supports Fp (base field) and FpExt (quartic extension).
 * 
 * Operations:
 * - init: Initialize field elements from raw u32 arrays
 * - add: Element-wise addition
 * - mul: Element-wise multiplication  
 * - inv: Element-wise inversion
 */

#include "fp.h"
#include "fpext.h"

// ============================================================================
// Launch Configuration
// ============================================================================

/// Block size for all benchmark kernels (threads per block)
constexpr int BENCH_BLOCK_SIZE = 512;

/// Calculate grid and block dimensions for a 1D kernel launch
inline dim3 get_launch_config(size_t n, int& grid_size) {
    grid_size = (n + BENCH_BLOCK_SIZE - 1) / BENCH_BLOCK_SIZE;
    return dim3(BENCH_BLOCK_SIZE);
}

// ============================================================================
// Templated Kernels
// ============================================================================

/// Initialize field elements from raw u32 data
/// For Fp: 1 u32 per element
/// For FpExt: 4 u32s per element
template<typename T, int ELEMS_PER_FIELD>
__global__ void bench_init_kernel(T* out, const uint32_t* raw_data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if constexpr (ELEMS_PER_FIELD == 1) {
        // Fp: single u32
        out[idx] = Fp(raw_data[idx]);
    } else if constexpr (ELEMS_PER_FIELD == 4) {
        // FpExt: 4 u32s
        size_t base = idx * 4;
        FpExt& elem = reinterpret_cast<FpExt*>(out)[idx];
        elem = FpExt(Fp(raw_data[base]), Fp(raw_data[base+1]), 
                     Fp(raw_data[base+2]), Fp(raw_data[base+3]));
    }
}

/// Element-wise addition with repetition to amortize memory access
/// Does `reps` additions per element to stress ALU
template<typename T>
__global__ void bench_add_kernel(T* out, const T* a, const T* b, size_t n, int reps) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    T acc = a[idx];
    T val_b = b[idx];
    #pragma unroll 1
    for (int i = 0; i < reps; i++) {
        acc = acc + val_b;
    }
    out[idx] = acc;
}

/// Element-wise multiplication with repetition to amortize memory access
template<typename T>
__global__ void bench_mul_kernel(T* out, const T* a, const T* b, size_t n, int reps) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    T acc = a[idx];
    T val_b = b[idx];
    #pragma unroll 1
    for (int i = 0; i < reps; i++) {
        acc = acc * val_b;
    }
    out[idx] = acc;
}

/// Element-wise inversion with repetition
/// Note: inv(inv(x)) = x, so we alternate to prevent trivial optimization
template<typename T>
__global__ void bench_inv_kernel(T* out, const T* a, size_t n, int reps) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    T acc = a[idx];
    #pragma unroll 1
    for (int i = 0; i < reps; i++) {
        acc = inv(acc);
    }
    out[idx] = acc;
}

// ============================================================================
// Extern "C" Wrappers for Fp (base field)
// ============================================================================

extern "C" int launch_bench_init_fp(void* out, const uint32_t* raw_data, size_t n) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_init_kernel<Fp, 1><<<grid_size, block>>>(static_cast<Fp*>(out), raw_data, n);
    return cudaGetLastError();
}

extern "C" int launch_bench_add_fp(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_add_kernel<Fp><<<grid_size, block>>>(
        static_cast<Fp*>(out), static_cast<const Fp*>(a), static_cast<const Fp*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int launch_bench_mul_fp(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_mul_kernel<Fp><<<grid_size, block>>>(
        static_cast<Fp*>(out), static_cast<const Fp*>(a), static_cast<const Fp*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int launch_bench_inv_fp(void* out, const void* a, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_inv_kernel<Fp><<<grid_size, block>>>(static_cast<Fp*>(out), static_cast<const Fp*>(a), n, reps);
    return cudaGetLastError();
}

// ============================================================================
// Extern "C" Wrappers for FpExt (quartic extension)
// ============================================================================

extern "C" int launch_bench_init_fpext(void* out, const uint32_t* raw_data, size_t n) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_init_kernel<FpExt, 4><<<grid_size, block>>>(static_cast<FpExt*>(out), raw_data, n);
    return cudaGetLastError();
}

extern "C" int launch_bench_add_fpext(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_add_kernel<FpExt><<<grid_size, block>>>(
        static_cast<FpExt*>(out), static_cast<const FpExt*>(a), static_cast<const FpExt*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int launch_bench_mul_fpext(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_mul_kernel<FpExt><<<grid_size, block>>>(
        static_cast<FpExt*>(out), static_cast<const FpExt*>(a), static_cast<const FpExt*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int launch_bench_inv_fpext(void* out, const void* a, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_inv_kernel<FpExt><<<grid_size, block>>>(static_cast<FpExt*>(out), static_cast<const FpExt*>(a), n, reps);
    return cudaGetLastError();
}
