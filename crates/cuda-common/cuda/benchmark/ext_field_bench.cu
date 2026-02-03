/**
 * Extension Field Benchmark Kernels
 * 
 * Provides templated kernels for benchmarking field arithmetic operations.
 * Supports: Fp, FpExt (Fp4), Fp5, Fp6, Fp2x3 (2×3 tower), Fp3x2 (3×2 tower)
 *           Kb (KoalaBear base)
 * 
 * Operations:
 * - init: Initialize field elements from raw u32 arrays
 * - add: Element-wise addition
 * - mul: Element-wise multiplication  
 * - inv: Element-wise inversion
 */

#include "fp.h"
#include "fpext.h"
#include "fp5.h"
#include "fp6.h"
#include "fp2x3.h"
#include "fp3x2.h"
#include "kb.h"

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
/// For Fp5: 5 u32s per element
/// For Fp6: 6 u32s per element
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
    } else if constexpr (ELEMS_PER_FIELD == 5) {
        // Fp5: 5 u32s
        size_t base = idx * 5;
        Fp5& elem = reinterpret_cast<Fp5*>(out)[idx];
        elem = Fp5(Fp(raw_data[base]), Fp(raw_data[base+1]), 
                   Fp(raw_data[base+2]), Fp(raw_data[base+3]), Fp(raw_data[base+4]));
    } else if constexpr (ELEMS_PER_FIELD == 6) {
        // Fp6: 6 u32s
        size_t base = idx * 6;
        Fp6& elem = reinterpret_cast<Fp6*>(out)[idx];
        elem = Fp6(Fp(raw_data[base]), Fp(raw_data[base+1]), Fp(raw_data[base+2]),
                   Fp(raw_data[base+3]), Fp(raw_data[base+4]), Fp(raw_data[base+5]));
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

extern "C" int init_fp(void* out, const uint32_t* raw_data, size_t n) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_init_kernel<Fp, 1><<<grid_size, block>>>(static_cast<Fp*>(out), raw_data, n);
    return cudaGetLastError();
}

extern "C" int add_fp(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_add_kernel<Fp><<<grid_size, block>>>(
        static_cast<Fp*>(out), static_cast<const Fp*>(a), static_cast<const Fp*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int mul_fp(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_mul_kernel<Fp><<<grid_size, block>>>(
        static_cast<Fp*>(out), static_cast<const Fp*>(a), static_cast<const Fp*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int inv_fp(void* out, const void* a, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_inv_kernel<Fp><<<grid_size, block>>>(static_cast<Fp*>(out), static_cast<const Fp*>(a), n, reps);
    return cudaGetLastError();
}

// ============================================================================
// Extern "C" Wrappers for FpExt (quartic extension)
// ============================================================================

extern "C" int init_fpext(void* out, const uint32_t* raw_data, size_t n) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_init_kernel<FpExt, 4><<<grid_size, block>>>(static_cast<FpExt*>(out), raw_data, n);
    return cudaGetLastError();
}

extern "C" int add_fpext(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_add_kernel<FpExt><<<grid_size, block>>>(
        static_cast<FpExt*>(out), static_cast<const FpExt*>(a), static_cast<const FpExt*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int mul_fpext(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_mul_kernel<FpExt><<<grid_size, block>>>(
        static_cast<FpExt*>(out), static_cast<const FpExt*>(a), static_cast<const FpExt*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int inv_fpext(void* out, const void* a, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_inv_kernel<FpExt><<<grid_size, block>>>(static_cast<FpExt*>(out), static_cast<const FpExt*>(a), n, reps);
    return cudaGetLastError();
}

// ============================================================================
// Extern "C" Wrappers for Fp5 (quintic extension)
// ============================================================================

extern "C" int init_fp5(void* out, const uint32_t* raw_data, size_t n) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_init_kernel<Fp5, 5><<<grid_size, block>>>(static_cast<Fp5*>(out), raw_data, n);
    return cudaGetLastError();
}

extern "C" int add_fp5(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_add_kernel<Fp5><<<grid_size, block>>>(
        static_cast<Fp5*>(out), static_cast<const Fp5*>(a), static_cast<const Fp5*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int mul_fp5(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_mul_kernel<Fp5><<<grid_size, block>>>(
        static_cast<Fp5*>(out), static_cast<const Fp5*>(a), static_cast<const Fp5*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int inv_fp5(void* out, const void* a, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_inv_kernel<Fp5><<<grid_size, block>>>(static_cast<Fp5*>(out), static_cast<const Fp5*>(a), n, reps);
    return cudaGetLastError();
}

// ============================================================================
// Extern "C" Wrappers for Fp6 (sextic extension)
// ============================================================================

extern "C" int init_fp6(void* out, const uint32_t* raw_data, size_t n) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_init_kernel<Fp6, 6><<<grid_size, block>>>(static_cast<Fp6*>(out), raw_data, n);
    return cudaGetLastError();
}

extern "C" int add_fp6(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_add_kernel<Fp6><<<grid_size, block>>>(
        static_cast<Fp6*>(out), static_cast<const Fp6*>(a), static_cast<const Fp6*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int mul_fp6(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_mul_kernel<Fp6><<<grid_size, block>>>(
        static_cast<Fp6*>(out), static_cast<const Fp6*>(a), static_cast<const Fp6*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int inv_fp6(void* out, const void* a, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_inv_kernel<Fp6><<<grid_size, block>>>(static_cast<Fp6*>(out), static_cast<const Fp6*>(a), n, reps);
    return cudaGetLastError();
}

// ============================================================================
// Extern "C" Wrappers for Fp2x3 (2×3 tower: Fp → Fp2 → Fp6)
// ============================================================================

// Generic init for 6-element tower types
template<typename T>
__global__ void bench_init_tower6_kernel(T* out, const uint32_t* raw_data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    size_t base = idx * 6;
    out[idx] = T(Fp(raw_data[base]), Fp(raw_data[base+1]), Fp(raw_data[base+2]),
                 Fp(raw_data[base+3]), Fp(raw_data[base+4]), Fp(raw_data[base+5]));
}

extern "C" int init_fp2x3(void* out, const uint32_t* raw_data, size_t n) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_init_tower6_kernel<Fp2x3><<<grid_size, block>>>(static_cast<Fp2x3*>(out), raw_data, n);
    return cudaGetLastError();
}

extern "C" int add_fp2x3(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_add_kernel<Fp2x3><<<grid_size, block>>>(
        static_cast<Fp2x3*>(out), static_cast<const Fp2x3*>(a), static_cast<const Fp2x3*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int mul_fp2x3(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_mul_kernel<Fp2x3><<<grid_size, block>>>(
        static_cast<Fp2x3*>(out), static_cast<const Fp2x3*>(a), static_cast<const Fp2x3*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int inv_fp2x3(void* out, const void* a, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_inv_kernel<Fp2x3><<<grid_size, block>>>(static_cast<Fp2x3*>(out), static_cast<const Fp2x3*>(a), n, reps);
    return cudaGetLastError();
}

// ============================================================================
// Extern "C" Wrappers for Fp3x2 (3×2 tower: Fp → Fp3 → Fp6)
// ============================================================================

extern "C" int init_fp3x2(void* out, const uint32_t* raw_data, size_t n) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_init_tower6_kernel<Fp3x2><<<grid_size, block>>>(static_cast<Fp3x2*>(out), raw_data, n);
    return cudaGetLastError();
}

extern "C" int add_fp3x2(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_add_kernel<Fp3x2><<<grid_size, block>>>(
        static_cast<Fp3x2*>(out), static_cast<const Fp3x2*>(a), static_cast<const Fp3x2*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int mul_fp3x2(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_mul_kernel<Fp3x2><<<grid_size, block>>>(
        static_cast<Fp3x2*>(out), static_cast<const Fp3x2*>(a), static_cast<const Fp3x2*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int inv_fp3x2(void* out, const void* a, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_inv_kernel<Fp3x2><<<grid_size, block>>>(static_cast<Fp3x2*>(out), static_cast<const Fp3x2*>(a), n, reps);
    return cudaGetLastError();
}

// ============================================================================
// Extern "C" Wrappers for Kb (KoalaBear base field)
// ============================================================================

/// Initialize Kb elements from raw u32 data
__global__ void bench_init_kb_kernel(Kb* out, const uint32_t* raw_data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = Kb(raw_data[idx]);
}

extern "C" int init_kb(void* out, const uint32_t* raw_data, size_t n) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_init_kb_kernel<<<grid_size, block>>>(static_cast<Kb*>(out), raw_data, n);
    return cudaGetLastError();
}

extern "C" int add_kb(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_add_kernel<Kb><<<grid_size, block>>>(
        static_cast<Kb*>(out), static_cast<const Kb*>(a), static_cast<const Kb*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int mul_kb(void* out, const void* a, const void* b, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_mul_kernel<Kb><<<grid_size, block>>>(
        static_cast<Kb*>(out), static_cast<const Kb*>(a), static_cast<const Kb*>(b), n, reps);
    return cudaGetLastError();
}

extern "C" int inv_kb(void* out, const void* a, size_t n, int reps) {
    int grid_size;
    dim3 block = get_launch_config(n, grid_size);
    bench_inv_kernel<Kb><<<grid_size, block>>>(static_cast<Kb*>(out), static_cast<const Kb*>(a), n, reps);
    return cudaGetLastError();
}
