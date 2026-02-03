/**
 * Verification Kernels for Extension Fields
 * 
 * Provides correctness tests for field arithmetic:
 * - Inversion test: a * inv(a) = 1
 * - Distributivity test: (a + b) * c = a*c + b*c
 * 
 * Supported fields:
 * - Baby Bear: Fp5, Fp6, Fp2x3, Fp3x2
 * - KoalaBear: Kb (base), Kb5 (quintic extension)
 * 
 * Note: Uses init functions from ext_field_bench.cu for initialization.
 */

#include "fp.h"
#include "fpext.h"
#include "fp5.h"
#include "fp6.h"
#include "fp2x3.h"
#include "fp3x2.h"
#include "kb.h"
#include "kb5.h"

// ============================================================================
// Launch Configuration
// ============================================================================

constexpr int VERIFY_BLOCK_SIZE = 256;

inline dim3 get_verify_config(size_t n, int& grid_size) {
    grid_size = (n + VERIFY_BLOCK_SIZE - 1) / VERIFY_BLOCK_SIZE;
    return dim3(VERIFY_BLOCK_SIZE);
}

// ============================================================================
// Templated Verification Kernels
// ============================================================================

/// Test: a * inv(a) = 1 for all non-zero elements
template<typename T>
__global__ void verify_inv_kernel(uint32_t* failures, const T* a, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    T val = a[idx];
    if (val == T::zero()) return;
    
    T val_inv = inv(val);
    T product = val * val_inv;
    
    if (product != T::one()) {
        atomicAdd(failures, 1);
    }
}

/// Test: (a + b) * c = a*c + b*c (distributivity)
template<typename T>
__global__ void verify_distrib_kernel(uint32_t* failures, const T* a, const T* b, const T* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    T av = a[idx];
    T bv = b[idx];
    T cv = c[idx];
    
    T lhs = (av + bv) * cv;
    T rhs = av * cv + bv * cv;
    
    if (lhs != rhs) {
        atomicAdd(failures, 1);
    }
}

// ============================================================================
// Fp5 Verification
// ============================================================================

extern "C" int verify_inv_fp5(uint32_t* failures, const void* a, size_t n) {
    int grid_size;
    dim3 block = get_verify_config(n, grid_size);
    verify_inv_kernel<Fp5><<<grid_size, block>>>(failures, static_cast<const Fp5*>(a), n);
    return cudaGetLastError();
}

extern "C" int verify_distrib_fp5(uint32_t* failures, const void* a, const void* b, const void* c, size_t n) {
    int grid_size;
    dim3 block = get_verify_config(n, grid_size);
    verify_distrib_kernel<Fp5><<<grid_size, block>>>(
        failures, static_cast<const Fp5*>(a), static_cast<const Fp5*>(b), static_cast<const Fp5*>(c), n);
    return cudaGetLastError();
}

// ============================================================================
// Fp6 Verification
// ============================================================================

extern "C" int verify_inv_fp6(uint32_t* failures, const void* a, size_t n) {
    int grid_size;
    dim3 block = get_verify_config(n, grid_size);
    verify_inv_kernel<Fp6><<<grid_size, block>>>(failures, static_cast<const Fp6*>(a), n);
    return cudaGetLastError();
}

extern "C" int verify_distrib_fp6(uint32_t* failures, const void* a, const void* b, const void* c, size_t n) {
    int grid_size;
    dim3 block = get_verify_config(n, grid_size);
    verify_distrib_kernel<Fp6><<<grid_size, block>>>(
        failures, static_cast<const Fp6*>(a), static_cast<const Fp6*>(b), static_cast<const Fp6*>(c), n);
    return cudaGetLastError();
}

// ============================================================================
// Fp2x3 Verification (2×3 tower)
// ============================================================================

extern "C" int verify_inv_fp2x3(uint32_t* failures, const void* a, size_t n) {
    int grid_size;
    dim3 block = get_verify_config(n, grid_size);
    verify_inv_kernel<Fp2x3><<<grid_size, block>>>(failures, static_cast<const Fp2x3*>(a), n);
    return cudaGetLastError();
}

extern "C" int verify_distrib_fp2x3(uint32_t* failures, const void* a, const void* b, const void* c, size_t n) {
    int grid_size;
    dim3 block = get_verify_config(n, grid_size);
    verify_distrib_kernel<Fp2x3><<<grid_size, block>>>(
        failures, static_cast<const Fp2x3*>(a), static_cast<const Fp2x3*>(b), static_cast<const Fp2x3*>(c), n);
    return cudaGetLastError();
}

// ============================================================================
// Fp3x2 Verification (3×2 tower)
// ============================================================================

extern "C" int verify_inv_fp3x2(uint32_t* failures, const void* a, size_t n) {
    int grid_size;
    dim3 block = get_verify_config(n, grid_size);
    verify_inv_kernel<Fp3x2><<<grid_size, block>>>(failures, static_cast<const Fp3x2*>(a), n);
    return cudaGetLastError();
}

extern "C" int verify_distrib_fp3x2(uint32_t* failures, const void* a, const void* b, const void* c, size_t n) {
    int grid_size;
    dim3 block = get_verify_config(n, grid_size);
    verify_distrib_kernel<Fp3x2><<<grid_size, block>>>(
        failures, static_cast<const Fp3x2*>(a), static_cast<const Fp3x2*>(b), static_cast<const Fp3x2*>(c), n);
    return cudaGetLastError();
}

// ============================================================================
// Kb Verification (KoalaBear base)
// ============================================================================

extern "C" int verify_inv_kb(uint32_t* failures, const void* a, size_t n) {
    int grid_size;
    dim3 block = get_verify_config(n, grid_size);
    verify_inv_kernel<Kb><<<grid_size, block>>>(failures, static_cast<const Kb*>(a), n);
    return cudaGetLastError();
}

extern "C" int verify_distrib_kb(uint32_t* failures, const void* a, const void* b, const void* c, size_t n) {
    int grid_size;
    dim3 block = get_verify_config(n, grid_size);
    verify_distrib_kernel<Kb><<<grid_size, block>>>(
        failures, static_cast<const Kb*>(a), static_cast<const Kb*>(b), static_cast<const Kb*>(c), n);
    return cudaGetLastError();
}

// ============================================================================
// Kb5 Verification (KoalaBear quintic)
// ============================================================================

extern "C" int verify_inv_kb5(uint32_t* failures, const void* a, size_t n) {
    int grid_size;
    dim3 block = get_verify_config(n, grid_size);
    verify_inv_kernel<Kb5><<<grid_size, block>>>(failures, static_cast<const Kb5*>(a), n);
    return cudaGetLastError();
}

extern "C" int verify_distrib_kb5(uint32_t* failures, const void* a, const void* b, const void* c, size_t n) {
    int grid_size;
    dim3 block = get_verify_config(n, grid_size);
    verify_distrib_kernel<Kb5><<<grid_size, block>>>(
        failures, static_cast<const Kb5*>(a), static_cast<const Kb5*>(b), static_cast<const Kb5*>(c), n);
    return cudaGetLastError();
}
