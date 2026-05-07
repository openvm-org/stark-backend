#pragma once

#include <algorithm>
#include <cuda_runtime.h>
#ifdef CUDA_DEBUG
#include <cstdio>
#endif

// =============================================================================
// SHADOW CTA profiler hooks.
//
// When SHADOW_CTA_PROFILE is defined (set by cuda-builder when the consumer
// crate enables the `profiler` feature on openvm-cuda-backend), every
// instrumented kernel takes one extra parameter (a CtaProbeCtx) and records a
// per-CTA start/end timestamp + %smid. When the macro is undefined, all of
// these expand to nothing — kernel signatures, launch sites, and SASS are
// byte-identical to the unprofiled build.
//
// Use pattern:
//
//   __global__ void my_kernel(int *a, int *b SHADOW_KERNEL_PARAM) {
//       SHADOW_KERNEL_BEGIN(KID_MY_KERNEL);
//       // ... body ...
//       SHADOW_KERNEL_END(KID_MY_KERNEL);
//   }
//
//   my_kernel<<<grid, block, 0, stream>>>(a, b SHADOW_LAUNCH_ARG);
//
// `KID_*` constants are defined in the consumer crate's CUDA sources and
// mirrored on the Rust side via openvm-cuda-profiler::kernel_ids.
// =============================================================================
#ifdef SHADOW_CTA_PROFILE
#include "cta_probe.cuh"
// Implemented in the openvm-cuda-profiler crate; resolves to a process-global
// CtaProbeCtx that the profiler's init publishes once. Returns a zeroed ctx
// (mask == 0) when the profiler is compiled in but the runtime switch is off,
// which the probe macros treat as a no-op.
extern "C" CtaProbeCtx shadow_cta_ctx();

// Used in kernel signatures: `__global__ void k(args... SHADOW_KERNEL_PARAM)`.
#define SHADOW_KERNEL_PARAM , CtaProbeCtx __cta_ctx

// Used at kernel body start/end: `SHADOW_KERNEL_BEGIN("kernel_name")`. The
// name is hashed at compile time by `cta_kid_fnv1a`; the host-side
// `register_kernel(name)` records the inverse mapping.
#define SHADOW_KERNEL_BEGIN(NAME) CTA_PROBE_BEGIN(::cta_kid_fnv1a(NAME), __cta_ctx)
#define SHADOW_KERNEL_END(NAME) CTA_PROBE_END(::cta_kid_fnv1a(NAME), __cta_ctx)

// Used at the launch site: `kern<<<g,b,0,s>>>(args... SHADOW_LAUNCH_ARG)`.
#define SHADOW_LAUNCH_ARG , shadow_cta_ctx()
#else
#define SHADOW_KERNEL_PARAM
#define SHADOW_KERNEL_BEGIN(NAME) ((void)0)
#define SHADOW_KERNEL_END(NAME) ((void)0)
#define SHADOW_LAUNCH_ARG
#endif

static const size_t MAX_THREADS = 1024;
static const size_t WARP_SIZE = 32;

inline size_t div_ceil(size_t a, size_t b) { return (a + b - 1) / b; }

inline std::pair<dim3, dim3> kernel_launch_params(
    size_t count,
    size_t threads_per_block = MAX_THREADS
) {
    if (count == 0) {
        return std::make_pair(dim3(0, 1, 1), dim3(1, 1, 1));
    }
    size_t block = std::min(count, threads_per_block);
    size_t grid = div_ceil(count, block);
    return std::make_pair(dim3(grid, 1, 1), dim3(block, 1, 1));
}

inline std::pair<dim3, dim3> kernel_launch_2d_params(size_t x, size_t y) {
    if (x == 0 || y == 0) {
        return std::make_pair(dim3(0, 1, 1), dim3(1, 1, 1));
    }
    dim3 block = dim3(std::min(x, WARP_SIZE), std::min(y, WARP_SIZE));
    dim3 grid = dim3(div_ceil(x, block.x), div_ceil(y, block.y));
    return std::make_pair(grid, block);
}

#define CUDA_OK(expr) do {                                  \
    cudaError_t err = expr;                                 \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA kernel error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err));   \
    }                                                       \
} while(0)

#ifdef CUDA_DEBUG
    inline int cuda_check_kernel(const char* kernel_name) {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "[ERROR] Kernel '%s' failed: %s\n",
                    kernel_name, cudaGetErrorString(err));
        }
        return err;
    }
#   define CHECK_KERNEL() cuda_check_kernel(__func__)
#else
#   define CHECK_KERNEL() cudaGetLastError()
#endif
