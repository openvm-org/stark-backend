// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Test kernel for the in-process profiler smoke test.
//
// Defines:
//
//   shadow_test_kernel_launch(n, stream)
//     Launches a tiny no-op kernel `n` times under different grid sizes so the
//     profiler captures a healthy mix of CTAs across SMs.
//
// We bypass launcher.cuh's macros and use cta_probe.cuh directly so the test
// kernel doesn't depend on SHADOW_CTA_PROFILE being defined at compile time —
// it's always instrumented (the runtime mask gate makes it a no-op when the
// profiler is off).

#include "cta_probe.cuh"
#include <cuda_runtime.h>

extern "C" CtaProbeCtx shadow_cta_ctx();

namespace shadow_test {
constexpr uint32_t kid_test_kernel() {
    // FNV-1a("shadow_test_kernel"); identical to register_kernel host call.
    uint32_t h = 0x811c9dc5u;
    const char *s = "shadow_test_kernel";
    while (*s) {
        h ^= static_cast<uint32_t>(static_cast<unsigned char>(*s++));
        h *= 0x01000193u;
    }
    return h == 0u ? 0xdeadbeefu : h;
}

__global__ void shadow_test_kernel(uint32_t n, uint32_t *dummy, CtaProbeCtx ctx) {
    CTA_PROBE_BEGIN(kid_test_kernel(), ctx);
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Trivial work to keep the kernel from being optimised away. Writes
        // to a per-thread dummy slot (caller guarantees `dummy` length).
        dummy[tid] = tid * 0xa5a5a5a5u;
    }
    CTA_PROBE_END(kid_test_kernel(), ctx);
}
} // namespace shadow_test

// Launch `iters` rounds, each with `blocks` blocks of `threads_per_block`
// threads, so the test gets a deterministic CTA count.
extern "C" int shadow_test_kernel_launch(
    uint32_t blocks,
    uint32_t threads_per_block,
    uint32_t iters,
    uint32_t *dummy,
    void *stream_ptr
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    dim3 grid(blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);
    uint32_t n = blocks * threads_per_block;
    for (uint32_t i = 0; i < iters; ++i) {
        shadow_test::shadow_test_kernel<<<grid, block, 0, stream>>>(n, dummy, shadow_cta_ctx());
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return static_cast<int>(err);
        }
    }
    return 0;
}
