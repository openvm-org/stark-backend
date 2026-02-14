#include "device_ntt.cuh"
#include "fp.h"
#include "launcher.cuh"
#include "utils.cuh"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector_types.h>

using namespace device_ntt;

// ============================================================================
// Constant Memory Definition and Initialization
// ============================================================================

namespace device_ntt {

// Define the constant memory (declared extern in device_ntt.cuh)
__constant__ Fp DEVICE_NTT_TWIDDLES[DEVICE_NTT_TWIDDLES_SIZE];

} // namespace device_ntt

// Kernel to generate twiddles for all levels 1..MAX_NTT_LEVEL
// Each thread handles one twiddle: omega_level^index
__global__ void generate_device_ntt_twiddles_kernel(Fp *d_twiddles) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= DEVICE_NTT_TWIDDLES_SIZE)
        return;

    // Find which level this tid belongs to
    // Level L starts at offset 2^L - 2 and has 2^L elements
    // tid in [2^L - 2, 2^(L+1) - 2) => level L
    uint32_t level = 1;
    uint32_t offset = 0;
    while (level <= MAX_NTT_LEVEL) {
        uint32_t level_size = 1u << level;
        if (tid < offset + level_size) {
            break;
        }
        offset += level_size;
        level++;
    }

    uint32_t index = tid - offset;
    // Compute omega_level^index where omega_level = TWO_ADIC_GENERATORS[level]
    d_twiddles[tid] = pow(TWO_ADIC_GENERATORS[level], index);
}

// Generate twiddles into the provided device buffer and copy to constant memory.
// `d_twiddles` must have capacity for DEVICE_NTT_TWIDDLES_SIZE elements.
// Returns 0 on success, non-zero on error.
extern "C" int _generate_device_ntt_twiddles(Fp *d_twiddles) {
    // Generate twiddles on GPU
    constexpr uint32_t BLOCK_SIZE = 256;
    uint32_t num_blocks = div_ceil(DEVICE_NTT_TWIDDLES_SIZE, BLOCK_SIZE);
    generate_device_ntt_twiddles_kernel<<<num_blocks, BLOCK_SIZE>>>(d_twiddles);

    // Copy to constant memory using per-thread stream
    cudaMemcpyToSymbolAsync(
        DEVICE_NTT_TWIDDLES,
        d_twiddles,
        DEVICE_NTT_TWIDDLES_SIZE * sizeof(Fp),
        0,
        cudaMemcpyDeviceToDevice,
        cudaStreamPerThread
    );
    cudaStreamSynchronize(cudaStreamPerThread);

    return CHECK_KERNEL();
}

// ============================================================================
// Batch NTT Kernels
// ============================================================================

template <bool intt, bool needs_shmem>
__global__ void batch_ntt_kernel(
    Fp *__restrict__ buffer,
    uint32_t const l_skip,
    uint32_t const cnt_blocks
) {
    uint32_t const block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    bool const active_thread = (block_idx < cnt_blocks);
    buffer += block_idx << l_skip;

#ifdef CUDA_DEBUG
    assert(blockDim.x <= (1 << l_skip));
#endif
    uint32_t const i = threadIdx.x;

    Fp this_thread_value;
    if constexpr (needs_shmem) {
        extern __shared__ Fp smem[];
        auto sbuf = smem + (threadIdx.y << l_skip);
        if (active_thread) {
            sbuf[i] = buffer[i];
        }
        __syncthreads();

        ntt_natural_to_bitrev<intt, true>(this_thread_value, sbuf, i, l_skip, active_thread);
    } else if (active_thread) {
        this_thread_value = buffer[i];
        ntt_natural_to_bitrev<intt, false>(this_thread_value, nullptr, i, l_skip, true);
    }

    if (active_thread) {
        auto const j = rev_len(i, l_skip);
        buffer[j] = this_thread_value;
    }
}

template <bool intt, bool needs_shmem>
int launch_batch_ntt_small(Fp *buffer, size_t const l_skip, size_t const cnt_blocks) {
    uint32_t const threads_per_block = 1024;
    uint32_t const threads_x = 1 << l_skip;
    assert(threads_per_block >> l_skip);
    uint32_t const threads_y = threads_per_block / threads_x;
    size_t const smem_size = needs_shmem ? (sizeof(Fp) * threads_per_block) : 0;

    batch_ntt_kernel<intt, needs_shmem>
        <<<div_ceil(cnt_blocks, threads_y), dim3{threads_x, threads_y, 1}, smem_size>>>(
            buffer, l_skip, cnt_blocks
        );

    return CHECK_KERNEL();
}

extern "C" int _batch_ntt_small(
    Fp *buffer,
    size_t const l_skip,
    size_t const cnt_blocks,
    bool const is_intt
) {
    bool const needs_shmem = l_skip > LOG_WARP_SIZE;
    assert((1 << l_skip) <= 1024);
    return DISPATCH_BOOL_PAIR(
        launch_batch_ntt_small, is_intt, needs_shmem, buffer, l_skip, cnt_blocks
    );
}
