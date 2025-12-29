#include "device_ntt.cuh"
#include "fp.h"
#include "launcher.cuh"
#include "utils.cuh"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector_types.h>

using namespace device_ntt;

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
