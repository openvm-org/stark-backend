#include "fp.h"
#include "launcher.cuh"
#include "utils.cuh"

constexpr uint32_t LOG_WARP_SIZE = 5;

template <bool intt>
__device__ __forceinline__ Fp sum_or_semi_sum(Fp&& x) {
    if constexpr (intt) {
        return x.halve();
    } else {
        return x;
    }
}

template <bool intt>
__global__ void batch_ntt_kernel(
    Fp* __restrict__ buffer,
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
    uint32_t const buf_idx = intt ? (i ? (1 << l_skip) - i : 0) : i;

    uint32_t const log_interwarp = min(l_skip, LOG_WARP_SIZE);
    Fp inv_twiddle = pow(TWO_ADIC_GENERATORS[l_skip], i);
    Fp this_thread_value;
    if (l_skip > log_interwarp) {
        extern __shared__ Fp smem[];
        auto sbuf = smem + (threadIdx.y << l_skip);

        if (active_thread) {
            sbuf[i] = buffer[buf_idx];
        }
        __syncthreads();

        for (uint32_t log_len = l_skip; log_len --> log_interwarp;) {
            if (active_thread) {
                uint32_t const len = 1u << log_len;
                if (!(i & len)) {
                    Fp const sum = sbuf[i];
                    Fp const diff = sbuf[i + len];
                    sbuf[i] = sum_or_semi_sum<intt>(sum + diff);
                    sbuf[i + len] = sum_or_semi_sum<intt>(sum - diff) * inv_twiddle;
                }
                inv_twiddle *= inv_twiddle;
            }
            __syncthreads();
        }
        this_thread_value = sbuf[i];
    } else if (active_thread) {
        this_thread_value = buffer[buf_idx];
    }

    if (active_thread) {
        for (uint32_t log_len = log_interwarp; log_len --> 0;) {
            uint32_t const len = 1u << log_len;
            Fp const other_value = Fp::fromRaw(__shfl_xor_sync(0xffffffff, this_thread_value.asRaw(), len));
            if (!(i & len)) {
                // this_thread_value = sum, other_value = diff
                this_thread_value = sum_or_semi_sum<intt>(this_thread_value + other_value);
            } else {
                // this_thread_value = diff, other_value = sum
                this_thread_value = sum_or_semi_sum<intt>(this_thread_value - other_value) * inv_twiddle;
            }
            inv_twiddle *= inv_twiddle;
        }
    }

    if (active_thread) {
        auto const j = rev_len(i, l_skip);
        buffer[j] = this_thread_value;
    }
}

extern "C" int _batch_ntt_small(
    Fp* buffer,
    size_t const l_skip,
    size_t const cnt_blocks,
    bool const is_intt
) {
    uint32_t const threads_per_block = 1024;
    uint32_t const threads_x = 1 << l_skip;
    assert(threads_per_block >> l_skip);
    uint32_t const threads_y = threads_per_block / threads_x;
    size_t const smem_size = (l_skip > LOG_WARP_SIZE) ? (sizeof(Fp) * threads_per_block) : 0;
    assert((1 << l_skip) <= 1024);
    if (is_intt) {
        batch_ntt_kernel<true><<<(cnt_blocks + threads_y - 1) / threads_y, dim3{threads_x, threads_y, 1}, smem_size>>>(buffer, l_skip, cnt_blocks);
    } else {
        batch_ntt_kernel<false><<<(cnt_blocks + threads_y - 1) / threads_y, dim3{threads_x, threads_y, 1}, smem_size>>>(buffer, l_skip, cnt_blocks);
    }

    return CHECK_KERNEL();
}