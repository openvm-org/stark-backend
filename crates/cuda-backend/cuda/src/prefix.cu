/*
 * Source: https://github.com/scroll-tech/plonky3-gpu (private repo)
 * Status: BASED ON plonky3-gpu/gpu-backend/src/cuda/kernels/prefix.cu
 * Imported: 2025-01-25 by @gaxiom
 */

#include "fpext.h"
#include "launcher.cuh"

static const uint32_t ACC_PER_THREAD = 16;
static const uint32_t SHARED_DATA = 256;

__global__ void prefix_scan_block_ext(FpExt *d_inout, uint64_t length, uint64_t round_stride) {
    FpExt acc_res = FpExt(0);
    FpExt acc_data[ACC_PER_THREAD];

#ifndef __clang_analyzer__
    __shared__ FpExt shared_mem[SHARED_DATA];
#else
    __shared__ __align__(alignof(FpExt)) unsigned char shared_mem_raw[SHARED_DATA * sizeof(FpExt)];
    FpExt *shared_mem = reinterpret_cast<FpExt *>(shared_mem_raw);
#endif

    uint32_t tile_idx = threadIdx.x;
    uint32_t shared_elem_per_block = blockDim.x;
    uint64_t index = (blockIdx.x * blockDim.x + threadIdx.x) * ACC_PER_THREAD;
    bool first_round = (round_stride == 1);

#pragma unroll
    for (uint32_t i = 0; i < ACC_PER_THREAD; i++) {
        FpExt data_in = FpExt(0);
        uint64_t offset = first_round ? index + i : (index + i + 1) * round_stride - 1;
        if (offset < length) {
            data_in = d_inout[offset];
        }
        acc_res += data_in;
        acc_data[i] = acc_res;
    }
    shared_mem[tile_idx] = acc_res;
    __syncthreads();

    // https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf
    // Brent-Kung construction in Fig.1d
    // upsweep
    uint32_t stride = 2;
    uint32_t src_offset = 0;
    uint32_t dst_offset = stride - 1;
    for (uint32_t idx = shared_elem_per_block >> 1; idx > 0; idx = idx >> 1) {
        if (tile_idx < idx) {
            FpExt dst = shared_mem[tile_idx * stride + dst_offset];
            FpExt src = shared_mem[tile_idx * stride + src_offset];
            dst += src;
            shared_mem[tile_idx * stride + dst_offset] = dst;
        }
        src_offset = stride - 1;
        stride = stride << 1;
        dst_offset = stride - 1;
        __syncthreads();
    }

    // downsweep
    for (uint32_t stride = shared_elem_per_block; stride > 1; stride = stride >> 1) {
        src_offset = stride - 1;
        dst_offset = src_offset + (stride >> 1);
        uint32_t thread_in_round = shared_elem_per_block / stride - 1;
        if (tile_idx < thread_in_round) {
            FpExt dst = shared_mem[tile_idx * stride + dst_offset];
            FpExt src = shared_mem[tile_idx * stride + src_offset];
            dst += src;
            shared_mem[tile_idx * stride + dst_offset] = dst;
        }
        __syncthreads();
    }

    // scan and write back
    FpExt prefix_sum = FpExt(0);
    if (tile_idx > 0)
        prefix_sum = shared_mem[tile_idx - 1];

#pragma unroll
    for (uint32_t i = 0; i < ACC_PER_THREAD; i++) {
        uint64_t offset = first_round ? index + i : (index + i + 1) * round_stride - 1;
        if (offset < length) {
            d_inout[offset] = prefix_sum + acc_data[i];
        }
    }
}

__global__ void prefix_scan_block_downsweep_ext(
    FpExt *d_inout,
    uint64_t length,
    uint64_t round_stride,
    uint64_t basic_level
) {
    uint64_t low_level_round_stride = round_stride / basic_level;
    uint64_t dst_index = (blockIdx.x * blockDim.x + threadIdx.x) * low_level_round_stride - 1;
    uint64_t src_index =
        (dst_index / round_stride) * round_stride - 1; // last element in last "data block"
    bool is_last_elem = dst_index % round_stride == round_stride - 1; // bypass
    uint64_t level_offset = dst_index / round_stride;
    if (dst_index >= length || is_last_elem || level_offset == 0)
        return;

    FpExt dst = d_inout[dst_index];
    FpExt src = d_inout[src_index];
    d_inout[dst_index] = dst + src;
}

__global__ void prefix_scan_epilogue_ext(FpExt *d_inout, uint64_t length, uint64_t basic_level) {
    uint64_t dst_index = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t level_offset = dst_index / basic_level;
    uint64_t src_index = level_offset * basic_level - 1;
    bool is_last_elem_in_level = dst_index % basic_level == basic_level - 1;
    if (level_offset == 0 || dst_index >= length)
        return;
    if (is_last_elem_in_level)
        return;

    FpExt dst = d_inout[dst_index];
    FpExt prefix_sum = d_inout[src_index];
    d_inout[dst_index] = dst + prefix_sum;
}

// LAUNCHERS

extern "C" int _prefix_scan_block_ext(
    FpExt *d_inout,
    uint64_t length,
    uint64_t round_stride,
    uint64_t block_num
) {
    prefix_scan_block_ext<<<block_num, SHARED_DATA>>>(d_inout, length, round_stride);
    return cudaGetLastError();
}

extern "C" int _prefix_scan_block_downsweep_ext(
    FpExt *d_inout,
    uint64_t length,
    uint64_t round_stride
) {
    auto element_per_block = ACC_PER_THREAD * SHARED_DATA;
    auto low_level_round_stride = round_stride / element_per_block;
    auto node_num = div_ceil(length, low_level_round_stride);
    auto block_num = div_ceil(node_num, SHARED_DATA);
    prefix_scan_block_downsweep_ext<<<block_num, SHARED_DATA>>>(
        d_inout, length, round_stride, element_per_block
    );
    return cudaGetLastError();
}

extern "C" int _prefix_scan_epilogue_ext(FpExt *d_inout, uint64_t length) {
    auto element_per_block = ACC_PER_THREAD * SHARED_DATA;
    auto epilogue_block_num = div_ceil(length, SHARED_DATA);
    prefix_scan_epilogue_ext<<<epilogue_block_num, SHARED_DATA>>>(
        d_inout, length, element_per_block
    );
    return cudaGetLastError();
}