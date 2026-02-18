// Brent-Kung parallel prefix scan kernels for Metal
// Translated from CUDA: cuda-backend/cuda/src/prefix.cu
#include <metal_stdlib>
using namespace metal;

#include "../include/baby_bear.h"
#include "../include/baby_bear_ext.h"

constant uint32_t ACC_PER_THREAD = 16;

kernel void prefix_scan_block_ext(
    device FpExt *d_inout                   [[buffer(0)]],
    constant uint64_t &length               [[buffer(1)]],
    constant uint64_t &round_stride         [[buffer(2)]],
    threadgroup FpExt *shared_mem           [[threadgroup(0)]],
    uint tid                                [[thread_index_in_threadgroup]],
    uint tpg                                [[threads_per_threadgroup]],
    uint group_id                           [[threadgroup_position_in_grid]]
) {
    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt acc_res = zero;
    FpExt acc_data[ACC_PER_THREAD];

    uint32_t tile_idx = tid;
    uint32_t shared_elem_per_block = tpg;
    uint64_t index = (uint64_t(group_id) * tpg + tid) * ACC_PER_THREAD;
    bool first_round = (round_stride == 1);

    for (uint32_t i = 0; i < ACC_PER_THREAD; i++) {
        FpExt data_in = zero;
        uint64_t offset = first_round ? index + i : (index + i + 1) * round_stride - 1;
        if (offset < length) {
            data_in = d_inout[offset];
        }
        acc_res = acc_res + data_in;
        acc_data[i] = acc_res;
    }
    shared_mem[tile_idx] = acc_res;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Brent-Kung upsweep
    uint32_t stride = 2;
    uint32_t src_offset = 0;
    uint32_t dst_offset = stride - 1;
    for (uint32_t idx = shared_elem_per_block >> 1; idx > 0; idx = idx >> 1) {
        if (tile_idx < idx) {
            FpExt dst = shared_mem[tile_idx * stride + dst_offset];
            FpExt src = shared_mem[tile_idx * stride + src_offset];
            dst = dst + src;
            shared_mem[tile_idx * stride + dst_offset] = dst;
        }
        src_offset = stride - 1;
        stride = stride << 1;
        dst_offset = stride - 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // downsweep
    for (uint32_t s = shared_elem_per_block; s > 1; s = s >> 1) {
        src_offset = s - 1;
        dst_offset = src_offset + (s >> 1);
        uint32_t thread_in_round = shared_elem_per_block / s - 1;
        if (tile_idx < thread_in_round) {
            FpExt dst = shared_mem[tile_idx * s + dst_offset];
            FpExt src = shared_mem[tile_idx * s + src_offset];
            dst = dst + src;
            shared_mem[tile_idx * s + dst_offset] = dst;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // scan and write back
    FpExt prefix_sum = zero;
    if (tile_idx > 0)
        prefix_sum = shared_mem[tile_idx - 1];

    for (uint32_t i = 0; i < ACC_PER_THREAD; i++) {
        uint64_t offset = first_round ? index + i : (index + i + 1) * round_stride - 1;
        if (offset < length) {
            d_inout[offset] = prefix_sum + acc_data[i];
        }
    }
}

kernel void prefix_scan_block_downsweep_ext(
    device FpExt *d_inout                   [[buffer(0)]],
    constant uint64_t &length               [[buffer(1)]],
    constant uint64_t &round_stride         [[buffer(2)]],
    constant uint64_t &basic_level          [[buffer(3)]],
    uint gid                                [[thread_position_in_grid]]
) {
    uint64_t low_level_round_stride = round_stride / basic_level;
    uint64_t dst_index = (uint64_t(gid) + 1) * low_level_round_stride - 1;
    uint64_t src_index = (dst_index / round_stride) * round_stride - 1;
    bool is_last_elem = dst_index % round_stride == round_stride - 1;
    uint64_t level_offset = dst_index / round_stride;
    if (dst_index >= length || is_last_elem || level_offset == 0)
        return;

    FpExt dst = d_inout[dst_index];
    FpExt src = d_inout[src_index];
    d_inout[dst_index] = dst + src;
}

kernel void prefix_scan_epilogue_ext(
    device FpExt *d_inout                   [[buffer(0)]],
    constant uint64_t &length               [[buffer(1)]],
    constant uint64_t &basic_level          [[buffer(2)]],
    uint gid                                [[thread_position_in_grid]]
) {
    uint64_t dst_index = gid;
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
