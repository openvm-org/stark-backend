// Sumcheck protocol kernels for Metal
// Translated from CUDA: cuda-backend/cuda/src/sumcheck.cu
#include <metal_stdlib>
using namespace metal;

#include "../include/baby_bear.h"
#include "../include/baby_bear_ext.h"
#include "../include/sumcheck.h"

inline FpExt load_fpext_words(const device Fp *ptr, uint32_t idx) {
    uint32_t base = idx * 4;
    return FpExt{ptr[base], ptr[base + 1], ptr[base + 2], ptr[base + 3]};
}

inline void store_fpext_words(device Fp *ptr, uint32_t idx, FpExt value) {
    uint32_t base = idx * 4;
    ptr[base] = value.elems[0];
    ptr[base + 1] = value.elems[1];
    ptr[base + 2] = value.elems[2];
    ptr[base + 3] = value.elems[3];
}

// Reduces evaluations over x and column dimensions for PLE round 0
kernel void reduce_over_x_and_cols_kernel(
    const device Fp *input                  [[buffer(0)]],
    device Fp *output                       [[buffer(1)]],
    constant uint32_t &num_x               [[buffer(2)]],
    constant uint32_t &num_cols            [[buffer(3)]],
    constant uint32_t &large_domain_size   [[buffer(4)]],
    uint gid                                [[thread_position_in_grid]]
) {
    uint32_t z = gid;
    if (z >= large_domain_size) return;

    Fp sum = Fp(0u);
    for (uint32_t x = 0; x < num_x; x++) {
        for (uint32_t col = 0; col < num_cols; col++) {
            uint32_t offset = (x * num_cols + col) * large_domain_size;
            sum = sum + input[offset + z];
        }
    }
    output[z] = sum;
}

// MLE sumcheck round kernel (WD=1)
// Computes s(X) = sum_{y in H_{n-1}} f_hat(X, y) for X in {1, ..., d}
kernel void sumcheck_mle_round_kernel(
    const device uint64_t *input_matrices   [[buffer(0)]],  // array of pointers
    device FpExt *block_sums               [[buffer(1)]],
    const device uint32_t *widths          [[buffer(2)]],
    constant uint32_t &num_matrices        [[buffer(3)]],
    constant uint32_t &height              [[buffer(4)]],
    constant uint32_t &d                   [[buffer(5)]],
    threadgroup FpExt *shared              [[threadgroup(0)]],
    uint tid                                [[thread_index_in_threadgroup]],
    uint tpg                                [[threads_per_threadgroup]],
    uint group_id                           [[threadgroup_position_in_grid]],
    uint groups_per_grid                    [[threadgroups_per_grid]]
) {
    uint32_t half_height = height >> 1;
    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};

    // Local accumulators: max degree 4+1=5
    FpExt local_sums[5];
    for (uint i = 0; i < d; i++) {
        local_sums[i] = zero;
    }

    uint32_t grid_threads = tpg * groups_per_grid;
    for (uint32_t y = group_id * tpg + tid; y < half_height; y += grid_threads) {
        for (uint32_t x_int = 1; x_int <= d; x_int++) {
            FpExt X = FpExt{Fp(x_int), Fp(0u), Fp(0u), Fp(0u)};

            for (uint32_t mat_idx = 0; mat_idx < num_matrices; mat_idx++) {
                const device Fp *input = reinterpret_cast<const device Fp *>(input_matrices[mat_idx]);
                uint32_t width = widths[mat_idx];

                for (uint32_t col = 0; col < width; col++) {
                    uint32_t col_offset = col * height;
                    uint32_t idx_0 = col_offset + (y << 1);
                    uint32_t idx_1 = col_offset + (y << 1) + 1;

                    FpExt eval_0 = load_fpext_words(input, idx_0);
                    FpExt eval_1 = load_fpext_words(input, idx_1);
                    FpExt eval_X = eval_0 + X * (eval_1 - eval_0);

                    local_sums[x_int - 1] = local_sums[x_int - 1] + eval_X;
                }
            }
        }
    }

    // Block reduction for each x_int
    for (uint32_t x_int = 0; x_int < d; x_int++) {
        FpExt reduced = block_reduce_sum(local_sums[x_int], shared, tid, tpg);
        if (tid == 0) {
            block_sums[group_id * d + x_int] = reduced;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Specialized single-matrix MLE round for plain multilinear sumcheck.
// Computes s(1) = sum_y input[2*y+1].
kernel void sumcheck_mle_round_single_kernel(
    const device FpExt *input             [[buffer(0)]],
    device FpExt *block_sums              [[buffer(1)]],
    constant uint32_t &height             [[buffer(2)]],
    threadgroup FpExt *shared             [[threadgroup(0)]],
    uint tid                               [[thread_index_in_threadgroup]],
    uint tpg                               [[threads_per_threadgroup]],
    uint group_id                          [[threadgroup_position_in_grid]],
    uint groups_per_grid                   [[threadgroups_per_grid]]
) {
    uint32_t half_height = height >> 1;
    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt local_sum = zero;

    uint32_t grid_threads = tpg * groups_per_grid;
    for (uint32_t y = group_id * tpg + tid; y < half_height; y += grid_threads) {
        local_sum = local_sum + input[(y << 1) + 1];
    }

    FpExt reduced = block_reduce_sum(local_sum, shared, tid, tpg);
    if (tid == 0) {
        block_sums[group_id] = reduced;
    }
}

// Final reduction: combines partial block sums into final result
kernel void final_reduce_block_sums_kernel(
    const device FpExt *block_sums         [[buffer(0)]],
    device FpExt *output                   [[buffer(1)]],
    constant uint32_t &num_blocks          [[buffer(2)]],
    constant uint32_t &d                   [[buffer(3)]],
    threadgroup FpExt *shared              [[threadgroup(0)]],
    uint tid                                [[thread_index_in_threadgroup]],
    uint tpg                                [[threads_per_threadgroup]],
    uint out_idx                            [[threadgroup_position_in_grid]]
) {
    if (out_idx >= d) return;

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt sum = zero;
    for (uint32_t block_id = tid; block_id < num_blocks; block_id += tpg) {
        sum = sum + block_sums[block_id * d + out_idx];
    }
    sum = block_reduce_sum(sum, shared, tid, tpg);
    if (tid == 0) {
        output[out_idx] = sum;
    }
}

// Batched final reduction for multiple segments (e.g., AIRs)
kernel void batched_final_reduce_block_sums_kernel(
    const device FpExt *block_sums         [[buffer(0)]],
    device FpExt *output                   [[buffer(1)]],
    const device uint32_t *segment_offsets [[buffer(2)]],
    constant uint32_t &d                   [[buffer(3)]],
    threadgroup FpExt *shared              [[threadgroup(0)]],
    uint2 tid2                              [[thread_position_in_threadgroup]],
    uint2 tpg2                              [[threads_per_threadgroup]],
    uint2 group_id                          [[threadgroup_position_in_grid]]
) {
    uint tid = tid2.x;
    uint tpg = tpg2.x;
    uint32_t seg_idx = group_id.x;
    uint32_t out_idx = group_id.y;

    uint32_t start = segment_offsets[seg_idx];
    uint32_t end = segment_offsets[seg_idx + 1];
    uint32_t num_blocks = end - start;

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt sum = zero;
    for (uint32_t i = tid; i < num_blocks; i += tpg) {
        sum = sum + block_sums[(start + i) * d + out_idx];
    }
    sum = block_reduce_sum(sum, shared, tid, tpg);
    if (tid == 0) {
        output[seg_idx * d + out_idx] = sum;
    }
}

// Fold MLE evaluations: output[y] = input[2*y] + r*(input[2*y+1] - input[2*y])
kernel void fold_mle_kernel(
    const device uint64_t *input_matrices  [[buffer(0)]],
    const device uint64_t *output_matrices [[buffer(1)]],
    const device uint32_t *widths          [[buffer(2)]],
    constant uint8_t &log_output_height    [[buffer(3)]],
    constant FpExt &r_val                  [[buffer(4)]],
    uint2 gid                               [[thread_position_in_grid]]
) {
    uint32_t tidx = gid.x;
    uint32_t mat_idx = gid.y;

    uint32_t width = widths[mat_idx];
    uint32_t output_height = 1u << log_output_height;
    if (tidx >= output_height * width) return;

    uint32_t row_idx = tidx & (output_height - 1);
    uint32_t col_idx = tidx >> log_output_height;

    const device Fp *input = reinterpret_cast<const device Fp *>(input_matrices[mat_idx]);
    device Fp *output = reinterpret_cast<device Fp *>(output_matrices[mat_idx]);

    uint32_t col_offset_out = col_idx * output_height;
    uint32_t col_offset_in = col_offset_out << 1;

    uint32_t idx_0 = col_offset_in + (row_idx << 1);
    uint32_t idx_1 = col_offset_in + (row_idx << 1) + 1;
    uint32_t out_idx = col_offset_out + row_idx;

    FpExt t0 = load_fpext_words(input, idx_0);
    FpExt t1 = load_fpext_words(input, idx_1);
    store_fpext_words(output, out_idx, t0 + (t1 - t0) * r_val);
}

// Specialized single-matrix fold for plain multilinear sumcheck.
kernel void fold_mle_single_kernel(
    const device FpExt *input             [[buffer(0)]],
    device FpExt *output                  [[buffer(1)]],
    constant uint8_t &log_output_height   [[buffer(2)]],
    constant FpExt &r_val                 [[buffer(3)]],
    uint gid                               [[thread_position_in_grid]]
) {
    uint32_t output_height = 1u << log_output_height;
    if (gid >= output_height) return;

    uint32_t idx_0 = gid << 1;
    uint32_t idx_1 = idx_0 + 1;
    FpExt t0 = input[idx_0];
    FpExt t1 = input[idx_1];
    output[gid] = t0 + (t1 - t0) * r_val;
}

// Fold a single matrix with explicit width and output height.
kernel void fold_mle_matrix_kernel(
    const device FpExt *input             [[buffer(0)]],
    device FpExt *output                  [[buffer(1)]],
    constant uint32_t &width              [[buffer(2)]],
    constant uint8_t &log_output_height   [[buffer(3)]],
    constant FpExt &r_val                 [[buffer(4)]],
    uint gid                               [[thread_position_in_grid]]
) {
    uint32_t output_height = 1u << log_output_height;
    uint32_t total = output_height * width;
    if (gid >= total) return;

    uint32_t row_idx = gid & (output_height - 1);
    uint32_t col_idx = gid >> log_output_height;

    uint32_t col_offset_out = col_idx * output_height;
    uint32_t col_offset_in = col_offset_out << 1;

    uint32_t idx_0 = col_offset_in + (row_idx << 1);
    uint32_t idx_1 = idx_0 + 1;
    uint32_t out_idx = col_offset_out + row_idx;

    FpExt t0 = input[idx_0];
    FpExt t1 = input[idx_1];
    output[out_idx] = t0 + (t1 - t0) * r_val;
}

// Fold a single column in-place
kernel void fold_mle_column_kernel(
    device FpExt *buffer                   [[buffer(0)]],
    constant uint32_t &half_len            [[buffer(1)]],
    constant FpExt &r_val                  [[buffer(2)]],
    uint gid                                [[thread_position_in_grid]]
) {
    if (gid >= half_len) return;

    FpExt t0 = buffer[gid];
    FpExt t1 = buffer[gid + half_len];
    buffer[gid] = t0 + (t1 - t0) * r_val;
}

// Batch fold MLE with per-matrix heights
kernel void batch_fold_mle_kernel(
    const device uint64_t *input_matrices  [[buffer(0)]],
    const device uint64_t *output_matrices [[buffer(1)]],
    const device uint32_t *widths          [[buffer(2)]],
    const device uint8_t *log_output_heights [[buffer(3)]],
    constant FpExt &r_val                  [[buffer(4)]],
    uint2 gid                               [[thread_position_in_grid]]
) {
    uint32_t tidx = gid.x;
    uint32_t mat_idx = gid.y;

    uint8_t log_output_height = log_output_heights[mat_idx];
    uint32_t width = widths[mat_idx];
    uint32_t output_height = 1u << log_output_height;
    if (tidx >= output_height * width) return;

    uint32_t row_idx = tidx & (output_height - 1);
    uint32_t col_idx = tidx >> log_output_height;

    const device Fp *input = reinterpret_cast<const device Fp *>(input_matrices[mat_idx]);
    device Fp *output = reinterpret_cast<device Fp *>(output_matrices[mat_idx]);

    uint32_t col_offset_out = col_idx * output_height;
    uint32_t col_offset_in = col_offset_out << 1;

    uint32_t idx_0 = col_offset_in + (row_idx << 1);
    uint32_t idx_1 = col_offset_in + (row_idx << 1) + 1;
    uint32_t out_idx = col_offset_out + row_idx;

    FpExt t0 = load_fpext_words(input, idx_0);
    FpExt t1 = load_fpext_words(input, idx_1);
    store_fpext_words(output, out_idx, t0 + (t1 - t0) * r_val);
}

// Evaluate polynomial at r using Horner's method (from coefficients after iDFT)
kernel void fold_ple_from_coeffs_kernel(
    const device Fp *input_coeffs          [[buffer(0)]],
    device FpExt *output                   [[buffer(1)]],
    constant uint32_t &num_x               [[buffer(2)]],
    constant uint32_t &width               [[buffer(3)]],
    constant uint32_t &domain_size         [[buffer(4)]],
    constant FpExt &r_val                  [[buffer(5)]],
    uint gid                                [[thread_position_in_grid]]
) {
    uint32_t total_polys = num_x * width;
    if (gid >= total_polys) return;

    const device Fp *coeffs = input_coeffs + gid * domain_size;

    // Horner's method
    FpExt result = FpExt{coeffs[domain_size - 1], Fp(0u), Fp(0u), Fp(0u)};
    for (int i = int(domain_size) - 2; i >= 0; i--) {
        result = result * r_val + FpExt{coeffs[i], Fp(0u), Fp(0u), Fp(0u)};
    }
    output[gid] = result;
}

// Triangular fold for segment tree
kernel void triangular_fold_mle_kernel(
    device FpExt *output                   [[buffer(0)]],
    const device FpExt *input              [[buffer(1)]],
    constant FpExt &r                      [[buffer(2)]],
    uint2 gid                               [[thread_position_in_grid]]
) {
    uint32_t x = gid.x;
    uint32_t n = gid.y;
    uint32_t out_num_x = 1 << n;
    uint32_t in_num_x = 2 * out_num_x;

    if (x >= out_num_x) return;

    FpExt t0 = input[in_num_x + (x << 1)];
    FpExt t1 = input[in_num_x + (x << 1) + 1];
    output[out_num_x + x] = t0 + (t1 - t0) * r;
}
