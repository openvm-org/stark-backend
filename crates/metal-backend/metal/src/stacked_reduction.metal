// Stacked reduction kernels for Metal.
// Translated from CUDA: cuda-backend/cuda/src/stacked_reduction.cu
#include <metal_stdlib>
using namespace metal;

#include "../include/baby_bear.h"
#include "../include/baby_bear_ext.h"
#include "../include/sumcheck.h"

constant int S_DEG = 2;
constant int NUM_G = 3;
constant int D_EF = 4;

struct UnstackedSlice {
    uint32_t commit_idx;
    uint32_t log_height;
    uint32_t stacked_row_idx;
    uint32_t stacked_col_idx;
};

inline FpExt zero_ext() {
    return FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
}

inline FpExt get_eq_cube(const device FpExt *eq_r_ns, uint32_t cube_size, uint32_t x) {
    return eq_r_ns[cube_size + x];
}

inline uint32_t rot_prev(uint32_t x_int, uint32_t cube_size) {
    return x_int == 0 ? cube_size - 1 : x_int - 1;
}

inline Fp eq1(Fp x, Fp y) {
    Fp one = Fp(1u);
    Fp two = Fp(2u);
    return one - x - y + two * x * y;
}

// SP_DEG=1 round-0 block reduction.
kernel void stacked_reduction_round0_block_sum_kernel(
    const device FpExt *eq_r_ns                 [[buffer(0)]],
    const device Fp *trace_ptr                  [[buffer(1)]],
    const device FpExt *lambda_pows             [[buffer(2)]],
    device FpExt *block_sums                    [[buffer(3)]],
    constant uint32_t &height                   [[buffer(4)]],
    constant uint32_t &width                    [[buffer(5)]],
    constant uint32_t &l_skip                   [[buffer(6)]],
    constant uint32_t &skip_mask                [[buffer(7)]],
    constant uint32_t &num_x                    [[buffer(8)]],
    constant uint32_t &log_stride               [[buffer(9)]],
    threadgroup FpExt *shared_sum               [[threadgroup(0)]],
    uint2 tid2                                  [[thread_position_in_threadgroup]],
    uint2 tpg2                                  [[threads_per_threadgroup]],
    uint2 group_id                              [[threadgroup_position_in_grid]],
    uint2 groups_per_grid                       [[threadgroups_per_grid]]
) {
    uint tid = tid2.x;
    uint tpg = tpg2.x;
    uint32_t PADDED_X = tpg + 1;
    uint32_t tidx = group_id.x * tpg + tid;
    uint32_t z_idx = tidx & skip_mask;
    uint32_t x_int = tidx >> l_skip;
    uint32_t col_idx = group_id.y;
    FpExt zero = zero_ext();

    if (col_idx >= width) {
        shared_sum[0 * PADDED_X + tid] = zero;
        shared_sum[1 * PADDED_X + tid] = zero;
        shared_sum[2 * PADDED_X + tid] = zero;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return;
    }

    FpExt g0_term = zero;
    FpExt g1_term = zero;
    FpExt g2_term = zero;

    if (x_int < num_x) {
        FpExt eq_cube = get_eq_cube(eq_r_ns, num_x, x_int);
        FpExt eq_cube_rot_prev = get_eq_cube(eq_r_ns, num_x, rot_prev(x_int, num_x));
        FpExt k_rot_diff = eq_cube_rot_prev - eq_cube;

        FpExt coeff_eq = lambda_pows[2 * col_idx];
        FpExt coeff_rot = lambda_pows[2 * col_idx + 1];
        FpExt w0 = coeff_eq * eq_cube;
        FpExt w1 = coeff_rot * eq_cube;
        FpExt w2 = coeff_rot * k_rot_diff;

        const device Fp *evals = trace_ptr + col_idx * height + (x_int << (l_skip - log_stride));
        uint32_t stride_mask = (1u << log_stride) - 1;
        Fp q = ((z_idx & stride_mask) == 0) ? evals[z_idx >> log_stride] : Fp(0u);
        g0_term = w0 * q;
        g1_term = w1 * q;
        g2_term = w2 * q;
    }

    shared_sum[0 * PADDED_X + tid] = g0_term;
    shared_sum[1 * PADDED_X + tid] = g1_term;
    shared_sum[2 * PADDED_X + tid] = g2_term;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((tid >> l_skip) == 0) {
        FpExt g0 = shared_sum[0 * PADDED_X + z_idx];
        FpExt g1 = shared_sum[1 * PADDED_X + z_idx];
        FpExt g2 = shared_sum[2 * PADDED_X + z_idx];

        for (uint32_t lane = 1; lane < (tpg >> l_skip); ++lane) {
            g0 = g0 + shared_sum[0 * PADDED_X + (lane << l_skip) + z_idx];
            g1 = g1 + shared_sum[1 * PADDED_X + (lane << l_skip) + z_idx];
            g2 = g2 + shared_sum[2 * PADDED_X + (lane << l_skip) + z_idx];
        }

        uint32_t skip_domain = 1u << l_skip;
        uint32_t grid_x = groups_per_grid.x;
        device FpExt *out_ptr =
            block_sums + (col_idx * grid_x + group_id.x) * (NUM_G * skip_domain);
        out_ptr[0 * skip_domain + z_idx] = g0;
        out_ptr[1 * skip_domain + z_idx] = g1;
        out_ptr[2 * skip_domain + z_idx] = g2;
    }
}

// Final reduction for round-0 block sums with ADD-to-output semantics.
kernel void stacked_reduction_final_reduce_block_sums_add_kernel(
    const device FpExt *block_sums             [[buffer(0)]],
    device FpExt *output                       [[buffer(1)]],
    constant uint32_t &num_blocks              [[buffer(2)]],
    constant uint32_t &d                       [[buffer(3)]],
    threadgroup FpExt *shared                  [[threadgroup(0)]],
    uint tid                                   [[thread_index_in_threadgroup]],
    uint tpg                                   [[threads_per_threadgroup]],
    uint out_idx                               [[threadgroup_position_in_grid]]
) {
    if (out_idx >= d) {
        return;
    }

    FpExt sum = zero_ext();
    for (uint32_t block_id = tid; block_id < num_blocks; block_id += tpg) {
        sum = sum + block_sums[block_id * d + out_idx];
    }

    sum = block_reduce_sum(sum, shared, tid, tpg);
    if (tid == 0) {
        output[out_idx] = output[out_idx] + sum;
    }
}

// Non-degenerate MLE round with block reductions and atomic u64 accumulation.
kernel void stacked_reduction_sumcheck_mle_round_kernel(
    const device FpExt *q                      [[buffer(0)]],
    const device FpExt *eq_r_ns                [[buffer(1)]],
    const device FpExt *k_rot_ns               [[buffer(2)]],
    const device UnstackedSlice *unstacked_cols [[buffer(3)]],
    const device FpExt *lambda_pows            [[buffer(4)]],
    device atomic_uint *output                 [[buffer(5)]],
    constant uint32_t &q_height                [[buffer(6)]],
    constant uint32_t &window_len              [[buffer(7)]],
    constant uint32_t &num_y                   [[buffer(8)]],
    threadgroup FpExt *shared                  [[threadgroup(0)]],
    uint2 tid2                                  [[thread_position_in_threadgroup]],
    uint2 tpg2                                  [[threads_per_threadgroup]],
    uint2 group_id                              [[threadgroup_position_in_grid]],
    uint2 groups_per_grid                       [[threadgroups_per_grid]]
) {
    uint tid = tid2.x;
    uint tpg = tpg2.x;
    uint32_t window_idx_base = group_id.y;
    uint32_t y_int = group_id.x * tpg + tid;
    bool active_thread = (y_int < num_y);

    FpExt local_sums[S_DEG];
    for (uint idx = 0; idx < S_DEG; ++idx) {
        local_sums[idx] = zero_ext();
    }

    if (active_thread) {
        uint32_t num_evals = num_y * 2;

        FpExt eq_0 = get_eq_cube(eq_r_ns, num_evals, y_int << 1);
        FpExt eq_1 = get_eq_cube(eq_r_ns, num_evals, (y_int << 1) | 1);
        FpExt eq_c1 = eq_1 - eq_0;
        FpExt k_rot_0 = get_eq_cube(k_rot_ns, num_evals, y_int << 1);
        FpExt k_rot_1 = get_eq_cube(k_rot_ns, num_evals, (y_int << 1) | 1);
        FpExt k_rot_c1 = k_rot_1 - k_rot_0;

        uint32_t grid_y = groups_per_grid.y;
        for (uint32_t window_idx = window_idx_base; window_idx < window_len; window_idx += grid_y) {
            UnstackedSlice s = unstacked_cols[window_idx];

            uint32_t log_h = s.log_height;
            uint32_t col_idx = s.stacked_col_idx;
            uint32_t row_start = (s.stacked_row_idx >> log_h) * num_evals;
            uint32_t q_offset = col_idx * q_height + row_start;

            FpExt q_0 = q[q_offset + (y_int << 1)];
            FpExt q_1 = q[q_offset + (y_int << 1) + 1];
            FpExt q_c1 = q_1 - q_0;

            for (uint32_t x_int = 1; x_int <= uint32_t(S_DEG); ++x_int) {
                Fp x = Fp(x_int);
                FpExt x_ext = FpExt(x);
                FpExt q_x = q_0 + q_c1 * x_ext;
                FpExt eq_val = eq_0 + eq_c1 * x_ext;
                FpExt k_rot_val = k_rot_0 + k_rot_c1 * x_ext;

                FpExt contrib =
                    (lambda_pows[2 * window_idx] * eq_val + lambda_pows[2 * window_idx + 1] * k_rot_val) *
                    q_x;
                local_sums[x_int - 1] = local_sums[x_int - 1] + contrib;
            }
        }
    }

    for (uint idx = 0; idx < S_DEG; ++idx) {
        FpExt reduced = block_reduce_sum(local_sums[idx], shared, tid, tpg);
        if (tid == 0) {
            atomic_add_fpext_to_u64(output + idx * (D_EF * 2), reduced);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Degenerate MLE round.
kernel void stacked_reduction_sumcheck_mle_round_degenerate_kernel(
    const device FpExt *q                      [[buffer(0)]],
    const device FpExt *eq_ub_ptr              [[buffer(1)]],
    constant FpExt &eq_r                        [[buffer(2)]],
    constant FpExt &k_rot_r                     [[buffer(3)]],
    const device UnstackedSlice *unstacked_cols [[buffer(4)]],
    const device FpExt *lambda_pows             [[buffer(5)]],
    device atomic_uint *output                  [[buffer(6)]],
    constant uint32_t &q_height                 [[buffer(7)]],
    constant uint32_t &window_len               [[buffer(8)]],
    constant uint32_t &shift_factor             [[buffer(9)]],
    threadgroup FpExt *shared                   [[threadgroup(0)]],
    uint tid                                    [[thread_index_in_threadgroup]],
    uint tpg                                    [[threads_per_threadgroup]]
) {
    FpExt local_sums[S_DEG];
    for (uint idx = 0; idx < S_DEG; ++idx) {
        local_sums[idx] = zero_ext();
    }

    for (uint32_t window_idx = tid; window_idx < window_len; window_idx += tpg) {
        UnstackedSlice s = unstacked_cols[window_idx];

        uint32_t col_idx = s.stacked_col_idx;
        uint32_t row_idx = s.stacked_row_idx;
        uint32_t row_start = (row_idx >> shift_factor) << 1;
        uint32_t q_offset = col_idx * q_height + row_start;

        FpExt q_0 = q[q_offset];
        FpExt q_1 = q[q_offset + 1];
        FpExt q_c1 = q_1 - q_0;

        uint32_t b_bool = (row_idx >> (shift_factor - 1)) & 1;
        Fp b = Fp(b_bool);
        FpExt eq_ub = eq_ub_ptr[window_idx];

        for (uint32_t x_int = 1; x_int <= uint32_t(S_DEG); ++x_int) {
            Fp x = Fp(x_int);
            FpExt eq_ub_x = eq_ub * eq1(x, b);
            FpExt eq_val = eq_r * eq_ub_x;
            FpExt k_rot_val = k_rot_r * eq_ub_x;
            FpExt q_x = q_0 + q_c1 * x;

            FpExt contrib =
                (lambda_pows[2 * window_idx] * eq_val + lambda_pows[2 * window_idx + 1] * k_rot_val) *
                q_x;
            local_sums[x_int - 1] = local_sums[x_int - 1] + contrib;
        }
    }

    for (uint idx = 0; idx < S_DEG; ++idx) {
        FpExt reduced = block_reduce_sum(local_sums[idx], shared, tid, tpg);
        if (tid == 0) {
            atomic_add_fpext_to_u64(output + idx * (D_EF * 2), reduced);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Triangular sweep.
kernel void initialize_k_rot_from_eq_segments_kernel(
    const device FpExt *eq_r_ns                 [[buffer(0)]],
    device FpExt *k_rot_ns                      [[buffer(1)]],
    constant FpExt &k_rot_uni_0                 [[buffer(2)]],
    constant FpExt &k_rot_uni_1                 [[buffer(3)]],
    uint2 gid                                   [[thread_position_in_grid]]
) {
    uint32_t x = gid.x;
    uint32_t n = gid.y;
    uint32_t num_x = 1u << n;

    if (x >= num_x) {
        return;
    }

    FpExt eq_cube = get_eq_cube(eq_r_ns, num_x, x);
    FpExt k_rot_cube = get_eq_cube(eq_r_ns, num_x, rot_prev(x, num_x));
    k_rot_ns[num_x + x] = k_rot_uni_0 * eq_cube + k_rot_uni_1 * (k_rot_cube - eq_cube);
}

// Single-trace fold PLE kernel.
kernel void stacked_reduction_fold_ple_kernel(
    const device Fp *src                        [[buffer(0)]],
    device FpExt *dst                           [[buffer(1)]],
    const device Fp *omega_skip_pows            [[buffer(2)]],
    const device FpExt *inv_lagrange_denoms     [[buffer(3)]],
    constant uint32_t &trace_height             [[buffer(4)]],
    constant uint32_t &new_height               [[buffer(5)]],
    constant uint32_t &skip_domain              [[buffer(6)]],
    uint2 tid2                                  [[thread_position_in_threadgroup]],
    uint2 tpg2                                  [[threads_per_threadgroup]],
    uint2 group_id                              [[threadgroup_position_in_grid]]
) {
    uint tid = tid2.x;
    uint tpg = tpg2.x;
    uint32_t col_idx = group_id.y;
    uint32_t row_idx = group_id.x * tpg + tid;
    if (row_idx >= new_height) {
        return;
    }

    uint32_t src_len = min(trace_height, skip_domain);
    uint32_t stride = skip_domain / src_len;
    const device Fp *cell_src = src + col_idx * trace_height + row_idx * src_len;

    FpExt sum = zero_ext();
    for (uint32_t idx = 0; idx < src_len; ++idx) {
        FpExt eval_ext = FpExt(cell_src[idx]);
        FpExt omega_ext = FpExt(omega_skip_pows[idx * stride]);
        sum = sum + eval_ext * omega_ext * inv_lagrange_denoms[idx * stride];
    }

    dst[col_idx * new_height + row_idx] = sum;
}
