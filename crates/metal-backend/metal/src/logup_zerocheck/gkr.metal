// logup_zerocheck/gkr - GKR fractional sumcheck kernels
// Ported from CUDA: cuda-backend/cuda/src/logup_zerocheck/gkr.cu
#include <metal_stdlib>
using namespace metal;

#include "baby_bear.h"
#include "baby_bear_ext.h"
#include "frac_ext.h"
#include "sumcheck.h"

// Degree of s' polynomial (factored out the first eq term)
constant int GKR_SP_DEG = 2;

// Build fractional tree layer: combine pairs by frac_add or frac_unadd
kernel void frac_build_tree_layer_kernel(
    device FpExt *layer_p [[buffer(0)]],
    device FpExt *layer_q [[buffer(1)]],
    constant uint32_t &half_len [[buffer(2)]],
    constant uint32_t &do_revert [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= half_len) return;

    FpExt lhs_p = layer_p[idx];
    FpExt lhs_q = layer_q[idx];
    FpExt rhs_p = layer_p[idx + half_len];
    FpExt rhs_q = layer_q[idx + half_len];

    if (do_revert != 0) {
        // frac_unadd: find c such that frac_add(c, rhs) == lhs
        FpExt rhs_q_inv = inv(rhs_q);
        FpExt new_q = lhs_q * rhs_q_inv;
        FpExt new_p = (lhs_p - new_q * rhs_p) * rhs_q_inv;
        layer_p[idx] = new_p;
        layer_q[idx] = new_q;
    } else {
        // frac_add: lhs = lhs.p * rhs.q + lhs.q * rhs.p, lhs.q = lhs.q * rhs.q
        layer_p[idx] = lhs_p * rhs_q + lhs_q * rhs_p;
        layer_q[idx] = lhs_q * rhs_q;
    }
}

// Reconstruct eq weight from sqrt-decomposed buffers
inline FpExt sqrt_buffer_get(
    const device FpExt *eq_xi_low,
    const device FpExt *eq_xi_high,
    uint32_t log_eq_low_cap,
    uint32_t idx
) {
    return eq_xi_low[idx & ((1u << log_eq_low_cap) - 1)] * eq_xi_high[idx >> log_eq_low_cap];
}

// Helper for reversed bit indexing into pq buffer
inline uint32_t with_rev_bits(uint32_t idx, uint32_t pq_size, uint32_t bit0, uint32_t bit1) {
    // For 2-poly layout: pq_buffer index with specific bit pattern
    uint32_t half_sz = pq_size >> 1;
    uint32_t quarter = pq_size >> 2;
    return idx + bit0 * half_sz + bit1 * quarter;
}

// Compute round block sum: evaluates s' polynomial at points 1 and 2
kernel void compute_round_block_sum_kernel(
    const device FpExt *eq_xi_low [[buffer(0)]],
    const device FpExt *eq_xi_high [[buffer(1)]],
    const device FpExt *pq_p [[buffer(2)]],
    const device FpExt *pq_q [[buffer(3)]],
    device FpExt *block_sums [[buffer(4)]],
    constant uint32_t &num_x [[buffer(5)]],
    constant uint32_t &log_eq_low_cap [[buffer(6)]],
    constant FpExt &lambda [[buffer(7)]],
    threadgroup FpExt *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tg_grid_size [[threadgroups_per_grid]]
) {
    uint32_t pq_size = 2 * num_x;
    uint32_t half_sz = pq_size >> 1;
    uint32_t quarter = pq_size >> 2;

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt local0 = zero;
    FpExt local1 = zero;

    uint32_t grid_stride = tg_size * tg_grid_size;
    for (uint32_t idx = tid + gid * tg_size; idx < num_x / 2; idx += grid_stride) {
        FpExt eq_val = sqrt_buffer_get(eq_xi_low, eq_xi_high, log_eq_low_cap, idx);

        // Load p,q pairs for (poly0, even), (poly1, even), (poly0, odd), (poly1, odd)
        FpExt p0_even = pq_p[idx];
        FpExt q0_even = pq_q[idx];
        FpExt p1_even = pq_p[idx + half_sz];
        FpExt q1_even = pq_q[idx + half_sz];
        FpExt p0_odd = pq_p[idx + quarter];
        FpExt q0_odd = pq_q[idx + quarter];
        FpExt p1_odd = pq_p[idx + half_sz + quarter];
        FpExt q1_odd = pq_q[idx + half_sz + quarter];

        FpExt p0_diff = p0_odd - p0_even;
        FpExt q0_diff = q0_odd - q0_even;
        FpExt p1_diff = p1_odd - p1_even;
        FpExt q1_diff = q1_odd - q1_even;

        FpExt p_j0 = p0_even + lambda * q0_even;
        FpExt q_j0 = q0_even;
        FpExt p_j1 = p1_even;
        FpExt q_j1 = q1_even;

        FpExt lambda_times_q0_diff = lambda * q0_diff;

        // Evaluate at point 1
        p_j0 = p_j0 + p0_diff + lambda_times_q0_diff;
        q_j0 = q_j0 + q0_diff;
        p_j1 = p_j1 + p1_diff;
        q_j1 = q_j1 + q1_diff;
        local0 = local0 + eq_val * (p_j0 * q_j1 + p_j1 * q_j0);

        // Evaluate at point 2
        p_j0 = p_j0 + p0_diff + lambda_times_q0_diff;
        q_j0 = q_j0 + q0_diff;
        p_j1 = p_j1 + p1_diff;
        q_j1 = q_j1 + q1_diff;
        local1 = local1 + eq_val * (p_j0 * q_j1 + p_j1 * q_j0);
    }

    // Reduce and write block sums
    FpExt reduced0 = block_reduce_sum(local0, shared, tid, tg_size);
    if (tid == 0) {
        block_sums[gid * GKR_SP_DEG + 0] = reduced0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FpExt reduced1 = block_reduce_sum(local1, shared, tid, tg_size);
    if (tid == 0) {
        block_sums[gid * GKR_SP_DEG + 1] = reduced1;
    }
}

// Fold FpExt columns: interpolate pairs and write folded result
kernel void fold_ef_columns_kernel(
    const device FpExt *src [[buffer(0)]],
    device FpExt *dst [[buffer(1)]],
    constant uint32_t &quarter [[buffer(2)]],
    constant FpExt &r [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= quarter) return;

    FpExt v0 = src[idx];
    FpExt v1 = src[idx + quarter];
    dst[idx] = v0 + (v1 - v0) * r;

    uint32_t half_sz = quarter << 1;
    FpExt v2 = src[idx + half_sz];
    FpExt v3 = src[idx + half_sz + quarter];
    dst[idx + quarter] = v2 + (v3 - v2) * r;
}

// Fused compute round + fold kernel (out-of-place)
kernel void compute_round_and_fold_kernel(
    const device FpExt *eq_xi_low [[buffer(0)]],
    const device FpExt *eq_xi_high [[buffer(1)]],
    const device FpExt *src_pq_p [[buffer(2)]],
    const device FpExt *src_pq_q [[buffer(3)]],
    device FpExt *block_sums [[buffer(4)]],
    device FpExt *dst_pq_p [[buffer(5)]],
    device FpExt *dst_pq_q [[buffer(6)]],
    constant uint32_t &num_x [[buffer(7)]],
    constant uint32_t &log_eq_low_cap [[buffer(8)]],
    constant FpExt &lambda [[buffer(9)]],
    constant FpExt &r_prev [[buffer(10)]],
    threadgroup FpExt *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tg_grid_size [[threadgroups_per_grid]]
) {
    uint32_t pq_size = 2 * num_x;
    uint32_t half_sz = pq_size;
    uint32_t quarter = pq_size >> 1;
    uint32_t eighth = pq_size >> 2;
    uint32_t three_eighths = eighth * 3;

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt local0 = zero;
    FpExt local1 = zero;

    uint32_t grid_stride = tg_size * tg_grid_size;
    for (uint32_t idx = tid + gid * tg_size; idx < num_x / 2; idx += grid_stride) {
        // Load and fold pairs
        FpExt p0_even, q0_even, p1_even, q1_even;
        FpExt p0_odd, q0_odd, p1_odd, q1_odd;

        // f00
        {
            FpExt ap = src_pq_p[idx]; FpExt aq = src_pq_q[idx];
            FpExt bp = src_pq_p[idx + quarter]; FpExt bq = src_pq_q[idx + quarter];
            p0_even = ap + r_prev * (bp - ap);
            q0_even = aq + r_prev * (bq - aq);
            dst_pq_p[idx] = p0_even; dst_pq_q[idx] = q0_even;
        }
        // f10
        {
            FpExt ap = src_pq_p[idx + half_sz]; FpExt aq = src_pq_q[idx + half_sz];
            FpExt bp = src_pq_p[idx + half_sz + quarter]; FpExt bq = src_pq_q[idx + half_sz + quarter];
            p1_even = ap + r_prev * (bp - ap);
            q1_even = aq + r_prev * (bq - aq);
            dst_pq_p[idx + quarter] = p1_even; dst_pq_q[idx + quarter] = q1_even;
        }
        // f01
        {
            FpExt ap = src_pq_p[idx + eighth]; FpExt aq = src_pq_q[idx + eighth];
            FpExt bp = src_pq_p[idx + three_eighths]; FpExt bq = src_pq_q[idx + three_eighths];
            p0_odd = ap + r_prev * (bp - ap);
            q0_odd = aq + r_prev * (bq - aq);
            dst_pq_p[idx + eighth] = p0_odd; dst_pq_q[idx + eighth] = q0_odd;
        }
        // f11
        {
            FpExt ap = src_pq_p[idx + half_sz + eighth]; FpExt aq = src_pq_q[idx + half_sz + eighth];
            FpExt bp = src_pq_p[idx + half_sz + three_eighths]; FpExt bq = src_pq_q[idx + half_sz + three_eighths];
            p1_odd = ap + r_prev * (bp - ap);
            q1_odd = aq + r_prev * (bq - aq);
            dst_pq_p[idx + three_eighths] = p1_odd; dst_pq_q[idx + three_eighths] = q1_odd;
        }

        // Accumulate contributions
        FpExt eq_val = sqrt_buffer_get(eq_xi_low, eq_xi_high, log_eq_low_cap, idx);
        FpExt p0_diff = p0_odd - p0_even;
        FpExt q0_diff = q0_odd - q0_even;
        FpExt p1_diff = p1_odd - p1_even;
        FpExt q1_diff = q1_odd - q1_even;

        FpExt p_j0 = p0_even + lambda * q0_even;
        FpExt q_j0 = q0_even;
        FpExt p_j1 = p1_even;
        FpExt q_j1 = q1_even;
        FpExt lambda_times_q0_diff = lambda * q0_diff;

        // Point 1
        p_j0 = p_j0 + p0_diff + lambda_times_q0_diff;
        q_j0 = q_j0 + q0_diff;
        p_j1 = p_j1 + p1_diff;
        q_j1 = q_j1 + q1_diff;
        local0 = local0 + eq_val * (p_j0 * q_j1 + p_j1 * q_j0);

        // Point 2
        p_j0 = p_j0 + p0_diff + lambda_times_q0_diff;
        q_j0 = q_j0 + q0_diff;
        p_j1 = p_j1 + p1_diff;
        q_j1 = q_j1 + q1_diff;
        local1 = local1 + eq_val * (p_j0 * q_j1 + p_j1 * q_j0);
    }

    FpExt reduced0 = block_reduce_sum(local0, shared, tid, tg_size);
    if (tid == 0) block_sums[gid * GKR_SP_DEG + 0] = reduced0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FpExt reduced1 = block_reduce_sum(local1, shared, tid, tg_size);
    if (tid == 0) block_sums[gid * GKR_SP_DEG + 1] = reduced1;
}

// Fused compute round + fold kernel (in-place).
// Reads from pre-fold pq buffer (size 2*pq_size) and writes folded output
// back into the first pq_size entries.
kernel void compute_round_and_fold_inplace_kernel(
    const device FpExt *eq_xi_low [[buffer(0)]],
    const device FpExt *eq_xi_high [[buffer(1)]],
    device FpExt *pq_p [[buffer(2)]],
    device FpExt *pq_q [[buffer(3)]],
    device FpExt *block_sums [[buffer(4)]],
    constant uint32_t &num_x [[buffer(5)]],
    constant uint32_t &log_eq_low_cap [[buffer(6)]],
    constant FpExt &lambda [[buffer(7)]],
    constant FpExt &r_prev [[buffer(8)]],
    threadgroup FpExt *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tg_grid_size [[threadgroups_per_grid]]
) {
    uint32_t pq_size = 2 * num_x;
    uint32_t half_sz = pq_size;
    uint32_t quarter = pq_size >> 1;
    uint32_t eighth = pq_size >> 2;
    uint32_t three_eighths = eighth * 3;

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt local0 = zero;
    FpExt local1 = zero;

    uint32_t grid_stride = tg_size * tg_grid_size;
    for (uint32_t idx = tid + gid * tg_size; idx < num_x / 2; idx += grid_stride) {
        FpExt p0_even, q0_even, p1_even, q1_even;
        FpExt p0_odd, q0_odd, p1_odd, q1_odd;

        {
            FpExt ap = pq_p[idx]; FpExt aq = pq_q[idx];
            FpExt bp = pq_p[idx + quarter]; FpExt bq = pq_q[idx + quarter];
            p0_even = ap + r_prev * (bp - ap);
            q0_even = aq + r_prev * (bq - aq);
        }
        {
            FpExt ap = pq_p[idx + half_sz]; FpExt aq = pq_q[idx + half_sz];
            FpExt bp = pq_p[idx + half_sz + quarter]; FpExt bq = pq_q[idx + half_sz + quarter];
            p1_even = ap + r_prev * (bp - ap);
            q1_even = aq + r_prev * (bq - aq);
        }
        {
            FpExt ap = pq_p[idx + eighth]; FpExt aq = pq_q[idx + eighth];
            FpExt bp = pq_p[idx + three_eighths]; FpExt bq = pq_q[idx + three_eighths];
            p0_odd = ap + r_prev * (bp - ap);
            q0_odd = aq + r_prev * (bq - aq);
        }
        {
            FpExt ap = pq_p[idx + half_sz + eighth]; FpExt aq = pq_q[idx + half_sz + eighth];
            FpExt bp = pq_p[idx + half_sz + three_eighths]; FpExt bq = pq_q[idx + half_sz + three_eighths];
            p1_odd = ap + r_prev * (bp - ap);
            q1_odd = aq + r_prev * (bq - aq);
        }

        // Write folded values into the first half after all required reads.
        pq_p[idx] = p0_even; pq_q[idx] = q0_even;
        pq_p[idx + quarter] = p1_even; pq_q[idx + quarter] = q1_even;
        pq_p[idx + eighth] = p0_odd; pq_q[idx + eighth] = q0_odd;
        pq_p[idx + three_eighths] = p1_odd; pq_q[idx + three_eighths] = q1_odd;

        FpExt eq_val = sqrt_buffer_get(eq_xi_low, eq_xi_high, log_eq_low_cap, idx);
        FpExt p0_diff = p0_odd - p0_even;
        FpExt q0_diff = q0_odd - q0_even;
        FpExt p1_diff = p1_odd - p1_even;
        FpExt q1_diff = q1_odd - q1_even;

        FpExt p_j0 = p0_even + lambda * q0_even;
        FpExt q_j0 = q0_even;
        FpExt p_j1 = p1_even;
        FpExt q_j1 = q1_even;
        FpExt lambda_times_q0_diff = lambda * q0_diff;

        p_j0 = p_j0 + p0_diff + lambda_times_q0_diff;
        q_j0 = q_j0 + q0_diff;
        p_j1 = p_j1 + p1_diff;
        q_j1 = q_j1 + q1_diff;
        local0 = local0 + eq_val * (p_j0 * q_j1 + p_j1 * q_j0);

        p_j0 = p_j0 + p0_diff + lambda_times_q0_diff;
        q_j0 = q_j0 + q0_diff;
        p_j1 = p_j1 + p1_diff;
        q_j1 = q_j1 + q1_diff;
        local1 = local1 + eq_val * (p_j0 * q_j1 + p_j1 * q_j0);
    }

    FpExt reduced0 = block_reduce_sum(local0, shared, tid, tg_size);
    if (tid == 0) block_sums[gid * GKR_SP_DEG + 0] = reduced0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FpExt reduced1 = block_reduce_sum(local1, shared, tid, tg_size);
    if (tid == 0) block_sums[gid * GKR_SP_DEG + 1] = reduced1;
}

// Fused compute round + tree layer revert kernel (for first inner round)
kernel void compute_round_and_revert_kernel(
    const device FpExt *eq_xi_low [[buffer(0)]],
    const device FpExt *eq_xi_high [[buffer(1)]],
    device FpExt *layer_p [[buffer(2)]],
    device FpExt *layer_q [[buffer(3)]],
    device FpExt *block_sums [[buffer(4)]],
    constant uint32_t &num_x [[buffer(5)]],
    constant uint32_t &log_eq_low_cap [[buffer(6)]],
    constant FpExt &lambda [[buffer(7)]],
    threadgroup FpExt *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tg_grid_size [[threadgroups_per_grid]]
) {
    uint32_t pq_size = 2 * num_x;
    uint32_t half_sz = pq_size >> 1;
    uint32_t quarter = pq_size >> 2;

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt local0 = zero;
    FpExt local1 = zero;

    uint32_t grid_stride = tg_size * tg_grid_size;
    for (uint32_t idx = tid + gid * tg_size; idx < num_x / 2; idx += grid_stride) {
        FpExt p0_even, q0_even, p1_even, q1_even;
        FpExt p0_odd, q0_odd, p1_odd, q1_odd;

        // Revert: frac_unadd for first half entries
        {
            FpExt lp = layer_p[idx]; FpExt lq = layer_q[idx];
            FpExt rp = layer_p[idx + half_sz]; FpExt rq = layer_q[idx + half_sz];
            FpExt rq_inv = inv(rq);
            q0_even = lq * rq_inv;
            p0_even = (lp - q0_even * rp) * rq_inv;
            p1_even = rp; q1_even = rq;
            layer_p[idx] = p0_even; layer_q[idx] = q0_even;
        }
        {
            FpExt lp = layer_p[idx + quarter]; FpExt lq = layer_q[idx + quarter];
            FpExt rp = layer_p[idx + half_sz + quarter]; FpExt rq = layer_q[idx + half_sz + quarter];
            FpExt rq_inv = inv(rq);
            q0_odd = lq * rq_inv;
            p0_odd = (lp - q0_odd * rp) * rq_inv;
            p1_odd = rp; q1_odd = rq;
            layer_p[idx + quarter] = p0_odd; layer_q[idx + quarter] = q0_odd;
        }

        // Accumulate contributions
        FpExt eq_val = sqrt_buffer_get(eq_xi_low, eq_xi_high, log_eq_low_cap, idx);
        FpExt p0_diff = p0_odd - p0_even;
        FpExt q0_diff = q0_odd - q0_even;
        FpExt p1_diff = p1_odd - p1_even;
        FpExt q1_diff = q1_odd - q1_even;

        FpExt p_j0 = p0_even + lambda * q0_even;
        FpExt q_j0 = q0_even;
        FpExt p_j1 = p1_even;
        FpExt q_j1 = q1_even;
        FpExt lambda_times_q0_diff = lambda * q0_diff;

        p_j0 = p_j0 + p0_diff + lambda_times_q0_diff;
        q_j0 = q_j0 + q0_diff;
        p_j1 = p_j1 + p1_diff;
        q_j1 = q_j1 + q1_diff;
        local0 = local0 + eq_val * (p_j0 * q_j1 + p_j1 * q_j0);

        p_j0 = p_j0 + p0_diff + lambda_times_q0_diff;
        q_j0 = q_j0 + q0_diff;
        p_j1 = p_j1 + p1_diff;
        q_j1 = q_j1 + q1_diff;
        local1 = local1 + eq_val * (p_j0 * q_j1 + p_j1 * q_j0);
    }

    FpExt reduced0 = block_reduce_sum(local0, shared, tid, tg_size);
    if (tid == 0) block_sums[gid * GKR_SP_DEG + 0] = reduced0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FpExt reduced1 = block_reduce_sum(local1, shared, tid, tg_size);
    if (tid == 0) block_sums[gid * GKR_SP_DEG + 1] = reduced1;
}

// Add alpha to q component of FracExt buffer
kernel void add_alpha_kernel(
    device FpExt *data_q [[buffer(0)]],
    constant uint32_t &len [[buffer(1)]],
    constant FpExt &alpha [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < len) {
        data_q[idx] = data_q[idx] + alpha;
    }
}

// Multiply p component of FracExt vector by a scalar
kernel void frac_vector_scalar_multiply_kernel(
    device FpExt *frac_p [[buffer(0)]],
    constant Fp &scalar [[buffer(1)]],
    constant uint32_t &length [[buffer(2)]],
    uint tidx [[thread_position_in_grid]]
) {
    if (tidx >= length) return;
    frac_p[tidx] = frac_p[tidx] * scalar;
}

// Static final reduce for GKR: reduces block sums to output[0..GKR_SP_DEG-1]
kernel void static_final_reduce_block_sums_kernel(
    const device FpExt *block_sums [[buffer(0)]],
    device FpExt *output [[buffer(1)]],
    constant uint32_t &num_blocks [[buffer(2)]],
    threadgroup FpExt *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    uint d_idx = gid;
    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt sum = zero;

    for (uint b = tid; b < num_blocks; b += tg_size) {
        sum = sum + block_sums[b * GKR_SP_DEG + d_idx];
    }

    FpExt reduced = block_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) {
        output[d_idx] = reduced;
    }
}

// Multifold kernel: fold w rounds at once using precomputed eq_r_window
kernel void multifold_kernel(
    const device FpExt *src_p [[buffer(0)]],
    const device FpExt *src_q [[buffer(1)]],
    device FpExt *dst_p [[buffer(2)]],
    device FpExt *dst_q [[buffer(3)]],
    const device FpExt *eq_r_window [[buffer(4)]],
    constant uint32_t &tail_size [[buffer(5)]],
    constant uint32_t &beta_size [[buffer(6)]],
    uint out_idx [[thread_position_in_grid]]
) {
    if (out_idx >= tail_size) return;

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt acc0_p = zero, acc0_q = zero;
    FpExt acc1_p = zero, acc1_q = zero;

    uint32_t poly1_offset = tail_size * beta_size;
    for (uint32_t beta = 0; beta < beta_size; ++beta) {
        uint32_t idx = beta * tail_size + out_idx;
        FpExt v0p = src_p[idx]; FpExt v0q = src_q[idx];
        FpExt v1p = src_p[poly1_offset + idx]; FpExt v1q = src_q[poly1_offset + idx];
        FpExt eq_r = eq_r_window[beta];
        acc0_p = acc0_p + eq_r * v0p;
        acc0_q = acc0_q + eq_r * v0q;
        acc1_p = acc1_p + eq_r * v1p;
        acc1_q = acc1_q + eq_r * v1q;
    }
    dst_p[out_idx] = acc0_p; dst_q[out_idx] = acc0_q;
    dst_p[tail_size + out_idx] = acc1_p; dst_q[tail_size + out_idx] = acc1_q;
}

// Precompute M eval round: evaluate round polynomial from precomputed M matrix
kernel void precompute_m_eval_round_kernel(
    const device FpExt *m_total [[buffer(0)]],
    const device FpExt *eq_r_prefix [[buffer(1)]],
    const device FpExt *eq_suffix [[buffer(2)]],
    device FpExt *out [[buffer(3)]],
    constant uint32_t &w [[buffer(4)]],
    constant uint32_t &t [[buffer(5)]],
    threadgroup FpExt *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint32_t m = 1u << w;
    uint32_t prefix_bits = t;
    uint32_t suffix_bits = w - t - 1;
    uint32_t prefix_size = 1u << prefix_bits;
    uint32_t suffix_size = 1u << suffix_bits;

    uint64_t total = uint64_t(prefix_size) * uint64_t(prefix_size) * uint64_t(suffix_size);
    uint32_t cur_bit = 1u << suffix_bits;

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt one = FpExt(1u);
    FpExt two = one + one;

    FpExt local_s1 = zero;
    FpExt local_s2 = zero;

    for (uint64_t i = tid; i < total; i += tg_size) {
        uint32_t suffix = uint32_t(i % suffix_size);
        uint64_t tmp = i / suffix_size;
        uint32_t b2 = uint32_t(tmp % prefix_size);
        uint32_t b1 = uint32_t(tmp / prefix_size);

        FpExt weight = eq_r_prefix[b1] * eq_r_prefix[b2] * eq_suffix[suffix];

        uint32_t prefix_shift = suffix_bits + 1;
        uint32_t beta1_0 = (b1 << prefix_shift) | suffix;
        uint32_t beta1_1 = beta1_0 | cur_bit;
        uint32_t beta2_0 = (b2 << prefix_shift) | suffix;
        uint32_t beta2_1 = beta2_0 | cur_bit;

        FpExt m00 = m_total[beta1_0 * m + beta2_0];
        FpExt m01 = m_total[beta1_0 * m + beta2_1];
        FpExt m10 = m_total[beta1_1 * m + beta2_0];
        FpExt m11 = m_total[beta1_1 * m + beta2_1];

        local_s1 = local_s1 + weight * m11;
        local_s2 = local_s2 + weight * (m00 - two * (m01 + m10 - m11 - m11));
    }

    FpExt reduced_s1 = block_reduce_sum(local_s1, shared, tid, tg_size);
    if (tid == 0) out[0] = reduced_s1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FpExt reduced_s2 = block_reduce_sum(local_s2, shared, tid, tg_size);
    if (tid == 0) out[1] = reduced_s2;
}

// Reduce partial M blocks into final M matrix
kernel void precompute_m_reduce_partials_kernel(
    const device FpExt *partial [[buffer(0)]],
    device FpExt *m_total [[buffer(1)]],
    constant uint32_t &num_blocks [[buffer(2)]],
    constant uint32_t &total_entries [[buffer(3)]],
    uint entry [[thread_position_in_grid]]
) {
    if (entry >= total_entries) return;

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt acc = zero;
    for (uint32_t b = 0; b < num_blocks; ++b) {
        acc = acc + partial[uint64_t(b) * total_entries + entry];
    }
    m_total[entry] = acc;
}
