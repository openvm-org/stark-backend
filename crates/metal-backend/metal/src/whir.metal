// WHIR polynomial commitment kernels for Metal
// Translated from CUDA: cuda-backend/cuda/src/whir.cu
#include <metal_stdlib>
using namespace metal;

#include "../include/baby_bear.h"
#include "../include/baby_bear_ext.h"
#include "../include/sumcheck.h"

constant int S_DEG = 2;

struct BatchingTracePacket {
    uint64_t ptr; // device pointer to Fp trace data
    uint32_t height;
    uint32_t width;
    uint32_t stacked_row_start;
    uint32_t mu_idx;
};

// Algebraically batch unstacked traces
kernel void whir_algebraic_batch_traces_kernel(
    device Fp *output                              [[buffer(0)]],
    const device BatchingTracePacket *packets       [[buffer(1)]],
    const device FpExt *mu_powers                  [[buffer(2)]],
    constant uint32_t &stacked_height              [[buffer(3)]],
    constant uint32_t &num_packets                 [[buffer(4)]],
    constant uint32_t &skip_domain                 [[buffer(5)]],
    uint gid                                        [[thread_position_in_grid]]
) {
    uint32_t row = gid;
    if (row >= stacked_height) return;

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt res = zero;

    for (uint32_t idx = 0; idx < num_packets; idx++) {
        BatchingTracePacket packet = packets[idx];
        const device Fp *trace = reinterpret_cast<const device Fp *>(packet.ptr);
        uint32_t h = packet.height;
        uint32_t lifted_height = max(h, skip_domain);
        uint32_t w = packet.width;
        uint32_t row_start = packet.stacked_row_start;
        uint32_t mu_idx_start = packet.mu_idx;
        uint32_t stride = max(skip_domain / h, 1u);

        uint32_t stacked_end = row_start + lifted_height * w;
        if (row >= stacked_end) continue;

        uint32_t offset_start = row_start <= row ? 0 : 1;
        for (uint32_t row_offset = offset_start * stacked_height + row; row_offset < stacked_end;
             row_offset += stacked_height) {
            uint32_t offset = (row_offset - row) / stacked_height;
            uint32_t tmp = row_offset - row_start;
            uint32_t trace_col = tmp / lifted_height;
            uint32_t strided_trace_row = tmp % lifted_height;
            Fp trace_val = (strided_trace_row % stride == 0)
                               ? trace[trace_col * h + (strided_trace_row / stride)]
                               : Fp(0u);
            FpExt mu_pow = mu_powers[mu_idx_start + offset];
            res = res + mu_pow * trace_val;
        }
    }

    for (uint i = 0; i < 4; i++) {
        output[i * stacked_height + row] = res.elems[i];
    }
}

// WHIR sumcheck round using coefficient/moment form
kernel void whir_sumcheck_coeff_moments_round_kernel(
    const device FpExt *f_coeffs           [[buffer(0)]],
    const device FpExt *w_moments          [[buffer(1)]],
    device FpExt *block_sums               [[buffer(2)]],
    constant uint32_t &height              [[buffer(3)]],
    threadgroup FpExt *shared              [[threadgroup(0)]],
    uint tid                                [[thread_index_in_threadgroup]],
    uint tpg                                [[threads_per_threadgroup]],
    uint group_id                           [[threadgroup_position_in_grid]],
    uint grid_size                          [[threadgroups_per_grid]]
) {
    uint32_t half_height = height >> 1;
    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt local_sums_0 = zero;
    FpExt local_sums_1 = zero;

    for (uint32_t y = group_id * tpg + tid; y < half_height; y += grid_size * tpg) {
        uint32_t idx0 = y << 1;
        uint32_t idx1 = idx0 + 1;

        FpExt c0 = f_coeffs[idx0];
        FpExt c1 = f_coeffs[idx1];
        FpExt m0 = w_moments[idx0];
        FpExt m1 = w_moments[idx1];

        // X = 1: f_1 = c0 + c1, w_1 = m1
        FpExt f_1 = c0 + c1;
        local_sums_0 = local_sums_0 + f_1 * m1;

        // X = 2: f_2 = c0 + 2*c1, w_2 = -m0 + 3*m1
        FpExt f_2 = c0 + c1 + c1;
        FpExt fp3 = FpExt{Fp(3u), Fp(0u), Fp(0u), Fp(0u)};
        FpExt m_2 = m1 * fp3 - m0;
        local_sums_1 = local_sums_1 + f_2 * m_2;
    }

    for (uint idx = 0; idx < S_DEG; idx++) {
        FpExt val = (idx == 0) ? local_sums_0 : local_sums_1;
        FpExt reduced = block_reduce_sum(val, shared, tid, tpg);
        if (tid == 0) {
            block_sums[group_id * S_DEG + idx] = reduced;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Fold both f (coefficients) and w (moments)
kernel void whir_fold_coeffs_and_moments_kernel(
    const device FpExt *f_coeffs           [[buffer(0)]],
    const device FpExt *w_moments          [[buffer(1)]],
    device FpExt *f_folded_coeffs          [[buffer(2)]],
    device FpExt *w_folded_moments         [[buffer(3)]],
    constant FpExt &alpha                  [[buffer(4)]],
    constant uint32_t &half_height         [[buffer(5)]],
    uint gid                                [[thread_position_in_grid]]
) {
    if (gid >= half_height) return;

    uint32_t idx0 = gid << 1;
    uint32_t idx1 = idx0 + 1;

    FpExt c0 = f_coeffs[idx0];
    FpExt c1 = f_coeffs[idx1];
    f_folded_coeffs[gid] = c0 + alpha * c1;

    FpExt m0 = w_moments[idx0];
    FpExt m1 = w_moments[idx1];
    FpExt one = FpExt{Fp(1u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt one_minus_alpha = one - alpha;
    FpExt two_alpha_minus_one = alpha + alpha - one;
    w_folded_moments[gid] = one_minus_alpha * m0 + two_alpha_minus_one * m1;
}

// Power from squared powers (extension field)
inline FpExt whir_pow_from_pows2_ext(
    const device FpExt *pows2,
    uint32_t log_height,
    uint32_t exponent
) {
    FpExt one = FpExt{Fp(1u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt acc = one;
    for (uint32_t bit = 0; bit < log_height; bit++) {
        if (exponent & (1u << bit)) {
            acc = acc * pows2[bit];
        }
    }
    return acc;
}

// Power from squared powers (base field)
inline Fp whir_pow_from_pows2_base(
    const device Fp *pows2,
    uint32_t log_height,
    uint32_t exponent
) {
    Fp acc = Fp(1u);
    for (uint32_t bit = 0; bit < log_height; bit++) {
        if (exponent & (1u << bit)) {
            acc = acc * pows2[bit];
        }
    }
    return acc;
}

// Accumulate w-moments from query points
kernel void w_moments_accumulate_kernel(
    device FpExt *w_moments                [[buffer(0)]],
    const device FpExt *z0_pows2           [[buffer(1)]],
    const device Fp *z_pows2               [[buffer(2)]],
    constant FpExt &gamma                  [[buffer(3)]],
    constant uint32_t &num_queries         [[buffer(4)]],
    constant uint32_t &log_height          [[buffer(5)]],
    constant uint32_t &height              [[buffer(6)]],
    uint gid                                [[thread_position_in_grid]]
) {
    if (gid >= height) return;

    uint32_t exponent = gid;

    FpExt acc = gamma * whir_pow_from_pows2_ext(z0_pows2, log_height, exponent);
    FpExt gamma_pow = gamma;
    for (uint32_t i = 0; i < num_queries; i++) {
        gamma_pow = gamma_pow * gamma;
        const device Fp *query_pows2 = z_pows2 + i * log_height;
        Fp z_i_pow = whir_pow_from_pows2_base(query_pows2, log_height, exponent);
        FpExt z_i_ext = FpExt{z_i_pow, Fp(0u), Fp(0u), Fp(0u)};
        acc = acc + gamma_pow * z_i_ext;
    }
    w_moments[gid] = w_moments[gid] + acc;
}
