// logup_zerocheck/zerocheck_round0 - NTT-based zerocheck round0 constraint evaluation
// Ported from CUDA: cuda-backend/cuda/src/logup_zerocheck/zerocheck_round0.cu
//
// The CUDA version uses C++ templates (NttEvalContext<NUM_COSETS>, ntt_eval_dag_entry, etc.)
// for compile-time specialization over NUM_COSETS, GLOBAL, NEEDS_SHMEM.
// Metal does not support templates in compute kernels, so we implement:
// - NTT coset interpolation as inline functions
// - Separate kernel variants for different configurations
// - The Rust FFI layer selects the appropriate kernel at dispatch time
#include <metal_stdlib>
using namespace metal;

#include "baby_bear.h"
#include "baby_bear_ext.h"
#include "codec.h"
#include "device_ntt.h"
#include "sumcheck.h"
#include "utils.h"

// ============================================================================
// NTT COSET EVALUATION HELPERS
// ============================================================================

// Given x and n (power of 2), computes 1/n * (1 + x + ... + x^{n-1}).
// This is the average of a geometric progression.
inline Fp avg_gp(Fp x, uint32_t n) {
    Fp res = Fp(1);
    for (uint32_t i = 1; i < n; i <<= 1) {
        res = res * (Fp(1) + x);
        res = res.halve();
        x = x * x;
    }
    return res;
}

// NTT coset interpolation with threadgroup memory (skip_domain > 32).
inline Fp ntt_coset_interpolate_tg(
    const device Fp *evals,
    const device Fp *twiddles,
    Fp omega_shift,
    threadgroup Fp *ntt_buffer,
    uint32_t ntt_idx,
    uint32_t x_int,
    uint32_t skip_domain,
    uint32_t height,
    uint8_t offset,
    bool skip_ntt
) {
    uint32_t l_skip = accel_ffs(skip_domain) - 1;
    uint32_t base = x_int * skip_domain;
    uint32_t idx = (base + ntt_idx + offset) & (height - 1);
    Fp coeff = evals[idx];

    if (skip_ntt) {
        return coeff;
    }

    // iNTT: natural to bit-reversed
    ntt_buffer[ntt_idx] = coeff;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ntt_natural_to_bitrev<true, true>(coeff, ntt_buffer, twiddles, ntt_idx, l_skip, true);

    // Apply shift
    Fp shifted = coeff * omega_shift;

    // Forward NTT: bit-reversed to natural
    ntt_bitrev_to_natural<false, true>(shifted, ntt_buffer, twiddles, ntt_idx, l_skip);

    return shifted;
}

// NTT coset interpolation with SIMD only (skip_domain <= 32).
inline Fp ntt_coset_interpolate_simd(
    const device Fp *evals,
    const device Fp *twiddles,
    Fp omega_shift,
    uint32_t ntt_idx,
    uint32_t x_int,
    uint32_t skip_domain,
    uint32_t height,
    uint8_t offset,
    bool skip_ntt
) {
    uint32_t l_skip = accel_ffs(skip_domain) - 1;
    uint32_t base = x_int * skip_domain;
    uint32_t idx = (base + ntt_idx + offset) & (height - 1);
    Fp coeff = evals[idx];

    if (skip_ntt) {
        return coeff;
    }

    ntt_natural_to_bitrev<true, false>(coeff, nullptr, twiddles, ntt_idx, l_skip, true);
    Fp shifted = coeff * omega_shift;
    ntt_bitrev_to_natural<false, false>(shifted, nullptr, twiddles, ntt_idx, l_skip);

    return shifted;
}

// NTT-based DAG entry evaluation for a single coset.
inline Fp ntt_eval_dag_entry(
    SourceInfo src,
    const device Fp *preprocessed,
    const device uint64_t *main_parts_ptrs,
    const device Fp *public_values,
    const device Fp *twiddles,
    thread Fp *inter_buffer,
    uint32_t buffer_stride,
    uint32_t buffer_size,
    threadgroup Fp *ntt_buffer,
    Fp omega_shift,
    uint32_t ntt_idx,
    uint32_t x_int,
    uint32_t skip_domain,
    uint32_t height,
    Fp is_first,
    Fp is_last,
    bool needs_tg_mem,
    bool skip_ntt
) {
    switch (src.type) {
    case ENTRY_PREPROCESSED: {
        const device Fp *col = preprocessed + height * src.index;
        if (needs_tg_mem) {
            return ntt_coset_interpolate_tg(col, twiddles, omega_shift, ntt_buffer, ntt_idx, x_int, skip_domain, height, src.offset, skip_ntt);
        } else {
            return ntt_coset_interpolate_simd(col, twiddles, omega_shift, ntt_idx, x_int, skip_domain, height, src.offset, skip_ntt);
        }
    }
    case ENTRY_MAIN: {
        const device Fp *col = reinterpret_cast<const device Fp *>(main_parts_ptrs[src.part]) + height * src.index;
        if (needs_tg_mem) {
            return ntt_coset_interpolate_tg(col, twiddles, omega_shift, ntt_buffer, ntt_idx, x_int, skip_domain, height, src.offset, skip_ntt);
        } else {
            return ntt_coset_interpolate_simd(col, twiddles, omega_shift, ntt_idx, x_int, skip_domain, height, src.offset, skip_ntt);
        }
    }
    case ENTRY_PUBLIC:
        return public_values[src.index];
    case SRC_CONSTANT:
        return Fp(src.index);
    case SRC_INTERMEDIATE:
        return inter_buffer[src.index * buffer_stride];
    case SRC_IS_FIRST:
        return is_first;
    case SRC_IS_LAST:
        return is_last;
    case SRC_IS_TRANSITION:
        return Fp(1) - is_last;
    default:
        break;
    }
    return Fp(0);
}

// ============================================================================
// KERNELS
// ============================================================================

// Fold selectors for round0: multiply base field selectors by extension field multipliers.
kernel void fold_selectors_round0_kernel(
    device FpExt *out [[buffer(0)]],
    const device Fp *in [[buffer(1)]],
    constant FpExt &is_first [[buffer(2)]],
    constant FpExt &is_last [[buffer(3)]],
    constant uint32_t &num_x [[buffer(4)]],
    uint tidx [[thread_position_in_grid]]
) {
    if (tidx >= num_x) return;

    out[tidx] = is_first * in[tidx];                             // is_first
    out[2 * num_x + tidx] = is_last * in[2 * num_x + tidx];      // is_last
    out[num_x + tidx] = FpExt(1u) - out[2 * num_x + tidx];       // is_transition
}

// Coset-parallel zerocheck round0 kernel.
// Each threadgroup handles ONE coset (identified by coset_idx).
// Thread layout: ntt_idx = tidx % skip_domain, x_int = tidx / skip_domain
kernel void zerocheck_ntt_eval_constraints_kernel(
    device FpExt *tmp_sums_buffer [[buffer(0)]],
    const device Fp *selectors_cube [[buffer(1)]],
    const device Fp *preprocessed [[buffer(2)]],
    const device uint64_t *main_parts_ptrs [[buffer(3)]],
    const device FpExt *eq_cube [[buffer(4)]],
    const device FpExt *d_lambda_pows [[buffer(5)]],
    const device Fp *public_values [[buffer(6)]],
    const device Rule *d_rules [[buffer(7)]],
    const device uint64_t *d_used_nodes [[buffer(8)]],
    const device Fp *twiddles [[buffer(9)]],
    constant uint32_t &rules_len [[buffer(10)]],
    constant uint32_t &used_nodes_len [[buffer(11)]],
    constant uint32_t &lambda_len [[buffer(12)]],
    constant uint32_t &buffer_size [[buffer(13)]],
    constant uint32_t &skip_domain [[buffer(14)]],
    constant uint32_t &num_x [[buffer(15)]],
    constant uint32_t &height [[buffer(16)]],
    constant uint32_t &coset_idx [[buffer(17)]],
    constant Fp &g_shift [[buffer(18)]],
    constant uint32_t &needs_tg_mem_flag [[buffer(19)]],
    threadgroup FpExt *shared_sum [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid_x [[threadgroup_position_in_grid]]
) {
    bool needs_tg_mem = (needs_tg_mem_flag != 0);
    uint32_t l_skip = accel_ffs(skip_domain) - 1;

    uint32_t tidx = tid + gid_x * tg_size;
    uint32_t ntt_idx = tidx & (skip_domain - 1);
    uint32_t x_int_base = tidx >> l_skip;

    uint32_t ntt_idx_rev = rev_len(ntt_idx, l_skip);

    uint32_t log_height_total = accel_ffs(height) - 1;
    uint32_t log_segment = min(l_skip, log_height_total);
    uint32_t segment_size = 1u << log_segment;
    uint32_t log_stride = l_skip - log_segment;

    Fp eta = TWO_ADIC_GENERATORS[l_skip - log_stride];
    Fp omega_skip_ntt = (l_skip == 0) ? Fp(1) : get_twiddle(twiddles, l_skip, ntt_idx);

    // g_coset = g_shift^(coset_idx + 1)
    Fp g_coset = pow(g_shift, coset_idx + 1);
    Fp eval_point = g_coset * omega_skip_ntt;
    Fp omega = exp_power_of_2(eval_point, log_stride);
    Fp is_first_mult = avg_gp(omega, segment_size);
    Fp is_last_mult = avg_gp(omega * eta, segment_size);
    Fp omega_shift = pow(g_coset, ntt_idx_rev);

    // Intermediate buffer (local)
    Fp local_buffer[16];
    thread Fp *inter_buffer = local_buffer;
    uint32_t buffer_stride_val = 1;

    FpExt sum = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};

    uint32_t x_int_stride = tg_size / skip_domain;

    // NTT buffer in threadgroup memory (after shared_sum area)
    threadgroup Fp *ntt_buffer = needs_tg_mem ?
        reinterpret_cast<threadgroup Fp *>(shared_sum + tg_size) + (tid >> l_skip) * skip_domain
        : nullptr;

    for (uint32_t x_int = x_int_base; x_int < num_x; x_int += x_int_stride) {
        Fp is_first = is_first_mult * selectors_cube[x_int];
        Fp is_last = is_last_mult * selectors_cube[2 * num_x + x_int];

        uint32_t lambda_idx = 0;
        FpExt constraint_sum = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};

        for (uint32_t node = 0; node < rules_len; ++node) {
            Rule rule = d_rules[node];
            RuleHeader header = decode_rule_header(rule);

            Fp x_val = ntt_eval_dag_entry(
                header.x, preprocessed, main_parts_ptrs, public_values, twiddles,
                inter_buffer, buffer_stride_val, buffer_size,
                ntt_buffer, omega_shift, ntt_idx, x_int, skip_domain, height,
                is_first, is_last, needs_tg_mem, false
            );

            Fp result;
            switch (header.op) {
            case OP_ADD: {
                SourceInfo y_src = decode_y(rule);
                Fp y_val = ntt_eval_dag_entry(
                    y_src, preprocessed, main_parts_ptrs, public_values, twiddles,
                    inter_buffer, buffer_stride_val, buffer_size,
                    ntt_buffer, omega_shift, ntt_idx, x_int, skip_domain, height,
                    is_first, is_last, needs_tg_mem, false
                );
                result = x_val + y_val;
                break;
            }
            case OP_SUB: {
                SourceInfo y_src = decode_y(rule);
                Fp y_val = ntt_eval_dag_entry(
                    y_src, preprocessed, main_parts_ptrs, public_values, twiddles,
                    inter_buffer, buffer_stride_val, buffer_size,
                    ntt_buffer, omega_shift, ntt_idx, x_int, skip_domain, height,
                    is_first, is_last, needs_tg_mem, false
                );
                result = x_val - y_val;
                break;
            }
            case OP_MUL: {
                SourceInfo y_src = decode_y(rule);
                Fp y_val = ntt_eval_dag_entry(
                    y_src, preprocessed, main_parts_ptrs, public_values, twiddles,
                    inter_buffer, buffer_stride_val, buffer_size,
                    ntt_buffer, omega_shift, ntt_idx, x_int, skip_domain, height,
                    is_first, is_last, needs_tg_mem, false
                );
                result = x_val * y_val;
                break;
            }
            case OP_NEG:
                result = -x_val;
                break;
            case OP_VAR:
                result = x_val;
                break;
            case OP_INV:
                result = inv(x_val);
                break;
            default:
                result = Fp(0);
                break;
            }

            if (header.buffer_result && buffer_size > 0) {
                uint32_t z_index = decode_z_index(rule);
                inter_buffer[z_index * buffer_stride_val] = result;
            }

            if (header.is_constraint) {
                while (lambda_idx < lambda_len && lambda_idx < used_nodes_len &&
                       d_used_nodes[lambda_idx] == node) {
                    FpExt lambda = d_lambda_pows[lambda_idx];
                    lambda_idx++;
                    constraint_sum = constraint_sum + lambda * result;
                }
            }
        }

        sum = sum + constraint_sum * eq_cube[x_int];
    }

    // Divide by zerofier
    Fp zerofier = exp_power_of_2(eval_point, l_skip) - Fp(1);
    shared_sum[tid] = sum * FpExt(inv(zerofier));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction: threads with same ntt_idx sum their results
    if (tid < skip_domain) {
        FpExt tile_sum = shared_sum[tid];
        for (uint32_t lane = 1; lane < (tg_size >> l_skip); ++lane) {
            tile_sum = tile_sum + shared_sum[(lane << l_skip) + tid];
        }
        tmp_sums_buffer[gid_x * skip_domain + ntt_idx] = tile_sum;
    }
}

// Final reduction for round0: sum partial block sums.
// Grid: (d, 1) where d = num_cosets * skip_domain
kernel void zerocheck_r0_final_reduce_kernel(
    const device FpExt *block_sums [[buffer(0)]],
    device FpExt *output [[buffer(1)]],
    constant uint32_t &num_blocks [[buffer(2)]],
    constant uint32_t &stride [[buffer(3)]],
    threadgroup FpExt *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt sum = zero;

    for (uint32_t b = tid; b < num_blocks; b += tg_size) {
        sum = sum + block_sums[b * stride + gid];
    }

    FpExt reduced = block_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) {
        output[gid] = reduced;
    }
}
