// logup_zerocheck/logup_round0 - NTT-based logup round0 interaction evaluation
// Ported from CUDA: cuda-backend/cuda/src/logup_zerocheck/logup_round0.cu
//
// Similar to zerocheck_round0 but evaluates logup interactions (numerator + denominator)
// instead of zerocheck constraints. Uses the same NTT coset evaluation infrastructure.
#include <metal_stdlib>
using namespace metal;

#include "baby_bear.h"
#include "baby_bear_ext.h"
#include "codec.h"
#include "device_ntt.h"
#include "frac_ext.h"
#include "sumcheck.h"
#include "utils.h"

// ============================================================================
// NTT COSET EVALUATION HELPERS
// ============================================================================

// Given x and n (power of 2), computes 1/n * (1 + x + ... + x^{n-1}).
inline Fp logup_avg_gp(Fp x, uint32_t n) {
    Fp res = Fp(1);
    for (uint32_t i = 1; i < n; i <<= 1) {
        res = res * (Fp(1) + x);
        res = res.halve();
        x = x * x;
    }
    return res;
}

// NTT coset interpolation with threadgroup memory
inline Fp logup_ntt_coset_interpolate_tg(
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

    ntt_buffer[ntt_idx] = coeff;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ntt_natural_to_bitrev<true, true>(coeff, ntt_buffer, twiddles, ntt_idx, l_skip, true);

    Fp shifted = coeff * omega_shift;
    ntt_bitrev_to_natural<false, true>(shifted, ntt_buffer, twiddles, ntt_idx, l_skip);

    return shifted;
}

// NTT coset interpolation with SIMD only (skip_domain <= 32)
inline Fp logup_ntt_coset_interpolate_simd(
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

// NTT-based DAG entry evaluation for logup
inline Fp logup_ntt_eval_dag_entry(
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
            return logup_ntt_coset_interpolate_tg(col, twiddles, omega_shift, ntt_buffer, ntt_idx, x_int, skip_domain, height, src.offset, skip_ntt);
        } else {
            return logup_ntt_coset_interpolate_simd(col, twiddles, omega_shift, ntt_idx, x_int, skip_domain, height, src.offset, skip_ntt);
        }
    }
    case ENTRY_MAIN: {
        const device Fp *col = reinterpret_cast<const device Fp *>(main_parts_ptrs[src.part]) + height * src.index;
        if (needs_tg_mem) {
            return logup_ntt_coset_interpolate_tg(col, twiddles, omega_shift, ntt_buffer, ntt_idx, x_int, skip_domain, height, src.offset, skip_ntt);
        } else {
            return logup_ntt_coset_interpolate_simd(col, twiddles, omega_shift, ntt_idx, x_int, skip_domain, height, src.offset, skip_ntt);
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
// KERNEL
// ============================================================================

// Coset-parallel logup round0 kernel.
// Each threadgroup handles ONE coset. The Rust FFI dispatches across cosets.
// Evaluates interaction DAG to produce numerator and denominator sums.
kernel void logup_r0_ntt_eval_interactions_kernel(
    device FpExt *tmp_sums_p [[buffer(0)]],
    device FpExt *tmp_sums_q [[buffer(1)]],
    const device Fp *selectors_cube [[buffer(2)]],
    const device Fp *preprocessed [[buffer(3)]],
    const device uint64_t *main_parts_ptrs [[buffer(4)]],
    const device FpExt *eq_cube [[buffer(5)]],
    const device Fp *public_values [[buffer(6)]],
    const device FpExt *numer_weights [[buffer(7)]],
    const device FpExt *denom_weights [[buffer(8)]],
    constant FpExt &denom_sum_init [[buffer(9)]],
    const device Rule *d_rules [[buffer(10)]],
    const device Fp *twiddles [[buffer(11)]],
    constant uint32_t &rules_len [[buffer(12)]],
    constant uint32_t &buffer_size [[buffer(13)]],
    constant uint32_t &skip_domain [[buffer(14)]],
    constant uint32_t &num_x [[buffer(15)]],
    constant uint32_t &height [[buffer(16)]],
    constant uint32_t &coset_idx [[buffer(17)]],
    constant Fp &g_shift [[buffer(18)]],
    constant uint32_t &needs_tg_mem_flag [[buffer(19)]],
    constant uint32_t &is_identity_coset_flag [[buffer(20)]],
    threadgroup FpExt *shared_sum [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid_x [[threadgroup_position_in_grid]]
) {
    bool needs_tg_mem = (needs_tg_mem_flag != 0);
    bool is_identity_coset = (is_identity_coset_flag != 0);
    uint32_t l_skip = accel_ffs(skip_domain) - 1;

    uint32_t tidx = tid + gid_x * tg_size;
    uint32_t ntt_idx = tidx & (skip_domain - 1);
    uint32_t x_int_base = tidx >> l_skip;

    uint32_t ntt_idx_rev = rev_len(ntt_idx, l_skip);
    Fp omega_skip = TWO_ADIC_GENERATORS[l_skip];

    uint32_t log_height_total = accel_ffs(height) - 1;
    uint32_t log_segment = min(l_skip, log_height_total);
    uint32_t segment_size = 1u << log_segment;
    uint32_t log_stride = l_skip - log_segment;

    Fp eta = TWO_ADIC_GENERATORS[l_skip - log_stride];
    Fp omega_skip_ntt = pow(omega_skip, ntt_idx);

    Fp g_coset = is_identity_coset ? Fp(1) : pow(g_shift, coset_idx);
    Fp eval_point = is_identity_coset ? omega_skip_ntt : (g_coset * omega_skip_ntt);
    Fp omega = exp_power_of_2(eval_point, log_stride);
    Fp is_first_mult = logup_avg_gp(omega, segment_size);
    Fp is_last_mult = logup_avg_gp(omega * eta, segment_size);
    Fp omega_shift = is_identity_coset ? Fp(1) : pow(g_coset, ntt_idx_rev);

    Fp local_buffer[16];
    thread Fp *inter_buffer = local_buffer;
    uint32_t buffer_stride_val = 1;

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt numer_sum = zero;
    FpExt denom_sum = zero;

    uint32_t x_int_stride = tg_size / skip_domain;

    // NTT buffer in threadgroup memory (after shared_sum)
    threadgroup Fp *ntt_buffer = needs_tg_mem ?
        reinterpret_cast<threadgroup Fp *>(shared_sum + tg_size) + (tid >> l_skip) * skip_domain
        : nullptr;

    bool skip_ntt = is_identity_coset;

    for (uint32_t x_int = x_int_base; x_int < num_x; x_int += x_int_stride) {
        Fp is_first = is_first_mult * selectors_cube[x_int];
        Fp is_last = is_last_mult * selectors_cube[2 * num_x + x_int];

        FpExt numer_result = zero;
        FpExt denom_result = zero;

        for (uint32_t node = 0; node < rules_len; ++node) {
            Rule rule = d_rules[node];
            RuleHeader header = decode_rule_header(rule);

            Fp x_val = logup_ntt_eval_dag_entry(
                header.x, preprocessed, main_parts_ptrs, public_values, twiddles,
                inter_buffer, buffer_stride_val, buffer_size,
                ntt_buffer, omega_shift, ntt_idx, x_int, skip_domain, height,
                is_first, is_last, needs_tg_mem, skip_ntt
            );

            Fp result;
            switch (header.op) {
            case OP_ADD: {
                SourceInfo y_src = decode_y(rule);
                Fp y_val = logup_ntt_eval_dag_entry(
                    y_src, preprocessed, main_parts_ptrs, public_values, twiddles,
                    inter_buffer, buffer_stride_val, buffer_size,
                    ntt_buffer, omega_shift, ntt_idx, x_int, skip_domain, height,
                    is_first, is_last, needs_tg_mem, skip_ntt
                );
                result = x_val + y_val;
                break;
            }
            case OP_SUB: {
                SourceInfo y_src = decode_y(rule);
                Fp y_val = logup_ntt_eval_dag_entry(
                    y_src, preprocessed, main_parts_ptrs, public_values, twiddles,
                    inter_buffer, buffer_stride_val, buffer_size,
                    ntt_buffer, omega_shift, ntt_idx, x_int, skip_domain, height,
                    is_first, is_last, needs_tg_mem, skip_ntt
                );
                result = x_val - y_val;
                break;
            }
            case OP_MUL: {
                SourceInfo y_src = decode_y(rule);
                Fp y_val = logup_ntt_eval_dag_entry(
                    y_src, preprocessed, main_parts_ptrs, public_values, twiddles,
                    inter_buffer, buffer_stride_val, buffer_size,
                    ntt_buffer, omega_shift, ntt_idx, x_int, skip_domain, height,
                    is_first, is_last, needs_tg_mem, skip_ntt
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
                FpExt numer_w = numer_weights[node];
                FpExt denom_w = denom_weights[node];
                numer_result = numer_result + numer_w * result;
                denom_result = denom_result + denom_w * result;
            }
        }

        FpExt eq = eq_cube[x_int];
        numer_sum = numer_sum + eq * numer_result;
        denom_sum = denom_sum + eq * (denom_result + denom_sum_init);
    }

    // Reduction for numerator
    shared_sum[tid] = numer_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < skip_domain) {
        FpExt tile_sum = shared_sum[tid];
        for (uint32_t lane = 1; lane < (tg_size >> l_skip); ++lane) {
            tile_sum = tile_sum + shared_sum[(lane << l_skip) + tid];
        }
        tmp_sums_p[gid_x * skip_domain + ntt_idx] = tile_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction for denominator
    shared_sum[tid] = denom_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < skip_domain) {
        FpExt tile_sum = shared_sum[tid];
        for (uint32_t lane = 1; lane < (tg_size >> l_skip); ++lane) {
            tile_sum = tile_sum + shared_sum[(lane << l_skip) + tid];
        }
        tmp_sums_q[gid_x * skip_domain + ntt_idx] = tile_sum;
    }
}

// Final reduction for logup round0
kernel void logup_r0_final_reduce_kernel(
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
