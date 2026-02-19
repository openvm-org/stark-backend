// logup_zerocheck/batch_mle - Batched MLE evaluation kernels for multiple AIRs
// Ported from CUDA: cuda-backend/cuda/src/logup_zerocheck/batch_mle.cu
#include <metal_stdlib>
using namespace metal;

#include "baby_bear.h"
#include "baby_bear_ext.h"
#include "codec.h"
#include "eval_ctx.h"
#include "frac_ext.h"
#include "sumcheck.h"

// Context structs for batched evaluation

struct ZerocheckCtx {
    EvalCoreCtx eval_ctx;
    uint64_t d_intermediates;
    uint32_t num_y;
    uint64_t d_eq_xi;
    uint64_t d_rules;
    uint32_t rules_len;
    uint64_t d_used_nodes;
    uint32_t used_nodes_len;
    uint32_t buffer_size;
};

struct LogupCtx {
    EvalCoreCtx eval_ctx;
    uint64_t d_intermediates;
    uint32_t num_y;
    uint64_t d_eq_xi;
    uint64_t d_challenges;
    uint64_t d_eq_3bs;
    uint64_t d_rules;
    uint32_t rules_len;
    uint64_t d_used_nodes;
    uint64_t d_pair_idxs;
    uint32_t used_nodes_len;
    uint32_t buffer_size;
};

// Local context for device use only
struct EvalCtx {
    const device FpExt *d_selectors;
    MainMatrixPtrsExt d_preprocessed;
    const device MainMatrixPtrsExt *d_main;
    const device Fp *d_public;
    device FpExt *d_intermediates;
    uint32_t height;
};

inline FpExt evaluate_mle_entry_batch(
    SourceInfo src,
    uint32_t row,
    thread EvalCtx &ctx,
    uint32_t buffer_stride,
    const device FpExt *d_challenges
) {
    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    switch (src.type) {
    case ENTRY_PREPROCESSED: {
        uint32_t stride = ctx.height * ctx.d_preprocessed.air_width;
        const device FpExt *matrix = as_fpext_ptr(ctx.d_preprocessed.data) + stride * src.offset;
        const device FpExt *column = matrix + ctx.height * src.index;
        return column[row];
    }
    case ENTRY_MAIN: {
        MainMatrixPtrsExt main_ptr = ctx.d_main[src.part];
        uint32_t stride = ctx.height * main_ptr.air_width;
        const device FpExt *matrix = as_fpext_ptr(main_ptr.data) + stride * src.offset;
        const device FpExt *column = matrix + ctx.height * src.index;
        return column[row];
    }
    case SRC_INTERMEDIATE:
        return ctx.d_intermediates[src.index * buffer_stride];
    case SRC_IS_FIRST:
        return ctx.d_selectors[row];
    case SRC_IS_LAST:
        return ctx.d_selectors[2 * ctx.height + row];
    case SRC_IS_TRANSITION:
        return ctx.d_selectors[ctx.height + row];
    case ENTRY_CHALLENGE:
        return d_challenges[src.index];
    case ENTRY_PUBLIC:
        return FpExt{ctx.d_public[src.index], Fp(0u), Fp(0u), Fp(0u)};
    case SRC_CONSTANT:
        return FpExt{Fp(src.index), Fp(0u), Fp(0u), Fp(0u)};
    default:
        break;
    }
    return zero;
}

// Batched zerocheck MLE kernel: evaluates constraints for multiple AIRs
kernel void zerocheck_batch_mle_kernel(
    device FpExt *tmp_sums_buffer [[buffer(0)]],
    const device BlockCtx *d_block_ctxs [[buffer(1)]],
    const device ZerocheckCtx *d_zc_ctxs [[buffer(2)]],
    const device FpExt *d_lambda_pows [[buffer(3)]],
    constant uint32_t &lambda_len [[buffer(4)]],
    constant uint32_t &num_x [[buffer(5)]],
    threadgroup FpExt *shared [[threadgroup(0)]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tg_size2 [[threads_per_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    BlockCtx block_ctx = d_block_ctxs[gid.x];
    ZerocheckCtx zc_ctx = d_zc_ctxs[block_ctx.air_idx];

    uint32_t x_int = gid.y;
    uint32_t y_int = tid + block_ctx.local_block_idx_x * tg_size;
    bool active_thread = (y_int < zc_ctx.num_y);

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt sum = zero;

    if (active_thread) {
        uint32_t height = num_x * zc_ctx.num_y;
        uint32_t row = x_int * zc_ctx.num_y + y_int;

        device FpExt *d_intermediates = reinterpret_cast<device FpExt *>(zc_ctx.d_intermediates);
        const device FpExt *d_eq_xi = as_fpext_ptr(zc_ctx.d_eq_xi);
        const device Rule *d_rules = reinterpret_cast<const device Rule *>(zc_ctx.d_rules);
        const device uint64_t *d_used_nodes =
            reinterpret_cast<const device uint64_t *>(zc_ctx.d_used_nodes);

        uint32_t buffer_stride = 0;
        device FpExt *intermediates = nullptr;
        if (zc_ctx.buffer_size > 0) {
            intermediates = d_intermediates + row;
            buffer_stride = height;
        }

        EvalCtx eval_ctx;
        eval_ctx.d_selectors = as_fpext_ptr(zc_ctx.eval_ctx.d_selectors);
        eval_ctx.d_preprocessed = zc_ctx.eval_ctx.d_preprocessed;
        eval_ctx.d_main = as_main_matrix_ptrs_ext(zc_ctx.eval_ctx.d_main);
        eval_ctx.d_public = as_fp_ptr(zc_ctx.eval_ctx.d_public);
        eval_ctx.d_intermediates = intermediates;
        eval_ctx.height = height;

        uint32_t lambda_idx = 0;

        for (uint32_t node = 0; node < zc_ctx.rules_len; ++node) {
            Rule rule = d_rules[node];
            DecodedRule decoded = decode_rule(rule);

            FpExt x_val = evaluate_mle_entry_batch(decoded.x, row, eval_ctx, buffer_stride, nullptr);
            FpExt result;
            switch (decoded.op) {
            case OP_ADD: {
                FpExt y_val = evaluate_mle_entry_batch(decoded.y, row, eval_ctx, buffer_stride, nullptr);
                result = x_val + y_val;
                break;
            }
            case OP_SUB: {
                FpExt y_val = evaluate_mle_entry_batch(decoded.y, row, eval_ctx, buffer_stride, nullptr);
                result = x_val - y_val;
                break;
            }
            case OP_MUL: {
                FpExt y_val = evaluate_mle_entry_batch(decoded.y, row, eval_ctx, buffer_stride, nullptr);
                result = x_val * y_val;
                break;
            }
            case OP_NEG:
                result = fpext_neg(x_val);
                break;
            case OP_VAR:
                result = x_val;
                break;
            default:
                result = zero;
                break;
            }

            if (decoded.buffer_result && zc_ctx.buffer_size > 0) {
                eval_ctx.d_intermediates[decoded.z_index * buffer_stride] = result;
            }

            if (decoded.is_constraint) {
                while (lambda_idx < lambda_len && lambda_idx < zc_ctx.used_nodes_len &&
                       d_used_nodes[lambda_idx] == node) {
                    FpExt lambda = d_lambda_pows[lambda_idx];
                    lambda_idx++;
                    sum = sum + lambda * result;
                }
            }
        }
        sum = sum * d_eq_xi[y_int];
    }

    FpExt reduced = block_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) {
        tmp_sums_buffer[gid.x * num_x + x_int] = reduced;
    }
}

// Batched logup MLE kernel: evaluates logup interactions for multiple AIRs
kernel void logup_batch_mle_kernel(
    device FpExt *tmp_sums_p [[buffer(0)]],
    device FpExt *tmp_sums_q [[buffer(1)]],
    const device BlockCtx *d_block_ctxs [[buffer(2)]],
    const device LogupCtx *d_logup_ctxs [[buffer(3)]],
    constant uint32_t &num_x [[buffer(4)]],
    threadgroup FpExt *shared [[threadgroup(0)]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tg_size2 [[threads_per_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    BlockCtx block_ctx = d_block_ctxs[gid.x];
    LogupCtx logup_ctx = d_logup_ctxs[block_ctx.air_idx];

    uint32_t x_int = gid.y;
    uint32_t y_int = tid + block_ctx.local_block_idx_x * tg_size;
    bool active_thread = (y_int < logup_ctx.num_y);

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt numer_sum = zero;
    FpExt denom_sum = zero;

    if (active_thread) {
        uint32_t height = num_x * logup_ctx.num_y;
        uint32_t row = x_int * logup_ctx.num_y + y_int;

        device FpExt *d_intermediates = reinterpret_cast<device FpExt *>(logup_ctx.d_intermediates);
        const device FpExt *d_eq_xi = as_fpext_ptr(logup_ctx.d_eq_xi);
        const device FpExt *d_challenges = as_fpext_ptr(logup_ctx.d_challenges);
        const device FpExt *d_eq_3bs = as_fpext_ptr(logup_ctx.d_eq_3bs);
        const device Rule *d_rules = reinterpret_cast<const device Rule *>(logup_ctx.d_rules);
        const device uint64_t *d_used_nodes =
            reinterpret_cast<const device uint64_t *>(logup_ctx.d_used_nodes);
        const device uint32_t *d_pair_idxs =
            reinterpret_cast<const device uint32_t *>(logup_ctx.d_pair_idxs);

        uint32_t buffer_stride = 0;
        device FpExt *intermediates = nullptr;
        if (logup_ctx.buffer_size > 0) {
            intermediates = d_intermediates + row;
            buffer_stride = height;
        }

        EvalCtx eval_ctx;
        eval_ctx.d_selectors = as_fpext_ptr(logup_ctx.eval_ctx.d_selectors);
        eval_ctx.d_preprocessed = logup_ctx.eval_ctx.d_preprocessed;
        eval_ctx.d_main = as_main_matrix_ptrs_ext(logup_ctx.eval_ctx.d_main);
        eval_ctx.d_public = as_fp_ptr(logup_ctx.eval_ctx.d_public);
        eval_ctx.d_intermediates = intermediates;
        eval_ctx.height = height;

        uint32_t rules_evaluated = 0;

        for (uint32_t used_idx = 0; used_idx < logup_ctx.used_nodes_len; ++used_idx) {
            uint64_t node_idx = d_used_nodes[used_idx];
            FpExt result = zero;

            if (node_idx < rules_evaluated) {
                Rule rule = d_rules[node_idx];
                DecodedRule decoded = decode_rule(rule);
                if (decoded.op == OP_VAR) {
                    result = evaluate_mle_entry_batch(decoded.x, row, eval_ctx, buffer_stride, d_challenges);
                } else if (logup_ctx.buffer_size > 0 && decoded.buffer_result) {
                    result = eval_ctx.d_intermediates[decoded.z_index * buffer_stride];
                }
            } else {
                for (; rules_evaluated <= node_idx; ++rules_evaluated) {
                    Rule rule = d_rules[rules_evaluated];
                    DecodedRule decoded = decode_rule(rule);

                    FpExt x_val = evaluate_mle_entry_batch(decoded.x, row, eval_ctx, buffer_stride, d_challenges);
                    FpExt node_result;
                    switch (decoded.op) {
                    case OP_ADD: {
                        FpExt y_val = evaluate_mle_entry_batch(decoded.y, row, eval_ctx, buffer_stride, d_challenges);
                        node_result = x_val + y_val;
                        break;
                    }
                    case OP_SUB: {
                        FpExt y_val = evaluate_mle_entry_batch(decoded.y, row, eval_ctx, buffer_stride, d_challenges);
                        node_result = x_val - y_val;
                        break;
                    }
                    case OP_MUL: {
                        FpExt y_val = evaluate_mle_entry_batch(decoded.y, row, eval_ctx, buffer_stride, d_challenges);
                        node_result = x_val * y_val;
                        break;
                    }
                    case OP_NEG:
                        node_result = fpext_neg(x_val);
                        break;
                    case OP_VAR:
                        node_result = x_val;
                        break;
                    default:
                        node_result = zero;
                        break;
                    }

                    if (decoded.buffer_result && logup_ctx.buffer_size > 0) {
                        eval_ctx.d_intermediates[decoded.z_index * buffer_stride] = node_result;
                    }

                    if (rules_evaluated == node_idx) {
                        result = node_result;
                    }
                }
            }

            uint32_t pair_idx = d_pair_idxs[used_idx];
            result = result * d_eq_3bs[pair_idx >> 1];

            if (pair_idx & 1) {
                denom_sum = denom_sum + result;
            } else {
                numer_sum = numer_sum + result;
            }
        }

        FpExt eq_val = d_eq_xi[y_int];
        numer_sum = numer_sum * eq_val;
        denom_sum = denom_sum * eq_val;
    }

    FpExt numer_reduced = block_reduce_sum(numer_sum, shared, tid, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FpExt denom_reduced = block_reduce_sum(denom_sum, shared, tid, tg_size);
    if (tid == 0) {
        uint out_idx = gid.x * num_x + x_int;
        tmp_sums_p[out_idx] = numer_reduced;
        tmp_sums_q[out_idx] = denom_reduced;
    }
}

// Batched final reduction: sum partial sums across blocks for each (air, x) pair
kernel void batch_mle_final_reduce_block_sums_kernel(
    const device FpExt *block_sums [[buffer(0)]],
    device FpExt *output [[buffer(1)]],
    const device uint32_t *air_block_offsets [[buffer(2)]],
    constant uint32_t &stride [[buffer(3)]],
    threadgroup FpExt *shared [[threadgroup(0)]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tg_size2 [[threads_per_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    uint air_idx = gid.x;
    uint d_idx = gid.y;

    uint block_start = air_block_offsets[air_idx];
    uint block_end = air_block_offsets[air_idx + 1];
    uint num_blocks = block_end - block_start;

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt sum = zero;

    for (uint b = tid; b < num_blocks; b += tg_size) {
        uint block_idx = block_start + b;
        sum = sum + block_sums[block_idx * stride + d_idx];
    }

    FpExt reduced = block_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) {
        output[air_idx * stride + d_idx] = reduced;
    }
}
