// logup_zerocheck/gkr_input - GKR input evaluation kernels
// Ported from CUDA: cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu
#include <metal_stdlib>
using namespace metal;

#include "baby_bear.h"
#include "baby_bear_ext.h"
#include "codec.h"
#include "frac_ext.h"

// Evaluate a DAG entry for GKR input phase.
// Unlike MLE evaluation, this operates on base field traces (Fp) and wraps row with modular
// arithmetic for rotation offsets. Selectors are boolean (is_first/is_last/is_transition).
inline FpExt evaluate_dag_entry_gkr(
    SourceInfo src,
    uint32_t row_index,
    const device Fp *d_preprocessed,
    const device uint64_t *d_main,
    const device Fp *d_public_values,
    const device FpExt *d_challenges,
    thread FpExt *d_intermediates,
    uint32_t intermediate_stride,
    uint32_t height
) {
    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    switch (src.type) {
    case ENTRY_PREPROCESSED: {
        uint32_t idx = height * src.index + ((row_index + src.offset) % height);
        return FpExt(d_preprocessed[idx]);
    }
    case ENTRY_MAIN: {
        const device Fp *d_main_fp = reinterpret_cast<const device Fp *>(d_main[src.part]);
        uint32_t idx = height * src.index + ((row_index + src.offset) % height);
        return FpExt(d_main_fp[idx]);
    }
    case ENTRY_PUBLIC:
        return FpExt(d_public_values[src.index]);
    case ENTRY_CHALLENGE:
        return d_challenges[src.index];
    case SRC_INTERMEDIATE:
        return d_intermediates[intermediate_stride * src.index];
    case SRC_CONSTANT:
        return FpExt{Fp(src.index), Fp(0u), Fp(0u), Fp(0u)};
    case SRC_IS_FIRST:
        return make_bool_ext(row_index == 0);
    case SRC_IS_LAST:
        return make_bool_ext(row_index == height - 1);
    case SRC_IS_TRANSITION:
        return make_bool_ext(row_index != height - 1);
    default:
        break;
    }
    return zero;
}

// GKR input evaluation kernel (local intermediates mode).
// Each thread evaluates the interaction DAG for one row, producing FracExt pairs.
// Output: d_fracs_p[interaction_idx * height + row] = numerator
//         d_fracs_q[interaction_idx * height + row] = denominator
kernel void evaluate_interactions_gkr_local_kernel(
    device FpExt *d_fracs_p [[buffer(0)]],
    device FpExt *d_fracs_q [[buffer(1)]],
    const device Fp *d_preprocessed [[buffer(2)]],
    const device uint64_t *d_main [[buffer(3)]],
    const device Fp *d_public_values [[buffer(4)]],
    const device FpExt *d_challenges [[buffer(5)]],
    const device Rule *d_rules [[buffer(6)]],
    const device uint64_t *d_used_nodes [[buffer(7)]],
    const device uint32_t *d_pair_idxs [[buffer(8)]],
    constant uint32_t &used_nodes_len [[buffer(9)]],
    constant uint32_t &permutation_height [[buffer(10)]],
    constant uint32_t &rules_len [[buffer(11)]],
    uint tidx [[thread_position_in_grid]]
) {
    if (tidx >= permutation_height) return;

    uint32_t row = tidx;

    FpExt local_intermediates[10];
    thread FpExt *intermediates_ptr = local_intermediates;
    uint32_t intermediate_stride = 1;

    uint32_t rules_evaluated = 0;

    for (uint32_t used_idx = 0; used_idx < used_nodes_len; used_idx++) {
        uint64_t node_idx = d_used_nodes[used_idx];
        FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
        FpExt result = zero;

        if (node_idx < rules_evaluated) {
            Rule rule = d_rules[node_idx];
            RuleHeader header = decode_rule_header(rule);
            if (header.op == OP_VAR) {
                result = evaluate_dag_entry_gkr(
                    header.x, row,
                    d_preprocessed, d_main, d_public_values, d_challenges,
                    intermediates_ptr, intermediate_stride, permutation_height
                );
            } else if (header.buffer_result) {
                uint32_t z_index = decode_z_index(rule);
                result = intermediates_ptr[z_index * intermediate_stride];
            }
        } else {
            for (; rules_evaluated <= node_idx; rules_evaluated++) {
                Rule rule = d_rules[rules_evaluated];
                RuleHeader header = decode_rule_header(rule);

                FpExt x = evaluate_dag_entry_gkr(
                    header.x, row,
                    d_preprocessed, d_main, d_public_values, d_challenges,
                    intermediates_ptr, intermediate_stride, permutation_height
                );
                FpExt node_result;

                switch (header.op) {
                case OP_ADD: {
                    SourceInfo y_src = decode_y(rule);
                    FpExt y = evaluate_dag_entry_gkr(
                        y_src, row,
                        d_preprocessed, d_main, d_public_values, d_challenges,
                        intermediates_ptr, intermediate_stride, permutation_height
                    );
                    node_result = x + y;
                    break;
                }
                case OP_SUB: {
                    SourceInfo y_src = decode_y(rule);
                    FpExt y = evaluate_dag_entry_gkr(
                        y_src, row,
                        d_preprocessed, d_main, d_public_values, d_challenges,
                        intermediates_ptr, intermediate_stride, permutation_height
                    );
                    node_result = x - y;
                    break;
                }
                case OP_MUL: {
                    SourceInfo y_src = decode_y(rule);
                    FpExt y = evaluate_dag_entry_gkr(
                        y_src, row,
                        d_preprocessed, d_main, d_public_values, d_challenges,
                        intermediates_ptr, intermediate_stride, permutation_height
                    );
                    node_result = x * y;
                    break;
                }
                case OP_NEG:
                    node_result = fpext_neg(x);
                    break;
                case OP_VAR:
                    node_result = x;
                    break;
                default:
                    node_result = zero;
                    break;
                }

                if (header.buffer_result) {
                    uint32_t z_index = decode_z_index(rule);
                    intermediates_ptr[z_index * intermediate_stride] = node_result;
                }

                if (rules_evaluated == node_idx) {
                    result = node_result;
                }
            }
        }

        uint32_t pair_idx = d_pair_idxs[used_idx];
        uint32_t interaction_idx = pair_idx >> 1;
        uint32_t out_idx = interaction_idx * permutation_height + row;
        if (pair_idx & 1) {
            d_fracs_q[out_idx] = result;
        } else {
            d_fracs_p[out_idx] = result;
        }
    }
}

// GKR input evaluation kernel (global intermediates mode).
// Same logic but intermediates stored in global device memory for large buffer sizes.
kernel void evaluate_interactions_gkr_global_kernel(
    device FpExt *d_fracs_p [[buffer(0)]],
    device FpExt *d_fracs_q [[buffer(1)]],
    const device Fp *d_preprocessed [[buffer(2)]],
    const device uint64_t *d_main [[buffer(3)]],
    const device Fp *d_public_values [[buffer(4)]],
    const device FpExt *d_challenges [[buffer(5)]],
    device FpExt *d_intermediates [[buffer(6)]],
    const device Rule *d_rules [[buffer(7)]],
    const device uint64_t *d_used_nodes [[buffer(8)]],
    const device uint32_t *d_pair_idxs [[buffer(9)]],
    constant uint32_t &used_nodes_len [[buffer(10)]],
    constant uint32_t &permutation_height [[buffer(11)]],
    constant uint32_t &rules_len [[buffer(12)]],
    constant uint32_t &total_threads [[buffer(13)]],
    uint tidx [[thread_position_in_grid]]
) {
    if (tidx >= permutation_height) return;

    uint32_t row = tidx;
    uint32_t intermediate_stride = total_threads;

    // Point into global intermediates at this thread's offset
    device FpExt *intermediates_device = d_intermediates + tidx;

    // We need a thread-local wrapper since evaluate_dag_entry_gkr expects thread FpExt*
    // For global mode, we read/write directly to device memory
    FpExt local_intermediates[10]; // temporary local cache
    thread FpExt *intermediates_ptr = local_intermediates;

    uint32_t rules_evaluated = 0;

    for (uint32_t used_idx = 0; used_idx < used_nodes_len; used_idx++) {
        uint64_t node_idx = d_used_nodes[used_idx];
        FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
        FpExt result = zero;

        if (node_idx < rules_evaluated) {
            Rule rule = d_rules[node_idx];
            RuleHeader header = decode_rule_header(rule);
            if (header.op == OP_VAR) {
                result = evaluate_dag_entry_gkr(
                    header.x, row,
                    d_preprocessed, d_main, d_public_values, d_challenges,
                    intermediates_ptr, 1, permutation_height
                );
            } else if (header.buffer_result) {
                uint32_t z_index = decode_z_index(rule);
                result = intermediates_device[z_index * intermediate_stride];
            }
        } else {
            for (; rules_evaluated <= node_idx; rules_evaluated++) {
                Rule rule = d_rules[rules_evaluated];
                RuleHeader header = decode_rule_header(rule);

                FpExt x = evaluate_dag_entry_gkr(
                    header.x, row,
                    d_preprocessed, d_main, d_public_values, d_challenges,
                    intermediates_ptr, 1, permutation_height
                );
                FpExt node_result;

                switch (header.op) {
                case OP_ADD: {
                    SourceInfo y_src = decode_y(rule);
                    FpExt y = evaluate_dag_entry_gkr(
                        y_src, row,
                        d_preprocessed, d_main, d_public_values, d_challenges,
                        intermediates_ptr, 1, permutation_height
                    );
                    node_result = x + y;
                    break;
                }
                case OP_SUB: {
                    SourceInfo y_src = decode_y(rule);
                    FpExt y = evaluate_dag_entry_gkr(
                        y_src, row,
                        d_preprocessed, d_main, d_public_values, d_challenges,
                        intermediates_ptr, 1, permutation_height
                    );
                    node_result = x - y;
                    break;
                }
                case OP_MUL: {
                    SourceInfo y_src = decode_y(rule);
                    FpExt y = evaluate_dag_entry_gkr(
                        y_src, row,
                        d_preprocessed, d_main, d_public_values, d_challenges,
                        intermediates_ptr, 1, permutation_height
                    );
                    node_result = x * y;
                    break;
                }
                case OP_NEG:
                    node_result = fpext_neg(x);
                    break;
                case OP_VAR:
                    node_result = x;
                    break;
                default:
                    node_result = zero;
                    break;
                }

                if (header.buffer_result) {
                    uint32_t z_index = decode_z_index(rule);
                    intermediates_device[z_index * intermediate_stride] = node_result;
                    intermediates_ptr[z_index] = node_result;
                }

                if (rules_evaluated == node_idx) {
                    result = node_result;
                }
            }
        }

        uint32_t pair_idx = d_pair_idxs[used_idx];
        uint32_t interaction_idx = pair_idx >> 1;
        uint32_t out_idx = interaction_idx * permutation_height + row;
        if (pair_idx & 1) {
            d_fracs_q[out_idx] = result;
        } else {
            d_fracs_p[out_idx] = result;
        }
    }
}
