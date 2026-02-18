// logup_zerocheck/gkr_input - GKR input evaluation kernels
// Ported from CUDA: cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu
#include <metal_stdlib>
using namespace metal;

#include "baby_bear.h"
#include "baby_bear_ext.h"
#include "codec.h"
#include "frac_ext.h"

struct MainPtrs4 {
    const device Fp *parts[4];
};

// Evaluate a DAG entry for GKR input phase.
// Unlike MLE evaluation, this operates on base field traces (Fp) and wraps row with modular
// arithmetic for rotation offsets. Selectors are boolean (is_first/is_last/is_transition).
inline FpExt evaluate_dag_entry_gkr_local(
    SourceInfo src,
    uint32_t row_index,
    const device Fp *d_preprocessed,
    MainPtrs4 mains,
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
        const device Fp *d_main_fp = (src.part < 4) ? mains.parts[src.part] : nullptr;
        if (d_main_fp == nullptr) {
            return zero;
        }
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

inline FpExt evaluate_dag_entry_gkr_global(
    SourceInfo src,
    uint32_t row_index,
    const device Fp *d_preprocessed,
    MainPtrs4 mains,
    const device Fp *d_public_values,
    const device FpExt *d_challenges,
    const device FpExt *d_intermediates,
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
        const device Fp *d_main_fp = (src.part < 4) ? mains.parts[src.part] : nullptr;
        if (d_main_fp == nullptr) {
            return zero;
        }
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
    const device Fp *d_public_values [[buffer(3)]],
    const device FpExt *d_challenges [[buffer(4)]],
    const device Rule *d_rules [[buffer(5)]],
    const device uint64_t *d_used_nodes [[buffer(6)]],
    const device uint32_t *d_pair_idxs [[buffer(7)]],
    constant uint32_t &used_nodes_len [[buffer(8)]],
    constant uint32_t &permutation_height [[buffer(9)]],
    constant uint32_t &rules_len [[buffer(10)]],
    constant uint32_t &num_rows_per_tile [[buffer(11)]],
    constant uint32_t &total_threads [[buffer(12)]],
    const device Fp *main0 [[buffer(13)]],
    const device Fp *main1 [[buffer(14)]],
    const device Fp *main2 [[buffer(15)]],
    const device Fp *main3 [[buffer(16)]],
    uint tidx [[thread_position_in_grid]]
) {
    (void)rules_len;
    uint32_t task_offset = tidx;
    uint32_t task_stride = total_threads;

    FpExt local_intermediates[10];
    thread FpExt *intermediates_ptr = local_intermediates;
    uint32_t intermediate_stride = 1;
    MainPtrs4 mains;
    mains.parts[0] = main0;
    mains.parts[1] = main1;
    mains.parts[2] = main2;
    mains.parts[3] = main3;

    for (uint32_t j = 0; j < num_rows_per_tile; j++) {
        uint32_t row = task_offset + j * task_stride;
        if (row >= permutation_height) {
            continue;
        }
        uint32_t rules_evaluated = 0;
        for (uint32_t used_idx = 0; used_idx < used_nodes_len; used_idx++) {
            uint32_t node_idx = (uint32_t)d_used_nodes[used_idx];
            FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
            FpExt result = zero;

            if (node_idx < rules_evaluated) {
                Rule rule = d_rules[node_idx];
                RuleHeader header = decode_rule_header(rule);
                if (header.op == OP_VAR) {
                    result = evaluate_dag_entry_gkr_local(
                        header.x,
                        row,
                        d_preprocessed,
                        mains,
                        d_public_values,
                        d_challenges,
                        intermediates_ptr,
                        intermediate_stride,
                        permutation_height
                    );
                } else {
                    uint32_t z_index = decode_z_index(rule);
                    result = intermediates_ptr[z_index * intermediate_stride];
                }
            } else {
                for (; rules_evaluated <= node_idx; rules_evaluated++) {
                    Rule rule = d_rules[rules_evaluated];
                    RuleHeader header = decode_rule_header(rule);

                    FpExt x = evaluate_dag_entry_gkr_local(
                        header.x,
                        row,
                        d_preprocessed,
                        mains,
                        d_public_values,
                        d_challenges,
                        intermediates_ptr,
                        intermediate_stride,
                        permutation_height
                    );
                    FpExt node_result;

                    switch (header.op) {
                    case OP_ADD: {
                        SourceInfo y_src = decode_y(rule);
                        FpExt y = evaluate_dag_entry_gkr_local(
                            y_src,
                            row,
                            d_preprocessed,
                            mains,
                            d_public_values,
                            d_challenges,
                            intermediates_ptr,
                            intermediate_stride,
                            permutation_height
                        );
                        node_result = x + y;
                        break;
                    }
                    case OP_SUB: {
                        SourceInfo y_src = decode_y(rule);
                        FpExt y = evaluate_dag_entry_gkr_local(
                            y_src,
                            row,
                            d_preprocessed,
                            mains,
                            d_public_values,
                            d_challenges,
                            intermediates_ptr,
                            intermediate_stride,
                            permutation_height
                        );
                        node_result = x - y;
                        break;
                    }
                    case OP_MUL: {
                        SourceInfo y_src = decode_y(rule);
                        FpExt y = evaluate_dag_entry_gkr_local(
                            y_src,
                            row,
                            d_preprocessed,
                            mains,
                            d_public_values,
                            d_challenges,
                            intermediates_ptr,
                            intermediate_stride,
                            permutation_height
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
            size_t interaction_idx = pair_idx >> 1;
            size_t out_idx = interaction_idx * permutation_height + row;
            if (pair_idx & 1) {
                d_fracs_q[out_idx] = result;
            } else {
                d_fracs_p[out_idx] = result;
            }
        }
    }
}

// GKR input evaluation kernel (global intermediates mode).
// Same logic but intermediates stored in global device memory for large buffer sizes.
kernel void evaluate_interactions_gkr_global_kernel(
    device FpExt *d_fracs_p [[buffer(0)]],
    device FpExt *d_fracs_q [[buffer(1)]],
    const device Fp *d_preprocessed [[buffer(2)]],
    const device Fp *d_public_values [[buffer(3)]],
    const device FpExt *d_challenges [[buffer(4)]],
    device FpExt *d_intermediates [[buffer(5)]],
    const device Rule *d_rules [[buffer(6)]],
    const device uint64_t *d_used_nodes [[buffer(7)]],
    const device uint32_t *d_pair_idxs [[buffer(8)]],
    constant uint32_t &used_nodes_len [[buffer(9)]],
    constant uint32_t &permutation_height [[buffer(10)]],
    constant uint32_t &rules_len [[buffer(11)]],
    constant uint32_t &num_rows_per_tile [[buffer(12)]],
    constant uint32_t &total_threads [[buffer(13)]],
    const device Fp *main0 [[buffer(14)]],
    const device Fp *main1 [[buffer(15)]],
    const device Fp *main2 [[buffer(16)]],
    const device Fp *main3 [[buffer(17)]],
    uint tidx [[thread_position_in_grid]]
) {
    (void)rules_len;
    uint32_t task_offset = tidx;
    uint32_t task_stride = total_threads;
    uint32_t intermediate_stride = task_stride;
    device FpExt *intermediates_ptr = d_intermediates + task_offset;
    MainPtrs4 mains;
    mains.parts[0] = main0;
    mains.parts[1] = main1;
    mains.parts[2] = main2;
    mains.parts[3] = main3;

    for (uint32_t j = 0; j < num_rows_per_tile; j++) {
        uint32_t row = task_offset + j * task_stride;
        if (row >= permutation_height) {
            continue;
        }

        uint32_t rules_evaluated = 0;
        for (uint32_t used_idx = 0; used_idx < used_nodes_len; used_idx++) {
            uint32_t node_idx = (uint32_t)d_used_nodes[used_idx];
            FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
            FpExt result = zero;

            if (node_idx < rules_evaluated) {
                Rule rule = d_rules[node_idx];
                RuleHeader header = decode_rule_header(rule);
                if (header.op == OP_VAR) {
                    result = evaluate_dag_entry_gkr_global(
                        header.x,
                        row,
                        d_preprocessed,
                        mains,
                        d_public_values,
                        d_challenges,
                        intermediates_ptr,
                        intermediate_stride,
                        permutation_height
                    );
                } else {
                    uint32_t z_index = decode_z_index(rule);
                    result = intermediates_ptr[z_index * intermediate_stride];
                }
            } else {
                for (; rules_evaluated <= node_idx; rules_evaluated++) {
                    Rule rule = d_rules[rules_evaluated];
                    RuleHeader header = decode_rule_header(rule);

                    FpExt x = evaluate_dag_entry_gkr_global(
                        header.x,
                        row,
                        d_preprocessed,
                        mains,
                        d_public_values,
                        d_challenges,
                        intermediates_ptr,
                        intermediate_stride,
                        permutation_height
                    );
                    FpExt node_result;

                    switch (header.op) {
                    case OP_ADD: {
                        SourceInfo y_src = decode_y(rule);
                        FpExt y = evaluate_dag_entry_gkr_global(
                            y_src,
                            row,
                            d_preprocessed,
                            mains,
                            d_public_values,
                            d_challenges,
                            intermediates_ptr,
                            intermediate_stride,
                            permutation_height
                        );
                        node_result = x + y;
                        break;
                    }
                    case OP_SUB: {
                        SourceInfo y_src = decode_y(rule);
                        FpExt y = evaluate_dag_entry_gkr_global(
                            y_src,
                            row,
                            d_preprocessed,
                            mains,
                            d_public_values,
                            d_challenges,
                            intermediates_ptr,
                            intermediate_stride,
                            permutation_height
                        );
                        node_result = x - y;
                        break;
                    }
                    case OP_MUL: {
                        SourceInfo y_src = decode_y(rule);
                        FpExt y = evaluate_dag_entry_gkr_global(
                            y_src,
                            row,
                            d_preprocessed,
                            mains,
                            d_public_values,
                            d_challenges,
                            intermediates_ptr,
                            intermediate_stride,
                            permutation_height
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
            size_t interaction_idx = pair_idx >> 1;
            size_t out_idx = interaction_idx * permutation_height + row;
            if (pair_idx & 1) {
                d_fracs_q[out_idx] = result;
            } else {
                d_fracs_p[out_idx] = result;
            }
        }
    }
}
