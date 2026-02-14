#include "codec.cuh"
#include "fp.h"
#include "fpext.h"
#include "frac_ext.cuh"
#include "launcher.cuh"
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace logup_gkr_input_evaluation {

// ============================================================================
// KERNELS
// ============================================================================

// GKR phase interactions kernel (for permutation evaluation)
__device__ __forceinline__ FpExt evaluate_dag_entry_gkr(
    const SourceInfo &src,
    uint32_t row_index,
    const Fp *d_preprocessed,
    const uint64_t *d_main,
    const Fp *d_public_values,
    const FpExt *d_challenges,
    const FpExt *d_intermediates,
    const uint32_t intermediate_stride,
    const uint32_t height
) {
    FpExt result(0);
    switch (src.type) {
    case ENTRY_PREPROCESSED:
        result = FpExt(d_preprocessed[height * src.index + ((row_index + src.offset) % height)]);
        break;
    case ENTRY_MAIN: {
        Fp *d_main_fp = (Fp *)d_main[src.part];
        result = FpExt(d_main_fp[height * src.index + ((row_index + src.offset) % height)]);
        break;
    }
    case ENTRY_PUBLIC:
        result = FpExt(d_public_values[src.index]);
        break;
    case ENTRY_CHALLENGE:
        result = d_challenges[src.index];
        break;
    case SRC_INTERMEDIATE:
        result = d_intermediates[intermediate_stride * src.index];
        break;
    case SRC_CONSTANT:
        result = FpExt(Fp(src.index));
        break;
    case SRC_IS_FIRST:
        result = make_bool_ext(row_index == 0);
        break;
    case SRC_IS_LAST:
        result = make_bool_ext(row_index == height - 1);
        break;
    case SRC_IS_TRANSITION:
        result = make_bool_ext(row_index != height - 1);
        break;
    default:
        assert(0);
    }
    return result;
}

template <bool GLOBAL>
__global__ void evaluate_interactions_gkr_kernel(
    FracExt *__restrict__ d_fracs,
    const Fp *__restrict__ d_preprocessed,
    const uint64_t *__restrict__ d_main,
    const Fp *__restrict__ d_public_values,
    const FpExt *__restrict__ d_challenges,
    const FpExt *__restrict__ d_intermediates,
    const Rule *__restrict__ d_rules,
    const size_t *__restrict__ d_used_nodes,
    const uint32_t *__restrict__ d_pair_idxs,
    const size_t used_nodes_len,
    const uint32_t permutation_height,
    const uint32_t num_rows_per_tile
) {
    uint32_t task_offset = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t task_stride = gridDim.x * blockDim.x;

    FpExt *intermediates_ptr;
    uint32_t intermediate_stride;
    if constexpr (GLOBAL) {
        intermediates_ptr = (FpExt *)d_intermediates + task_offset;
        intermediate_stride = task_stride;
    } else {
        FpExt intermediates[10];
        intermediates_ptr = intermediates;
        intermediate_stride = 1;
    }

    for (uint32_t j = 0; j < num_rows_per_tile; j++) {
        uint32_t row = task_offset + j * task_stride;

        if (row < permutation_height) {
            uint32_t rules_evaluated = 0;
            // Iterate through used_nodes (alternates: numer, denom, numer, denom, ...)
            for (uint32_t used_idx = 0; used_idx < used_nodes_len; used_idx++) {
                uint32_t node_idx = d_used_nodes[used_idx];
                FpExt result(0);
                if (node_idx < rules_evaluated) {
                    Rule rule = d_rules[node_idx];
                    RuleHeader header = decode_rule_header(rule);
                    if (header.op == OP_VAR) {
                        result = evaluate_dag_entry_gkr(
                            header.x,
                            row,
                            d_preprocessed,
                            d_main,
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

                        FpExt x = evaluate_dag_entry_gkr(
                            header.x,
                            row,
                            d_preprocessed,
                            d_main,
                            d_public_values,
                            d_challenges,
                            intermediates_ptr,
                            intermediate_stride,
                            permutation_height
                        );
                        FpExt y;

                        switch (header.op) {
                        case OP_ADD:
                            y = evaluate_dag_entry_gkr(
                                decode_y(rule),
                                row,
                                d_preprocessed,
                                d_main,
                                d_public_values,
                                d_challenges,
                                intermediates_ptr,
                                intermediate_stride,
                                permutation_height
                            );
                            result = x + y;
                            break;
                        case OP_SUB:
                            y = evaluate_dag_entry_gkr(
                                decode_y(rule),
                                row,
                                d_preprocessed,
                                d_main,
                                d_public_values,
                                d_challenges,
                                intermediates_ptr,
                                intermediate_stride,
                                permutation_height
                            );
                            result = x - y;
                            break;
                        case OP_MUL:
                            y = evaluate_dag_entry_gkr(
                                decode_y(rule),
                                row,
                                d_preprocessed,
                                d_main,
                                d_public_values,
                                d_challenges,
                                intermediates_ptr,
                                intermediate_stride,
                                permutation_height
                            );
                            x *= y;
                            result = x;
                            break;
                        case OP_NEG:
                            result = -x;
                            break;
                        case OP_VAR:
                            result = x;
                            break;
                        default:
                            assert(0);
                        }

                        if (header.buffer_result) {
                            uint32_t z_index = decode_z_index(rule);
                            intermediates_ptr[z_index * intermediate_stride] = result;
                        }
                    }
                }

                uint32_t pair_idx = d_pair_idxs[used_idx];
                size_t interaction_idx = pair_idx >> 1;
                size_t out_idx = (interaction_idx * permutation_height + row) * 2;
                reinterpret_cast<FpExt *>(d_fracs)[out_idx + (pair_idx & 1)] = result;
            }
        }
    }
}

// ============================================================================
// LAUNCHERS
// ============================================================================

constexpr uint32_t TASK_SIZE = 1 << 16;

extern "C" int _logup_gkr_input_eval(
    bool is_global,
    FracExt *d_fracs,
    const Fp *d_preprocessed,
    const uint64_t *d_main,
    const Fp *d_public_values,
    const FpExt *d_challenges,
    const FpExt *d_intermediates,
    const Rule *d_rules,
    const size_t *d_used_nodes,
    const uint32_t *d_pair_idxs,
    size_t used_nodes_len,
    uint32_t permutation_height,
    uint32_t num_rows_per_tile
) {
    auto count = is_global ? TASK_SIZE : permutation_height;
    auto [grid, block] = kernel_launch_params(count, 256);
    if (is_global) {
        evaluate_interactions_gkr_kernel<true><<<grid, block>>>(
            d_fracs,
            d_preprocessed,
            d_main,
            d_public_values,
            d_challenges,
            d_intermediates,
            d_rules,
            d_used_nodes,
            d_pair_idxs,
            used_nodes_len,
            permutation_height,
            num_rows_per_tile
        );
    } else {
        evaluate_interactions_gkr_kernel<false><<<grid, block>>>(
            d_fracs,
            d_preprocessed,
            d_main,
            d_public_values,
            d_challenges,
            d_intermediates,
            d_rules,
            d_used_nodes,
            d_pair_idxs,
            used_nodes_len,
            permutation_height,
            num_rows_per_tile
        );
    }
    return CHECK_KERNEL();
}

} // namespace logup_gkr_input_evaluation
