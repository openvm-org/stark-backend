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
    const FpExt *__restrict__ d_challenges,
    const FpExt *__restrict__ d_intermediates,
    const Rule *__restrict__ d_rules,
    const size_t *__restrict__ d_used_nodes,
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
            bool is_denom = false;
            FpExt numerator(0);
            FpExt denom(0);

            // Iterate through used_nodes (alternates: numer, denom, numer, denom, ...)
            for (uint32_t used_idx = 0; used_idx < used_nodes_len; used_idx++) {
                uint32_t node_idx = d_used_nodes[used_idx];
                FpExt result(0);
                if (node_idx < rules_evaluated) {
                    Rule rule = d_rules[node_idx];
                    DecodedRule decoded_rule = decode_rule(rule);
                    if (decoded_rule.op == OP_VAR) {
                        result = evaluate_dag_entry_gkr(
                            decoded_rule.x,
                            row,
                            d_preprocessed,
                            d_main,
                            d_challenges,
                            intermediates_ptr,
                            intermediate_stride,
                            permutation_height
                        );
                    } else {
                        result = intermediates_ptr[decoded_rule.z_index * intermediate_stride];
                    }
                } else {
                    for (; rules_evaluated <= node_idx; rules_evaluated++) {
                        Rule rule = d_rules[rules_evaluated];
                        DecodedRule decoded_rule = decode_rule(rule);

                        FpExt x = evaluate_dag_entry_gkr(
                            decoded_rule.x,
                            row,
                            d_preprocessed,
                            d_main,
                            d_challenges,
                            intermediates_ptr,
                            intermediate_stride,
                            permutation_height
                        );
                        FpExt y = evaluate_dag_entry_gkr(
                            decoded_rule.y,
                            row,
                            d_preprocessed,
                            d_main,
                            d_challenges,
                            intermediates_ptr,
                            intermediate_stride,
                            permutation_height
                        );

                        switch (decoded_rule.op) {
                        case OP_ADD:
                            result = x + y;
                            break;
                        case OP_SUB:
                            result = x - y;
                            break;
                        case OP_MUL:
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

                        if (decoded_rule.buffer_result) {
                            intermediates_ptr[decoded_rule.z_index * intermediate_stride] = result;
                        }
                    }
                }

                if (is_denom) {
                    denom = result;
                    // We use the fact that used nodes come in (numer, denom) pairs
                    size_t interaction_idx = used_idx >> 1;
                    size_t out_idx = interaction_idx * permutation_height + row;
                    // Numerator is computed as FpExt through DAG evaluation
                    // For logup, the numerator (count) should be in the base field
                    // Extract the base field component (first element of extension field)
                    d_fracs[out_idx] = {FpExt(numerator.elems[0]), denom};
                } else {
                    numerator = result;
                }
                is_denom = !is_denom;
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
    const FpExt *d_challenges,
    const FpExt *d_intermediates,
    const Rule *d_rules,
    const size_t *d_used_nodes,
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
            d_challenges,
            d_intermediates,
            d_rules,
            d_used_nodes,
            used_nodes_len,
            permutation_height,
            num_rows_per_tile
        );
    } else {
        evaluate_interactions_gkr_kernel<false><<<grid, block>>>(
            d_fracs,
            d_preprocessed,
            d_main,
            d_challenges,
            d_intermediates,
            d_rules,
            d_used_nodes,
            used_nodes_len,
            permutation_height,
            num_rows_per_tile
        );
    }
    return CHECK_KERNEL();
}

} // namespace logup_gkr_input_evaluation
