#include "codec.cuh"
#include "dag_entry.cuh"
#include "fp.h"
#include "fpext.h"
#include "frac_ext.cuh"
#include "launcher.cuh"
#include "matrix.cuh"
#include <cstddef>
#include <cstdint>
#include <utility>

using namespace logup_round0;

// Device function equivalent to helper.eval_interactions without eq_* parts
// This computes the interaction numerator and denominator sums (weighted by eq_3b)
// The eq_* multiplication is done separately in the kernel
template <bool GLOBAL>
__device__ __forceinline__ void acc_interactions(
    uint32_t row,
    const Fp *__restrict__ d_selectors,
    const MainMatrixPtrs *__restrict__ d_main,
    uint32_t height,
    uint32_t selectors_width,
    const Fp *__restrict__ d_preprocessed,
    uint32_t preprocessed_width,
    const FpExt *__restrict__ d_eq_z,
    const FpExt *__restrict__ d_eq_x,
    const Fp *__restrict__ d_public,
    uint32_t public_len,
    const FpExt *__restrict__ d_eq_3b,
    const Rule *__restrict__ d_rules,
    size_t rules_len,
    const size_t *__restrict__ d_used_nodes,
    size_t used_nodes_len,
    uint32_t buffer_size,
    FpExt *__restrict__ inter_buffer,
    FpExt *__restrict__ local_buffer,
    uint32_t buffer_stride,
    uint32_t large_domain,
    const FpExt *__restrict__ d_challenges,
    FpExt &numer_sum,
    FpExt &denom_sum
) {
    // Track how many rules we've evaluated so far
    size_t rules_evaluated = 0;
    FpExt temp_numer(Fp::zero());

    // Iterate through used_nodes (alternates: numer, denom, numer, denom, ...)
    for (size_t used_idx = 0; used_idx < used_nodes_len; ++used_idx) {
        size_t node_idx = d_used_nodes[used_idx];
        FpExt result(Fp::zero());

        if (node_idx < rules_evaluated) {
            // Node already evaluated, get from buffer
            Rule rule = d_rules[node_idx];
            DecodedRule decoded = decode_rule(rule);
            if (decoded.op == OP_VAR) {
                result = evaluate_dag_entry(
                    decoded.x,
                    row,
                    d_selectors,
                    d_main,
                    height,
                    selectors_width,
                    d_preprocessed,
                    preprocessed_width,
                    d_eq_z,
                    d_eq_x,
                    d_public,
                    public_len,
                    inter_buffer,
                    buffer_stride,
                    buffer_size,
                    large_domain,
                    d_challenges
                );
            } else if (buffer_size > 0 && decoded.buffer_result) {
                if constexpr (GLOBAL) {
                    result = inter_buffer[decoded.z_index * buffer_stride];
                } else {
                    result = local_buffer[decoded.z_index];
                }
            }
        } else {
            // Need to evaluate this node (and all nodes up to it)
            for (; rules_evaluated <= node_idx; ++rules_evaluated) {
                Rule rule = d_rules[rules_evaluated];
                DecodedRule decoded = decode_rule(rule);

                FpExt x_val = evaluate_dag_entry(
                    decoded.x,
                    row,
                    d_selectors,
                    d_main,
                    height,
                    selectors_width,
                    d_preprocessed,
                    preprocessed_width,
                    d_eq_z,
                    d_eq_x,
                    d_public,
                    public_len,
                    inter_buffer,
                    buffer_stride,
                    buffer_size,
                    large_domain,
                    d_challenges
                );
                FpExt node_result;
                switch (decoded.op) {
                case OP_ADD: {
                    FpExt y_val = evaluate_dag_entry(
                        decoded.y,
                        row,
                        d_selectors,
                        d_main,
                        height,
                        selectors_width,
                        d_preprocessed,
                        preprocessed_width,
                        d_eq_z,
                        d_eq_x,
                        d_public,
                        public_len,
                        inter_buffer,
                        buffer_stride,
                        buffer_size,
                        large_domain,
                        d_challenges
                    );
                    node_result = x_val + y_val;
                    break;
                }
                case OP_SUB: {
                    FpExt y_val = evaluate_dag_entry(
                        decoded.y,
                        row,
                        d_selectors,
                        d_main,
                        height,
                        selectors_width,
                        d_preprocessed,
                        preprocessed_width,
                        d_eq_z,
                        d_eq_x,
                        d_public,
                        public_len,
                        inter_buffer,
                        buffer_stride,
                        buffer_size,
                        large_domain,
                        d_challenges
                    );
                    node_result = x_val - y_val;
                    break;
                }
                case OP_MUL: {
                    FpExt y_val = evaluate_dag_entry(
                        decoded.y,
                        row,
                        d_selectors,
                        d_main,
                        height,
                        selectors_width,
                        d_preprocessed,
                        preprocessed_width,
                        d_eq_z,
                        d_eq_x,
                        d_public,
                        public_len,
                        inter_buffer,
                        buffer_stride,
                        buffer_size,
                        large_domain,
                        d_challenges
                    );
                    node_result = x_val * y_val;
                    break;
                }
                case OP_NEG:
                    node_result = -x_val;
                    break;
                case OP_VAR:
                    node_result = x_val;
                    break;
                case OP_INV:
                    node_result = inv(x_val);
                    break;
                default:
                    node_result = FpExt(Fp::zero());
                    break;
                }

                if (decoded.buffer_result && buffer_size > 0) {
                    if constexpr (GLOBAL) {
                        inter_buffer[decoded.z_index * buffer_stride] = node_result;
                    } else {
                        local_buffer[decoded.z_index] = node_result;
                    }
                }

                if (rules_evaluated == node_idx) {
                    result = node_result;
                }
            }
        }

        // Alternate between numer and denom
        // Even indices (0, 2, 4...) are numerators
        // Odd indices (1, 3, 5...) are denominators
        if ((used_idx % 2) == 0) {
            // Numerator - store temporarily, will be weighted when we see denom
            temp_numer = result;
        } else {
            // Denominator - weight both numer and denom by eq_3b[interaction_idx]
            size_t interaction_idx = used_idx / 2;
            FpExt eq_3b_weight = d_eq_3b[interaction_idx];
            numer_sum += eq_3b_weight * temp_numer;
            denom_sum += eq_3b_weight * result;
        }
    }
}

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
    FracExt *__restrict__ d_output,
    const Fp *__restrict__ d_preprocessed,
    const uint64_t *__restrict__ d_main,
    const FpExt *__restrict__ d_challenges,
    const FpExt *__restrict__ d_intermediates,
    const Rule *__restrict__ d_rules,
    const size_t *__restrict__ d_used_nodes,
    const uint32_t *__restrict__ d_partition_lens,
    const size_t num_partitions,
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
        uint32_t col_offset = 0;
        uint32_t row = task_offset + j * task_stride;

        if (row < permutation_height) {
            uint32_t rules_evaluated = 0;
            uint32_t curr_node_idx = 0;
            bool is_denom = false;
            FpExt numerator(0);
            FpExt denom(0);

            for (uint32_t partition_idx = 0; partition_idx < num_partitions; partition_idx++) {
                uint32_t partition_len = 2 * d_partition_lens[partition_idx];

                for (uint32_t i = 0; i < partition_len; i++, curr_node_idx++) {
                    uint32_t node_idx = d_used_nodes[curr_node_idx];
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
                                intermediates_ptr[decoded_rule.z_index * intermediate_stride] =
                                    result;
                            }
                        }
                    }

                    if (is_denom) {
                        denom = result;
                        size_t interaction_idx = col_offset + (i >> 1);
                        size_t out_idx = interaction_idx * permutation_height + row;
                        d_output[out_idx].p = numerator;
                        d_output[out_idx].q = denom;
                    } else {
                        numerator = result;
                    }
                    is_denom = !is_denom;
                }
                col_offset += partition_len >> 1;
            }
        }
    }
}

// Round0 phase interactions kernel (for sumcheck round0)
template <bool GLOBAL>
__global__ void evaluate_interactions_round0_kernel(
    FpExt *__restrict__ d_output_numer,
    FpExt *__restrict__ d_output_denom,
    const Fp *__restrict__ d_selectors,
    uint32_t selectors_width,
    const MainMatrixPtrs *__restrict__ d_main,
    uint32_t main_count,
    const Fp *__restrict__ d_preprocessed,
    uint32_t preprocessed_width,
    const FpExt *__restrict__ d_eq_z,
    const FpExt *__restrict__ d_eq_x,
    const FpExt *__restrict__ d_eq_3b,
    const Fp *__restrict__ d_public,
    uint32_t public_len,
    const Rule *__restrict__ d_rules,
    size_t rules_len,
    const size_t *__restrict__ d_used_nodes,
    size_t used_nodes_len,
    uint32_t buffer_size,
    FpExt *__restrict__ d_intermediates,
    uint32_t large_domain,
    uint32_t num_x,
    uint32_t num_rows_per_tile,
    uint32_t skip_stride,
    const FpExt *__restrict__ d_challenges
) {
    uint32_t task_offset = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t task_stride = gridDim.x * blockDim.x;

    FpExt local_buffer[16];
    FpExt *inter_buffer;
    uint32_t buffer_stride;
    if constexpr (GLOBAL) {
        inter_buffer = d_intermediates + task_offset;
        buffer_stride = task_stride;
    } else {
        inter_buffer = local_buffer;
        buffer_stride = 1;
    }

    uint32_t height = large_domain * num_x;

    for (uint32_t tile = 0; tile < num_rows_per_tile; ++tile) {
        uint32_t row = task_offset + tile * task_stride;
        if (row >= height) {
            continue;
        }

        uint32_t z_idx = row % large_domain;
        uint32_t x_idx = row / large_domain;

        if (buffer_size > 0) {
            if constexpr (GLOBAL) {
                for (uint32_t idx = 0; idx < buffer_size; ++idx) {
                    inter_buffer[idx * buffer_stride] = FpExt(Fp::zero());
                }
            } else {
                uint32_t limit = buffer_size < 16 ? buffer_size : 16;
                for (uint32_t idx = 0; idx < limit; ++idx) {
                    local_buffer[idx] = FpExt(Fp::zero());
                }
            }
        }

        FpExt eq_val = d_eq_z[z_idx] * d_eq_x[x_idx];

        FpExt numer_sum(Fp::zero());
        FpExt denom_sum(Fp::zero());

        // Compute interaction sums (without eq_* multiplication)
        acc_interactions<GLOBAL>(
            row,
            d_selectors,
            d_main,
            height,
            selectors_width,
            d_preprocessed,
            preprocessed_width,
            d_eq_z,
            d_eq_x,
            d_public,
            public_len,
            d_eq_3b,
            d_rules,
            rules_len,
            d_used_nodes,
            used_nodes_len,
            buffer_size,
            inter_buffer,
            local_buffer,
            buffer_stride,
            large_domain,
            d_challenges,
            numer_sum,
            denom_sum
        );

        // Apply eq_val multiplier to both sums
        d_output_numer[row] = numer_sum * eq_val;
        d_output_denom[row] = denom_sum * eq_val;
    }
}

__global__ void add_alpha_kernel(FracExt *data, size_t len, FpExt alpha) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        data[idx].q = data[idx].q + alpha;
    }
}

template <typename F, typename EF>
__global__ void frac_vector_scalar_multiply_kernel(
    std::pair<EF, EF> *frac_vec,
    F scalar,
    uint32_t length
) {
    size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= length)
        return;

    frac_vec[tidx].first *= scalar;
}

static const size_t TASK_SIZE = 65536;

// ============================================================================
// LAUNCHERS
// ============================================================================

extern "C" int _zerocheck_eval_interactions_gkr(
    bool is_global,
    FracExt *d_output,
    const Fp *d_preprocessed,
    const uint64_t *d_main,
    const FpExt *d_challenges,
    const FpExt *d_intermediates,
    const Rule *d_rules,
    const size_t *d_used_nodes,
    const uint32_t *d_partition_lens,
    size_t num_partitions,
    uint32_t permutation_height,
    uint32_t num_rows_per_tile
) {
    auto [grid, block] = kernel_launch_params(TASK_SIZE, 256);
    if (is_global) {
        evaluate_interactions_gkr_kernel<true><<<grid, block>>>(
            d_output,
            d_preprocessed,
            d_main,
            d_challenges,
            d_intermediates,
            d_rules,
            d_used_nodes,
            d_partition_lens,
            num_partitions,
            permutation_height,
            num_rows_per_tile
        );
    } else {
        evaluate_interactions_gkr_kernel<false><<<grid, block>>>(
            d_output,
            d_preprocessed,
            d_main,
            d_challenges,
            d_intermediates,
            d_rules,
            d_used_nodes,
            d_partition_lens,
            num_partitions,
            permutation_height,
            num_rows_per_tile
        );
    }
    return CHECK_KERNEL();
}

extern "C" int _frac_add_alpha(FracExt *data, size_t len, FpExt alpha) {
    auto [grid, block] = kernel_launch_params(len);
    add_alpha_kernel<<<grid, block>>>(data, len, alpha);
    return CHECK_KERNEL();
}

extern "C" int _zerocheck_eval_interactions_round0(
    FpExt *output_numer,
    FpExt *output_denom,
    const Fp *selectors,
    uint32_t selectors_width,
    const MainMatrixPtrs *partitioned_main,
    uint32_t main_count,
    const Fp *preprocessed,
    uint32_t preprocessed_width,
    const FpExt *eq_z,
    const FpExt *eq_x,
    const FpExt *eq_3b,
    const Fp *public_values,
    uint32_t public_len,
    const Rule *rules,
    size_t rules_len,
    const size_t *used_nodes,
    size_t used_nodes_len,
    uint32_t buffer_size,
    FpExt *intermediates,
    uint32_t large_domain,
    uint32_t num_x,
    uint32_t num_rows_per_tile,
    uint32_t skip_stride,
    const FpExt *challenges
) {
    auto [grid, block] = kernel_launch_params(large_domain * num_x, 256);
#ifdef CUDA_DEBUG
    if (std::getenv("LOGUP_GPU_SINGLE_THREAD") != nullptr) {
        grid = dim3(1, 1, 1);
        block = dim3(1, 1, 1);
    }
#endif
    if (buffer_size > 16) {
        evaluate_interactions_round0_kernel<true><<<grid, block>>>(
            output_numer,
            output_denom,
            selectors,
            selectors_width,
            partitioned_main,
            main_count,
            preprocessed,
            preprocessed_width,
            eq_z,
            eq_x,
            eq_3b,
            public_values,
            public_len,
            rules,
            rules_len,
            used_nodes,
            used_nodes_len,
            buffer_size,
            intermediates,
            large_domain,
            num_x,
            num_rows_per_tile,
            skip_stride,
            challenges
        );
    } else {
        evaluate_interactions_round0_kernel<false><<<grid, block>>>(
            output_numer,
            output_denom,
            selectors,
            selectors_width,
            partitioned_main,
            main_count,
            preprocessed,
            preprocessed_width,
            eq_z,
            eq_x,
            eq_3b,
            public_values,
            public_len,
            rules,
            rules_len,
            used_nodes,
            used_nodes_len,
            buffer_size,
            intermediates,
            large_domain,
            num_x,
            num_rows_per_tile,
            skip_stride,
            challenges
        );
    }
    return CHECK_KERNEL();
}

extern "C" int _frac_vector_scalar_multiply_ext_fp(FracExt *frac_vec, Fp scalar, uint32_t length) {
    auto [grid, block] = kernel_launch_params(length);
    frac_vector_scalar_multiply_kernel<Fp, FpExt>
        <<<grid, block>>>(reinterpret_cast<std::pair<FpExt, FpExt> *>(frac_vec), scalar, length);
    return CHECK_KERNEL();
}
