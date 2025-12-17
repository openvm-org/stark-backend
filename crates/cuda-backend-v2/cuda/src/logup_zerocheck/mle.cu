#include "codec.cuh"
#include "eval_config.cuh"
#include "fp.h"
#include "fpext.h"
#include "frac_ext.cuh"
#include "launcher.cuh"
#include "matrix.cuh"
#include "sumcheck.cuh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdio.h>
#include <vector_types.h>

namespace logup_zerocheck_mle {

__device__ __forceinline__ FpExt evaluate_mle_entry(
    const SourceInfo &src,
    uint32_t row,
    const FpExt *__restrict__ d_selectors,
    const MainMatrixPtrs<FpExt> d_preprocessed,
    const MainMatrixPtrs<FpExt> *__restrict__ d_main,
    const Fp *__restrict__ d_public,
    uint32_t public_len,
    const FpExt *__restrict__ inter_buffer,
    uint32_t buffer_stride,
    const uint32_t height,
    const FpExt *__restrict__ d_challenges = nullptr
) {
    switch (src.type) {
    case ENTRY_PREPROCESSED: {
#ifdef CUDA_DEBUG
        assert(d_preprocessed.data);
#endif
        const auto stride = height * d_preprocessed.air_width;
        const FpExt *__restrict__ matrix = d_preprocessed.data + stride * src.offset;
        const FpExt *__restrict__ column = matrix + height * src.index;
        return column[row];
    }
    case ENTRY_MAIN: {
        auto main_ptr = d_main[src.part];
        const auto stride = height * main_ptr.air_width;
        const FpExt *__restrict__ matrix = main_ptr.data + stride * src.offset;
        const FpExt *__restrict__ column = matrix + height * src.index;
        return column[row];
    }
    case SRC_INTERMEDIATE:
        return inter_buffer[src.index * buffer_stride];
    case SRC_IS_FIRST: {
        const FpExt *__restrict__ column = d_selectors;
        return FpExt(column[row]);
    }
    case SRC_IS_LAST: {
        const FpExt *__restrict__ column = d_selectors + 2 * height;
        return FpExt(column[row]);
    }
    case SRC_IS_TRANSITION: {
        const FpExt *__restrict__ column = d_selectors + height;
        return FpExt(column[row]);
    }
    case ENTRY_CHALLENGE:
#ifdef CUDA_DEBUG
        assert(d_challenges);
#endif
        return d_challenges[src.index];
    case ENTRY_PUBLIC:
        return FpExt(d_public[src.index]);
    case SRC_CONSTANT:
        return FpExt(Fp(src.index));
    default:
        assert(false);
    }
    return FpExt(Fp::zero());
}

// ============================================================================
// KERNELS
// ============================================================================

constexpr uint32_t MAX_NUM_X = 5;
constexpr uint32_t ZEROCHECK_BUFFER_THRESHOLD = 16;
constexpr uint32_t LOGUP_BUFFER_THRESHOLD = 10;

template <bool GLOBAL>
__global__ void zerocheck_mle_kernel(
    FpExt *__restrict__ tmp_sums_buffer,
    const FpExt *__restrict__ d_eq_xi,
    const FpExt *__restrict__ d_selectors,
    const MainMatrixPtrs<FpExt> d_preprocessed,
    const MainMatrixPtrs<FpExt> *__restrict__ d_main,
    const FpExt *__restrict__ d_lambda_pows,
    const uint32_t *__restrict__ d_lambda_indices,
    const Fp *__restrict__ d_public,
    uint32_t public_len,
    const Rule *__restrict__ d_rules,
    size_t rules_len,
    const size_t *__restrict__ d_used_nodes,
    size_t used_nodes_len,
    size_t lambda_len,
    uint32_t buffer_size,
    FpExt *__restrict__ d_intermediates,
    uint32_t num_y,
    uint32_t num_x
) {
    extern __shared__ char smem[];
    FpExt *shared = (FpExt *)smem;

    uint32_t task_offset = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t task_stride = blockDim.x * gridDim.x;

    FpExt local_buffer[ZEROCHECK_BUFFER_THRESHOLD];
    FpExt *inter_buffer;
    uint32_t buffer_stride;
    if constexpr (GLOBAL) {
        inter_buffer = d_intermediates + task_offset;
        buffer_stride = task_stride;
    } else {
        inter_buffer = local_buffer;
        buffer_stride = 1;
    }

    uint32_t height = num_x * num_y;

    // Sumcheck sum, we sum over y_int's
    FpExt sum[MAX_NUM_X];
#pragma unroll
    for (uint32_t i = 0; i < MAX_NUM_X; ++i) {
        sum[i] = FpExt(Fp::zero());
    }

    for (uint32_t row = task_offset; row < height; row += task_stride) {
        FpExt constraint_sum(Fp::zero());
        size_t lambda_idx = 0; // Initialize lambda_idx

#define ZEROCHECK_EVAL_ARGS                                                                        \
    row, d_selectors, d_preprocessed, d_main, d_public, public_len, inter_buffer, buffer_stride,   \
        height

        for (size_t node = 0; node < rules_len; ++node) {
            Rule rule = d_rules[node];
            DecodedRule decoded = decode_rule(rule);

            FpExt x_val = evaluate_mle_entry(decoded.x, ZEROCHECK_EVAL_ARGS);
            FpExt result;
            switch (decoded.op) {
            case OP_ADD: {
                FpExt y_val = evaluate_mle_entry(decoded.y, ZEROCHECK_EVAL_ARGS);
                result = x_val + y_val;
                break;
            }
            case OP_SUB: {
                FpExt y_val = evaluate_mle_entry(decoded.y, ZEROCHECK_EVAL_ARGS);
                result = x_val - y_val;
                break;
            }
            case OP_MUL: {
                FpExt y_val = evaluate_mle_entry(decoded.y, ZEROCHECK_EVAL_ARGS);
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
            }

            if (decoded.buffer_result && buffer_size > 0) {
                if constexpr (GLOBAL) {
                    inter_buffer[decoded.z_index * buffer_stride] = result;
                } else {
                    local_buffer[decoded.z_index] = result;
                }
            }

            if (decoded.is_constraint) {
                while (lambda_idx < lambda_len && lambda_idx < used_nodes_len &&
                       d_used_nodes[lambda_idx] == node) {
                    uint32_t mapped_idx = d_lambda_indices != nullptr
                                              ? d_lambda_indices[lambda_idx]
                                              : static_cast<uint32_t>(lambda_idx);
                    FpExt lambda = d_lambda_pows[mapped_idx];
                    lambda_idx++;
                    constraint_sum += lambda * result;
                }
            }
        }

        FpExt eq_xi_val = d_eq_xi[row];
        uint32_t x_int = row % num_x;
        sum[x_int] += eq_xi_val * constraint_sum;
    }

    for (uint32_t i = 0; i < num_x; ++i) {
        FpExt reduced = sumcheck::block_reduce_sum(sum[i], shared);
        if (threadIdx.x == 0) {
            tmp_sums_buffer[blockIdx.x * num_x + i] = reduced;
        }
        __syncthreads();
    }
}

template <bool GLOBAL>
__global__ void logup_mle_kernel(
    FracExt *__restrict__ tmp_sums_buffer,
    const FpExt *__restrict__ d_eq_sharp,
    const FpExt *__restrict__ d_selectors,
    const MainMatrixPtrs<FpExt> d_preprocessed,
    const MainMatrixPtrs<FpExt> *__restrict__ d_main,
    const FpExt *__restrict__ d_challenges,
    const FpExt *__restrict__ d_eq_3bs,
    const Fp *__restrict__ d_public,
    uint32_t public_len,
    const Rule *__restrict__ d_rules,
    size_t rules_len,
    const size_t *__restrict__ d_used_nodes,
    size_t used_nodes_len,
    uint32_t buffer_size,
    FpExt *__restrict__ d_intermediates,
    uint32_t num_y,
    uint32_t num_x
) {
    extern __shared__ char smem[];
    FpExt *shared = reinterpret_cast<FpExt *>(smem);

    uint32_t task_offset = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t task_stride = blockDim.x * gridDim.x;

    FpExt local_buffer[LOGUP_BUFFER_THRESHOLD];
    FpExt *inter_buffer;
    uint32_t buffer_stride;
    if constexpr (GLOBAL) {
        inter_buffer = d_intermediates + task_offset;
        buffer_stride = task_stride;
    } else {
        inter_buffer = local_buffer;
        buffer_stride = 1;
    }

    uint32_t height = num_x * num_y;

    // Sumcheck sum, we sum over y_int's
    FracExt sum[MAX_NUM_X];
#pragma unroll
    for (uint32_t i = 0; i < MAX_NUM_X; i++) {
        sum[i] = {FpExt(Fp::zero()), FpExt(Fp::zero())};
    }

    for (uint32_t row = task_offset; row < height; row += task_stride) {
        FpExt numer_sum = FpExt(Fp::zero());
        FpExt denom_sum = FpExt(Fp::zero());

        // Track how many rules we've evaluated so far
        size_t rules_evaluated = 0;
        bool is_denom = false; // Alternates: false=numer, true=denom

#define LOGUP_EVAL_ARGS                                                                            \
    row, d_selectors, d_preprocessed, d_main, d_public, public_len, inter_buffer, buffer_stride,   \
        height, d_challenges

        // Iterate through used_nodes (alternates: numer, denom, numer, denom, ...)
        for (size_t used_idx = 0; used_idx < used_nodes_len; ++used_idx) {
            size_t node_idx = d_used_nodes[used_idx];
            FpExt result(Fp::zero());

            if (node_idx < rules_evaluated) {
                // Node already evaluated, get from buffer
                Rule rule = d_rules[node_idx];
                DecodedRule decoded = decode_rule(rule);
                if (decoded.op == OP_VAR) {
                    result = evaluate_mle_entry(decoded.x, LOGUP_EVAL_ARGS);
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

                    FpExt x_val = evaluate_mle_entry(decoded.x, LOGUP_EVAL_ARGS);
                    FpExt node_result;
                    switch (decoded.op) {
                    case OP_ADD: {
                        FpExt y_val = evaluate_mle_entry(decoded.y, LOGUP_EVAL_ARGS);
                        node_result = x_val + y_val;
                        break;
                    }
                    case OP_SUB: {
                        FpExt y_val = evaluate_mle_entry(decoded.y, LOGUP_EVAL_ARGS);
                        node_result = x_val - y_val;
                        break;
                    }
                    case OP_MUL: {
                        FpExt y_val = evaluate_mle_entry(decoded.y, LOGUP_EVAL_ARGS);
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
                    }

                    if (decoded.buffer_result && buffer_size > 0) {
                        if constexpr (GLOBAL) {
                            inter_buffer[decoded.z_index * buffer_stride] = node_result;
                        } else {
                            local_buffer[decoded.z_index] = node_result;
                        }
                    }

                    // If this is the node we're looking for, use it
                    if (rules_evaluated == node_idx) {
                        result = node_result;
                    }
                }
            }

            // Accumulate to numer or denom based on alternation
            // Weight by eq_3bs[used_idx / 2] (each interaction has numer and denom)
            result *= d_eq_3bs[used_idx / 2];

            if (is_denom) {
                denom_sum += result;
            } else {
                numer_sum += result;
            }
            is_denom = !is_denom;
        }

        FpExt eq_sharp_val = d_eq_sharp[row];
        uint32_t x_int = row % num_x;
        sum[x_int].p += eq_sharp_val * numer_sum;
        sum[x_int].q += eq_sharp_val * denom_sum;
    }

    for (uint32_t i = 0; i < num_x; ++i) {
        FpExt reduced = sumcheck::block_reduce_sum(sum[i].p, shared);
        if (threadIdx.x == 0) {
            tmp_sums_buffer[blockIdx.x * num_x + i].p = reduced;
        }
        __syncthreads();
        reduced = sumcheck::block_reduce_sum(sum[i].q, shared);
        if (threadIdx.x == 0) {
            tmp_sums_buffer[blockIdx.x * num_x + i].q = reduced;
        }
        __syncthreads();
    }
}

// ============================================================================
// LAUNCHERS
// ============================================================================
//
constexpr uint32_t MAX_THREADS = 256;

// (Not a launcher) Utility function to calculate required size of temp sum buffer.
// Required length of *temp_sum_buffer in FpExt elements
extern "C" size_t _zerocheck_mle_temp_sums_buffer_size(
    uint32_t buffer_size,
    uint32_t num_x,
    uint32_t num_y
) {
    return mle_rounds_config::temp_sums_buffer_size(
        buffer_size, num_x, num_y, ZEROCHECK_BUFFER_THRESHOLD, MAX_THREADS
    );
}

// In FpExt elements
extern "C" size_t _zerocheck_mle_intermediates_buffer_size(
    uint32_t buffer_size,
    uint32_t num_x,
    uint32_t num_y
) {
    return mle_rounds_config::intermediates_buffer_size(
        buffer_size, num_x, num_y, ZEROCHECK_BUFFER_THRESHOLD, MAX_THREADS
    );
}

extern "C" int _zerocheck_eval_mle(
    FpExt *tmp_sums_buffer,
    FpExt *output,
    const FpExt *eq_xi,
    const FpExt *selectors,
    const MainMatrixPtrs<FpExt> preprocessed,
    const MainMatrixPtrs<FpExt> *main,
    const FpExt *lambda_pows,
    const uint32_t *lambda_indices,
    const Fp *public_values,
    uint32_t public_len,
    const Rule *rules,
    size_t rules_len,
    const size_t *used_nodes,
    size_t used_nodes_len,
    size_t lambda_len,
    uint32_t buffer_size,
    FpExt *intermediates,
    uint32_t num_y,
    uint32_t num_x
) {
    assert(num_x <= MAX_NUM_X); // num_x = s_deg = max_constraint_degree + 1
    auto [grid, block] = mle_rounds_config::eval_constraints_launch_params(
        buffer_size, num_x, num_y, ZEROCHECK_BUFFER_THRESHOLD, MAX_THREADS
    );
    unsigned int num_warps = (block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t shmem_bytes = std::max(1u, num_warps) * sizeof(FpExt);

#define ZEROCHECK_KERNEL_ARGS                                                                      \
    tmp_sums_buffer, eq_xi, selectors, preprocessed, main, lambda_pows, lambda_indices,            \
        public_values, public_len, rules, rules_len, used_nodes, used_nodes_len, lambda_len,       \
        buffer_size, intermediates, num_y, num_x

    if (buffer_size > ZEROCHECK_BUFFER_THRESHOLD) {
        zerocheck_mle_kernel<true><<<grid, block, shmem_bytes>>>(ZEROCHECK_KERNEL_ARGS);
    } else {
        zerocheck_mle_kernel<false><<<grid, block, shmem_bytes>>>(ZEROCHECK_KERNEL_ARGS);
    }
    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Launch final reduction kernel - reads from block_sums, writes to output
    auto num_blocks = grid.x;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    sumcheck::final_reduce_block_sums<<<num_x, reduce_block, reduce_shmem>>>(
        tmp_sums_buffer, output, num_blocks
    );
    return CHECK_KERNEL();
}

// (Not a launcher) Utility function to calculate required size of temp sum buffer.
// Required length of *temp_sum_buffer in FracExt elements
extern "C" size_t _logup_mle_temp_sums_buffer_size(
    uint32_t buffer_size,
    uint32_t num_x,
    uint32_t num_y
) {
    return mle_rounds_config::temp_sums_buffer_size(
        buffer_size, num_x, num_y, LOGUP_BUFFER_THRESHOLD, MAX_THREADS
    );
}

// In FpExt elements
extern "C" size_t _logup_mle_intermediates_buffer_size(
    uint32_t buffer_size,
    uint32_t num_x,
    uint32_t num_y
) {
    return mle_rounds_config::intermediates_buffer_size(
        buffer_size, num_x, num_y, LOGUP_BUFFER_THRESHOLD, MAX_THREADS
    );
}

extern "C" int _logup_eval_mle(
    FracExt *tmp_sums_buffer,
    FracExt *output,
    const FpExt *eq_sharp,
    const FpExt *selectors,
    const MainMatrixPtrs<FpExt> preprocessed,
    const MainMatrixPtrs<FpExt> *main,
    const FpExt *challenges,
    const FpExt *eq_3bs,
    const Fp *public_values,
    uint32_t public_len,
    const Rule *rules,
    size_t rules_len,
    const size_t *used_nodes,
    size_t used_nodes_len,
    uint32_t buffer_size,
    FpExt *intermediates,
    uint32_t num_y,
    uint32_t num_x
) {
    assert(num_x <= MAX_NUM_X); // num_x = s_deg = max_constraint_degree + 1
    auto [grid, block] = mle_rounds_config::eval_constraints_launch_params(
        buffer_size, num_x, num_y, LOGUP_BUFFER_THRESHOLD, MAX_THREADS
    );
    unsigned int num_warps = (block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t shmem_bytes = std::max(1u, num_warps) * sizeof(FpExt);

#define LOGUP_KERNEL_ARGS                                                                          \
    tmp_sums_buffer, eq_sharp, selectors, preprocessed, main, challenges, eq_3bs, public_values,   \
        public_len, rules, rules_len, used_nodes, used_nodes_len, buffer_size, intermediates,      \
        num_y, num_x

    if (buffer_size > LOGUP_BUFFER_THRESHOLD) {
        logup_mle_kernel<true><<<grid, block, shmem_bytes>>>(LOGUP_KERNEL_ARGS);
    } else {
        logup_mle_kernel<false><<<grid, block, shmem_bytes>>>(LOGUP_KERNEL_ARGS);
    }
    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Launch final reduction kernel - reads from block_sums, writes to output
    auto num_blocks = grid.x;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    // FracExt = (FpExt, FpExt) so we set block = 2 * large_domain
    sumcheck::final_reduce_block_sums<<<2 * num_x, reduce_block, reduce_shmem>>>(
        reinterpret_cast<FpExt *>(tmp_sums_buffer), reinterpret_cast<FpExt *>(output), num_blocks
    );
    return CHECK_KERNEL();
}

} // namespace logup_zerocheck_mle
