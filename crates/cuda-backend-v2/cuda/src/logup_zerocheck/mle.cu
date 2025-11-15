#include "codec.cuh"
#include "dag_entry.cuh"
#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "matrix.cuh"
#include "sumcheck.cuh"

#include <cstdint>
#include <stdio.h>
#include <vector_types.h>

__device__ __forceinline__ FpExt evaluate_mle_entry(
    const SourceInfo &src,
    uint32_t row_index,
    const FpExt *d_selectors,
    const MainMatrixPtrs<FpExt> d_preprocessed,
    const MainMatrixPtrs<FpExt> *d_main,
    const Fp *__restrict__ d_public,
    uint32_t public_len,
    const FpExt *inter_buffer,
    uint32_t buffer_stride,
    const uint32_t height,
    const FpExt *__restrict__ d_challenges = nullptr
) {
    switch (src.type) {
    case ENTRY_PREPROCESSED: {
        if (d_preprocessed.data == nullptr) {
            return FpExt(Fp::zero());
        }
        const auto stride = height * d_preprocessed.air_width;
        const FpExt *matrix = d_preprocessed.data + stride * src.offset;
        return matrix[height * src.index + row_index];
    }
    case ENTRY_MAIN: {
        auto main_ptr = d_main[src.part];
        const auto stride = height * main_ptr.air_width;
        const FpExt *matrix = main_ptr.data + stride * src.offset;
        return matrix[height * src.index + row_index];
    }
    case SRC_INTERMEDIATE:
        return inter_buffer[src.index * buffer_stride];
    case SRC_IS_FIRST:
        return FpExt(d_selectors[row_index]);
    case SRC_IS_LAST:
        return FpExt(d_selectors[height * 2 + row_index]);
    case SRC_IS_TRANSITION:
        return FpExt(d_selectors[height + row_index]);
    case ENTRY_CHALLENGE:
        if (d_challenges != nullptr) {
            return d_challenges[src.index];
        }
        return FpExt(Fp::zero());
    case ENTRY_PERMUTATION:
    case ENTRY_EXPOSED:
        return FpExt(Fp::zero());
    case ENTRY_PUBLIC:
        return FpExt(d_public[src.index]);
    case SRC_CONSTANT:
        return FpExt(Fp(src.index));
    }
    return FpExt(Fp::zero());
}

template <bool GLOBAL>
__global__ void evaluate_mle_constraints_kernel(
    FpExt *__restrict__ d_output,
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
    uint32_t num_x,
    uint32_t num_rows_per_tile
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

    uint32_t height = num_x * num_y; // Total rows: s_deg * num_y

    for (uint32_t tile = 0; tile < num_rows_per_tile; ++tile) {
        uint32_t row = task_offset + tile * task_stride;
        if (row >= height) {
            continue;
        }

        FpExt eq_xi_val = d_eq_xi[row];

        FpExt constraint_sum = {0, 0, 0, 0};
        size_t lambda_idx = 0; // Initialize lambda_idx

        for (size_t node = 0; node < rules_len; ++node) {
            Rule rule = d_rules[node];
            DecodedRule decoded = decode_rule(rule);

            FpExt x_val = evaluate_mle_entry(
                decoded.x,
                row,
                d_selectors,
                d_preprocessed,
                d_main,
                d_public,
                public_len,
                inter_buffer,
                buffer_stride,
                height,
                nullptr
            );
            FpExt result;
            switch (decoded.op) {
            case OP_ADD: {
                FpExt y_val = evaluate_mle_entry(
                    decoded.y,
                    row,
                    d_selectors,
                    d_preprocessed,
                    d_main,
                    d_public,
                    public_len,
                    inter_buffer,
                    buffer_stride,
                    height,
                    nullptr
                );
                result = x_val + y_val;
                break;
            }
            case OP_SUB: {
                FpExt y_val = evaluate_mle_entry(
                    decoded.y,
                    row,
                    d_selectors,
                    d_preprocessed,
                    d_main,
                    d_public,
                    public_len,
                    inter_buffer,
                    buffer_stride,
                    height,
                    nullptr
                );
                result = x_val - y_val;
                break;
            }
            case OP_MUL: {
                FpExt y_val = evaluate_mle_entry(
                    decoded.y,
                    row,
                    d_selectors,
                    d_preprocessed,
                    d_main,
                    d_public,
                    public_len,
                    inter_buffer,
                    buffer_stride,
                    height,
                    nullptr
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

        uint32_t y_idx = row % num_y;
        uint32_t x_idx = row / num_y;
        d_output[x_idx * num_y + y_idx] = eq_xi_val * constraint_sum;
    }
}

template <bool GLOBAL>
__global__ void evaluate_mle_interactions_kernel(
    FpExt *__restrict__ d_output_numer,
    FpExt *__restrict__ d_output_denom,
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
    uint32_t num_x,
    uint32_t num_rows_per_tile
) {
    uint32_t task_offset = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t task_stride = gridDim.x * blockDim.x;

    FpExt local_buffer[10];
    FpExt *inter_buffer;
    uint32_t buffer_stride;
    if constexpr (GLOBAL) {
        inter_buffer = d_intermediates + task_offset;
        buffer_stride = task_stride;
    } else {
        inter_buffer = local_buffer;
        buffer_stride = 1;
    }

    uint32_t height = num_x * num_y; // Total rows: s_deg * num_y

    for (uint32_t tile = 0; tile < num_rows_per_tile; ++tile) {
        uint32_t row = task_offset + tile * task_stride;
        if (row >= height) {
            continue;
        }

        FpExt eq_sharp_val = d_eq_sharp[row];

        // Initialize buffers if needed
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

        FpExt numer_sum = FpExt(Fp::zero());
        FpExt denom_sum = FpExt(Fp::zero());

        // Track how many rules we've evaluated so far
        size_t rules_evaluated = 0;
        bool is_denom = false; // Alternates: false=numer, true=denom

        // Iterate through used_nodes (alternates: numer, denom, numer, denom, ...)
        for (size_t used_idx = 0; used_idx < used_nodes_len; ++used_idx) {
            size_t node_idx = d_used_nodes[used_idx];
            FpExt result(Fp::zero());

            if (node_idx < rules_evaluated) {
                // Node already evaluated, get from buffer
                Rule rule = d_rules[node_idx];
                DecodedRule decoded = decode_rule(rule);
                if (decoded.op == OP_VAR) {
                    result = evaluate_mle_entry(
                        decoded.x,
                        row,
                        d_selectors,
                        d_preprocessed,
                        d_main,
                        d_public,
                        public_len,
                        inter_buffer,
                        buffer_stride,
                        height,
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

                    FpExt x_val = evaluate_mle_entry(
                        decoded.x,
                        row,
                        d_selectors,
                        d_preprocessed,
                        d_main,
                        d_public,
                        public_len,
                        inter_buffer,
                        buffer_stride,
                        height,
                        d_challenges
                    );
                    FpExt node_result;
                    switch (decoded.op) {
                    case OP_ADD: {
                        FpExt y_val = evaluate_mle_entry(
                            decoded.y,
                            row,
                            d_selectors,
                            d_preprocessed,
                            d_main,
                            d_public,
                            public_len,
                            inter_buffer,
                            buffer_stride,
                            height,
                            d_challenges
                        );
                        node_result = x_val + y_val;
                        break;
                    }
                    case OP_SUB: {
                        FpExt y_val = evaluate_mle_entry(
                            decoded.y,
                            row,
                            d_selectors,
                            d_preprocessed,
                            d_main,
                            d_public,
                            public_len,
                            inter_buffer,
                            buffer_stride,
                            height,
                            d_challenges
                        );
                        node_result = x_val - y_val;
                        break;
                    }
                    case OP_MUL: {
                        FpExt y_val = evaluate_mle_entry(
                            decoded.y,
                            row,
                            d_selectors,
                            d_preprocessed,
                            d_main,
                            d_public,
                            public_len,
                            inter_buffer,
                            buffer_stride,
                            height,
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

        uint32_t y_idx = row % num_y;
        uint32_t x_idx = row / num_y;
        d_output_numer[x_idx * num_y + y_idx] = eq_sharp_val * numer_sum;
        d_output_denom[x_idx * num_y + y_idx] = eq_sharp_val * denom_sum;
    }
}

// Phase 1: Block-level reduction - parallelize over y_idx, reduce within blocks
// Layout: evaluated[y_idx * s_deg + x_idx] (column-major for x)
__global__ void reduce_hypercube_block_kernel(
    FpExt *block_sums,      // Output: [gridDim.x][s_deg] - partial sums per block
    const FpExt *evaluated, // [s_deg * num_y]
    uint32_t s_deg,
    uint32_t num_y
) {
    extern __shared__ char smem[];
    FpExt *shared = reinterpret_cast<FpExt *>(smem);

    uint32_t y_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid = threadIdx.x;

    // Each thread reads its y_idx values for all x_idx
    FpExt local_sums[5]; // s_deg is typically <= 5
    for (uint32_t x = 0; x < s_deg; ++x) {
        if (y_idx < num_y) {
            local_sums[x] = evaluated[y_idx * s_deg + x];
        } else {
            local_sums[x] = FpExt(Fp::zero());
        }
    }

    // Reduce each x_idx separately using block_reduce_sum
    for (uint32_t x = 0; x < s_deg; ++x) {
        FpExt reduced = sumcheck::block_reduce_sum(local_sums[x], shared);
        if (tid == 0) {
            block_sums[blockIdx.x * s_deg + x] = reduced;
        }
        __syncthreads(); // Needed before reusing shared memory
    }
}

// Phase 2: Final reduction - combine block sums into final result
__global__ void reduce_hypercube_final_kernel(
    FpExt *output,           // [s_deg] - final output
    const FpExt *block_sums, // [num_blocks][s_deg] - partial sums from phase 1
    uint32_t s_deg,
    uint32_t num_blocks
) {
    extern __shared__ char smem[];
    FpExt *shared = reinterpret_cast<FpExt *>(smem);

    uint32_t tid = threadIdx.x;
    uint32_t x_idx = blockIdx.x; // Each block handles one x_idx

    if (x_idx >= s_deg)
        return;

    // Each thread accumulates subset of blocks
    FpExt sum = FpExt(Fp::zero());
    for (uint32_t block_id = tid; block_id < num_blocks; block_id += blockDim.x) {
        sum += block_sums[block_id * s_deg + x_idx];
    }

    // Block-level reduction
    sum = sumcheck::block_reduce_sum(sum, shared);

    if (tid == 0) {
        output[x_idx] = sum;
    }
}

extern "C" int _reduce_hypercube_blocks(
    FpExt *block_sums,
    const FpExt *evaluated,
    uint32_t s_deg,
    uint32_t num_y
) {
    constexpr uint32_t BLOCK_SIZE = 256;
    uint32_t num_blocks = (num_y + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE);

    // Shared memory: enough for block_reduce_sum (needs space for warps)
    uint32_t num_warps = (BLOCK_SIZE + 31) / 32;
    size_t shmem_bytes = sizeof(FpExt) * num_warps;

    reduce_hypercube_block_kernel<<<grid, block, shmem_bytes>>>(
        block_sums, evaluated, s_deg, num_y
    );

    return CHECK_KERNEL();
}

extern "C" int _reduce_hypercube_final(
    FpExt *output,
    const FpExt *block_sums,
    uint32_t s_deg,
    uint32_t num_blocks
) {
    constexpr uint32_t BLOCK_SIZE = 256;

    dim3 grid(s_deg); // One block per x_idx
    dim3 block(BLOCK_SIZE);

    // Shared memory: enough for block_reduce_sum
    uint32_t num_warps = (BLOCK_SIZE + 31) / 32;
    size_t shmem_bytes = sizeof(FpExt) * num_warps;

    reduce_hypercube_final_kernel<<<grid, block, shmem_bytes>>>(
        output, block_sums, s_deg, num_blocks
    );

    return CHECK_KERNEL();
}

extern "C" int _zerocheck_eval_mle(
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
    uint32_t num_x,
    uint32_t num_rows_per_tile
) {
    auto count = constraint_evaluation::get_launcher_count(buffer_size, num_x * num_y);
    auto [grid, block] = kernel_launch_params(count, 256);

    if (buffer_size > constraint_evaluation::BUFFER_THRESHOLD) {
        evaluate_mle_constraints_kernel<true><<<grid, block>>>(
            output,
            eq_xi,
            selectors,
            preprocessed,
            main,
            lambda_pows,
            lambda_indices,
            public_values,
            public_len,
            rules,
            rules_len,
            used_nodes,
            used_nodes_len,
            lambda_len,
            buffer_size,
            intermediates,
            num_y,
            num_x,
            num_rows_per_tile
        );
    } else {
        evaluate_mle_constraints_kernel<false><<<grid, block>>>(
            output,
            eq_xi,
            selectors,
            preprocessed,
            main,
            lambda_pows,
            lambda_indices,
            public_values,
            public_len,
            rules,
            rules_len,
            used_nodes,
            used_nodes_len,
            lambda_len,
            buffer_size,
            intermediates,
            num_y,
            num_x,
            num_rows_per_tile
        );
    }
    return CHECK_KERNEL();
}

extern "C" int _zerocheck_eval_mle_interactions(
    FpExt *output_numer,
    FpExt *output_denom,
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
    uint32_t num_x,
    uint32_t num_rows_per_tile
) {
    auto count = interaction_evaluation::get_launcher_count(buffer_size, num_x * num_y);
    auto [grid, block] = kernel_launch_params(count, 256);

    if (buffer_size > interaction_evaluation::BUFFER_THRESHOLD) {
        evaluate_mle_interactions_kernel<true><<<grid, block>>>(
            output_numer,
            output_denom,
            eq_sharp,
            selectors,
            preprocessed,
            main,
            challenges,
            eq_3bs,
            public_values,
            public_len,
            rules,
            rules_len,
            used_nodes,
            used_nodes_len,
            buffer_size,
            intermediates,
            num_y,
            num_x,
            num_rows_per_tile
        );
    } else {
        evaluate_mle_interactions_kernel<false><<<grid, block>>>(
            output_numer,
            output_denom,
            eq_sharp,
            selectors,
            preprocessed,
            main,
            challenges,
            eq_3bs,
            public_values,
            public_len,
            rules,
            rules_len,
            used_nodes,
            used_nodes_len,
            buffer_size,
            intermediates,
            num_y,
            num_x,
            num_rows_per_tile
        );
    }

    return CHECK_KERNEL();
}
