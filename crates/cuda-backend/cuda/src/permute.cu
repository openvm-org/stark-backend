/*
 * Source: https://github.com/scroll-tech/plonky3-gpu (private repo)
 * Status: BASED ON plonky3-gpu/gpu-backend/src/cuda/kernels/permute.cu
 * Imported: 2025-01-25 by @gaxiom
 */

#include "codec.cuh"
#include "constraint_buffer.cuh"
#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#ifdef DEBUG
#include <cstdio>
#endif

// read the input operands
__device__ __forceinline__ FpExt permute_entry(
    const SourceInfo &src,
    uint32_t row_index,
    const Fp *d_preprocessed,
    const uint64_t *d_main, // partitioned main ptr
    const FpExt *d_challenges,
    ConstraintBuffer &buffer,
    const uint32_t task_stride,
    const uint32_t height
) {
    FpExt result(0); //{0, 0, 0, 0};
    switch (src.type) {
    case ENTRY_PREPROCESSED:
        result = FpExt(d_preprocessed[height * src.index + ((row_index + src.offset) % height)]);
        break;
    case ENTRY_MAIN: {
        // src.offset: {0,1}, 1: next_row
        Fp *d_main_fp = (Fp *)d_main[src.part];
        result = FpExt(d_main_fp[height * src.index + ((row_index + src.offset) % height)]);
        break;
    }
    case ENTRY_CHALLENGE:
        result = d_challenges[src.index];
        break;
    case SRC_INTERMEDIATE:
        result = buffer.read(src.index, task_stride);
        break;
    case SRC_CONSTANT:
        result = FpExt(Fp(src.index));
        break;
    default:
        assert(0);
    }
    return result;
}

__global__ void calculate_cumulative_sums(
    Fp *__restrict__ d_permutation,
    FpExt *__restrict__ d_cumulative_sums,
    const Fp *__restrict__ d_preprocessed,
    const uint64_t *__restrict__ d_main, // partitioned main ptr
    const FpExt *__restrict__ d_challenges,
    const FpExt *__restrict__ d_intermediates,
    // params
    const Rule *__restrict__ d_rules,
    const size_t *__restrict__ d_used_nodes,
    const uint32_t *__restrict__ d_partition_lens,
    const size_t num_partitions,
    const uint32_t permutation_height,
    const uint32_t permutation_width_ext,
    const uint32_t num_rows_per_tile
) {
    uint32_t task_offset = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t task_stride = gridDim.x * blockDim.x;

    ConstraintBuffer buffer((FpExt *)d_intermediates + task_offset);

    for (uint32_t j = 0; j < num_rows_per_tile; j++) {
        uint32_t row = task_offset + j * task_stride;

        if (row < permutation_height) {
            FpExt cumulative_sums(0);
            uint32_t rules_evaluated = 0;
            uint32_t curr_node_idx = 0;

            for (uint32_t partition_idx = 0; partition_idx < num_partitions; partition_idx++) {
                uint32_t partition_len = 2 * d_partition_lens[partition_idx];
                FpExt local_sum(0);
                FpExt numerator(0);
                for (uint32_t i = 0; i < partition_len; i++, curr_node_idx++) {
                    uint32_t node_idx = d_used_nodes[curr_node_idx];
                    FpExt result(0);
                    if (node_idx < rules_evaluated) {
                        Rule rule = d_rules[node_idx];
                        DecodedRule decoded_rule = decode_rule(rule);
                        if (decoded_rule.op == OP_VAR) {
                            result = permute_entry(
                                decoded_rule.x,
                                row,
                                d_preprocessed,
                                d_main,
                                d_challenges,
                                buffer,
                                task_stride,
                                permutation_height
                            );
                        } else {
                            result = buffer.read(decoded_rule.z.index, task_stride);
                        }
                    } else {
                        for (; rules_evaluated <= node_idx; rules_evaluated++) {
                            Rule rule = d_rules[rules_evaluated];
                            DecodedRule decoded_rule = decode_rule(rule);

                            // read input operands
                            FpExt x = permute_entry(
                                decoded_rule.x,
                                row,
                                d_preprocessed,
                                d_main,
                                d_challenges,
                                buffer,
                                task_stride,
                                permutation_height
                            );
                            FpExt y = permute_entry(
                                decoded_rule.y,
                                row,
                                d_preprocessed,
                                d_main,
                                d_challenges,
                                buffer,
                                task_stride,
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

                            if (decoded_rule.op != OP_VAR) {
                                buffer.write(decoded_rule.z.index, result, task_stride);
                            }
                        }
                    }
                    if ((curr_node_idx & 1) == 1) {
                        local_sum += (numerator * binomial_inversion(result));
                    } else {
                        numerator = result;
                    }
                }
                if (partition_idx < permutation_width_ext) {
                    cumulative_sums += local_sum;
                }
                // write to permutation ext matrix
                // each ext column is represented by `D` columns over base field
                uint32_t perm_idx =
                    partition_idx * permutation_height * 4 + row; // D=4: extension field
                d_permutation[permutation_height * 0 + perm_idx] = local_sum.elems[0];
                d_permutation[permutation_height * 1 + perm_idx] = local_sum.elems[1];
                d_permutation[permutation_height * 2 + perm_idx] = local_sum.elems[2];
                d_permutation[permutation_height * 3 + perm_idx] = local_sum.elems[3];
            }
            d_cumulative_sums[row] = cumulative_sums;
        } else {
            break;
        }
    }
}

__global__ void cukernel_permute_update(
    FpExt *__restrict__ d_sum,
    Fp *__restrict__ d_permutation,
    FpExt *__restrict__ d_cumulative_sums,
    const uint32_t permutation_height,
    const uint32_t permutation_width_ext
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    // write the last column: permutation_width_ext - 1
    if (row < permutation_height) {
        uint32_t col_offset = permutation_width_ext - 1;
        FpExt cumulative_sums = d_cumulative_sums[row];
        uint32_t perm_idx = col_offset * permutation_height * 4 + row; // D=4: extension field
        d_permutation[permutation_height * 0 + perm_idx] = cumulative_sums.elems[0];
        d_permutation[permutation_height * 1 + perm_idx] = cumulative_sums.elems[1];
        d_permutation[permutation_height * 2 + perm_idx] = cumulative_sums.elems[2];
        d_permutation[permutation_height * 3 + perm_idx] = cumulative_sums.elems[3];

        if (row == permutation_height - 1) {
            *d_sum = cumulative_sums;
        }
    }
}

// LAUNCHERS

static const size_t TASK_SIZE = 65536;

extern "C" int _calculate_cumulative_sums(
    Fp *d_permutation,
    FpExt *d_cumulative_sums,
    const Fp *d_preprocessed,
    const uint64_t *d_main, // partitioned main ptr
    const FpExt *d_challenges,
    const FpExt *d_intermediates,
    // params
    const Rule *d_rules,
    const size_t *d_used_nodes,
    const uint32_t *d_partition_lens,
    const size_t num_partitions,
    const uint32_t permutation_height,
    const uint32_t permutation_width_ext,
    const uint32_t num_rows_per_tile
) {
    auto [grid, block] = kernel_launch_params(TASK_SIZE, 256);

    #define PERMUTE_ARGUMENTS \
        d_permutation, \
        d_cumulative_sums, \
        d_preprocessed, \
        d_main, \
        d_challenges, \
        d_intermediates, \
        d_rules, \
        d_used_nodes, \
        d_partition_lens, \
        num_partitions, \
        permutation_height, \
        permutation_width_ext, \
        num_rows_per_tile

    calculate_cumulative_sums<<<grid, block>>>(PERMUTE_ARGUMENTS);
    return cudaGetLastError();
}

extern "C" int _permute_update(
    FpExt *d_sum,
    Fp *d_permutation,
    FpExt *d_cumulative_sums,
    const uint32_t permutation_height,
    const uint32_t permutation_width_ext
) {
    auto [grid, block] = kernel_launch_params(permutation_height);
    cukernel_permute_update<<<grid, block>>>(
        d_sum, d_permutation, d_cumulative_sums, permutation_height, permutation_width_ext
    );
    return cudaGetLastError();
}
