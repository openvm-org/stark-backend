// FROM https://github.com/scroll-tech/plonky3-gpu/blob/fa356768aad31f4cf27d724336bb0323bb9d66eb/gpu-backend/src/cuda/kernels/permute.cu

#include "codec.cuh"
#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include <cstdio>

// read the input operands
__host__ __device__ __forceinline__ FpExt permute_entry(
    const SourceInfo &src,
    uint32_t row_index,
    const Fp *d_preprocessed,
    const uint64_t *d_main, // partitioned main ptr
    const FpExt *d_challenges,
    const FpExt *d_intermediates,
    const uint32_t intermediate_stride,
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
        result = d_intermediates[intermediate_stride * src.index];
        break;
    case SRC_CONSTANT:
        result = FpExt(Fp(src.index));
        break;
    default:
        // Handle error
        ;
    }
    return result;
}

__global__ void cukernel_permute_trace_gen_global(
    Fp *d_permutation,
    FpExt *d_cumulative_sums,
    const Fp *d_preprocessed,
    const uint64_t *d_main, // partitioned main ptr
    const FpExt *d_challenges,
    const FpExt *d_intermediates,
    // params
    const Rule *d_rules,
    const uint32_t num_rules,
    const uint32_t permutation_height,
    const uint32_t permutation_width_ext,
    const uint32_t num_rows_per_tile
) {
    uint32_t task_offset = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t task_stride = gridDim.x * blockDim.x;

    FpExt *intermediates_ptr = (FpExt *)d_intermediates + task_offset;
    uint32_t intermediate_stride = task_stride;

    for (uint32_t j = 0; j < num_rows_per_tile; j++) {
        uint32_t col_offset = 0;
        uint32_t row = task_offset + j * task_stride;
        bool valid = row < permutation_height;

        FpExt cumulative_sums(0);
        if (valid) {
            for (uint32_t i = 0; i < num_rules; i++) {
                __syncthreads();
                Rule rule = d_rules[i];
                DecodedRule decoded_rule = decode_rule(rule);

                // read input operands
                FpExt x = permute_entry(
                    decoded_rule.x,
                    row,
                    d_preprocessed,
                    d_main,
                    d_challenges,
                    intermediates_ptr,
                    intermediate_stride,
                    permutation_height
                );
                FpExt y = permute_entry(
                    decoded_rule.y,
                    row,
                    d_preprocessed,
                    d_main,
                    d_challenges,
                    intermediates_ptr,
                    intermediate_stride,
                    permutation_height
                );
                FpExt result(0); // = {0, 0, 0, 0};
                switch (decoded_rule.op) {
                case OP_ADD:
                    result = x + y;
                    break;
                case OP_SUB:
                    result = x - y;
                    break;
                case OP_MUL:
                    x *= y;
                    result += x;
                    break;
                case OP_NEG:
                    result = -x;
                    break;
                case OP_VAR:
                    result = x;
                    break;
                case OP_INV:
                    result = binomial_inversion(x);
                    break;
                default:;
                }

                if (decoded_rule.op != OP_VAR) {
                    intermediates_ptr[decoded_rule.z.index * intermediate_stride] = result;
                }

                // `is_constraint` here is used to determine whether to write to
                // permutation matrix
                if (decoded_rule.is_constraint) {
                    if (col_offset < permutation_width_ext) {
                        cumulative_sums += result;
                    }
                    // write to permutation ext matrix
                    // each ext column is represented by `D` columns over base field
                    uint32_t perm_idx =
                        col_offset * permutation_height * 4 + row; // D=4: extension field
                    d_permutation[permutation_height * 0 + perm_idx] = result.elems[0];
                    d_permutation[permutation_height * 1 + perm_idx] = result.elems[1];
                    d_permutation[permutation_height * 2 + perm_idx] = result.elems[2];
                    d_permutation[permutation_height * 3 + perm_idx] = result.elems[3];

                    col_offset += 1;
                }
            }
            d_cumulative_sums[row] = cumulative_sums;
        }
    }
}

__global__ void cukernel_permute_trace_gen_register(
    Fp *d_permutation,
    FpExt *d_cumulative_sums,
    const Fp *d_preprocessed,
    const uint64_t *d_main, // partitioned main ptr
    const FpExt *d_challenges,
    const FpExt *d_intermediates,
    // params
    const Rule *d_rules,
    const uint32_t num_rules,
    const uint32_t permutation_height,
    const uint32_t permutation_width_ext,
    const uint32_t num_rows_per_tile
) {
    uint32_t task_offset = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t task_stride = gridDim.x * blockDim.x;

    FpExt intermediates[10];
    FpExt *intermediates_ptr = intermediates;
    const uint32_t intermediate_stride = 1;

    for (uint32_t j = 0; j < num_rows_per_tile; j++) {
        uint32_t col_offset = 0;
        uint32_t row = task_offset + j * task_stride;
        bool valid = row < permutation_height;

        FpExt cumulative_sums(0);
        if (valid) {
            for (uint32_t i = 0; i < num_rules; i++) {
                // coalesced memory access??
                __syncthreads();
                Rule rule = d_rules[i];
                DecodedRule decoded_rule = decode_rule(rule);

                FpExt x = permute_entry(
                    decoded_rule.x,
                    row,
                    d_preprocessed,
                    d_main,
                    d_challenges,
                    intermediates_ptr,
                    intermediate_stride,
                    permutation_height
                );
                FpExt y = permute_entry(
                    decoded_rule.y,
                    row,
                    d_preprocessed,
                    d_main,
                    d_challenges,
                    intermediates_ptr,
                    intermediate_stride,
                    permutation_height
                );

                FpExt result(0); // = {0, 0, 0, 0};
                switch (decoded_rule.op) {
                case OP_ADD:
                    result = x + y;
                    break;
                case OP_SUB:
                    result = x - y;
                    break;
                case OP_MUL:
                    x *= y;
                    result += x;
                    break;
                case OP_NEG:
                    result = -x;
                    break;
                case OP_VAR:
                    result = x;
                    break;
                case OP_INV:
                    result = binomial_inversion(x);
                    break;
                default:;
                }

                if (decoded_rule.op != OP_VAR) {
                    intermediates_ptr[decoded_rule.z.index * intermediate_stride] = result;
                }

                if (decoded_rule.is_constraint) {
                    if (col_offset < permutation_width_ext) {
                        cumulative_sums += result;
                    }
                    uint32_t perm_idx =
                        col_offset * permutation_height * 4 + row; // D=4: extension field
                    d_permutation[permutation_height * 0 + perm_idx] = result.elems[0];
                    d_permutation[permutation_height * 1 + perm_idx] = result.elems[1];
                    d_permutation[permutation_height * 2 + perm_idx] = result.elems[2];
                    d_permutation[permutation_height * 3 + perm_idx] = result.elems[3];
                    col_offset += 1;
                }
            }

            d_cumulative_sums[row] = cumulative_sums;
        }
    }
}

__global__ void cukernel_permute_update(
    FpExt *d_sum,
    Fp *d_permutation,
    FpExt *d_cumulative_sums,
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

// END OF FILE gpu-backend/src/cuda/kernels/permute.cu

static const size_t TASK_SIZE = 65536;

extern "C" int _permute_trace_gen_global(
    Fp *d_permutation,
    FpExt *d_cumulative_sums,
    const Fp *d_preprocessed,
    const uint64_t *d_main, // partitioned main ptr
    const FpExt *d_challenges,
    const FpExt *d_intermediates,
    // params
    const Rule *d_rules,
    const uint32_t num_rules,
    const uint32_t permutation_height,
    const uint32_t permutation_width_ext,
    const uint32_t num_rows_per_tile
) {
    auto [grid, block] = kernel_launch_params(TASK_SIZE, 256);
    cukernel_permute_trace_gen_global<<<grid, block>>>(
        d_permutation,
        d_cumulative_sums,
        d_preprocessed,
        d_main,
        d_challenges,
        d_intermediates,
        d_rules,
        num_rules,
        permutation_height,
        permutation_width_ext,
        num_rows_per_tile
    );
    return cudaGetLastError();
}

extern "C" int _permute_trace_gen_register(
    Fp *d_permutation,
    FpExt *d_cumulative_sums,
    const Fp *d_preprocessed,
    const uint64_t *d_main, // partitioned main ptr
    const FpExt *d_challenges,
    const FpExt *d_intermediates,
    // params
    const Rule *d_rules,
    const uint32_t num_rules,
    const uint32_t permutation_height,
    const uint32_t permutation_width_ext,
    const uint32_t num_rows_per_tile
) {
    auto [grid, block] = kernel_launch_params(TASK_SIZE, 256);
    cukernel_permute_trace_gen_register<<<grid, block>>>(
        d_permutation,
        d_cumulative_sums,
        d_preprocessed,
        d_main,
        d_challenges,
        d_intermediates,
        d_rules,
        num_rules,
        permutation_height,
        permutation_width_ext,
        num_rows_per_tile
    );
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