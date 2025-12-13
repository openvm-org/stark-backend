#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector_types.h>

#include "codec.cuh"
#include "dag_entry.cuh"
#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "sumcheck.cuh"
#include "utils.cuh"

using namespace symbolic_dag;

namespace zerocheck_round0 {
// Device function equivalent to helper.acc_constraints without eq_* parts
// This computes the constraint sum: sum(lambda_i * constraint_i) for all constraints
template <bool GLOBAL>
__device__ __forceinline__ FpExt acc_constraints(
    const DagPrismEvalContext &eval_ctx,
    const FpExt *__restrict__ d_lambda_pows,
    const uint32_t *__restrict__ d_lambda_indices,
    const Rule *__restrict__ d_rules,
    size_t rules_len,
    const size_t *__restrict__ d_used_nodes,
    size_t used_nodes_len,
    size_t lambda_len,
    Fp *__restrict__ local_buffer
) {
    size_t lambda_idx = 0;
    FpExt constraint_sum(Fp::zero());

    for (size_t node = 0; node < rules_len; ++node) {
        Rule rule = d_rules[node];
        DecodedRule decoded = decode_rule(rule);

        Fp x_val = bary_eval_dag_entry(decoded.x, eval_ctx);
        Fp result;
        switch (decoded.op) {
        case OP_ADD: {
            Fp y_val = bary_eval_dag_entry(decoded.y, eval_ctx);
            result = x_val + y_val;
            break;
        }
        case OP_SUB: {
            Fp y_val = bary_eval_dag_entry(decoded.y, eval_ctx);
            result = x_val - y_val;
            break;
        }
        case OP_MUL: {
            Fp y_val = bary_eval_dag_entry(decoded.y, eval_ctx);
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

        if (decoded.buffer_result && eval_ctx.buffer_size > 0) {
#ifdef CUDA_DEBUG
            assert(decoded.z_index < eval_ctx.buffer_size);
#endif
            if constexpr (GLOBAL) {
                // Note: decoded.z_index refers to a decoding index, it doesn't have to do with prism coordinate
                eval_ctx.inter_buffer[decoded.z_index * eval_ctx.buffer_stride] = result;
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

    return constraint_sum;
}

constexpr uint32_t BUFFER_THRESHOLD = 16;

template <bool GLOBAL>
__global__ void zerocheck_bary_evaluate_constraints_kernel(
    FpExt *__restrict__ tmp_sums_buffer,   // [blockDim.y * gridDim.y][large_domain]
    const Fp *__restrict__ selectors_cube, // [3][num_x]
    const Fp *__restrict__ preprocessed,
    const Fp *const *__restrict__ main_parts,
    const Fp *__restrict__ omega_skip_pows,     // [skip_domain]
    const Fp *__restrict__ inv_lagrange_denoms, // [large_domain][skip_domain]
    const FpExt *__restrict__ eq_uni,           // [large_domain]
    const FpExt *__restrict__ eq_cube,          // [num_x]
    const FpExt *__restrict__ d_lambda_pows,
    const uint32_t *__restrict__ d_lambda_indices,
    const Fp *__restrict__ public_values,
    const Rule *__restrict__ d_rules,
    size_t rules_len,
    const size_t *__restrict__ d_used_nodes,
    size_t used_nodes_len,
    size_t lambda_len,
    uint32_t buffer_size,
    Fp *__restrict__ d_intermediates,
    uint32_t large_domain,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t height,
    uint32_t
        expansion_factor, // large_domain.next_power_of_two() / skip_domain: determines when to skip barycentric eval
    std::pair<uint32_t, uint32_t> z_dim
) {
    extern __shared__ char smem[];
    FpExt *shared = reinterpret_cast<FpExt *>(smem);

    // We unlinearize the thread index
    auto [zs_per_grid, zs_per_block] = z_dim;
    uint32_t xs_per_block = blockDim.x / zs_per_block;
    uint32_t tidx_z = threadIdx.x % zs_per_block;
    uint32_t tidx_x = threadIdx.x / zs_per_block;
    uint32_t bidx_z = blockIdx.x % zs_per_grid;
    uint32_t bidx_x = blockIdx.x / zs_per_grid;
    uint32_t z_int = tidx_z + bidx_z * zs_per_block;
    bool const active_thread = (z_int < large_domain);

    if (active_thread) {
        // The hypercube coordinates: we want to sum over these.
        uint32_t x_int_base = tidx_x + bidx_x * xs_per_block;
        uint32_t x_int_stride = (gridDim.x / zs_per_grid) * xs_per_block;

        uint32_t task_offset = x_int_base * large_domain + z_int;
        // The maximal amount of rows done by different threads; hence we need enough global memory [buffer_size * task_stride] for intermediates
        uint32_t task_stride = x_int_stride * large_domain;

        Fp local_buffer[BUFFER_THRESHOLD];
        Fp *inter_buffer;
        uint32_t buffer_stride;
        if constexpr (GLOBAL) {
            inter_buffer = d_intermediates + task_offset;
            buffer_stride = task_stride;
        } else {
            inter_buffer = local_buffer;
            buffer_stride = 1;
        }

        uint32_t log_skip = __ffs(skip_domain) - 1;
        uint32_t log_height_total = __ffs(height) - 1;
        uint32_t log_segment = std::min(log_skip, log_height_total);
        uint32_t segment_size = 1u << log_segment;
        uint32_t log_stride = log_skip - log_segment;

        Fp omega_root = TWO_ADIC_GENERATORS[__ffs(expansion_factor * skip_domain) - 1];
        Fp omega = pow(omega_root, z_int << log_stride);

        Fp eta = TWO_ADIC_GENERATORS[log_skip - log_stride];

        Fp is_first_mult = avg_gp(omega, segment_size);
        Fp is_last_mult = avg_gp(omega * eta, segment_size);

        // Handle multiple x in same thread if there aren't enough blocks
        // NOTE: for sumcheck we can sum over all of these
        FpExt sum = FpExt(Fp::zero());
        for (uint32_t x_int = x_int_base; x_int < num_x; x_int += x_int_stride) {
            Fp is_first = is_first_mult * selectors_cube[x_int];
            Fp is_last = is_last_mult * selectors_cube[2 * num_x + x_int];
            const Fp *inv_lagrange_denoms_z = inv_lagrange_denoms + z_int * skip_domain;

            DagPrismEvalContext eval_ctx{
                preprocessed,
                main_parts,
                public_values,
                omega_skip_pows,
                inv_lagrange_denoms_z,
                inter_buffer,
                is_first,
                is_last,
                skip_domain,
                num_x,
                height,
                buffer_stride,
                buffer_size,
                z_int,
                x_int,
                expansion_factor
            };

            FpExt constraint_sum = acc_constraints<GLOBAL>(
                eval_ctx,
                d_lambda_pows,
                d_lambda_indices,
                d_rules,
                rules_len,
                d_used_nodes,
                used_nodes_len,
                lambda_len,
                local_buffer
            );

            FpExt eq_val = eq_uni[z_int] * eq_cube[x_int];
            sum += constraint_sum * eq_val;
        }
        shared[threadIdx.x] = sum;
    }
    __syncthreads();

    // Reduce phase: reduce all threadIdx.x in the same block, keeping z_int independent
    if (active_thread && tidx_x == 0) {
        FpExt tile_sum = shared[tidx_z];
        for (int lane = 1; lane < xs_per_block; ++lane) {
            tile_sum += shared[lane * zs_per_block + tidx_z];
        }
        tmp_sums_buffer[bidx_x * large_domain + z_int] = tile_sum;
    }
}

__global__ void fold_selectors_round0_kernel(
    FpExt *out,
    const Fp *in,
    FpExt is_first,
    FpExt is_last,
    uint32_t num_x
) {
    size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= num_x)
        return;

    out[tidx] = is_first * in[tidx];                              // is_first
    out[2 * num_x + tidx] = is_last * in[2 * num_x + tidx];       // is_last
    out[num_x + tidx] = FpExt(Fp::one()) - out[2 * num_x + tidx]; // is_transition
}

// Launchers
constexpr uint32_t MAX_THREADS = 256;

// (Not a launcher) Utility function to calculate required size of temp sum buffer.
// Required length of *temp_sum_buffer in FpExt elements
extern "C" uint32_t _zerocheck_r0_temp_sums_buffer_size(
    uint32_t buffer_size,
    uint32_t large_domain,
    uint32_t num_x,
    size_t max_temp_bytes
) {
    return round0_config::temp_sums_buffer_size(
        buffer_size, large_domain, num_x, max_temp_bytes, BUFFER_THRESHOLD, MAX_THREADS
    );
}

extern "C" uint32_t _zerocheck_r0_intermediates_buffer_size(
    uint32_t buffer_size,
    uint32_t large_domain,
    uint32_t num_x,
    size_t max_temp_bytes
) {
    return round0_config::intermediates_buffer_size(
        buffer_size, large_domain, num_x, max_temp_bytes, BUFFER_THRESHOLD, MAX_THREADS
    );
}

extern "C" int _zerocheck_bary_eval_constraints(
    FpExt *tmp_sums_buffer,   // [blockDim.y * gridDim.y][large_domain]
    FpExt *output,            // [large_domain]
    const Fp *selectors_cube, // [3][num_x]
    const Fp *preprocessed,
    const Fp *const *main_parts,
    const Fp *omega_skip_pows,     // [skip_domain]
    const Fp *inv_lagrange_denoms, // [large_domain][skip_domain]
    const FpExt *eq_uni,           // [large_domain]
    const FpExt *eq_cube,          // [num_x]
    const FpExt *d_lambda_pows,
    const uint32_t *d_lambda_indices,
    const Fp *public_values,
    const Rule *d_rules,
    size_t rules_len,
    const size_t *d_used_nodes,
    size_t used_nodes_len,
    size_t lambda_len,
    uint32_t buffer_size,
    Fp *d_intermediates,
    uint32_t large_domain,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t height,
    uint32_t expansion_factor,
    size_t max_temp_bytes
) {
    auto [grid, block] = round0_config::eval_constraints_launch_params(
        buffer_size, large_domain, num_x, max_temp_bytes, BUFFER_THRESHOLD, MAX_THREADS
    );
    auto z_dim = round0_config::get_z_dim(large_domain);
    auto xs_per_grid = grid.x / z_dim.first;
    size_t shmem_bytes = sizeof(FpExt) * block.x;

#define ARGUMENTS                                                                                  \
    tmp_sums_buffer, selectors_cube, preprocessed, main_parts, omega_skip_pows,                    \
        inv_lagrange_denoms, eq_uni, eq_cube, d_lambda_pows, d_lambda_indices, public_values,      \
        d_rules, rules_len, d_used_nodes, used_nodes_len, lambda_len, buffer_size,                 \
        d_intermediates, large_domain, skip_domain, num_x, height, expansion_factor, z_dim

    if (buffer_size > BUFFER_THRESHOLD) {
        zerocheck_bary_evaluate_constraints_kernel<true><<<grid, block, shmem_bytes>>>(ARGUMENTS);
    } else {
        zerocheck_bary_evaluate_constraints_kernel<false><<<grid, block, shmem_bytes>>>(ARGUMENTS);
    }

    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Launch final reduction kernel - reads from block_sums, writes to output
    auto num_blocks = xs_per_grid;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    sumcheck::final_reduce_block_sums<<<large_domain, reduce_block, reduce_shmem>>>(
        tmp_sums_buffer, output, num_blocks
    );
    return CHECK_KERNEL();
}

extern "C" int _fold_selectors_round0(
    FpExt *out,
    const Fp *in,
    FpExt is_first,
    FpExt is_last,
    uint32_t num_x
) {
    auto [grid, block] = kernel_launch_params(num_x);
    fold_selectors_round0_kernel<<<grid, block>>>(out, in, is_first, is_last, num_x);
    return CHECK_KERNEL();
}

} // namespace zerocheck_round0
