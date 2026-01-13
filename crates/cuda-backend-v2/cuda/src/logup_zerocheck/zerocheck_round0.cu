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
#include "eval_config.cuh"
#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "sumcheck.cuh"
#include "utils.cuh"

using namespace symbolic_dag;

namespace zerocheck_round0 {
// Device function equivalent to helper.acc_constraints without eq_* parts
// This computes the constraint sum: sum(lambda_i * constraint_i) for all constraints
template <bool NEEDS_SHMEM>
__device__ __forceinline__ FpExt acc_constraints(
    const NttEvalContext &eval_ctx,
    const FpExt *__restrict__ d_lambda_pows,
    const Rule *__restrict__ d_rules,
    size_t rules_len,
    const size_t *__restrict__ d_used_nodes,
    size_t used_nodes_len,
    size_t lambda_len
) {
    size_t lambda_idx = 0;
    FpExt constraint_sum(Fp::zero());

    for (size_t node = 0; node < rules_len; ++node) {
        Rule rule = d_rules[node];
        DecodedRule decoded = decode_rule(rule);

        Fp x_val = ntt_eval_dag_entry<NEEDS_SHMEM>(decoded.x, eval_ctx);
        Fp result;
        switch (decoded.op) {
        case OP_ADD: {
            Fp y_val = ntt_eval_dag_entry<NEEDS_SHMEM>(decoded.y, eval_ctx);
            result = x_val + y_val;
            break;
        }
        case OP_SUB: {
            Fp y_val = ntt_eval_dag_entry<NEEDS_SHMEM>(decoded.y, eval_ctx);
            result = x_val - y_val;
            break;
        }
        case OP_MUL: {
            Fp y_val = ntt_eval_dag_entry<NEEDS_SHMEM>(decoded.y, eval_ctx);
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
            // Note: decoded.z_index refers to a decoding index, it doesn't have to do with prism coordinate
            eval_ctx.inter_buffer[decoded.z_index * eval_ctx.buffer_stride] = result;
        }

        if (decoded.is_constraint) {
            while (lambda_idx < lambda_len && lambda_idx < used_nodes_len &&
                   d_used_nodes[lambda_idx] == node) {
                FpExt lambda = d_lambda_pows[lambda_idx];
                lambda_idx++;
                constraint_sum += lambda * result;
            }
        }
    }

    return constraint_sum;
}

constexpr uint32_t BUFFER_THRESHOLD = 16;

// NTT-based kernel: ALL cosets are handled within one block.
// blockDim.x = skip_domain * num_cosets (full 2^k domain)
// Thread layout: z_idx = threadIdx.x % skip_domain, coset_idx = threadIdx.x / skip_domain
// Each block handles ONE x_int value; gridDim.x = num_x
//
// iNTT is done ONCE by coset_idx=0 threads, then all cosets share the coefficients.
// This saves d-1 redundant iNTTs per variable access.
template <bool GLOBAL, bool NEEDS_SHMEM>
__global__ void zerocheck_ntt_evaluate_constraints_kernel(
    FpExt *__restrict__ tmp_sums_buffer,   // [num_x][large_domain]
    const Fp *__restrict__ selectors_cube, // [3][num_x]
    const Fp *__restrict__ preprocessed,
    const Fp *const *__restrict__ main_parts,
    const FpExt *__restrict__ eq_cube, // [num_x]
    const FpExt *__restrict__ d_lambda_pows,
    const Fp *__restrict__ public_values,
    const Rule *__restrict__ d_rules,
    size_t rules_len,
    const size_t *__restrict__ d_used_nodes,
    size_t used_nodes_len,
    size_t lambda_len,
    uint32_t buffer_size,
    Fp *__restrict__ d_intermediates,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t height,
    Fp g_shift
) {
    extern __shared__ char smem[];
    // Shared memory layout:
    // - FpExt[blockDim.x]: shared_sum for reduction
    // - Fp[blockDim.x]: ntt_buffers (one skip_domain-sized buffer per x_int group in block)
    FpExt *shared_sum = reinterpret_cast<FpExt *>(smem);
    Fp *ntt_buffers_base = reinterpret_cast<Fp *>(smem + blockDim.x * sizeof(FpExt));

    uint32_t l_skip = __ffs(skip_domain) - 1;
    // Each x_int group within the block gets its own ntt_buffer slice
    uint32_t x_int_in_block = threadIdx.x >> l_skip;
    Fp *ntt_buffer = ntt_buffers_base + x_int_in_block * skip_domain;

    // Thread layout:
    // x-dim for (ntt_idx, x_int) where ntt_idx is index in skip_domain and varies fastest.
    // y-dim for coset_idx.
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t ntt_idx = tidx & (skip_domain - 1);
    uint32_t x_int_base = tidx >> l_skip;
    uint32_t coset_idx = blockIdx.y;

    auto num_cosets = gridDim.y;

    // Compute omega_shift for coset evaluation: g^(coset_idx * z_idx_rev)
    uint32_t const ntt_idx_rev = rev_len(ntt_idx, l_skip);
    Fp const g_coset = pow(g_shift, coset_idx + 1);
    Fp const omega_shift = pow(g_coset, ntt_idx_rev);

    // Compute is_first_mult, is_last_mult for this z point
    uint32_t const log_height_total = __ffs(height) - 1;
    uint32_t const log_segment = std::min(l_skip, log_height_total);
    uint32_t const segment_size = 1u << log_segment;
    uint32_t const log_stride = l_skip - log_segment;

    Fp const omega_skip = TWO_ADIC_GENERATORS[l_skip];
    Fp const eval_point = g_coset * pow(omega_skip, ntt_idx);
    // For selectors, we need eval_point^(stride) where stride handles lifted traces
    Fp const omega = pow(eval_point, 1u << log_stride);
    Fp const eta = TWO_ADIC_GENERATORS[l_skip - log_stride];
    Fp const is_first_mult = avg_gp(omega, segment_size);
    Fp const is_last_mult = avg_gp(omega * eta, segment_size);

    Fp local_buffer[BUFFER_THRESHOLD];
    Fp *inter_buffer;
    uint32_t buffer_stride;
    if constexpr (GLOBAL) {
        uint32_t task_offset = coset_idx * gridDim.x * blockDim.x + tidx;
        uint32_t task_stride = gridDim.y * gridDim.x * blockDim.x;
        inter_buffer = d_intermediates + task_offset;
        buffer_stride = task_stride;
    } else {
        inter_buffer = local_buffer;
        buffer_stride = 1;
    }

    FpExt sum = FpExt(Fp::zero());
    // blockDim.x is guaranteed to be a multiple of 2^l_skip
    uint32_t x_int_stride = (gridDim.x * blockDim.x) >> l_skip;
    // Tile across x_int to save intermediate buffer size.
    // NOTE: for sumcheck we can sum over all of these
    for (uint32_t x_int = x_int_base; x_int < num_x; x_int += x_int_stride) {

        Fp const is_first = is_first_mult * selectors_cube[x_int];
        Fp const is_last = is_last_mult * selectors_cube[2 * num_x + x_int];

        NttEvalContext eval_ctx{
            preprocessed,
            main_parts,
            public_values,
            inter_buffer,
            ntt_buffer,
            is_first,
            is_last,
            omega_shift,
            skip_domain,
            num_cosets,
            num_x,
            height,
            buffer_stride,
            buffer_size,
            ntt_idx,
            coset_idx,
            x_int
        };

        FpExt constraint_sum = acc_constraints<NEEDS_SHMEM>(
            eval_ctx, d_lambda_pows, d_rules, rules_len, d_used_nodes, used_nodes_len, lambda_len
        );
        sum += constraint_sum * eq_cube[x_int];
    }
    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    if (threadIdx.x < skip_domain) {
        auto tile_sum = shared_sum[threadIdx.x];
        for (int lane = 1; lane < (blockDim.x >> l_skip); ++lane) {
            tile_sum += shared_sum[(lane << l_skip) + threadIdx.x];
        }
        tmp_sums_buffer[blockIdx.x * num_cosets * skip_domain + (coset_idx << l_skip) + ntt_idx] =
            tile_sum;
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

// ============================================================================
// LAUNCHERS
// ============================================================================
constexpr uint32_t MAX_THREADS = 256;

// (Not a launcher) Utility function to calculate required size of temp sum buffer.
// Required length of *temp_sum_buffer in FpExt elements
extern "C" size_t _zerocheck_r0_temp_sums_buffer_size(
    uint32_t buffer_size,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t num_cosets,
    size_t max_temp_bytes
) {
    return coset_round0_config::temp_sums_buffer_size(
        buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes, BUFFER_THRESHOLD, MAX_THREADS
    );
}

extern "C" size_t _zerocheck_r0_intermediates_buffer_size(
    uint32_t buffer_size,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t num_cosets,
    size_t max_temp_bytes
) {
    return coset_round0_config::intermediates_buffer_size(
        buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes, BUFFER_THRESHOLD, MAX_THREADS
    );
}

template <bool GLOBAL, bool NEEDS_SHMEM>
int launch_zerocheck_ntt_evaluate_constraints(
    FpExt *tmp_sums_buffer,
    FpExt *output,
    const Fp *selectors_cube, // [3][num_x]
    const Fp *preprocessed,
    const Fp *const *main_parts,
    const FpExt *eq_cube, // [num_x]
    const FpExt *d_lambda_pows,
    const Fp *public_values,
    const Rule *d_rules,
    size_t rules_len,
    const size_t *d_used_nodes,
    size_t used_nodes_len,
    size_t lambda_len,
    uint32_t buffer_size,
    Fp *d_intermediates,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t height,
    uint32_t num_cosets,
    Fp g_shift,
    size_t max_temp_bytes
) {
    auto [grid, block] = coset_round0_config::eval_constraints_launch_params(
        buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes, BUFFER_THRESHOLD, MAX_THREADS
    );

    size_t shared_sum_size = sizeof(FpExt) * block.x;
    // Each x_int group in the block needs its own ntt_buffer of size skip_domain (only when NEEDS_SHMEM)
    size_t ntt_buffers_size = NEEDS_SHMEM ? sizeof(Fp) * block.x : 0;
    size_t shmem_bytes = shared_sum_size + ntt_buffers_size;

    zerocheck_ntt_evaluate_constraints_kernel<GLOBAL, NEEDS_SHMEM><<<grid, block, shmem_bytes>>>(
        tmp_sums_buffer,
        selectors_cube,
        preprocessed,
        main_parts,
        eq_cube,
        d_lambda_pows,
        public_values,
        d_rules,
        rules_len,
        d_used_nodes,
        used_nodes_len,
        lambda_len,
        buffer_size,
        d_intermediates,
        skip_domain,
        num_x,
        height,
        g_shift
    );
    int err = CHECK_KERNEL();
    if (err != 0) {
        return err;
    }

    // Final reduction: sum across all x_int blocks for each z_int
    auto num_blocks = grid.x;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = div_ceil(reduce_block.x, WARP_SIZE);
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    sumcheck::final_reduce_block_sums<<<num_cosets * skip_domain, reduce_block, reduce_shmem>>>(
        tmp_sums_buffer, output, num_blocks
    );

    return CHECK_KERNEL();
}

extern "C" int _zerocheck_ntt_eval_constraints(
    FpExt *tmp_sums_buffer,   // [num_x][large_domain]
    FpExt *output,            // [large_domain]
    const Fp *selectors_cube, // [3][num_x]
    const Fp *preprocessed,
    const Fp *const *main_parts,
    const FpExt *eq_cube, // [num_x]
    const FpExt *d_lambda_pows,
    const Fp *public_values,
    const Rule *d_rules,
    size_t rules_len,
    const size_t *d_used_nodes,
    size_t used_nodes_len,
    size_t lambda_len,
    uint32_t buffer_size,
    Fp *d_intermediates,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t height,
    uint32_t num_cosets,
    Fp g_shift,
    size_t max_temp_bytes
) {
    bool is_global = buffer_size > BUFFER_THRESHOLD;
    bool needs_shmem = skip_domain > WARP_SIZE;

    return DISPATCH_BOOL_PAIR(
        launch_zerocheck_ntt_evaluate_constraints,
        is_global,
        needs_shmem,
        tmp_sums_buffer,
        output,
        selectors_cube,
        preprocessed,
        main_parts,
        eq_cube,
        d_lambda_pows,
        public_values,
        d_rules,
        rules_len,
        d_used_nodes,
        used_nodes_len,
        lambda_len,
        buffer_size,
        d_intermediates,
        skip_domain,
        num_x,
        height,
        num_cosets,
        g_shift,
        max_temp_bytes
    );
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
