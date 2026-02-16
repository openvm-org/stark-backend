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

constexpr uint32_t BUFFER_THRESHOLD = 16;
// Threshold for switching between coset-parallel and lockstep modes.
// When num_x * skip_domain < threshold, use coset-parallel (grid.y = num_cosets) for better GPU utilization.
// When >= threshold, use lockstep (single thread handles all cosets) to avoid redundant iNTT.
constexpr uint32_t COSET_PARALLEL_THRESHOLD = 32768; // 2^15

// Device function to compute constraint sums for all cosets in lockstep with DAG traversal.
// This computes: sum(lambda_i * constraint_i) for all constraints, for each coset.
// NOTE: Using __inline__ instead of __forceinline__ to let compiler decide based on register pressure.
template <uint32_t NUM_COSETS, bool NEEDS_SHMEM>
__device__ __inline__ void acc_constraints(
    FpExt *__restrict__ constraint_sums, // output [NUM_COSETS]
    const NttEvalContext<NUM_COSETS> &eval_ctx,
    const FpExt *__restrict__ d_lambda_pows,
    const Rule *__restrict__ d_rules,
    size_t rules_len,
    const size_t *__restrict__ d_used_nodes,
    size_t used_nodes_len,
    size_t lambda_len
) {
    size_t lambda_idx = 0;

    // Initialize constraint sums to zero
#pragma unroll
    for (uint32_t c = 0; c < NUM_COSETS; c++) {
        constraint_sums[c] = FpExt(Fp::zero());
    }

    for (size_t node = 0; node < rules_len; ++node) {
        Rule rule = d_rules[node];
        // Lazy decoding: only decode header (op, flags, x) upfront
        RuleHeader header = decode_rule_header(rule);

        // Evaluate x operand for all cosets
        Fp x_vals[NUM_COSETS];
        ntt_eval_dag_entry<NUM_COSETS, NEEDS_SHMEM>(x_vals, header.x, eval_ctx);

        switch (header.op) {
        case OP_ADD: {
            // Decode y only for binary ops
            SourceInfo y_src = decode_y(rule);
            Fp y_vals[NUM_COSETS];
            ntt_eval_dag_entry<NUM_COSETS, NEEDS_SHMEM>(y_vals, y_src, eval_ctx);
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                x_vals[c] += y_vals[c];
            }
            break;
        }
        case OP_SUB: {
            SourceInfo y_src = decode_y(rule);
            Fp y_vals[NUM_COSETS];
            ntt_eval_dag_entry<NUM_COSETS, NEEDS_SHMEM>(y_vals, y_src, eval_ctx);
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                x_vals[c] -= y_vals[c];
            }
            break;
        }
        case OP_MUL: {
            SourceInfo y_src = decode_y(rule);
            Fp y_vals[NUM_COSETS];
            ntt_eval_dag_entry<NUM_COSETS, NEEDS_SHMEM>(y_vals, y_src, eval_ctx);
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                x_vals[c] *= y_vals[c];
            }
            break;
        }
        case OP_NEG:
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                x_vals[c] = -x_vals[c];
            }
            break;
        case OP_VAR:
            break;
        case OP_INV:
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                x_vals[c] = inv(x_vals[c]);
            }
            break;
        }

        // Store intermediate results for all cosets
        // Decode z_index only when buffering
        if (header.buffer_result && eval_ctx.buffer_size > 0) {
            uint32_t z_index = decode_z_index(rule);
#ifdef CUDA_DEBUG
            assert(z_index < eval_ctx.buffer_size);
#endif
            // Intermediate buffer layout: [buffer_size][NUM_COSETS] per thread
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                eval_ctx.inter_buffer[z_index * eval_ctx.buffer_stride + c] = x_vals[c];
            }
        }

        // Accumulate constraint sums
        if (header.is_constraint) {
            while (lambda_idx < lambda_len && lambda_idx < used_nodes_len &&
                   d_used_nodes[lambda_idx] == node) {
                FpExt lambda = d_lambda_pows[lambda_idx];
                lambda_idx++;
#pragma unroll
                for (uint32_t c = 0; c < NUM_COSETS; c++) {
                    constraint_sums[c] += lambda * x_vals[c];
                }
            }
        }
    }
}

// NTT-based kernel: Each thread handles ALL cosets in lockstep.
// Grid: (num_x_blocks, 1) - no coset dimension
// Block: (skip_domain * x_per_block)
// Thread layout: ntt_idx = threadIdx.x % skip_domain, varies fastest
//
// Key optimization: iNTT is done ONCE per trace access, then coefficient is
// used for all coset shifts + forward NTTs.
template <uint32_t NUM_COSETS, bool GLOBAL, bool NEEDS_SHMEM>
__global__ void zerocheck_ntt_evaluate_constraints_kernel(
    FpExt *__restrict__ tmp_sums_buffer,   // [num_blocks][NUM_COSETS][skip_domain]
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
    // - FpExt[blockDim.x]: shared_sum for reduction (reused per coset)
    // - Fp[blockDim.x]: ntt_buffers (one skip_domain-sized buffer per x_int group, only when NEEDS_SHMEM)
    FpExt *shared_sum = reinterpret_cast<FpExt *>(smem);
    Fp *ntt_buffers_base =
        NEEDS_SHMEM ? reinterpret_cast<Fp *>(smem + blockDim.x * sizeof(FpExt)) : nullptr;

    uint32_t const l_skip = __ffs(skip_domain) - 1;

    // Each x_int group within the block gets its own ntt_buffer slice
    uint32_t const x_int_in_block = threadIdx.x >> l_skip;
    Fp *ntt_buffer = NEEDS_SHMEM ? (ntt_buffers_base + x_int_in_block * skip_domain) : nullptr;

    // Thread layout: ntt_idx varies fastest, then x_int
    uint32_t const tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t const ntt_idx = tidx & (skip_domain - 1);
    uint32_t const x_int_base = tidx >> l_skip;

    // Precompute values needed for all cosets
    uint32_t const ntt_idx_rev = rev_len(ntt_idx, l_skip);

    // Compute is_first_mult, is_last_mult for all cosets
    uint32_t const log_height_total = __ffs(height) - 1;
    uint32_t const log_segment = min(l_skip, log_height_total);
    uint32_t const segment_size = 1u << log_segment;
    uint32_t const log_stride = l_skip - log_segment;

    Fp is_first_mult[NUM_COSETS];
    Fp is_last_mult[NUM_COSETS];
    Fp omega_trans[NUM_COSETS];
    Fp const eta = TWO_ADIC_GENERATORS[l_skip - log_stride];
    Fp const omega_skip_ntt =
        (l_skip == 0) ? Fp::one() : device_ntt::get_twiddle(l_skip, ntt_idx);

    Fp g_coset = g_shift; // g^1, will iterate g^2, g^3, ...
#pragma unroll
    for (uint32_t c = 0; c < NUM_COSETS; c++) {
        Fp eval_point = g_coset * omega_skip_ntt;
        Fp omega = exp_power_of_2(eval_point, log_stride);
        Fp omega_z = omega * eta;
        if (log_height_total == 0) {
            omega_z = Fp::one();
        }
        is_first_mult[c] = avg_gp(omega, segment_size);
        is_last_mult[c] = avg_gp(omega_z, segment_size);
        omega_trans[c] = omega_z;
        g_coset *= g_shift; // g^(c+2) for next iteration
    }

    // Intermediate buffer setup
    Fp local_buffer[GLOBAL ? 1 : BUFFER_THRESHOLD * NUM_COSETS];
    Fp *inter_buffer;
    uint32_t buffer_stride;
    if constexpr (GLOBAL) {
        // Layout: [buffer_size][num_threads][NUM_COSETS]
        inter_buffer = d_intermediates + tidx * NUM_COSETS;
        buffer_stride = gridDim.x * blockDim.x * NUM_COSETS;
    } else {
        inter_buffer = local_buffer;
        buffer_stride = NUM_COSETS;
    }

    // Accumulate sums for all cosets
    FpExt sums[NUM_COSETS] = {};

    // blockDim.x is guaranteed to be a multiple of 2^l_skip
    uint32_t const x_int_stride = (gridDim.x * blockDim.x) >> l_skip;

    // Initialize eval_ctx once with loop-invariant fields
    NttEvalContext<NUM_COSETS> eval_ctx{
        preprocessed,
        main_parts,
        public_values,
        inter_buffer,
        ntt_buffer,
        {}, // is_first - updated per x_int
        {}, // is_last - updated per x_int
        {}, // is_transition - updated per x_int
        {}, // omega_shifts - set once below
        skip_domain,
        height,
        buffer_stride,
        buffer_size,
        ntt_idx,
        0 // x_int - updated per iteration
    };
    // Compute omega_shifts directly into context using iterative multiplication
    Fp const omega_shift_base = pow(g_shift, ntt_idx_rev);
    Fp omega_shift = omega_shift_base;
#pragma unroll
    for (uint32_t c = 0; c < NUM_COSETS; c++) {
        eval_ctx.omega_shifts[c] = omega_shift;
        omega_shift *= omega_shift_base;
    }

    // Tile across x_int to save intermediate buffer size
    for (uint32_t x_int = x_int_base; x_int < num_x; x_int += x_int_stride) {
        // Update only the fields that change per x_int
        eval_ctx.x_int = x_int;
#pragma unroll
        for (uint32_t c = 0; c < NUM_COSETS; c++) {
            eval_ctx.is_first[c] = is_first_mult[c] * selectors_cube[x_int];
            eval_ctx.is_last[c] = is_last_mult[c] * selectors_cube[2 * num_x + x_int];
            eval_ctx.is_transition[c] = omega_trans[c] - selectors_cube[2 * num_x + x_int];
        }

        FpExt constraint_sums[NUM_COSETS];
        acc_constraints<NUM_COSETS, NEEDS_SHMEM>(
            constraint_sums,
            eval_ctx,
            d_lambda_pows,
            d_rules,
            rules_len,
            d_used_nodes,
            used_nodes_len,
            lambda_len
        );

        FpExt eq = eq_cube[x_int];
#pragma unroll
        for (uint32_t c = 0; c < NUM_COSETS; c++) {
            sums[c] += constraint_sums[c] * eq;
        }
    }

    // Reduction: one coset at a time to minimize shared memory
    // Recompute eval_point and zerofier on demand (saves registers during main loop)
    g_coset = g_shift; // reset to g^1
#pragma unroll
    for (uint32_t c = 0; c < NUM_COSETS; c++) {
        Fp eval_point = g_coset * omega_skip_ntt;
        Fp zerofier = exp_power_of_2(eval_point, l_skip) - Fp::one();

        // We assume g_shift is not in skip domain and since we use (c+1) power,
        // eval_point is never in skip domain. Hence zerofier is never zero.
        shared_sum[threadIdx.x] = sums[c] * inv(zerofier);
        __syncthreads();

        if (threadIdx.x < skip_domain) {
            FpExt tile_sum = shared_sum[threadIdx.x];
            for (uint32_t lane = 1; lane < (blockDim.x >> l_skip); ++lane) {
                tile_sum += shared_sum[(lane << l_skip) + threadIdx.x];
            }
            // Output layout: [num_blocks][NUM_COSETS * skip_domain]
            // This matches final_reduce_block_sums expected layout [num_blocks][d]
            tmp_sums_buffer[blockIdx.x * NUM_COSETS * skip_domain + c * skip_domain + ntt_idx] =
                tile_sum;
        }
        __syncthreads();
        g_coset *= g_shift;
    }
}

__global__ void fold_selectors_round0_kernel(
    FpExt *out,
    const Fp *in,
    FpExt is_first,
    FpExt is_last,
    FpExt omega_r0,
    uint32_t num_x,
    bool log_height_is_zero
) {
    size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= num_x)
        return;

    out[tidx] = is_first * in[tidx];                        // is_first
    out[2 * num_x + tidx] = is_last * in[2 * num_x + tidx]; // is_last
    out[num_x + tidx] =
        log_height_is_zero ? FpExt(0) : (omega_r0 - FpExt(in[2 * num_x + tidx])); // is_transition
}

// Coset-parallel kernel: grid.y = num_cosets, each block handles ONE coset.
// Reuses acc_constraints<1, NEEDS_SHMEM> for maximum code sharing.
// Use when num_x * skip_domain is small for better GPU utilization.
template <bool GLOBAL, bool NEEDS_SHMEM>
__global__ void zerocheck_ntt_evaluate_constraints_coset_parallel_kernel(
    FpExt *__restrict__ tmp_sums_buffer,   // [num_blocks][num_cosets * skip_domain]
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
    FpExt *shared_sum = reinterpret_cast<FpExt *>(smem);
    Fp *ntt_buffers_base =
        NEEDS_SHMEM ? reinterpret_cast<Fp *>(smem + blockDim.x * sizeof(FpExt)) : nullptr;

    uint32_t const l_skip = __ffs(skip_domain) - 1;
    uint32_t const num_cosets = gridDim.y;

    // Each x_int group within the block gets its own ntt_buffer slice
    uint32_t const x_int_in_block = threadIdx.x >> l_skip;
    Fp *ntt_buffer = NEEDS_SHMEM ? (ntt_buffers_base + x_int_in_block * skip_domain) : nullptr;

    // Thread layout: ntt_idx varies fastest, then x_int
    uint32_t const tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t const ntt_idx = tidx & (skip_domain - 1);
    uint32_t const x_int_base = tidx >> l_skip;

    // KEY DIFFERENCE: coset_idx from blockIdx.y
    uint32_t const coset_idx = blockIdx.y;

    // Precompute values for this single coset
    uint32_t const ntt_idx_rev = rev_len(ntt_idx, l_skip);

    uint32_t const log_height_total = __ffs(height) - 1;
    uint32_t const log_segment = min(l_skip, log_height_total);
    uint32_t const segment_size = 1u << log_segment;
    uint32_t const log_stride = l_skip - log_segment;

    Fp const eta = TWO_ADIC_GENERATORS[l_skip - log_stride];
    Fp const omega_skip_ntt =
        (l_skip == 0) ? Fp::one() : device_ntt::get_twiddle(l_skip, ntt_idx);

    // Compute for single coset: g^(coset_idx + 1)
    Fp const g_coset = pow(g_shift, coset_idx + 1);
    Fp const eval_point = g_coset * omega_skip_ntt;
    Fp const omega = exp_power_of_2(eval_point, log_stride);
    Fp omega_z = omega * eta;
    if (log_height_total == 0) {
        omega_z = Fp::one();
    }
    Fp const is_first_mult = avg_gp(omega, segment_size);
    Fp const is_last_mult = avg_gp(omega_z, segment_size);
    Fp const omega_shift = pow(g_coset, ntt_idx_rev);

    // Intermediate buffer setup (single coset, no NUM_COSETS multiplier)
    Fp local_buffer[GLOBAL ? 1 : BUFFER_THRESHOLD];
    Fp *inter_buffer;
    uint32_t buffer_stride;
    if constexpr (GLOBAL) {
        // Layout: [buffer_size][num_cosets * num_threads_per_coset]
        uint32_t global_tidx = coset_idx * gridDim.x * blockDim.x + tidx;
        inter_buffer = d_intermediates + global_tidx;
        buffer_stride = gridDim.x * gridDim.y * blockDim.x;
    } else {
        inter_buffer = local_buffer;
        buffer_stride = 1;
    }

    FpExt sum = FpExt(Fp::zero());

    uint32_t const x_int_stride = (gridDim.x * blockDim.x) >> l_skip;

    // Initialize single-coset context (NUM_COSETS=1)
    NttEvalContext<1> eval_ctx{
        preprocessed,
        main_parts,
        public_values,
        inter_buffer,
        ntt_buffer,
        {Fp::zero()},  // is_first[1] - updated per x_int
        {Fp::zero()},  // is_last[1] - updated per x_int
        {Fp::zero()},  // is_transition[1] - updated per x_int
        {omega_shift}, // omega_shifts[1]
        skip_domain,
        height,
        buffer_stride,
        buffer_size,
        ntt_idx,
        0 // x_int - updated per iteration
    };

    // Main loop - reuses acc_constraints<1, NEEDS_SHMEM>
    for (uint32_t x_int = x_int_base; x_int < num_x; x_int += x_int_stride) {
        eval_ctx.x_int = x_int;
        eval_ctx.is_first[0] = is_first_mult * selectors_cube[x_int];
        eval_ctx.is_last[0] = is_last_mult * selectors_cube[2 * num_x + x_int];
        eval_ctx.is_transition[0] = omega_z - selectors_cube[2 * num_x + x_int];

        FpExt constraint_sums[1];
        acc_constraints<1, NEEDS_SHMEM>(
            constraint_sums,
            eval_ctx,
            d_lambda_pows,
            d_rules,
            rules_len,
            d_used_nodes,
            used_nodes_len,
            lambda_len
        );

        sum += constraint_sums[0] * eq_cube[x_int];
    }

    // Single-coset reduction (no loop over cosets)
    Fp zerofier = exp_power_of_2(eval_point, l_skip) - Fp::one();
    // We assume g_shift is not in skip domain and since we use (c+1) power,
    // eval_point is never in skip domain. Hence zerofier is never zero.
    shared_sum[threadIdx.x] = sum * inv(zerofier);
    __syncthreads();

    if (threadIdx.x < skip_domain) {
        FpExt tile_sum = shared_sum[threadIdx.x];
        for (uint32_t lane = 1; lane < (blockDim.x >> l_skip); ++lane) {
            tile_sum += shared_sum[(lane << l_skip) + threadIdx.x];
        }
        // Output layout: [num_x_blocks][num_cosets * skip_domain]
        // Same as lockstep to ensure final_reduce_block_sums works unchanged
        tmp_sums_buffer[blockIdx.x * num_cosets * skip_domain + coset_idx * skip_domain + ntt_idx] =
            tile_sum;
    }
}

// ============================================================================
// LAUNCHERS
// ============================================================================
constexpr uint32_t MAX_THREADS = 128;

// Helper to determine which mode to use based on threshold
inline bool use_coset_parallel_mode(uint32_t num_x, uint32_t skip_domain) {
    return (num_x * skip_domain) < COSET_PARALLEL_THRESHOLD;
}

// (Not a launcher) Utility function to calculate required size of temp sum buffer.
// Required length of *temp_sum_buffer in FpExt elements
extern "C" size_t _zerocheck_r0_temp_sums_buffer_size(
    uint32_t buffer_size,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t num_cosets,
    size_t max_temp_bytes
) {
    // Both modes use the same buffer size formula
    return DISPATCH_BOOL(
        round0_config_impl::temp_sums_buffer_size,
        use_coset_parallel_mode(num_x, skip_domain),
        buffer_size,
        skip_domain,
        num_x,
        num_cosets,
        max_temp_bytes,
        BUFFER_THRESHOLD,
        MAX_THREADS
    );
}

extern "C" size_t _zerocheck_r0_intermediates_buffer_size(
    uint32_t buffer_size,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t num_cosets,
    size_t max_temp_bytes
) {
    // Both modes use the same buffer size formula
    return DISPATCH_BOOL(
        round0_config_impl::intermediates_buffer_size,
        use_coset_parallel_mode(num_x, skip_domain),
        buffer_size,
        skip_domain,
        num_x,
        num_cosets,
        max_temp_bytes,
        BUFFER_THRESHOLD,
        MAX_THREADS
    );
}

template <uint32_t NUM_COSETS, bool GLOBAL, bool NEEDS_SHMEM>
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
    Fp g_shift,
    size_t max_temp_bytes
) {
    auto [grid, block] = coset_round0_config::eval_constraints_launch_params(
        buffer_size, skip_domain, num_x, NUM_COSETS, max_temp_bytes, BUFFER_THRESHOLD, MAX_THREADS
    );

    size_t shared_sum_size = sizeof(FpExt) * block.x;
    // NTT buffers: one skip_domain-sized buffer per x_int group in the block
    size_t ntt_buffers_size = NEEDS_SHMEM ? sizeof(Fp) * block.x : 0;
    size_t shmem_bytes = shared_sum_size + ntt_buffers_size;

    zerocheck_ntt_evaluate_constraints_kernel<NUM_COSETS, GLOBAL, NEEDS_SHMEM>
        <<<grid, block, shmem_bytes>>>(
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

    // Final reduction: sum across all x_int blocks for each (coset, ntt_idx)
    auto num_blocks = grid.x;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = div_ceil(reduce_block.x, WARP_SIZE);
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    sumcheck::final_reduce_block_sums<<<NUM_COSETS * skip_domain, reduce_block, reduce_shmem>>>(
        tmp_sums_buffer, output, num_blocks
    );

    return CHECK_KERNEL();
}

// Generate dispatcher for num_cosets (1-4) x is_global x needs_shmem
DEFINE_DISPATCH_N_B1_B2(dispatch_zerocheck, launch_zerocheck_ntt_evaluate_constraints, 4)

// Coset-parallel launcher: uses grid.y = num_cosets, each block handles one coset
template <bool GLOBAL, bool NEEDS_SHMEM>
int launch_zerocheck_coset_parallel(
    FpExt *tmp_sums_buffer,
    FpExt *output,
    const Fp *selectors_cube,
    const Fp *preprocessed,
    const Fp *const *main_parts,
    const FpExt *eq_cube,
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
    auto [grid, block] = coset_parallel_round0_config::eval_constraints_launch_params(
        buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes, BUFFER_THRESHOLD, MAX_THREADS
    );

    size_t shared_sum_size = sizeof(FpExt) * block.x;
    size_t ntt_buffers_size = NEEDS_SHMEM ? sizeof(Fp) * block.x : 0;
    size_t shmem_bytes = shared_sum_size + ntt_buffers_size;

    zerocheck_ntt_evaluate_constraints_coset_parallel_kernel<GLOBAL, NEEDS_SHMEM>
        <<<grid, block, shmem_bytes>>>(
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

    // Final reduction: same as lockstep - sum across all x_int blocks for each (coset, ntt_idx)
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
    FpExt *tmp_sums_buffer,   // [num_blocks][num_cosets * skip_domain]
    FpExt *output,            // [num_cosets * skip_domain]
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

#define KERNEL_ARGS                                                                                \
    tmp_sums_buffer, output, selectors_cube, preprocessed, main_parts, eq_cube, d_lambda_pows,     \
        public_values, d_rules, rules_len, d_used_nodes, used_nodes_len, lambda_len, buffer_size,  \
        d_intermediates, skip_domain, num_x, height

    // Threshold-based dispatch: use coset-parallel for small workloads
    if (use_coset_parallel_mode(num_x, skip_domain)) {
        // Coset-parallel mode: grid.y = num_cosets, each block handles one coset
        return DISPATCH_BOOL_PAIR(
            launch_zerocheck_coset_parallel,
            is_global,
            needs_shmem,
            KERNEL_ARGS,
            num_cosets,
            g_shift,
            max_temp_bytes
        );
    } else {
        // Lockstep mode: single thread handles all cosets
        return dispatch_zerocheck(
            num_cosets, is_global, needs_shmem, KERNEL_ARGS, g_shift, max_temp_bytes
        );
    }
#undef KERNEL_ARGS
}

extern "C" int _fold_selectors_round0(
    FpExt *out,
    const Fp *in,
    FpExt is_first,
    FpExt is_last,
    FpExt omega_r0,
    uint32_t num_x,
    bool log_height_is_zero
) {
    auto [grid, block] = kernel_launch_params(num_x);
    fold_selectors_round0_kernel<<<grid, block>>>(
        out, in, is_first, is_last, omega_r0, num_x, log_height_is_zero
    );
    return CHECK_KERNEL();
}

} // namespace zerocheck_round0
