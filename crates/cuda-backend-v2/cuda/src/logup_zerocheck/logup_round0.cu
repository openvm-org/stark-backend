#include "codec.cuh"
#include "dag_entry.cuh"
#include "eval_config.cuh"
#include "fp.h"
#include "fpext.h"
#include "frac_ext.cuh"
#include "launcher.cuh"
#include "sumcheck.cuh"
#include "utils.cuh"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector_types.h>

using namespace symbolic_dag;

namespace logup_round0 {

// NOTE[jpw]: keeping separate from zerocheck so it can be tuned separately
constexpr uint32_t BUFFER_THRESHOLD = 16;
// Threshold for switching between coset-parallel and lockstep modes.
// When num_x * skip_domain < threshold, use coset-parallel (grid.y = num_cosets) for better GPU utilization.
// When >= threshold, use lockstep (single thread handles all cosets) to avoid redundant iNTT.
constexpr uint32_t COSET_PARALLEL_THRESHOLD = 32768; // 2^15

// Device function to evaluate interactions for all cosets in lockstep with DAG traversal.
// Computes numerator and denominator sums for each coset.
// NOTE: Using __inline__ instead of __forceinline__ to let compiler decide based on register pressure.
template <uint32_t NUM_COSETS, bool NEEDS_SHMEM>
__device__ __inline__ void acc_interactions(
    const NttEvalContext<NUM_COSETS> &eval_ctx,
    const FpExt *__restrict__ numer_weights,
    const FpExt *__restrict__ denom_weights,
    const Rule *__restrict__ d_rules,
    size_t rules_len,
    FpExt *__restrict__ numer_sums, // output [NUM_COSETS]
    FpExt *__restrict__ denom_sums  // output [NUM_COSETS]
) {
    // Initialize sums to zero
#pragma unroll
    for (uint32_t c = 0; c < NUM_COSETS; c++) {
        numer_sums[c] = FpExt(Fp::zero());
        denom_sums[c] = FpExt(Fp::zero());
    }

    for (size_t node = 0; node < rules_len; ++node) {
        Rule rule = d_rules[node];
        // Lazy decoding: only decode header (op, flags, x) upfront
        RuleHeader header = decode_rule_header(rule);

        // Evaluate x operand for all cosets
        Fp x_vals[NUM_COSETS];
        ntt_eval_dag_entry<NUM_COSETS, NEEDS_SHMEM>(x_vals, header.x, eval_ctx);

        Fp results[NUM_COSETS];

        switch (header.op) {
        case OP_ADD: {
            // Decode y only for binary ops
            SourceInfo y_src = decode_y(rule);
            Fp y_vals[NUM_COSETS];
            ntt_eval_dag_entry<NUM_COSETS, NEEDS_SHMEM>(y_vals, y_src, eval_ctx);
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                results[c] = x_vals[c] + y_vals[c];
            }
            break;
        }
        case OP_SUB: {
            SourceInfo y_src = decode_y(rule);
            Fp y_vals[NUM_COSETS];
            ntt_eval_dag_entry<NUM_COSETS, NEEDS_SHMEM>(y_vals, y_src, eval_ctx);
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                results[c] = x_vals[c] - y_vals[c];
            }
            break;
        }
        case OP_MUL: {
            SourceInfo y_src = decode_y(rule);
            Fp y_vals[NUM_COSETS];
            ntt_eval_dag_entry<NUM_COSETS, NEEDS_SHMEM>(y_vals, y_src, eval_ctx);
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                results[c] = x_vals[c] * y_vals[c];
            }
            break;
        }
        case OP_NEG:
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                results[c] = -x_vals[c];
            }
            break;
        case OP_VAR:
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                results[c] = x_vals[c];
            }
            break;
        case OP_INV:
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                results[c] = inv(x_vals[c]);
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
                eval_ctx.inter_buffer[z_index * eval_ctx.buffer_stride + c] = results[c];
            }
        }

        // Accumulate interaction sums
        if (header.is_constraint) {
            FpExt numer_w = numer_weights[node];
            FpExt denom_w = denom_weights[node];
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                numer_sums[c] += numer_w * results[c];
                denom_sums[c] += denom_w * results[c];
            }
        }
    }
}

// ============================================================================
// KERNELS
// ============================================================================

// Round0 phase interactions kernel - each thread handles ALL cosets in lockstep
template <uint32_t NUM_COSETS, bool GLOBAL, bool NEEDS_SHMEM>
__global__ void logup_r0_ntt_eval_interactions_kernel(
    FracExt *__restrict__ tmp_sums_buffer, // [NUM_COSETS][gridDim.x][skip_domain]
    const Fp *__restrict__ selectors_cube, // [3][num_x]
    const Fp *__restrict__ preprocessed,
    const Fp *const *__restrict__ main_parts,
    const FpExt *__restrict__ eq_cube, // [num_x]
    const Fp *__restrict__ public_values,
    const FpExt *__restrict__ numer_weights,
    const FpExt *__restrict__ denom_weights,
    FpExt denom_sum_init,
    const Rule *__restrict__ d_rules,
    size_t rules_len,
    uint32_t buffer_size,
    Fp *__restrict__ d_intermediates,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t height,
    Fp g_shift
) {
    extern __shared__ char smem[];
    // Shared memory layout:
    // - FracExt[blockDim.x]: shared_sum for reduction (reused per coset)
    // - Fp[blockDim.x]: ntt_buffers (one skip_domain-sized buffer per x_int group, only when NEEDS_SHMEM)
    FracExt *shared_sum = reinterpret_cast<FracExt *>(smem);
    Fp *ntt_buffers_base =
        NEEDS_SHMEM ? reinterpret_cast<Fp *>(smem + blockDim.x * sizeof(FracExt)) : nullptr;

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
    Fp const omega_skip = TWO_ADIC_GENERATORS[l_skip];

    // Compute is_first_mult, is_last_mult for all cosets
    uint32_t const log_height_total = __ffs(height) - 1;
    uint32_t const log_segment = min(l_skip, log_height_total);
    uint32_t const segment_size = 1u << log_segment;
    uint32_t const log_stride = l_skip - log_segment;

    Fp is_first_mult[NUM_COSETS];
    Fp is_last_mult[NUM_COSETS];
    Fp const eta = TWO_ADIC_GENERATORS[l_skip - log_stride];
    Fp const omega_skip_ntt = pow(omega_skip, ntt_idx);

    Fp g_coset = g_shift; // g^1, will iterate g^2, g^3, ...
#pragma unroll
    for (uint32_t c = 0; c < NUM_COSETS; c++) {
        Fp eval_point = g_coset * omega_skip_ntt;
        Fp omega = exp_power_of_2(eval_point, log_stride);
        is_first_mult[c] = avg_gp(omega, segment_size);
        is_last_mult[c] = avg_gp(omega * eta, segment_size);
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
    FracExt sums[NUM_COSETS];
#pragma unroll
    for (uint32_t c = 0; c < NUM_COSETS; c++) {
        sums[c] = {FpExt(Fp::zero()), FpExt(Fp::zero())};
    }

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
        {}, // omega_shifts - set once below
        skip_domain,
        height,
        buffer_stride,
        buffer_size,
        ntt_idx,
        0 // x_int - updated per iteration
    };
    // Compute omega_shifts directly into context using iterative multiplication
    g_coset = g_shift; // reset to g^1
#pragma unroll
    for (uint32_t c = 0; c < NUM_COSETS; c++) {
        eval_ctx.omega_shifts[c] = pow(g_coset, ntt_idx_rev);
        g_coset *= g_shift;
    }

    // Tile across x_int
    for (uint32_t x_int = x_int_base; x_int < num_x; x_int += x_int_stride) {
        // Update only the fields that change per x_int
        eval_ctx.x_int = x_int;
#pragma unroll
        for (uint32_t c = 0; c < NUM_COSETS; c++) {
            eval_ctx.is_first[c] = is_first_mult[c] * selectors_cube[x_int];
            eval_ctx.is_last[c] = is_last_mult[c] * selectors_cube[2 * num_x + x_int];
        }

        FpExt numer_results[NUM_COSETS];
        FpExt denom_results[NUM_COSETS];
        acc_interactions<NUM_COSETS, NEEDS_SHMEM>(
            eval_ctx, numer_weights, denom_weights, d_rules, rules_len, numer_results, denom_results
        );

        FpExt eq = eq_cube[x_int];
#pragma unroll
        for (uint32_t c = 0; c < NUM_COSETS; c++) {
            sums[c].p += eq * numer_results[c];
            sums[c].q += eq * (denom_results[c] + denom_sum_init);
        }
    }

    // Reduction: one coset at a time to minimize shared memory
#pragma unroll
    for (uint32_t c = 0; c < NUM_COSETS; c++) {
        shared_sum[threadIdx.x] = sums[c];
        __syncthreads();

        if (threadIdx.x < skip_domain) {
            FracExt tile_sum = shared_sum[threadIdx.x];
            for (uint32_t lane = 1; lane < (blockDim.x >> l_skip); ++lane) {
                auto lane_offset = (lane << l_skip) + threadIdx.x;
                tile_sum.p += shared_sum[lane_offset].p;
                tile_sum.q += shared_sum[lane_offset].q;
            }
            // Output layout: [num_blocks][NUM_COSETS * skip_domain]
            // This matches final_reduce_block_sums expected [num_blocks][d] layout
            tmp_sums_buffer[blockIdx.x * NUM_COSETS * skip_domain + c * skip_domain + ntt_idx] =
                tile_sum;
        }
        __syncthreads();
    }
}

// Coset-parallel kernel: grid.y = num_cosets, each block handles ONE coset.
// Reuses acc_interactions<1, NEEDS_SHMEM> for maximum code sharing.
// Use when num_x * skip_domain is small for better GPU utilization.
template <bool GLOBAL, bool NEEDS_SHMEM>
__global__ void logup_r0_ntt_eval_interactions_coset_parallel_kernel(
    FracExt *__restrict__ tmp_sums_buffer, // [num_blocks][num_cosets * skip_domain]
    const Fp *__restrict__ selectors_cube, // [3][num_x]
    const Fp *__restrict__ preprocessed,
    const Fp *const *__restrict__ main_parts,
    const FpExt *__restrict__ eq_cube, // [num_x]
    const Fp *__restrict__ public_values,
    const FpExt *__restrict__ numer_weights,
    const FpExt *__restrict__ denom_weights,
    FpExt denom_sum_init,
    const Rule *__restrict__ d_rules,
    size_t rules_len,
    uint32_t buffer_size,
    Fp *__restrict__ d_intermediates,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t height,
    Fp g_shift
) {
    extern __shared__ char smem[];
    FracExt *shared_sum = reinterpret_cast<FracExt *>(smem);
    Fp *ntt_buffers_base =
        NEEDS_SHMEM ? reinterpret_cast<Fp *>(smem + blockDim.x * sizeof(FracExt)) : nullptr;

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
    Fp const omega_skip = TWO_ADIC_GENERATORS[l_skip];

    uint32_t const log_height_total = __ffs(height) - 1;
    uint32_t const log_segment = min(l_skip, log_height_total);
    uint32_t const segment_size = 1u << log_segment;
    uint32_t const log_stride = l_skip - log_segment;

    Fp const eta = TWO_ADIC_GENERATORS[l_skip - log_stride];
    Fp const omega_skip_ntt = pow(omega_skip, ntt_idx);

    // Compute for single coset: g^(coset_idx + 1)
    Fp const g_coset = pow(g_shift, coset_idx + 1);
    Fp const eval_point = g_coset * omega_skip_ntt;
    Fp const omega = exp_power_of_2(eval_point, log_stride);
    Fp const is_first_mult = avg_gp(omega, segment_size);
    Fp const is_last_mult = avg_gp(omega * eta, segment_size);
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

    FracExt sum = {FpExt(Fp::zero()), FpExt(Fp::zero())};

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
        {omega_shift}, // omega_shifts[1]
        skip_domain,
        height,
        buffer_stride,
        buffer_size,
        ntt_idx,
        0 // x_int - updated per iteration
    };

    // Main loop - reuses acc_interactions<1, NEEDS_SHMEM>
    for (uint32_t x_int = x_int_base; x_int < num_x; x_int += x_int_stride) {
        eval_ctx.x_int = x_int;
        eval_ctx.is_first[0] = is_first_mult * selectors_cube[x_int];
        eval_ctx.is_last[0] = is_last_mult * selectors_cube[2 * num_x + x_int];

        FpExt numer_results[1];
        FpExt denom_results[1];
        acc_interactions<1, NEEDS_SHMEM>(
            eval_ctx, numer_weights, denom_weights, d_rules, rules_len, numer_results, denom_results
        );

        FpExt eq = eq_cube[x_int];
        sum.p += eq * numer_results[0];
        sum.q += eq * (denom_results[0] + denom_sum_init);
    }

    // Single-coset reduction (no loop over cosets)
    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    if (threadIdx.x < skip_domain) {
        FracExt tile_sum = shared_sum[threadIdx.x];
        for (uint32_t lane = 1; lane < (blockDim.x >> l_skip); ++lane) {
            auto lane_offset = (lane << l_skip) + threadIdx.x;
            tile_sum.p += shared_sum[lane_offset].p;
            tile_sum.q += shared_sum[lane_offset].q;
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

constexpr uint32_t MAX_THREADS = 256;

// Helper to determine which mode to use based on threshold
inline bool use_coset_parallel_mode(uint32_t num_x, uint32_t skip_domain) {
    return (num_x * skip_domain) < COSET_PARALLEL_THRESHOLD;
}

// (Not a launcher) Utility function to calculate required size of temp sum buffer.
// Required length of *temp_sum_buffer in FracExt elements
extern "C" size_t _logup_r0_temp_sums_buffer_size(
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

extern "C" size_t _logup_r0_intermediates_buffer_size(
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
int launch_logup_ntt_eval_interactions(
    FracExt *tmp_sums_buffer,
    FracExt *output,
    const Fp *selectors_cube,
    const Fp *preprocessed,
    const Fp *const *main_parts,
    const FpExt *eq_cube,
    const Fp *public_values,
    const FpExt *numer_weights,
    const FpExt *denom_weights,
    FpExt denom_sum_init,
    const Rule *d_rules,
    size_t rules_len,
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

    size_t shared_sum_size = sizeof(FracExt) * block.x;
    // NTT buffers: one skip_domain-sized buffer per x_int group in the block
    size_t ntt_buffers_size = NEEDS_SHMEM ? sizeof(Fp) * block.x : 0;
    size_t shmem_bytes = shared_sum_size + ntt_buffers_size;

    logup_r0_ntt_eval_interactions_kernel<NUM_COSETS, GLOBAL, NEEDS_SHMEM>
        <<<grid, block, shmem_bytes>>>(
            tmp_sums_buffer,
            selectors_cube,
            preprocessed,
            main_parts,
            eq_cube,
            public_values,
            numer_weights,
            denom_weights,
            denom_sum_init,
            d_rules,
            rules_len,
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

    auto num_blocks = grid.x;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = div_ceil(reduce_block.x, WARP_SIZE);
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    auto large_domain = NUM_COSETS * skip_domain;
    sumcheck::final_reduce_block_sums<<<2 * large_domain, reduce_block, reduce_shmem>>>(
        reinterpret_cast<FpExt *>(tmp_sums_buffer), reinterpret_cast<FpExt *>(output), num_blocks
    );
    return CHECK_KERNEL();
}

// Generate dispatcher for num_cosets (1-4) x is_global x needs_shmem
DEFINE_DISPATCH_N_B1_B2(dispatch_logup, launch_logup_ntt_eval_interactions, 4)

// Coset-parallel launcher: uses grid.y = num_cosets, each block handles one coset
template <bool GLOBAL, bool NEEDS_SHMEM>
int launch_logup_coset_parallel(
    FracExt *tmp_sums_buffer,
    FracExt *output,
    const Fp *selectors_cube,
    const Fp *preprocessed,
    const Fp *const *main_parts,
    const FpExt *eq_cube,
    const Fp *public_values,
    const FpExt *numer_weights,
    const FpExt *denom_weights,
    FpExt denom_sum_init,
    const Rule *d_rules,
    size_t rules_len,
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

    size_t shared_sum_size = sizeof(FracExt) * block.x;
    size_t ntt_buffers_size = NEEDS_SHMEM ? sizeof(Fp) * block.x : 0;
    size_t shmem_bytes = shared_sum_size + ntt_buffers_size;

    logup_r0_ntt_eval_interactions_coset_parallel_kernel<GLOBAL, NEEDS_SHMEM>
        <<<grid, block, shmem_bytes>>>(
            tmp_sums_buffer,
            selectors_cube,
            preprocessed,
            main_parts,
            eq_cube,
            public_values,
            numer_weights,
            denom_weights,
            denom_sum_init,
            d_rules,
            rules_len,
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

    // Final reduction: same as lockstep
    auto num_blocks = grid.x;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = div_ceil(reduce_block.x, WARP_SIZE);
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    auto large_domain = num_cosets * skip_domain;
    sumcheck::final_reduce_block_sums<<<2 * large_domain, reduce_block, reduce_shmem>>>(
        reinterpret_cast<FpExt *>(tmp_sums_buffer), reinterpret_cast<FpExt *>(output), num_blocks
    );

    return CHECK_KERNEL();
}

extern "C" int _logup_bary_eval_interactions_round0(
    FracExt *tmp_sums_buffer, // [num_blocks][num_cosets * skip_domain]
    FracExt *output,          // [num_cosets * skip_domain]
    const Fp *selectors_cube, // [3][num_x]
    const Fp *preprocessed,
    const Fp *const *main_parts,
    const FpExt *eq_cube, // [num_x]
    const Fp *public_values,
    const FpExt *numer_weights,
    const FpExt *denom_weights,
    FpExt denom_sum_init,
    const Rule *d_rules,
    size_t rules_len,
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

    // Threshold-based dispatch: use coset-parallel for small workloads
    if (use_coset_parallel_mode(num_x, skip_domain)) {
        // Coset-parallel mode: grid.y = num_cosets, each block handles one coset
        if (is_global) {
            if (needs_shmem) {
                return launch_logup_coset_parallel<true, true>(
                    tmp_sums_buffer,
                    output,
                    selectors_cube,
                    preprocessed,
                    main_parts,
                    eq_cube,
                    public_values,
                    numer_weights,
                    denom_weights,
                    denom_sum_init,
                    d_rules,
                    rules_len,
                    buffer_size,
                    d_intermediates,
                    skip_domain,
                    num_x,
                    height,
                    num_cosets,
                    g_shift,
                    max_temp_bytes
                );
            } else {
                return launch_logup_coset_parallel<true, false>(
                    tmp_sums_buffer,
                    output,
                    selectors_cube,
                    preprocessed,
                    main_parts,
                    eq_cube,
                    public_values,
                    numer_weights,
                    denom_weights,
                    denom_sum_init,
                    d_rules,
                    rules_len,
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
        } else {
            if (needs_shmem) {
                return launch_logup_coset_parallel<false, true>(
                    tmp_sums_buffer,
                    output,
                    selectors_cube,
                    preprocessed,
                    main_parts,
                    eq_cube,
                    public_values,
                    numer_weights,
                    denom_weights,
                    denom_sum_init,
                    d_rules,
                    rules_len,
                    buffer_size,
                    d_intermediates,
                    skip_domain,
                    num_x,
                    height,
                    num_cosets,
                    g_shift,
                    max_temp_bytes
                );
            } else {
                return launch_logup_coset_parallel<false, false>(
                    tmp_sums_buffer,
                    output,
                    selectors_cube,
                    preprocessed,
                    main_parts,
                    eq_cube,
                    public_values,
                    numer_weights,
                    denom_weights,
                    denom_sum_init,
                    d_rules,
                    rules_len,
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
        }
    } else {
        // Lockstep mode: single thread handles all cosets
        return dispatch_logup(
            num_cosets,
            is_global,
            needs_shmem,
            tmp_sums_buffer,
            output,
            selectors_cube,
            preprocessed,
            main_parts,
            eq_cube,
            public_values,
            numer_weights,
            denom_weights,
            denom_sum_init,
            d_rules,
            rules_len,
            buffer_size,
            d_intermediates,
            skip_domain,
            num_x,
            height,
            g_shift,
            max_temp_bytes
        );
    }
}

} // namespace logup_round0
