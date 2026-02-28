#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "sumcheck.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector_types.h>

namespace {
constexpr int S_DEG = 2;
}

struct BatchingTracePacket {
    const Fp *__restrict__ ptr;
    uint32_t height;
    uint32_t width;
    uint32_t stacked_row_start;
    uint32_t mu_idx;
};

/// Algebraically batch _unstacked_ traces to get equivalent behavior to stacking traces and then algebraically batching stacked columns.
__global__ void whir_algebraic_batch_traces_kernel(
    Fp *__restrict__ output,                         // Length is stacked_height
    const BatchingTracePacket *__restrict__ packets, // packets, one per unstacked trace
    const FpExt *mu_powers,                          // Len is sum of widths of all matrices
    size_t stacked_height,
    size_t num_packets,
    uint32_t skip_domain
) {

    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= stacked_height) {
        return;
    }

    // NOTE(perf): Depending on the number of columns, we may want to use shared memory to reduce this sum.
    FpExt res(Fp::zero());
    for (int idx = 0; idx < num_packets; idx++) {
        auto packet = packets[idx];
        auto trace = packet.ptr;
        uint32_t height = packet.height;
        uint32_t lifted_height = std::max(height, skip_domain);
        uint32_t width = packet.width;
        uint32_t row_start = packet.stacked_row_start;
        uint32_t mu_idx_start = packet.mu_idx;
        uint32_t stride = std::max(skip_domain / height, 1u);
        // Determine if there are entries between `row_start..row_start + lifted_height * width (mod stacked_height)` that equal `row`
        // This is the same as all `offset` satisfying
        // row_start <= offset * stacked_height + row < row_start + lifted_height * width
        auto stacked_end = row_start + lifted_height * width;
        if (row >= stacked_end)
            continue;
        uint32_t offset_start = row_start <= row ? 0 : 1;
        for (uint32_t row_offset = offset_start * stacked_height + row; row_offset < stacked_end;
             row_offset += stacked_height) {
            auto offset = (row_offset - row) / stacked_height;
            uint32_t tmp = row_offset - row_start; // < lifted_height * width
            uint32_t trace_col = tmp / lifted_height;
            uint32_t strided_trace_row = tmp % lifted_height;
            // match the striding that happens in stacking matrix
            auto trace_val = strided_trace_row % stride == 0
                                 ? trace[trace_col * height + (strided_trace_row / stride)]
                                 : Fp::zero();
            auto mu_pow = mu_powers[mu_idx_start + offset];
            res += mu_pow * trace_val;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        output[i * stacked_height + row] = res.elems[i];
    }
}

// Computes one WHIR sumcheck round using:
// - `f` in MLE coefficient form
// - `w` in moment form: m[T] = sum_{x superset T} w(x)
//
// For X in {1,2}, this evaluates
// s(X) = sum_y f(X,y) * w(X,y)
// with pairwise formulas derived from coefficient/moment slices.
__global__ void whir_sumcheck_coeff_moments_round_kernel(
    const FpExt *f_coeffs,
    const FpExt *w_moments,
    FpExt *block_sums,
    const uint32_t height
) {
    extern __shared__ char smem[];
    FpExt *shared = (FpExt *)smem;

    int half_height = height >> 1;

    FpExt local_sums[S_DEG];

#pragma unroll
    for (int i = 0; i < S_DEG; i++) {
        local_sums[i] = FpExt(0);
    }

    for (int y = blockIdx.x * blockDim.x + threadIdx.x; y < half_height;
         y += gridDim.x * blockDim.x) {
        int idx0 = y << 1;
        int idx1 = idx0 + 1;

        FpExt c0 = f_coeffs[idx0];
        FpExt c1 = f_coeffs[idx1];
        FpExt m0 = w_moments[idx0];
        FpExt m1 = w_moments[idx1];

        // X = 1:
        // f_1 = c0 + c1
        // w_1 moments reduce to m1
        FpExt f_1 = c0 + c1;
        FpExt term_1 = f_1 * m1;

        // X = 2:
        // f_2 = c0 + 2*c1
        // w_2 moments: -m0 + 3*m1
        FpExt f_2 = c0 + c1 + c1;
        FpExt m_2 = m1 * Fp(3) - m0;
        FpExt term_2 = f_2 * m_2;

        local_sums[0] += term_1;
        local_sums[1] += term_2;
    }

#pragma unroll
    for (int idx = 0; idx < S_DEG; idx++) {
        FpExt reduced = sumcheck::block_reduce_sum(local_sums[idx], shared);
        if (threadIdx.x == 0) {
            block_sums[blockIdx.x * S_DEG + idx] = reduced;
        }
        __syncthreads();
    }
}

// Folds both:
// - `f` in MLE coefficient form
// - `w` in moment form m[T] = sum_{x superset T} w(x)
__global__ void whir_fold_coeffs_and_moments_kernel(
    const FpExt *f_coeffs,
    const FpExt *w_moments,
    FpExt *f_folded_coeffs,
    FpExt *w_folded_moments,
    FpExt alpha,
    uint32_t half_height
) {
    uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= half_height) {
        return;
    }

    uint32_t idx0 = y << 1;
    uint32_t idx1 = idx0 + 1;

    FpExt c0 = f_coeffs[idx0];
    FpExt c1 = f_coeffs[idx1];
    f_folded_coeffs[y] = c0 + alpha * c1;

    FpExt m0 = w_moments[idx0];
    FpExt m1 = w_moments[idx1];
    FpExt one_minus_alpha = FpExt(Fp(1)) - alpha;
    FpExt two_alpha_minus_one = alpha + alpha - FpExt(Fp(1));
    w_folded_moments[y] = one_minus_alpha * m0 + two_alpha_minus_one * m1;
}

// Fused fold (previous round) + sumcheck (current round) kernel.
//
// Given pre-fold buffers f[pre_fold_height] and w[pre_fold_height] and the fold
// challenge alpha, this kernel simultaneously:
//   1. Folds f and w by alpha, writing results to f_folded[pre_fold_height/2] and
//      w_folded[pre_fold_height/2].
//   2. Computes the WHIR sumcheck polynomial s(X) for the folded round at X=1 and X=2,
//      accumulating partial sums into block_sums[gridDim.x * S_DEG].
//
// Memory savings: eliminates the separate read pass of f_folded/w_folded that would
// otherwise be needed as input to the next round's standalone sumcheck kernel.
//
// Each thread processes a group of 4 consecutive pre-fold elements (indices 4t..4t+3),
// which fold to the post-fold pair (f_folded[2t], f_folded[2t+1]) used in sumcheck.
// Total work items: pre_fold_height / 4.
//
// Shared memory requirement: max(num_warps, 1) * sizeof(FpExt)
__global__ void whir_fused_fold_and_sumcheck_kernel(
    const FpExt *__restrict__ f_coeffs,  // Pre-fold f, size pre_fold_height
    const FpExt *__restrict__ w_moments, // Pre-fold w, size pre_fold_height
    FpExt *__restrict__ f_folded,        // Post-fold f, size pre_fold_height/2
    FpExt *__restrict__ w_folded,        // Post-fold w, size pre_fold_height/2
    FpExt *__restrict__ block_sums,      // Partial sumcheck sums, size gridDim.x * S_DEG
    FpExt alpha,
    uint32_t quarter_height              // = pre_fold_height / 4 (total work items)
) {
    extern __shared__ char smem[];
    FpExt *shared = (FpExt *)smem;

    FpExt local_sums[S_DEG];
#pragma unroll
    for (int i = 0; i < S_DEG; i++) {
        local_sums[i] = FpExt(0);
    }

    const FpExt one_minus_alpha = FpExt(Fp(1)) - alpha;
    const FpExt two_alpha_minus_one = alpha + alpha - FpExt(Fp(1));

    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < quarter_height;
         t += gridDim.x * blockDim.x) {
        // Load 4 consecutive pre-fold elements
        uint32_t base = t << 2; // 4*t
        FpExt f0 = f_coeffs[base];
        FpExt f1 = f_coeffs[base + 1];
        FpExt f2 = f_coeffs[base + 2];
        FpExt f3 = f_coeffs[base + 3];

        FpExt w0 = w_moments[base];
        FpExt w1 = w_moments[base + 1];
        FpExt w2 = w_moments[base + 2];
        FpExt w3 = w_moments[base + 3];

        // Fold: f_folded[2t] = f[4t] + alpha*f[4t+1]
        //       f_folded[2t+1] = f[4t+2] + alpha*f[4t+3]
        FpExt c0 = f0 + alpha * f1;
        FpExt c1 = f2 + alpha * f3;

        // Fold: w_folded[2t]   = (1-alpha)*w[4t]   + (2*alpha-1)*w[4t+1]
        //       w_folded[2t+1] = (1-alpha)*w[4t+2] + (2*alpha-1)*w[4t+3]
        FpExt m0 = one_minus_alpha * w0 + two_alpha_minus_one * w1;
        FpExt m1 = one_minus_alpha * w2 + two_alpha_minus_one * w3;

        // Write folded output (same layout as standalone fold kernel)
        uint32_t out_base = t << 1; // 2*t
        f_folded[out_base]     = c0;
        f_folded[out_base + 1] = c1;
        w_folded[out_base]     = m0;
        w_folded[out_base + 1] = m1;

        // Sumcheck on folded pair (identical formula to whir_sumcheck_coeff_moments_round_kernel):
        // X = 1: f_1 = c0 + c1,       term_1 = f_1 * m1
        // X = 2: f_2 = c0 + 2*c1,     term_2 = f_2 * (3*m1 - m0)
        FpExt f_1 = c0 + c1;
        FpExt term_1 = f_1 * m1;

        FpExt f_2 = c0 + c1 + c1;
        FpExt m_2 = m1 * Fp(3) - m0;
        FpExt term_2 = f_2 * m_2;

        local_sums[0] += term_1;
        local_sums[1] += term_2;
    }

#pragma unroll
    for (int idx = 0; idx < S_DEG; idx++) {
        FpExt reduced = sumcheck::block_reduce_sum(local_sums[idx], shared);
        if (threadIdx.x == 0) {
            block_sums[blockIdx.x * S_DEG + idx] = reduced;
        }
        __syncthreads();
    }
}

__device__ __forceinline__ FpExt whir_pow_from_pows2_ext(
    const FpExt *pows2,
    uint32_t log_height,
    uint32_t exponent
) {
    FpExt acc = FpExt(Fp(1));
    for (uint32_t bit = 0; bit < log_height; bit++) {
        if (exponent & (1u << bit)) {
            acc *= pows2[bit];
        }
    }
    return acc;
}

__device__ __forceinline__ Fp whir_pow_from_pows2_base(
    const Fp *pows2,
    uint32_t log_height,
    uint32_t exponent
) {
    Fp acc = Fp(1);
    for (uint32_t bit = 0; bit < log_height; bit++) {
        if (exponent & (1u << bit)) {
            acc *= pows2[bit];
        }
    }
    return acc;
}

__global__ void w_moments_accumulate_kernel(
    FpExt *w_moments,
    const FpExt *z0_pows2, // [log_height], where z0_pows2[i] = z0^(2^i)
    const Fp *z_pows2,     // [num_queries][log_height], row-major
    FpExt gamma,
    uint32_t num_queries,
    uint32_t log_height,
    uint32_t height
) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) {
        return;
    }

    uint32_t exponent = static_cast<uint32_t>(row);

    FpExt acc = gamma * whir_pow_from_pows2_ext(z0_pows2, log_height, exponent);
    FpExt gamma_pow = gamma;
    for (uint32_t i = 0; i < num_queries; i++) {
        gamma_pow *= gamma;
        const Fp *query_pows2 = z_pows2 + i * log_height;
        Fp z_i_pow = whir_pow_from_pows2_base(query_pows2, log_height, exponent);
        acc += gamma_pow * z_i_pow;
    }
    w_moments[row] += acc;
}

// ============================================================================
// LAUNCHERS
// ============================================================================

extern "C" int _whir_algebraic_batch_traces(
    Fp *output, // Length is stacked_height
    const BatchingTracePacket *packets,
    const FpExt *mu_powers, // Len is sum of widths of all matrices
    size_t stacked_height,
    size_t num_packets,
    uint32_t skip_domain
) {
    auto [grid, block] = kernel_launch_params(stacked_height);
    whir_algebraic_batch_traces_kernel<<<grid, block>>>(
        output, packets, mu_powers, stacked_height, num_packets, skip_domain
    );
    return CHECK_KERNEL();
}

inline std::pair<dim3, dim3> whir_sumcheck_coeff_moments_launch_params(uint32_t height) {
    return kernel_launch_params(height >> 1, 256);
}

extern "C" uint32_t _whir_sumcheck_coeff_moments_required_temp_buffer_size(uint32_t height) {
    auto [grid, block] = whir_sumcheck_coeff_moments_launch_params(height);
    return grid.x * S_DEG;
}

extern "C" int _whir_sumcheck_coeff_moments_round(
    const FpExt *f_coeffs,
    const FpExt *w_moments,
    FpExt *output,         // Output: [d=2] final results
    FpExt *tmp_block_sums, // Temporary buffer: [num_blocks * d]
    const uint32_t height
) {
    auto [grid, block] = whir_sumcheck_coeff_moments_launch_params(height);
    unsigned int num_warps = (block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t shmem_bytes = std::max(1u, num_warps) * sizeof(FpExt);

    whir_sumcheck_coeff_moments_round_kernel<<<grid, block, shmem_bytes>>>(
        f_coeffs, w_moments, tmp_block_sums, height
    );

    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    auto num_blocks = grid.x;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    sumcheck::static_final_reduce_block_sums<S_DEG>
        <<<S_DEG, reduce_block, reduce_shmem>>>(tmp_block_sums, output, num_blocks);

    return CHECK_KERNEL();
}

extern "C" int _whir_fold_coeffs_and_moments(
    const FpExt *f_coeffs,
    const FpExt *w_moments,
    FpExt *f_folded_coeffs,
    FpExt *w_folded_moments,
    FpExt alpha,
    uint32_t height
) {
    auto [grid, block] = kernel_launch_params(height >> 1);
    whir_fold_coeffs_and_moments_kernel<<<grid, block>>>(
        f_coeffs, w_moments, f_folded_coeffs, w_folded_moments, alpha, height >> 1
    );
    return CHECK_KERNEL();
}

inline std::pair<dim3, dim3> whir_fused_fold_and_sumcheck_launch_params(uint32_t pre_fold_height) {
    return kernel_launch_params(pre_fold_height >> 2, 256);
}

extern "C" uint32_t _whir_fused_fold_and_sumcheck_required_temp_buffer_size(uint32_t pre_fold_height) {
    auto [grid, block] = whir_fused_fold_and_sumcheck_launch_params(pre_fold_height);
    return grid.x * S_DEG;
}

// Fused fold + sumcheck launcher.
// pre_fold_height is the size of the input f/w buffers.
// Post-fold size: pre_fold_height / 2.
// Sumcheck computed on post-fold values.
extern "C" int _whir_fused_fold_and_sumcheck(
    const FpExt *f_coeffs,      // Pre-fold f, size pre_fold_height
    const FpExt *w_moments,     // Pre-fold w, size pre_fold_height
    FpExt *f_folded,            // Post-fold f, size pre_fold_height/2
    FpExt *w_folded,            // Post-fold w, size pre_fold_height/2
    FpExt *output,              // Output: [S_DEG] final sumcheck results
    FpExt *tmp_block_sums,      // Temporary buffer: [num_blocks * S_DEG]
    FpExt alpha,
    uint32_t pre_fold_height
) {
    auto [grid, block] = whir_fused_fold_and_sumcheck_launch_params(pre_fold_height);
    unsigned int num_warps = (block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t shmem_bytes = std::max(1u, num_warps) * sizeof(FpExt);

    whir_fused_fold_and_sumcheck_kernel<<<grid, block, shmem_bytes>>>(
        f_coeffs, w_moments, f_folded, w_folded, tmp_block_sums, alpha, pre_fold_height >> 2
    );

    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    auto num_blocks = grid.x;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    sumcheck::static_final_reduce_block_sums<S_DEG>
        <<<S_DEG, reduce_block, reduce_shmem>>>(tmp_block_sums, output, num_blocks);

    return CHECK_KERNEL();
}

extern "C" int _w_moments_accumulate(
    FpExt *w_moments,
    const FpExt *z0_pows2,
    const Fp *z_pows2,
    FpExt gamma,
    uint32_t num_queries,
    uint32_t log_height,
    uint32_t height
) {
    auto [grid, block] = kernel_launch_params(height, 256);
    w_moments_accumulate_kernel<<<grid, block>>>(
        w_moments, z0_pows2, z_pows2, gamma, num_queries, log_height, height
    );
    return CHECK_KERNEL();
}
