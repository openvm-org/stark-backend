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

// Computes round of sumcheck on the polynomial `\hat{f} * \hat{w}` where
// `\hat{f}, \hat{w}` are multlinear. Hence the `s` poly has degree 2.
//
// Inputs:
// - `f_evals`, `w_evals`: evaluations of `\hat{f}, \hat{w}` on hypercube
//
// Memory layout: Column-major matrices with height = 2^n
// - For each y: reads indices [2*y, 2*y+1] (even/odd pairs)
// - Outputs partial sums per block, final reduction done in separate kernel
__global__ void whir_sumcheck_mle_round_kernel(
    const FpExt *f_evals,
    const FpExt *w_evals,
    FpExt *block_sums,    // Output: [gridDim.x][S_DEG] partial sums
    const uint32_t height // = 2^n
) {
    extern __shared__ char smem[];
    FpExt *shared = (FpExt *)smem;

    int half_height = height >> 1;

    FpExt local_sums[S_DEG];

// Initialize accumulators
#pragma unroll
    for (int i = 0; i < S_DEG; i++) {
        local_sums[i] = FpExt(0);
    }

    // Map phase: each thread processes multiple y values
    for (int y = blockIdx.x * blockDim.x + threadIdx.x; y < half_height;
         y += gridDim.x * blockDim.x) {

        // For each evaluation point X in {1, 2, ..., d}
        for (int x_int = 1; x_int <= S_DEG; x_int++) {
            Fp x = Fp(x_int);
            FpExt f_0 = f_evals[y << 1];
            FpExt f_1 = f_evals[(y << 1) + 1];
            FpExt f_x = f_0 + (f_1 - f_0) * x;
            FpExt w_0 = w_evals[y << 1];
            FpExt w_1 = w_evals[(y << 1) + 1];
            FpExt w_x = w_0 + (w_1 - w_0) * x;
            local_sums[x_int - 1] = f_x * w_x;
        }
    }

    // Reduce phase: for each x_int
#pragma unroll
    for (int idx = 0; idx < S_DEG; idx++) {
        FpExt reduced = sumcheck::block_reduce_sum(local_sums[idx], shared);

        if (threadIdx.x == 0) {
            block_sums[blockIdx.x * S_DEG + idx] = reduced;
        }
        __syncthreads(); // Needed before reusing shared memory
    }
}

__global__ void w_evals_accumulate_kernel(
    FpExt *w_evals,
    const FpExt *eq_z0,
    const Fp *eq_zs,
    FpExt gamma,
    uint32_t num_queries, // Width of eq_zs
    uint32_t height
) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) {
        return;
    }

    // NOTE(perf): Depending on the number of columns, we may want to use shared memory to reduce this sum.
    w_evals[row] += gamma * eq_z0[row];
    FpExt gamma_pow = gamma;
    for (int i = 0; i < num_queries; i++) {
        gamma_pow *= gamma;
        Fp eq_zi = eq_zs[i * height + row];
        w_evals[row] += gamma_pow * eq_zi;
    }
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

inline std::pair<dim3, dim3> whir_sumcheck_launch_params(uint32_t height) {
    return kernel_launch_params(height >> 1);
}

extern "C" uint32_t _whir_sumcheck_required_temp_buffer_size(uint32_t height) {
    auto [grid, block] = whir_sumcheck_launch_params(height);
    return grid.x * S_DEG;
}

extern "C" int _whir_sumcheck_mle_round(
    const FpExt *f_evals,
    const FpExt *w_evals,
    FpExt *output,         // Output: [d=2] final results
    FpExt *tmp_block_sums, // Temporary buffer: [num_blocks * d]
    const uint32_t height
) {
    auto [grid, block] = whir_sumcheck_launch_params(height);
    unsigned int num_warps = (block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t shmem_bytes = std::max(1u, num_warps) * sizeof(FpExt);

    // Launch main kernel - writes to tmp_block_sums
    whir_sumcheck_mle_round_kernel<<<grid, block, shmem_bytes>>>(
        f_evals, w_evals, tmp_block_sums, height
    );

    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Launch final reduction kernel - reads from tmp_block_sums, writes to output
    auto num_blocks = grid.x;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    // d = 2, WD = 1 so we have gridDim.x = 2 for exactly 2 blocks
    sumcheck::static_final_reduce_block_sums<S_DEG>
        <<<S_DEG, reduce_block, reduce_shmem>>>(tmp_block_sums, output, num_blocks);

    return CHECK_KERNEL();
}

extern "C" int _w_evals_accumulate(
    FpExt *w_evals,
    const FpExt *eq_z0,
    const Fp *eq_zs,
    FpExt gamma,
    uint32_t num_queries,
    uint32_t height
) {
    auto [grid, block] = kernel_launch_params(height);

    w_evals_accumulate_kernel<<<grid, block>>>(w_evals, eq_z0, eq_zs, gamma, num_queries, height);

    return CHECK_KERNEL();
}
