#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "sumcheck.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdint>

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
    FpExt *block_sums,    // Output: [gridDim.x][d][WD] partial sums
    const uint32_t height // = 2^n
) {
    const int s_deg = 2;
    extern __shared__ char smem[];
    FpExt *shared = (FpExt *)smem;

    int half_height = height >> 1;

    FpExt local_sums[2];

// Initialize accumulators
#pragma unroll
    for (int i = 0; i < 2; i++) {
        local_sums[i] = {0, 0, 0, 0};
    }

    // Map phase: each thread processes multiple y values
    for (int y = blockIdx.x * blockDim.x + threadIdx.x; y < half_height;
         y += gridDim.x * blockDim.x) {

        // For each evaluation point X in {1, 2, ..., d}
        for (int x_int = 1; x_int <= s_deg; x_int++) {
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
    for (int idx = 0; idx < s_deg; idx++) {
        FpExt reduced = sumcheck::block_reduce_sum(local_sums[idx], shared);

        if (threadIdx.x == 0) {
            block_sums[blockIdx.x * s_deg + idx] = reduced;
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

extern "C" int _whir_sumcheck_mle_round(
    const FpExt *f_evals,
    const FpExt *w_evals,
    FpExt *output,         // Output: [d=2] final results
    FpExt *tmp_block_sums, // Temporary buffer: [num_blocks * d]
    const uint32_t height
) {
    int half_height = height >> 1;
    // NOTE: if you change the launch params, update `get_num_blocks` in src/cuda/whir.rs
    auto [grid, block] = kernel_launch_params(half_height);
    unsigned int num_warps = (block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t shmem_bytes = std::max(1u, num_warps) * sizeof(FpExt);

    // Launch main kernel - writes to tmp_block_sums
    whir_sumcheck_mle_round_kernel<<<grid, block, shmem_bytes>>>(f_evals, w_evals, tmp_block_sums, height);

    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Launch final reduction kernel - reads from tmp_block_sums, writes to output
    auto num_blocks = grid.x;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    // d = 2, WD = 1 so we have gridDim.x = 2 for exactly 2 blocks
    sumcheck::final_reduce_block_sums<2>
        <<<2, reduce_block, reduce_shmem>>>(tmp_block_sums, output, num_blocks);

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
