#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "sumcheck.cuh"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace {

// ============================================================================
// KERNELS
// ============================================================================

// Folds PLE evaluations by interpolating univariate polynomials on coset D and evaluating at r_0
// Input: column-major matrix [height * width] of evaluations
// Output: column-major matrix [new_height * width] of folded evaluations (original OR rotated)
// For each (x, col), collects 2^l_skip evaluations on coset D and interpolates
// Parallelizes barycentric interpolation: each chunk of skip_domain threads handles one output cell
// Grid: (num_row_blocks, width) where blockIdx.y selects the column
template <bool ROTATE>
__global__ void fold_ple_from_evals_kernel(
    const Fp *__restrict__ input_matrix, // [height * width] column-major
    FpExt *__restrict__ output_matrix,   // [new_height * width] column-major
    const Fp *__restrict__ omega_skip_pows, // [skip_domain]
    const FpExt *inv_lagrange_denoms,       // [skip_domain]
    uint32_t height,
    uint32_t skip_domain, // 2^l_skip
    uint32_t l_skip,      // log2(domain_size)
    uint32_t new_height   // lifted_height >> l_skip
) {
    extern __shared__ char smem_raw[];
    FpExt *smem = reinterpret_cast<FpExt *>(smem_raw);

    uint32_t col = blockIdx.y;
    uint32_t chunks_per_block = blockDim.x / skip_domain;
    uint32_t chunk_in_block = threadIdx.x / skip_domain;
    uint32_t tid_in_chunk = threadIdx.x % skip_domain;
    uint32_t x = blockIdx.x * chunks_per_block + chunk_in_block; // row index in output

    // Cannot early-return: all threads must participate in chunk_reduce_sum (uses __syncthreads)
    bool const active_chunk = (x < new_height);

    // Each thread handles exactly ONE term of the barycentric interpolation
    FpExt local_val(Fp::zero());
    if (active_chunk) {
        uint32_t z = tid_in_chunk;
        uint32_t offset = ROTATE ? 1 : 0;
        uint32_t row_idx = ((x << l_skip) + z + offset) % height;
        uint32_t input_idx = col * height + row_idx;
        Fp eval = input_matrix[input_idx];

        // Lagrange interpolation: eval * omega_skip_pows[z] * inv_lagrange_denoms[z]
        local_val = inv_lagrange_denoms[z] * omega_skip_pows[z] * eval;
    }

    // Reduce within chunk
    // All threads must participate; inactive chunks contribute zero
    FpExt result =
        sumcheck::chunk_reduce_sum(local_val, smem, tid_in_chunk, skip_domain, chunk_in_block);

    // No __syncthreads() needed: each chunk writes to a distinct output location
    // and kernel exits immediately after this write
    if (active_chunk && tid_in_chunk == 0) {
        output_matrix[col * new_height + x] = result;
    }
}

__global__ void interpolate_columns_kernel(
    FpExt *__restrict__ interpolated,
    const FpExt *__restrict__ const *__restrict__ columns,
    uint32_t s_deg,
    uint32_t num_y,
    uint32_t num_columns
) {
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t y = tidx % num_y;
    uint32_t col_idx = tidx / num_y;
    if (col_idx >= num_columns)
        return;

    const FpExt *__restrict__ column = columns[col_idx];
    auto t0 = column[y << 1];
    auto t1 = column[(y << 1) | 1];
    FpExt *__restrict__ this_interpolated = interpolated + col_idx * s_deg * num_y;

    for (int x = 0; x < s_deg; x++) {
        this_interpolated[x * num_y + y] = t0 + (t1 - t0) * FpExt(Fp(x + 1u));
    }
}

__global__ void frac_matrix_vertically_repeat_kernel(
    std::pair<FpExt, FpExt> *__restrict__ out,
    const std::pair<FpExt, FpExt> *__restrict__ in,
    const uint32_t width,
    const uint32_t lifted_height,
    const uint32_t height
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t col = blockIdx.y + blockIdx.z * gridDim.y;
    if (col >= width) {
        return;
    }
    out[col * lifted_height + row].first = in[col * height + (row % height)].first;
    out[col * lifted_height + row].second = in[col * height + (row % height)].second;
}

// Vertically repeat separate (F, EF) buffers (for GKR input optimization)
__global__ void frac_matrix_vertically_repeat_mixed_kernel(
    FpExt *__restrict__ out_numerators,
    FpExt *__restrict__ out_denominators,
    const FpExt *__restrict__ in_numerators,
    const FpExt *__restrict__ in_denominators,
    const uint32_t width,
    const uint32_t lifted_height,
    const uint32_t height
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t col = blockIdx.y + blockIdx.z * gridDim.y;
    if (col >= width) {
        return;
    }
    uint32_t src_row = row % height;
    size_t src_idx = col * height + src_row;
    size_t dst_idx = col * lifted_height + row;
    out_numerators[dst_idx] = in_numerators[src_idx];
    out_denominators[dst_idx] = in_denominators[src_idx];
}

// ============================================================================
// LAUNCHERS
// ============================================================================

constexpr uint32_t MAX_GRID_DIM = 65535u;

extern "C" int _fold_ple_from_evals(
    const Fp *input_matrix,
    FpExt *output_matrix,
    const Fp *omega_skip_pows,
    const FpExt *inv_lagrange_denoms,
    uint32_t height,
    uint32_t width,
    uint32_t l_skip,
    uint32_t new_height,
    bool rotate
) {
    uint32_t skip_domain = 1u << l_skip;

    // Block size: at least skip_domain, prefer 256 for occupancy
    uint32_t block_size = std::max(256u, skip_domain);
    uint32_t chunks_per_block = block_size / skip_domain;

    // 2D grid: x-dim for rows, y-dim for columns
    dim3 grid(div_ceil(new_height, chunks_per_block), width);
    dim3 block(block_size);

    // Shared memory for cross-warp reduction (needed when skip_domain > 32)
    // Each warp in the block needs one FpExt slot for cross-warp reduction
    uint32_t total_warps_in_block = block_size / WARP_SIZE;
    size_t smem_bytes = (skip_domain > WARP_SIZE) ? total_warps_in_block * sizeof(FpExt) : 0;

    if (rotate) {
        fold_ple_from_evals_kernel<true><<<grid, block, smem_bytes>>>(
            input_matrix,
            output_matrix,
            omega_skip_pows,
            inv_lagrange_denoms,
            height,
            skip_domain,
            l_skip,
            new_height
        );
    } else {
        fold_ple_from_evals_kernel<false><<<grid, block, smem_bytes>>>(
            input_matrix,
            output_matrix,
            omega_skip_pows,
            inv_lagrange_denoms,
            height,
            skip_domain,
            l_skip,
            new_height
        );
    }
    return CHECK_KERNEL();
}

extern "C" int _interpolate_columns(
    FpExt *interpolated,
    const FpExt *const *columns,
    size_t s_deg,
    size_t num_y,
    size_t num_columns
) {
    auto [grid, block] = kernel_launch_params(num_y * num_columns, 512);

    interpolate_columns_kernel<<<grid, block>>>(interpolated, columns, s_deg, num_y, num_columns);
    return CHECK_KERNEL();
}

extern "C" int _frac_matrix_vertically_repeat(
    std::pair<FpExt, FpExt> *out,
    const std::pair<FpExt, FpExt> *in,
    const uint32_t width,
    const uint32_t lifted_height,
    const uint32_t height
) {
    auto [grid, block] = kernel_launch_params(lifted_height);
    grid.y = std::min(width, MAX_GRID_DIM);
    grid.z = (width + grid.y - 1) / grid.y;
    assert(grid.z <= MAX_GRID_DIM);
    frac_matrix_vertically_repeat_kernel<<<grid, block>>>(out, in, width, lifted_height, height);
    return CHECK_KERNEL();
}

extern "C" int _frac_matrix_vertically_repeat_ext(
    FpExt *out_numerators,
    FpExt *out_denominators,
    const FpExt *in_numerators,
    const FpExt *in_denominators,
    const uint32_t width,
    const uint32_t lifted_height,
    const uint32_t height
) {
    auto [grid, block] = kernel_launch_params(lifted_height);
    grid.y = std::min(width, MAX_GRID_DIM);
    grid.z = (width + grid.y - 1) / grid.y;
    assert(grid.z <= MAX_GRID_DIM);
    frac_matrix_vertically_repeat_mixed_kernel<<<grid, block>>>(
        out_numerators,
        out_denominators,
        in_numerators,
        in_denominators,
        width,
        lifted_height,
        height
    );
    return CHECK_KERNEL();
}

} // namespace
