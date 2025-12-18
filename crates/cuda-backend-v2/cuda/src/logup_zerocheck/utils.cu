#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
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
// For each (x, col), collects 2^l_skip evaluations on coset D and interpolates for both offsets
template <bool ROTATE>
__global__ void fold_ple_from_evals_kernel(
    const Fp *__restrict__ input_matrix, // [height * width] column-major
    FpExt *__restrict__ output_matrix,   // [new_height * output_width] column-major
    // If ROTATE: output_width = width * 2, layout: [orig_cols, rot_cols]
    // If !ROTATE: output_width = width
    const Fp *__restrict__ omega_skip_pows, // [skip_domain]
    const FpExt *inv_lagrange_denoms,       // [skip_domain]
    uint32_t height,
    uint32_t width,
    uint32_t skip_domain, // 2^l_skip
    uint32_t l_skip,      // log2(domain_size)
    uint32_t new_height   // lifted_height >> l_skip
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = new_height * width;
    if (idx >= total_outputs)
        return;

    int x = idx % new_height;
    int col = idx / new_height;

    // Barycentric interpolation:
    // NOTE: there is no special handling of the case when lagrange denominator is zero because the random point `r_0` should lie outside of the skip domain with high probability.
    if constexpr (!ROTATE) {
        // Compute offset 0 if rotate=false
        FpExt result_0(Fp::zero());

        for (int z = 0; z < skip_domain; z++) {
            // Offset 0: ((x << l_skip) + z + 0) % height
            int row_idx_0 = ((x << l_skip) + z) % height;
            int input_idx_0 = col * height + row_idx_0;
            Fp eval_0 = input_matrix[input_idx_0];
            // Lagrange interpolation: eval * numerators[z] * inv_lagrange_denoms[z]
            result_0 += inv_lagrange_denoms[z] * omega_skip_pows[z] * eval_0;
        }

        // Write original columns: output_matrix[col * new_height + x]
        int output_idx_original = col * new_height + x;
        output_matrix[output_idx_original] = result_0;
    } else {
        // Compute offset 1 if rotate=true
        FpExt result_1(Fp::zero());
        for (int z = 0; z < skip_domain; z++) {
            // Offset 1: ((x << l_skip) + z + 1) % height
            int row_idx_1 = ((x << l_skip) + z + 1) % height;
            int input_idx_1 = col * height + row_idx_1;
            Fp eval_1 = input_matrix[input_idx_1];
            result_1 += inv_lagrange_denoms[z] * omega_skip_pows[z] * eval_1;
        }
        // Write rotated columns: output_matrix[(width + col) * new_height + x]
        // Layout: [orig_col0...orig_col{width-1}, rot_col0...rot_col{width-1}]
        int output_idx_rotated = col * new_height + x;
        output_matrix[output_idx_rotated] = result_1;
    }
}

// Combined kernel: mutates eq_xi in-place (eq_xi *= eq_r0) and computes eq_sharp (original_eq_xi * eq_sharp_r0)
// Note: eq_sharp uses the ORIGINAL eq_xi value, not the multiplied one (matches CPU behavior)
__global__ void compute_eq_sharp_kernel(
    FpExt *eq_xi,      // [count] input/output: mutated in-place to eq_xi * eq_r0
    FpExt *eq_sharp,   // [count] output: original_eq_xi * eq_sharp_r0
    FpExt eq_r0,       // scalar
    FpExt eq_sharp_r0, // scalar
    uint32_t count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    FpExt original_eq_xi = eq_xi[idx];

    // Mutate in-place: eq_xi *= eq_r0
    eq_xi[idx] = original_eq_xi * eq_r0;

    // Compute eq_sharp using the ORIGINAL eq_xi value
    eq_sharp[idx] = original_eq_xi * eq_sharp_r0;
}

__global__ void interpolate_columns_kernel(
    FpExt *__restrict__ interpolated,
    const FpExt *__restrict__ const *__restrict__ columns,
    uint32_t s_deg,
    uint32_t num_y,
    uint32_t num_columns
) {
    int y = threadIdx.x + blockIdx.x * blockDim.x;
    if (y >= num_y)
        return;

    int col_idx = threadIdx.y + blockIdx.y * blockDim.y;
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
    int total_outputs = new_height * width;
    auto [grid, block] = kernel_launch_params(total_outputs);

    if (rotate) {
        fold_ple_from_evals_kernel<true><<<grid, block>>>(
            input_matrix,
            output_matrix,
            omega_skip_pows,
            inv_lagrange_denoms,
            height,
            width,
            1 << l_skip,
            l_skip,
            new_height
        );
    } else {
        fold_ple_from_evals_kernel<false><<<grid, block>>>(
            input_matrix,
            output_matrix,
            omega_skip_pows,
            inv_lagrange_denoms,
            height,
            width,
            1 << l_skip,
            l_skip,
            new_height
        );
    }
    return CHECK_KERNEL();
}

extern "C" int _compute_eq_sharp(
    FpExt *eq_xi,
    FpExt *eq_sharp,
    const FpExt eq_r0,
    const FpExt eq_sharp_r0,
    uint32_t count
) {
    if (count == 0)
        return 0;
    auto [grid, block] = kernel_launch_params(count);
    compute_eq_sharp_kernel<<<grid, block>>>(eq_xi, eq_sharp, eq_r0, eq_sharp_r0, count);
    return CHECK_KERNEL();
}

extern "C" int _interpolate_columns(
    FpExt *interpolated,
    const FpExt *const *columns,
    size_t s_deg,
    size_t num_y,
    size_t num_columns
) {
    auto [grid, block] = kernel_launch_2d_params(num_y, num_columns);

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
