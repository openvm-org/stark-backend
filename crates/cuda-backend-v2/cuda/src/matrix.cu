#include "fp.h"
#include "frac_ext.cuh"
#include "launcher.cuh"
#include "utils.cuh"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>

// `in` has width equal to `width` and height `domain_size * num_x`.
// `out` has width equal to `width` and height `padded_size * num_x`.
//
// CAUTION: rotation is with respect to `domain_size * num_x`, but padding is with respect to `domain_size` to `padded_size`.
__global__ void batch_rotate_pad_kernel(
    Fp *__restrict__ out,
    const Fp *__restrict__ in,
    uint32_t width,
    uint32_t num_x,
    uint32_t domain_size,
    uint32_t padded_size
) {
    auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    auto pidx = blockIdx.y + blockIdx.z * gridDim.y;

    if (pidx >= width * num_x) {
        return;
    }
    if (tidx < domain_size) {
        auto tidx_rot = tidx + 1;
        auto pidx_rot = pidx;
        if (tidx_rot == domain_size) {
            tidx_rot = 0;
            pidx_rot += 1;
            // NOTE: rotation is over `domain_size * num_x`
            if (pidx_rot % num_x == 0) {
                pidx_rot -= num_x;
            }
        }
        out[padded_size * pidx + tidx] = in[domain_size * pidx_rot + tidx_rot];
    } else if (tidx < padded_size) {
        out[padded_size * pidx + tidx] = Fp(0);
    }
}

// `matrix` is `padded_height x width` matrix.
// This kernel cyclically repeats the first `height` rows of the matrix vertically for `lifted_height` rows.
// The rows `lifted_height..padded_height` are left untouched.
__global__ void lift_padded_matrix_evals_kernel(
    Fp *__restrict__ matrix,
    uint32_t width,
    uint32_t height,
    uint32_t lifted_height,
    uint32_t padded_height
) {
    auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    auto col = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx >= lifted_height || col >= width) {
        return;
    }
    // lhs = rhs when tidx < height
    matrix[col * padded_height + tidx] = matrix[col * padded_height + (tidx % height)];
}

// Required: lifted_height = height * stride
__global__ void collapse_strided_matrix_kernel(
    Fp *__restrict__ out,
    const Fp *__restrict__ in,
    const uint32_t width,
    const uint32_t lifted_height,
    const uint32_t height,
    const uint32_t stride
) {
    auto row = threadIdx.x + blockIdx.x * blockDim.x;
    auto col = threadIdx.y + blockIdx.y * blockDim.y;
    if (row >= height || col >= width) {
        return;
    }

    out[col * height + row] = in[col * lifted_height + row * stride];
}

// `out` is `padded_height x width` column-major matrix.
// `in` is `height x width` column-major matrix.
//
// This kernel is for use when `width` is large (~2^20) while `height` is small (<2^10)
__global__ void batch_expand_pad_wide_kernel(
    Fp *__restrict__ out,
    const Fp *__restrict__ in,
    const uint32_t width,
    const uint32_t padded_height,
    const uint32_t height
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t col = blockIdx.y + blockIdx.z * gridDim.y;
    if (col >= width) {
        return;
    }
    if (row < height) {
        out[col * padded_height + row] = in[col * height + row];
    } else if (row < padded_height) {
        out[col * padded_height + row] = Fp(0);
    }
}

template <typename T>
__global__ void bitrev_kernel(
    T* __restrict__ buffer,
    uint32_t const log_n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (1 << log_n)) {
        return;
    }
    uint32_t rev_idx = rev_len(idx, log_n);
    if (idx < rev_idx) {
        auto const tmp = buffer[idx];
        buffer[idx] = buffer[rev_idx];
        buffer[rev_idx] = tmp;
    }
}

// ============================================================================
// LAUNCHERS
// ============================================================================

constexpr uint32_t MAX_GRID_DIM = 65535u;

extern "C" int _batch_rotate_pad(
    Fp *out,
    const Fp *in,
    uint32_t width,
    uint32_t num_x, // = (in.height() / domain_size)
    uint32_t domain_size,
    uint32_t padded_size
) {
    auto [grid, block] = kernel_launch_params(padded_size);
    auto num_poly = width * num_x;
    grid.y = std::min(num_poly, MAX_GRID_DIM);
    grid.z = (num_poly + grid.y - 1) / grid.y;
    batch_rotate_pad_kernel<<<grid, block>>>(out, in, width, num_x, domain_size, padded_size);
    return CHECK_KERNEL();
}

extern "C" int _lift_padded_matrix_evals(
    Fp *matrix,
    uint32_t width,
    uint32_t height,
    uint32_t lifted_height,
    uint32_t padded_height
) {
    auto [grid, block] = kernel_launch_2d_params(lifted_height, width);
    lift_padded_matrix_evals_kernel<<<grid, block>>>(
        matrix, width, height, lifted_height, padded_height
    );
    return CHECK_KERNEL();
}

extern "C" int _collapse_strided_matrix(
    Fp *out,
    const Fp *in,
    uint32_t width,
    uint32_t height,
    uint32_t stride
) {
    auto lifted_height = height * stride;
    auto [grid, block] = kernel_launch_2d_params(height, width);
    collapse_strided_matrix_kernel<<<grid, block>>>(out, in, width, lifted_height, height, stride);
    return CHECK_KERNEL();
}

extern "C" int _batch_expand_pad_wide(
    Fp *out,
    const Fp *in,
    const uint32_t width,
    const uint32_t padded_height,
    const uint32_t height
) {
    auto [grid, block] = kernel_launch_params(padded_height);
    grid.y = std::min(width, MAX_GRID_DIM);
    grid.z = (width + grid.y - 1) / grid.y;
    assert(grid.z <= MAX_GRID_DIM);
    batch_expand_pad_wide_kernel<<<grid, block>>>(out, in, width, padded_height, height);
    return CHECK_KERNEL();
}

extern "C" int _bitrev(
    FracExt* buffer,
    size_t n
) {
    if (n == 0) {
        return 0;
    }
    assert((n & (n - 1)) == 0);
    uint32_t const log_n = __builtin_ctz(n);

    auto [grid, block] = kernel_launch_params(n);
    bitrev_kernel<FracExt><<<grid, block>>>(buffer, log_n);
    return CHECK_KERNEL();
}
