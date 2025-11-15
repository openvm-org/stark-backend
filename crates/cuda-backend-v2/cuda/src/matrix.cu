#include "fp.h"
#include "launcher.cuh"
#include <algorithm>
#include <cassert>
#include <cstdint>

// out has width equal to `2 * width`
__global__ void batch_rotate_pad_kernel(
    Fp *__restrict__ out,
    const Fp *__restrict__ in,
    uint32_t width,
    uint32_t num_x,
    uint32_t domain_size,
    uint32_t padded_size
) {
    auto row = blockIdx.y;
    if (row >= domain_size) {
        return;
    }
    auto col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col >= width * num_x) {
        return;
    }
    auto value = in[domain_size * col + row];
    out[padded_size * col + row] = value;
    // rotation over domain_size
    if (row == 0) {
        row = domain_size - 1;
        col = (col % num_x == 0) ? col + (num_x - 1) : col - 1;
    } else {
        row = row - 1;
    }
    out[padded_size * (width * num_x + col) + row] = value;
}

// This is a weird kernel for a specific purpose and may be deleted later.
// `in` is a `width x lifted_height` matrix that is **assumed** to be the lifting of a `width x height` matrix (so it is vertically repeating every `height` rows). This kernel will rotate the **unlifted** matrix, lift it back to `width x lifted_height`, and then zero-expand each column to `width x padded_height`.
//
// Required: `height` divides `lifted_height`, `lifted_height` divides `padded_height`
//
// NOTE: unlike `batch_rotate_pad_kernel`, this kernel does not do plain expand without rotate.
__global__ void batch_rotate_lift_and_pad_kernel(
    Fp *__restrict__ out,
    const Fp *__restrict__ in,
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
    auto row = tidx % height;
    auto row_rot = (row + 1) % height;
    auto value = in[col * lifted_height + row_rot];
    out[col * padded_height + tidx] = value;
}

// Required: lifted_height = height * stride
__global__ void collapse_and_lift_strided_matrix_kernel(
    Fp *out,
    const Fp *in,
    uint32_t width,
    uint32_t lifted_height,
    uint32_t height,
    uint32_t stride
) {
    auto row = threadIdx.x + blockIdx.x * blockDim.x;
    auto col = threadIdx.y + blockIdx.y * blockDim.y;
    if (row >= lifted_height || col >= width) {
        return;
    }

    out[col * lifted_height + row] = in[col * lifted_height + (row % height) * stride];
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

// ============================================================================
// LAUNCHERS
// ============================================================================

extern "C" int _batch_rotate_pad(
    Fp *out,
    const Fp *in,
    uint32_t width,
    uint32_t num_x, // = (matrix.height() / domain_size)
    uint32_t domain_size,
    uint32_t padded_size
) {
    auto [grid, block] = kernel_launch_params(width * num_x);
    grid.y = domain_size;
    batch_rotate_pad_kernel<<<grid, block>>>(out, in, width, num_x, domain_size, padded_size);
    return CHECK_KERNEL();
}

extern "C" int _batch_rotate_lift_and_pad(
    Fp *out,
    const Fp *in,
    uint32_t width,
    uint32_t height,
    uint32_t lifted_height,
    uint32_t padded_height
) {
    auto [grid, block] = kernel_launch_2d_params(lifted_height, width);
    batch_rotate_lift_and_pad_kernel<<<grid, block>>>(
        out, in, width, height, lifted_height, padded_height
    );
    return CHECK_KERNEL();
}

extern "C" int _collapse_and_lift_strided_matrix(
    Fp *out,
    const Fp *in,
    uint32_t width,
    uint32_t height,
    uint32_t stride
) {
    auto lifted_height = height * stride;
    auto [grid, block] = kernel_launch_2d_params(lifted_height, width);
    collapse_and_lift_strided_matrix_kernel<<<grid, block>>>(
        out, in, width, lifted_height, height, stride
    );
    return CHECK_KERNEL();
}

extern "C" int _batch_expand_pad_wide(
    Fp *out,
    const Fp *in,
    const uint32_t width,
    const uint32_t padded_height,
    const uint32_t height
) {
    constexpr uint32_t MAX_GRID_DIM = 65535u;
    auto [grid, block] = kernel_launch_params(padded_height);
    grid.y = std::min(width, MAX_GRID_DIM);
    grid.z = (width + grid.y - 1) / grid.y;
    assert(grid.z <= MAX_GRID_DIM);
    batch_expand_pad_wide_kernel<<<grid, block>>>(out, in, width, padded_height, height);
    return CHECK_KERNEL();
}
