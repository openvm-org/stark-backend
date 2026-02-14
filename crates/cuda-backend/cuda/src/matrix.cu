/*
 * Source: https://github.com/scroll-tech/plonky3-gpu (private repo)
 * Status: BASED ON plonky3-gpu/gpu-backend/src/cuda/kernels/matrix.cu
 * Imported: 2025-01-25 by @gaxiom
 */

#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>

const size_t TILE_SIZE = 32; // do not change,
// const uint64_t GROUP_SIZE = 32; // do not change,

template <typename T>
__global__ void __launch_bounds__(
    TILE_SIZE
) matrix_transpose_kernel(T *output, const T *input, size_t col_size, size_t row_size) {

    // NOTE: This is what builds, we need to use this for clang-tidy to work
    // __shared__ T s_mem[TILE_SIZE][TILE_SIZE + 1];
    __shared__ __align__(
        alignof(T)
    ) unsigned char s_mem_raw[TILE_SIZE * (TILE_SIZE + 1) * sizeof(T)];
    T(*s_mem)[TILE_SIZE + 1] = reinterpret_cast<T(*)[TILE_SIZE + 1]>(s_mem_raw);

    size_t dim_x = (col_size + TILE_SIZE - 1) / TILE_SIZE;
    size_t bid = blockIdx.x; // (x, 1, 1)
    size_t bid_y = bid / dim_x;
    size_t bid_x = bid % dim_x; // (bid_x, bid_y, 1)

    size_t tid = threadIdx.x;
    size_t index_i = bid_y * TILE_SIZE * col_size + bid_x * TILE_SIZE + tid;
    size_t index_o = bid_x * TILE_SIZE * row_size + bid_y * TILE_SIZE + tid;

    // input
    bool boundray_column = bid_x * TILE_SIZE + tid < col_size;
    size_t row_offset = bid_y * TILE_SIZE + 0;
    for (auto i = 0; i < TILE_SIZE; ++i) {
        bool boundray = boundray_column && (row_offset + i < row_size);
        s_mem[i][tid] = (boundray) ? input[index_i + i * col_size] : T(0);
    }
    __syncthreads();

    // output
    boundray_column = bid_y * TILE_SIZE + tid < row_size;
    row_offset = bid_x * TILE_SIZE + 0;
    for (auto i = 0; i < TILE_SIZE; ++i) {
        bool boundray = boundray_column && (row_offset + i < col_size);
        if (boundray)
            output[index_o + i * row_size] = s_mem[tid][i];
    }
}

// Explicit instantiations
template __global__ void matrix_transpose_kernel<Fp>(Fp *, const Fp *, size_t, size_t);
template __global__ void matrix_transpose_kernel<FpExt>(FpExt *, const FpExt *, size_t, size_t);

__global__ void matrix_get_rows_fp_kernel(
    Fp *__restrict__ output,
    const Fp *__restrict__ input,
    uint32_t *__restrict__ row_indices,
    uint64_t matrix_width,
    uint64_t matrix_height
) {
    uint32_t col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx >= matrix_width) {
        return;
    }
    // Note: blockIdx.y may >= matrix_height, do not assert

    uint64_t input_row = row_indices[blockIdx.y];
    uint64_t output_row = blockIdx.y;
    uint64_t input_idx = col_idx * matrix_height + input_row;  // col-major matrix
    uint64_t output_idx = output_row * matrix_width + col_idx; // row-major matrix
    output[output_idx] = input[input_idx];
}

__global__ void split_ext_to_base_col_major_matrix_kernel(
    Fp *__restrict__ d_matrix,
    const FpExt *__restrict__ d_poly,
    uint64_t poly_len,
    uint32_t matrix_height
) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= matrix_height) {
        return;
    }

    uint32_t col_num = (poly_len / matrix_height); // SPLIT_FACTOR = 2
    for (uint32_t col_idx = 0; col_idx < col_num; col_idx++) {
        FpExt ext_val = d_poly[col_idx * matrix_height + row_idx];
        d_matrix[(col_idx * 4 + 0) * matrix_height + row_idx] = ext_val.elems[0];
        d_matrix[(col_idx * 4 + 1) * matrix_height + row_idx] = ext_val.elems[1];
        d_matrix[(col_idx * 4 + 2) * matrix_height + row_idx] = ext_val.elems[2];
        d_matrix[(col_idx * 4 + 3) * matrix_height + row_idx] = ext_val.elems[3];
    }
}

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

// memory layout of in: column-major
// if polyCount = 4, then the layout of in is expected to be
// in:  |  poly0  |  poly1  |  poly2  |  poly3  |
//
// if outSize / inSize = 2, then the layout of out is expected to be
// out: |  poly0  |    0    |  poly1  |    0    |   poly2  |    0    |  poly3  |    0    |
__global__ void batch_expand_pad_kernel(
    Fp *out,
    const Fp *in,
    const uint32_t polyCount,
    const uint32_t outSize,
    const uint32_t inSize
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outSize) {
        for (uint32_t i = 0; i < polyCount; i++) {
            Fp res = Fp(0);
            if (idx < inSize) {
                res = in[i * inSize + idx];
            }
            out[i * outSize + idx] = res;
        }
    }
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

constexpr uint32_t MAX_GRID_DIM = 65535u;

template <typename T>
int matrix_transpose_impl(T *output, const T *input, size_t col_size, size_t row_size) {
    uint32_t grid_x = (col_size + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t grid_y = (row_size + TILE_SIZE - 1) / TILE_SIZE;

    dim3 grid(grid_x * grid_y);
    dim3 block(TILE_SIZE);

    matrix_transpose_kernel<T><<<grid, block>>>(output, input, col_size, row_size);

    return CHECK_KERNEL();
}

extern "C" int _matrix_transpose_fp(Fp *output, const Fp *input, size_t col_size, size_t row_size) {
    return matrix_transpose_impl(output, input, col_size, row_size);
}

extern "C" int _matrix_transpose_fpext(
    FpExt *output,
    const FpExt *input,
    size_t col_size,
    size_t row_size
) {
    return matrix_transpose_impl(output, input, col_size, row_size);
}

extern "C" int _matrix_get_rows_fp(
    Fp *output,
    const Fp *input,
    uint32_t *row_indices,
    uint64_t matrix_width,
    uint64_t matrix_height,
    uint32_t row_indices_len
) {
    auto block = WARP_SIZE;
    dim3 grid = dim3(div_ceil(matrix_width, WARP_SIZE), row_indices_len);
    matrix_get_rows_fp_kernel<<<grid, block>>>(
        output, input, row_indices, matrix_width, matrix_height
    );
    return CHECK_KERNEL();
}

extern "C" int _split_ext_to_base_col_major_matrix(
    Fp *d_matrix,
    FpExt *d_poly,
    uint64_t poly_len,
    uint32_t matrix_height
) {
    auto [grid, block] = kernel_launch_params(matrix_height);
    split_ext_to_base_col_major_matrix_kernel<<<grid, block>>>(
        d_matrix, d_poly, poly_len, matrix_height
    );
    return CHECK_KERNEL();
}

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

extern "C" int _batch_expand_pad(
    Fp *out,
    const Fp *in,
    const uint32_t polyCount,
    const uint32_t outSize,
    const uint32_t inSize
) {
    auto [grid, block] = kernel_launch_params(outSize);
    batch_expand_pad_kernel<<<grid, block>>>(out, in, polyCount, outSize, inSize);
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
