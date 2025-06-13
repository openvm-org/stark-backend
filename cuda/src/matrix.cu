// FROM https://github.com/scroll-tech/plonky3-gpu/blob/openvm-v2/gpu-backend/src/cuda/kernels/transpose.cu
#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"

const size_t TILE_SIZE = 32; // do not change,
// const uint64_t GROUP_SIZE = 32; // do not change,

template <typename T>
__global__ void __launch_bounds__(TILE_SIZE)
    cukernel_matrix_transpose(T *output, const T *input, size_t col_size, size_t row_size) {
    __shared__ T s_mem[TILE_SIZE][TILE_SIZE + 1];
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
template __global__ void cukernel_matrix_transpose<Fp>(Fp *, const Fp *, size_t, size_t);
template __global__ void cukernel_matrix_transpose<FpExt>(FpExt *, const FpExt *, size_t, size_t);

__global__ void cukernel_matrix_get_rows_fp(
    Fp *output,
    const Fp *input,
    uint32_t *row_indices,
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

// plonky3/matrix/src/lib.rs: fn vertically_strided
__global__ void cukernel_split_ext_poly_to_multiple_base_matrix(
    const uint64_t *d_matrix_ptr, // array of matrices over Fp
    FpExt *d_poly,
    uint64_t poly_len,
    uint64_t num_chunk
) {
    uint64_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= poly_len) { // [0, poly_len - 1]
        return;
    }

    uint64_t chunk_idx = row_idx % num_chunk;
    Fp *d_matrix = (Fp *)d_matrix_ptr[chunk_idx];
    // assumption:
    //   1. FpExt::D = 4, the input FpExt will be flatten to Fp
    //   2. the order of the matrix is column major
    // uint64_t matrix_width = 4; // FpExt::D
    uint64_t matrix_height = (poly_len / num_chunk);
    uint64_t remainder = poly_len % num_chunk;
    if (chunk_idx < remainder) {
        matrix_height += 1;
    }
    uint64_t chunk_row = row_idx / num_chunk;

    FpExt ext_val = d_poly[row_idx];
    d_matrix[matrix_height * 0 + chunk_row] = ext_val.elems[0];
    d_matrix[matrix_height * 1 + chunk_row] = ext_val.elems[1];
    d_matrix[matrix_height * 2 + chunk_row] = ext_val.elems[2];
    d_matrix[matrix_height * 3 + chunk_row] = ext_val.elems[3];
}

// END OF gpu-backend/src/cuda/kernels/matrix.cu

__global__ void unpack_matrix_kernel(Fp *output, const FpExt *input, size_t rows, size_t cols) {
    size_t row_idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t col_idx = threadIdx.y + blockIdx.y * blockDim.y;

    if (col_idx >= cols || row_idx >= rows)
        return;

    output += row_idx + col_idx * rows * 4;
    input += row_idx + col_idx * rows;

    FpExt local_ext = *input;

    output[0 * rows] = local_ext.elems[0];
    output[1 * rows] = local_ext.elems[1];
    output[2 * rows] = local_ext.elems[2];
    output[3 * rows] = local_ext.elems[3];
}

template <typename T>
int matrix_transpose_impl(T *output, const T *input, size_t col_size, size_t row_size) {
    uint32_t grid_x = (col_size + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t grid_y = (row_size + TILE_SIZE - 1) / TILE_SIZE;

    dim3 grid(grid_x * grid_y);
    dim3 block(TILE_SIZE);

    cukernel_matrix_transpose<T><<<grid, block>>>(output, input, col_size, row_size);

    return cudaGetLastError();
}

extern "C" int matrix_transpose_fp(Fp *output, const Fp *input, size_t col_size, size_t row_size) {
    return matrix_transpose_impl(output, input, col_size, row_size);
}

extern "C" int matrix_transpose_fpext(
    FpExt *output,
    const FpExt *input,
    size_t col_size,
    size_t row_size
) {
    return matrix_transpose_impl(output, input, col_size, row_size);
}

extern "C" int _split_ext_poly_to_multiple_base_matrix(
    const uint64_t *d_matrix_ptr,
    FpExt *d_poly,
    uint64_t poly_len,
    uint64_t num_chunk
) {
    auto [grid, block] = kernel_launch_params(poly_len);
    cukernel_split_ext_poly_to_multiple_base_matrix<<<grid, block>>>(
        d_matrix_ptr, d_poly, poly_len, num_chunk
    );
    return cudaGetLastError();
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
    cukernel_matrix_get_rows_fp<<<grid, block>>>(
        output, input, row_indices, matrix_width, matrix_height
    );
    return cudaGetLastError();
}

extern "C" int _unpack_matrix(
    Fp *output,
    const FpExt *input,
    size_t input_height,
    size_t input_width
) {
    auto [grid, block] = kernel_launch_2d_params(input_height, input_width);
    unpack_matrix_kernel<<<grid, block>>>(output, input, input_height, input_width);
    return cudaGetLastError();
}