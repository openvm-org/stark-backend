// FROM https://github.com/scroll-tech/plonky3-gpu/blob/openvm-v2/gpu-backend/src/cuda/kernels/transpose.cu
#include "fp.h"
#include "fpext.h"

const uint64_t TILE_SIZE = 32; // do not change,
// const uint64_t GROUP_SIZE = 32; // do not change,

template <typename T>
__global__ void __launch_bounds__(TILE_SIZE)
    cukernel_matrix_transpose(T *output, const T *input, uint64_t col_size, uint64_t row_size) {
    __shared__ T s_mem[TILE_SIZE][TILE_SIZE + 1];
    uint64_t dim_x = (col_size + TILE_SIZE - 1) / TILE_SIZE;
    uint64_t bid = blockIdx.x; // (x, 1, 1)
    uint64_t bid_y = bid / dim_x;
    uint64_t bid_x = bid % dim_x; // (bid_x, bid_y, 1)

    uint64_t tid = threadIdx.x;
    uint64_t index_i = bid_y * TILE_SIZE * col_size + bid_x * TILE_SIZE + tid;
    uint64_t index_o = bid_x * TILE_SIZE * row_size + bid_y * TILE_SIZE + tid;

    //input
    bool boundray_column = bid_x * TILE_SIZE + tid < col_size;
    uint64_t row_offset = bid_y * TILE_SIZE + 0;
    for (uint64_t i = 0; i < TILE_SIZE; ++i) {
        bool boundray = boundray_column && (row_offset + i < row_size);
        s_mem[i][tid] = (boundray) ? input[index_i + i * col_size] : T(0);
    }
    __syncthreads();

    //output
    boundray_column = bid_y * TILE_SIZE + tid < row_size;
    row_offset = bid_x * TILE_SIZE + 0;
    for (uint64_t i = 0; i < TILE_SIZE; ++i) {
        bool boundray = boundray_column && (row_offset + i < col_size);
        if (boundray)
            output[index_o + i * row_size] = s_mem[tid][i];
    }
}

// Explicit instantiations
template __global__ void cukernel_matrix_transpose<Fp>(Fp *, const Fp *, uint64_t, uint64_t);
template __global__ void cukernel_matrix_transpose<FpExt>(
    FpExt *,
    const FpExt *,
    uint64_t,
    uint64_t
);

// END OF gpu-backend/src/cuda/kernels/transpose.cu

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