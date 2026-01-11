#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "poseidon2.cuh"
#include <cstddef>
#include <cstdint>

struct digest_t {
    Fp cells[CELLS_OUT];
};

// This kernel computes the row hash of each row, and then computes the merkle root of `rows_per_query` strided rows at a time, where the stride is `height / rows_per_query`.
// - height = query_stride * 2^log_rows_per_query
// - blockDim.y = 2^log_rows_per_query
// - gridDim.y = 1
__global__ void poseidon2_compressing_row_hashes_kernel(
    digest_t *out,    // [query_stride]
    const Fp *matrix, // [width][height]
    size_t width,
    size_t height,
    size_t query_stride,
    size_t log_rows_per_query
) {
    extern __shared__ char smem[]; // Fp[CELLS_OUT][blockDim.y / 2 * blockDim.x]
    Fp *shared = reinterpret_cast<Fp *>(smem);
    const uint32_t shared_stride = blockDim.x * (blockDim.y >> 1);

    const uint32_t stride_idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t leaf_idx = threadIdx.y;
    const uint32_t row = leaf_idx * query_stride + stride_idx;

    size_t used = 0;
    Fp cells[CELLS];
#pragma unroll
    for (int i = 0; i < CELLS; i++) {
        cells[i] = Fp(0);
    }

    if (stride_idx < query_stride) {
        // compute row hash
        for (int col = 0; col < width; col++) {
            cells[used++] = matrix[col * height + row];
            if (used == CELLS_RATE) {
                poseidon2::poseidon2_mix(cells);
                used = 0;
            }
        }
        if (used != 0) {
            poseidon2::poseidon2_mix(cells);
        }
    }

    for (int layer = 0; layer < log_rows_per_query; ++layer) {
        uint32_t mask = (1 << (layer + 1)) - 1;
        auto shared_offset = ((leaf_idx >> (layer + 1)) << layer) * blockDim.x + threadIdx.x;
        if ((leaf_idx & mask) == (1 << layer)) {
#pragma unroll
            for (int i = 0; i < CELLS_OUT; i++) {
                shared[i * shared_stride + shared_offset] = cells[i];
            }
        }
        __syncthreads();
        if ((leaf_idx & mask) == 0) {
#pragma unroll
            for (int i = 0; i < CELLS_OUT; i++) {
                cells[CELLS_OUT + i] = shared[i * shared_stride + shared_offset];
            }
            poseidon2::poseidon2_mix(cells);
        }
        __syncthreads(); // Ensure all reads complete before next iteration's writes
    }

    if (leaf_idx == 0) {
#pragma unroll
        for (int i = 0; i < CELLS_OUT; i++) {
            out[stride_idx].cells[i] = cells[i];
        }
    }
}

__global__ void poseidon2_compressing_row_hashes_ext_kernel(
    digest_t *out,
    const FpExt *matrix,
    size_t width,
    size_t height,
    size_t query_stride,
    size_t log_rows_per_query
) {
    extern __shared__ char smem[]; // Fp[CELLS_OUT][blockDim.y / 2 * blockDim.x]
    Fp *shared = reinterpret_cast<Fp *>(smem);
    const uint32_t shared_stride = blockDim.x * (blockDim.y >> 1);

    const uint32_t stride_idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t leaf_idx = threadIdx.y;
    const uint32_t row = leaf_idx * query_stride + stride_idx;

    size_t used = 0;
    Fp cells[CELLS];
#pragma unroll
    for (int i = 0; i < CELLS; i++) {
        cells[i] = Fp(0);
    }

    if (stride_idx < query_stride) {
        // compute row hash
        for (int col = 0; col < width; col++) {
#pragma unroll
            // Extension field degree is 4
            for (int i = 0; i < 4; i++) {
                cells[used++] = matrix[col * height + row].elems[i];
            }
            if (used == CELLS_RATE) {
                poseidon2::poseidon2_mix(cells);
                used = 0;
            }
        }
        if (used != 0) {
            poseidon2::poseidon2_mix(cells);
        }
    }

    // This part is same as for poseidon2_compressing_row_hashes_kernel
    for (int layer = 0; layer < log_rows_per_query; ++layer) {
        uint32_t mask = (1 << (layer + 1)) - 1;
        auto shared_offset = ((leaf_idx >> (layer + 1)) << layer) * blockDim.x + threadIdx.x;
        if ((leaf_idx & mask) == (1 << layer)) {
#pragma unroll
            for (int i = 0; i < CELLS_OUT; i++) {
                shared[i * shared_stride + shared_offset] = cells[i];
            }
        }
        __syncthreads();
        if ((leaf_idx & mask) == 0) {
#pragma unroll
            for (int i = 0; i < CELLS_OUT; i++) {
                cells[CELLS_OUT + i] = shared[i * shared_stride + shared_offset];
            }
            poseidon2::poseidon2_mix(cells);
        }
        __syncthreads(); // Ensure all reads complete before next iteration's writes
    }

    if (leaf_idx == 0) {
#pragma unroll
        for (int i = 0; i < CELLS_OUT; i++) {
            out[stride_idx].cells[i] = cells[i];
        }
    }
}
static_assert(CELLS_RATE % 4 == 0, "CELLS_RATE must be multiple of FpExt degree (4)");

// Striding keeps memory coalesced with two cache lines within a warp
__global__ void poseidon2_strided_compress_layer_kernel(
    digest_t *output,
    const digest_t *prev_layer,
    size_t output_size,
    size_t stride
) {
    uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= output_size) {
        return;
    }
    uint32_t x = gid / stride;
    uint32_t y = gid % stride;

    Fp cells[CELLS];
    for (int i = 0; i < CELLS_OUT; i++) {
        cells[i] = prev_layer[2 * x * stride + y].cells[i];
        cells[i + CELLS_OUT] = prev_layer[(2 * x + 1) * stride + y].cells[i];
    }

    poseidon2::poseidon2_mix(cells);

    for (int i = 0; i < CELLS_OUT; i++) {
        output[gid].cells[i] = cells[i];
    }
}

// LAUNCHERS

extern "C" int _poseidon2_compressing_row_hashes(
    digest_t *out,
    const Fp *matrix,
    size_t width,
    size_t query_stride,
    size_t log_rows_per_query
) {
    auto [grid, block] = kernel_launch_params(query_stride, 512 >> log_rows_per_query);
    block.y = 1 << log_rows_per_query;
    size_t shared_stride = block.x * div_ceil(block.y, 2);
    size_t shmem_bytes = CELLS_OUT * shared_stride * sizeof(Fp);
    auto height = query_stride << log_rows_per_query;

    poseidon2_compressing_row_hashes_kernel<<<grid, block, shmem_bytes>>>(
        out, matrix, width, height, query_stride, log_rows_per_query
    );
    return CHECK_KERNEL();
}

extern "C" int _poseidon2_compressing_row_hashes_ext(
    digest_t *out,
    const FpExt *matrix,
    size_t width,
    size_t query_stride,
    size_t log_rows_per_query
) {
    auto [grid, block] = kernel_launch_params(query_stride, 512 >> log_rows_per_query);
    block.y = 1 << log_rows_per_query;
    size_t shared_stride = block.x * div_ceil(block.y, 2);
    size_t shmem_bytes = CELLS_OUT * shared_stride * sizeof(Fp);
    auto height = query_stride << log_rows_per_query;

    poseidon2_compressing_row_hashes_ext_kernel<<<grid, block, shmem_bytes>>>(
        out, matrix, width, height, query_stride, log_rows_per_query
    );
    return CHECK_KERNEL();
}

extern "C" int _poseidon2_strided_compress_layer(
    digest_t *output,
    const digest_t *prev_layer,
    size_t output_size,
    size_t stride
) {
    auto [grid, block] = kernel_launch_params(output_size);
    poseidon2_strided_compress_layer_kernel<<<grid, block>>>(
        output, prev_layer, output_size, stride
    );
    return CHECK_KERNEL();
}

// NOTE[jpw]: adding this function in CUDA to ensure the compiler can optimize stride = 1 (not sure this would happen across FFI boundary).
// For 32 byte digest, adjacent memory read still keeps memory coalesced in a warp
extern "C" int _poseidon2_adjacent_compress_layer(
    digest_t *output,
    const digest_t *prev_layer,
    size_t output_size
) {
    auto [grid, block] = kernel_launch_params(output_size);
    poseidon2_strided_compress_layer_kernel<<<grid, block>>>(output, prev_layer, output_size, 1);
    return CHECK_KERNEL();
}
