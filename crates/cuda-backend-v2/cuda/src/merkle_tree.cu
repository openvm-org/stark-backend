#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "poseidon2.cuh"
#include <cstddef>
#include <cstdint>

struct digest_t {
    Fp cells[CELLS_OUT];
};

__global__ void poseidon2_row_hashes_kernel(
    digest_t *out,
    const Fp *matrix,
    size_t width,
    size_t height
) {
    uint32_t row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= height) {
        return;
    }

    size_t used = 0;
    Fp cells[CELLS];
    for (int i = 0; i < CELLS; i++) {
        cells[i] = Fp(0);
    }

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

    for (int i = 0; i < CELLS_OUT; i++) {
        out[row].cells[i] = cells[i];
    }
}

__global__ void poseidon2_row_hashes_ext_kernel(
    digest_t *out,
    const FpExt *matrix,
    size_t width,
    size_t height
) {
    uint32_t row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= height) {
        return;
    }

    size_t used = 0;
    Fp cells[CELLS];
    for (int i = 0; i < CELLS; i++) {
        cells[i] = Fp(0);
    }

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

    for (int i = 0; i < CELLS_OUT; i++) {
        out[row].cells[i] = cells[i];
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

extern "C" int _poseidon2_row_hashes(digest_t *out, const Fp *matrix, size_t width, size_t height) {
    auto [grid, block] = kernel_launch_params(height);
    poseidon2_row_hashes_kernel<<<grid, block>>>(out, matrix, width, height);
    return CHECK_KERNEL();
}

extern "C" int _poseidon2_row_hashes_ext(
    digest_t *out,
    const FpExt *matrix,
    size_t width,
    size_t height
) {
    auto [grid, block] = kernel_launch_params(height);
    poseidon2_row_hashes_ext_kernel<<<grid, block>>>(out, matrix, width, height);
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
