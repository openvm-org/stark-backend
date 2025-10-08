/*
 * Source: https://github.com/scroll-tech/plonky3-gpu (private repo)
 * Status: BASED ON plonky3-gpu/gpu-backend/src/cuda/kernels/lde.cu
 * Imported: 2025-01-25 by @gaxiom
 */

#include "fp.h"
#include "launcher.cuh"

__global__ void multi_bit_reverse_kernel(Fp *io, const uint32_t nBits, const uint32_t count) {
    uint totIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (totIdx < count) {
        uint32_t rowSize = 1 << nBits;
        uint32_t idx = totIdx & (rowSize - 1);
        uint32_t s = totIdx >> nBits;
        uint32_t ridx = __brev(idx) >> (32 - nBits);
        if (idx < ridx) {
            size_t idx1 = s * rowSize + idx;
            size_t idx2 = s * rowSize + ridx;
            Fp tmp = io[idx1];
            io[idx1] = io[idx2];
            io[idx2] = tmp;
        }
    }
}

// io[j][i] *= shift^i
__global__ void zk_shift_kernel(
    Fp *io,
    const uint32_t io_size,
    const uint32_t log_n,
    const uint32_t shift
) {
    Fp base = Fp(shift);
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint n = 1 << log_n;
    uint polyCount = io_size >> log_n;
    if (idx < n) {
        // the degree of each polynomial is 2^log_n - 1
        uint32_t pos = idx & ((1 << log_n) - 1);
        Fp power = pow(base, pos);
        for (uint i = 0; i < polyCount; i++) {
            io[i * n + idx] *= power;
        }
    }
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

// LAUNCHERS

extern "C" int _multi_bit_reverse(Fp *io, const uint32_t nBits, const uint32_t count) {
    auto [grid, block] = kernel_launch_params(count);
    multi_bit_reverse_kernel<<<grid, block>>>(io, nBits, count);
    return CHECK_KERNEL();
}

extern "C" int _zk_shift(
    Fp *io,
    const uint32_t io_size,
    const uint32_t log_n,
    const uint32_t shift
) {
    const uint32_t n = 1 << log_n;
    auto [grid, block] = kernel_launch_params(n);
    zk_shift_kernel<<<grid, block>>>(io, io_size, log_n, shift);
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