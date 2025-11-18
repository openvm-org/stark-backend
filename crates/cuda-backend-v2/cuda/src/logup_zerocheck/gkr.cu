#include "fp.h"
#include "fpext.h"
#include "frac_ext.cuh"
#include "launcher.cuh"
#include "sumcheck.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <utility>

// Includes GKR for fractional sumcheck
// The LogUp specific input to GKR is calculated in interactions.cu
namespace fractional_sumcheck_gkr {
// ============================================================================
// KERNELS
// ============================================================================
__global__ void frac_build_tree_layer_kernel(
    FracExt *__restrict__ out_layer,
    const FracExt *__restrict__ in_layer,
    size_t out_layer_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_layer_size) {
        return;
    }

    FracExt left = in_layer[idx << 1];
    FracExt right = in_layer[(idx << 1) + 1];
    out_layer[idx] = frac_add(left, right);
}

__global__ void compute_round_kernel(
    const FpExt *__restrict__ eq_xi,
    const FracExt *__restrict__ pq,
    size_t stride,
    FpExt lambda,
    FpExt *__restrict__ out
) {
    extern __shared__ FpExt shared[];
    const size_t half = stride >> 1;
    const FpExt zero = {0, 0, 0, 0};

    FpExt local[3] = {zero, zero, zero};
    const FpExt xs[3] = {FpExt(Fp(1u)), FpExt(Fp(2u)), FpExt(Fp(3u))};

    for (size_t idx = threadIdx.x; idx < half; idx += blockDim.x) {
        size_t even = idx << 1;
        FpExt eq_even = eq_xi[even];
        FpExt eq_odd = eq_xi[even + 1];
        FpExt eq_diff = eq_odd - eq_even;

        // \hat p_j({0 or 1}, {even or odd}, ..y)
        // The {even=0, odd=1} evaluations are used to interpolate at xs
        auto pq0_even = pq[even << 1];
        FpExt p0_even = pq0_even.p;
        FpExt q0_even = pq0_even.q;

        auto pq1_even = pq[(even << 1) + 1];
        FpExt p1_even = pq1_even.p;
        FpExt q1_even = pq1_even.q;

        size_t odd = even + 1;
        auto pq0_odd = pq[odd << 1];
        FpExt p0_odd = pq0_odd.p;
        FpExt q0_odd = pq0_odd.q;

        auto pq1_odd = pq[(odd << 1) + 1];
        FpExt p1_odd = pq1_odd.p;
        FpExt q1_odd = pq1_odd.q;

        FpExt p0_diff = p0_odd - p0_even;
        FpExt q0_diff = q0_odd - q0_even;
        FpExt p1_diff = p1_odd - p1_even;
        FpExt q1_diff = q1_odd - q1_even;

#pragma unroll
        for (int i = 0; i < 3; ++i) {
            FpExt eq_val = eq_even + xs[i] * eq_diff;
            FpExt p_j0 = p0_even + xs[i] * p0_diff;
            FpExt q_j0 = q0_even + xs[i] * q0_diff;
            FpExt p_j1 = p1_even + xs[i] * p1_diff;
            FpExt q_j1 = q1_even + xs[i] * q1_diff;

            FpExt p_prev = p_j0 * q_j1 + p_j1 * q_j0;
            FpExt q_prev = q_j0 * q_j1;
            local[i] = local[i] + eq_val * (p_prev + lambda * q_prev);
        }
    }

    for (int i = 0; i < 3; ++i) {
        FpExt reduced = sumcheck::block_reduce_sum(local[i], shared);
        if (threadIdx.x == 0) {
            out[i] = reduced;
        }
        __syncthreads();
    }
}

__global__ void fold_columns_kernel(
    const FpExt *input,
    size_t in_stride,
    size_t width,
    FpExt r,
    FpExt *output
) {
    size_t out_stride = in_stride >> 1;
    size_t total = out_stride * width;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    size_t row = idx % out_stride;
    size_t col = idx / out_stride;

    size_t in_offset = col * in_stride + (row << 1);
    size_t out_offset = col * out_stride + row;

    FpExt t0 = input[in_offset];
    FpExt t1 = input[in_offset + 1];
    output[out_offset] = t0 + r * (t1 - t0);
}

// Caution: we think of `input, output` as row-major matrices in FracExt
__global__ void fold_frac_ext_columns_kernel(
    const FracExt *__restrict__ input,
    size_t in_stride,
    size_t out_stride,
    size_t width,
    FpExt r,
    FracExt *__restrict__ output
) {
    size_t total = out_stride * width;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    size_t row = idx / width;
    size_t col = idx % width;

    size_t in_offset = (row << 1) * width + col;
    size_t out_offset = row * width + col;

    FracExt t0 = input[in_offset];
    FracExt t1 = input[in_offset + width];
    FracExt dt;
    // vector difference
    dt.p = t1.p - t0.p;
    dt.q = t1.q - t0.q;
    output[out_offset].p = t0.p + r * dt.p;
    output[out_offset].q = t0.q + r * dt.q;
}

__global__ void extract_claims_kernel(const FpExt *data, size_t stride, FpExt *out) {
    if (threadIdx.x != 0) {
        return;
    }

    out[0] = data[0];
    out[1] = data[stride];
    out[2] = data[stride * 2];
    out[3] = data[stride * 3];
}

__global__ void add_alpha_kernel(FracExt *data, size_t len, FpExt alpha) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        data[idx].q = data[idx].q + alpha;
    }
}

template <typename F, typename EF>
__global__ void frac_vector_scalar_multiply_kernel(
    std::pair<EF, EF> *frac_vec,
    F scalar,
    uint32_t length
) {
    size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= length)
        return;

    frac_vec[tidx].first *= scalar;
}

// ============================================================================
// LAUNCHERS
// ============================================================================
extern "C" int _frac_build_tree_layer(
    FracExt *out_layer,
    const FracExt *in_layer,
    size_t out_layer_size
) {
    if (out_layer_size == 0) {
        return 0;
    }

    auto [grid, block] = kernel_launch_params(out_layer_size);
    frac_build_tree_layer_kernel<<<grid, block>>>(out_layer, in_layer, out_layer_size);
    return CHECK_KERNEL();
}

extern "C" int _frac_compute_round(
    const FpExt *eq_xi,
    const FracExt *pq,
    size_t stride,
    FpExt lambda,
    FpExt *out_device
) {
    size_t half = stride >> 1;
    if (half == 0) {
        cudaError_t err = cudaMemsetAsync(out_device, 0, 3 * sizeof(FpExt), cudaStreamPerThread);
        return err == cudaSuccess ? 0 : err;
    }

    size_t threads = std::min<size_t>(std::max<size_t>(half, WARP_SIZE), 512);
    auto [grid, block] = kernel_launch_params(threads, threads);
    size_t num_warps = std::max<size_t>(1, (block.x + WARP_SIZE - 1) / WARP_SIZE);
    size_t shmem_bytes = num_warps * sizeof(FpExt);
    compute_round_kernel<<<grid, block, shmem_bytes>>>(eq_xi, pq, stride, lambda, out_device);
    return CHECK_KERNEL();
}

extern "C" int _frac_fold_columns(
    const FpExt *input,
    size_t in_stride,
    size_t width,
    FpExt r,
    FpExt *output
) {
    if (in_stride <= 1) {
        return 0;
    }
    size_t half = in_stride >> 1;
    auto [grid, block] = kernel_launch_params(half * width);
    fold_columns_kernel<<<grid, block>>>(input, in_stride, width, r, output);
    return CHECK_KERNEL();
}

extern "C" int _frac_fold_frac_ext_columns(
    const FracExt *__restrict__ input,
    size_t in_stride,
    size_t width,
    FpExt r,
    FracExt *__restrict__ output
) {
    if (in_stride <= 1) {
        return 0;
    }
    size_t half = in_stride >> 1;
    auto [grid, block] = kernel_launch_params(half * width);
    fold_frac_ext_columns_kernel<<<grid, block>>>(input, in_stride, half, width, r, output);
    return CHECK_KERNEL();
}

extern "C" int _frac_extract_claims(const FpExt *data, size_t stride, FpExt *out_device) {
    extract_claims_kernel<<<1, 32>>>(data, stride, out_device);
    return CHECK_KERNEL();
}

extern "C" int _frac_add_alpha(FracExt *data, size_t len, FpExt alpha) {
    auto [grid, block] = kernel_launch_params(len);
    add_alpha_kernel<<<grid, block>>>(data, len, alpha);
    return CHECK_KERNEL();
}

extern "C" int _frac_vector_scalar_multiply_ext_fp(FracExt *frac_vec, Fp scalar, uint32_t length) {
    auto [grid, block] = kernel_launch_params(length);
    frac_vector_scalar_multiply_kernel<Fp, FpExt>
        <<<grid, block>>>(reinterpret_cast<std::pair<FpExt, FpExt> *>(frac_vec), scalar, length);
    return CHECK_KERNEL();
}
} // namespace fractional_sumcheck_gkr
