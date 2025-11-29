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
template <bool revert>
__global__ void frac_build_tree_layer_kernel(
    FpExt *__restrict__ numerators,
    FpExt *__restrict__ denominators,
    size_t layer_size,
    size_t step
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= layer_size) {
        return;
    }

    FpExt& lhs_num = numerators[2 * idx * step];
    FpExt const& rhs_num = numerators[(2 * idx + 1) * step];
    FpExt& lhs_denom = denominators[2 * idx * step];
    FpExt const& rhs_denom = denominators[(2 * idx + 1) * step];
    if constexpr (revert) {
        frac_unadd_comps_inplace(lhs_num, lhs_denom, rhs_num, rhs_denom);
    } else {
        frac_add_comps_inplace(lhs_num, lhs_denom, rhs_num, rhs_denom);
    }
}

__global__ void compute_round_kernel(
    const FpExt *__restrict__ eq_xi,
    FpExt *__restrict__ pq_nums,
    FpExt *__restrict__ pq_denoms,
    size_t stride,
    size_t step,
    size_t pq_step,
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
        FpExt eq_even = eq_xi[even * step];
        FpExt eq_odd = eq_xi[(even + 1) * step];
        FpExt eq_diff = eq_odd - eq_even;

        // \hat p_j({0 or 1}, {even or odd}, ..y)
        // The {even=0, odd=1} evaluations are used to interpolate at xs
        uint32_t offset = even << 1;
        auto const& p0_even = pq_nums[offset * pq_step];
        auto const& q0_even = pq_denoms[offset * pq_step];
        auto const& p1_even = pq_nums[(offset + 1) * pq_step];
        auto const& q1_even = pq_denoms[(offset + 1) * pq_step];
        auto const& p0_odd = pq_nums[(offset + 2) * pq_step];
        auto const& q0_odd = pq_denoms[(offset + 2) * pq_step];
        auto const& p1_odd = pq_nums[(offset + 3) * pq_step];
        auto const& q1_odd = pq_denoms[(offset + 3) * pq_step];

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
    FpExt *buffer,
    size_t in_stride,
    size_t step,
    FpExt r
) {
    size_t out_stride = in_stride >> 1;
    size_t total = out_stride;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    FpExt& t0 = buffer[idx * 2 * step];
    FpExt const& t1 = buffer[(idx * 2 + 1) * step];
    t0 = t0 + r * (t1 - t0);
}

template <bool revert>
__global__ void fold_ef_columns_kernel(
    FpExt *__restrict__ buffer,
    size_t out_stride,
    size_t step,
    FpExt r
) {
    size_t total = out_stride;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    if constexpr (!revert) {
        auto tmp = buffer[(4 * idx + 1) * step];
        buffer[(4 * idx + 1) * step] = buffer[(4 * idx + 2) * step];
        buffer[(4 * idx + 2) * step] = tmp;
    }

#pragma unroll
    for (uint32_t i : {0, 1}) {
        FpExt& t0 = buffer[(4 * idx + 2 * i) * step];
        FpExt const& t1 = buffer[(4 * idx + 2 * i + 1) * step];
        if constexpr (revert) {
            assert(r != FpExt(Fp::one()));
            FpExt const r_inv = inv(FpExt(Fp::one()) - r);
            // z = x + r * (y - x) => x = (z - r * y) / (1 - r) = (z - y + (y - r * y)) / (1 - r)
            t0 = (t0 - t1) * r_inv + t1;
        } else {
            t0 = t0 + r * (t1 - t0);
        }
    }

    if constexpr (revert) {
        auto tmp = buffer[(4 * idx + 1) * step];
        buffer[(4 * idx + 1) * step] = buffer[(4 * idx + 2) * step];
        buffer[(4 * idx + 2) * step] = tmp;
    }
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

// Add alpha to denominators (only affects denominators buffer)
__global__ void add_alpha_mixed_kernel(FpExt *denominators, size_t len, FpExt alpha) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        denominators[idx] = denominators[idx] + alpha;
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

// Multiply numerators (base field) by scalar
__global__ void frac_vector_scalar_multiply_ext_kernel(
    FpExt *numerators,
    Fp scalar,
    uint32_t length
) {
    size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= length)
        return;

    numerators[tidx] *= scalar;
}

// ============================================================================
// LAUNCHERS
// ============================================================================
extern "C" int _frac_build_tree_layer(
    FpExt *numerators,
    FpExt *denominators,
    size_t layer_size,
    size_t step,
    bool revert
) {
    if (layer_size == 0) {
        return 0;
    }
    assert(layer_size % (2 * step) == 0);
    layer_size /= 2 * step;

    auto [grid, block] = kernel_launch_params(layer_size);
    if (revert) {
        frac_build_tree_layer_kernel<true><<<grid, block>>>(numerators, denominators, layer_size, step);
    } else {
        frac_build_tree_layer_kernel<false><<<grid, block>>>(numerators, denominators, layer_size, step);
    }
    return CHECK_KERNEL();
}

extern "C" int _frac_compute_round(
    const FpExt *eq_xi,
    FpExt *pq_nums,
    FpExt *pq_denoms,
    size_t stride,
    size_t step,
    size_t pq_extra_step,
    FpExt lambda,
    FpExt *out_device
) {
    size_t half = stride >> 1;
    if (half == 0) {
        cudaError_t err = cudaMemsetAsync(out_device, 0, 3 * sizeof(FpExt), cudaStreamPerThread);
        return err == cudaSuccess ? 0 : err;
    }

    size_t threads = std::min<size_t>(std::max<size_t>(half, WARP_SIZE), 256);
    auto [grid, block] = kernel_launch_params(threads, threads);
    size_t num_warps = std::max<size_t>(1, (block.x + WARP_SIZE - 1) / WARP_SIZE);
    size_t shmem_bytes = num_warps * sizeof(FpExt);
    compute_round_kernel<<<grid, block, shmem_bytes>>>(eq_xi, pq_nums, pq_denoms, stride, step, pq_extra_step * step, lambda, out_device);
    return CHECK_KERNEL();
}

extern "C" int _frac_fold_columns(
    FpExt *buffer,
    size_t in_stride,
    size_t step,
    FpExt r
) {
    if (in_stride <= 1) {
        return 0;
    }
    size_t half = in_stride >> 1;
    auto [grid, block] = kernel_launch_params(half);
    fold_columns_kernel<<<grid, block>>>(buffer, in_stride, step, r);
    return CHECK_KERNEL();
}

extern "C" int _frac_fold_ext_columns(
    FpExt *__restrict__ buffer,
    size_t in_stride,
    size_t step,
    FpExt r,
    bool revert
) {
    if (in_stride <= 1) {
        return 0;
    }
    size_t half = in_stride >> 1;
    auto [grid, block] = kernel_launch_params(half * 2);
    if (revert) {
        fold_ef_columns_kernel<true><<<grid, block>>>(buffer, half, step, r);
    } else {
        fold_ef_columns_kernel<false><<<grid, block>>>(buffer, half, step, r);
    }
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

extern "C" int _frac_add_alpha_mixed(FpExt *denominators, size_t len, FpExt alpha) {
    auto [grid, block] = kernel_launch_params(len);
    add_alpha_mixed_kernel<<<grid, block>>>(denominators, len, alpha);
    return CHECK_KERNEL();
}

extern "C" int _frac_vector_scalar_multiply_ext(FpExt *numerators, Fp scalar, uint32_t length) {
    auto [grid, block] = kernel_launch_params(length);
    frac_vector_scalar_multiply_ext_kernel<<<grid, block>>>(numerators, scalar, length);
    return CHECK_KERNEL();
}
} // namespace fractional_sumcheck_gkr
