#include "fp.h"
#include "fpext.h"
#include "frac_ext.cuh"
#include "launcher.cuh"
#include "sumcheck.cuh"
#include "utils.cuh"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <utility>
#include <vector_types.h>

// Includes GKR for fractional sumcheck
// The LogUp specific input to GKR is calculated in interactions.cu
namespace fractional_sumcheck_gkr {
constexpr int GKR_S_DEG = 3;
// ============================================================================
// KERNELS
// ============================================================================
template <bool revert>
__global__ void frac_build_tree_layer_kernel(
    FracExt *__restrict__ layer,
    uint32_t half
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half) {
        return;
    }

    FracExt &lhs = layer[idx];
    FracExt const &rhs = layer[idx | half];
    if constexpr (revert) {
        frac_unadd_inplace(lhs, rhs);
    } else {
        frac_add_inplace(lhs, rhs);
    }
}

// shared memory size requirement: max(num_warps,1) * sizeof(FpExt)
__global__ void compute_round_block_sum_kernel(
    const FpExt *__restrict__ eq_xi,
    FracExt *__restrict__ pq_buffer,
    uint32_t log_eq_size,
    uint32_t log_pq_size,
    FpExt lambda,
    FpExt *__restrict__ block_sums // Output: [gridDim.x][3]
) {
    extern __shared__ FpExt shared[];
    const uint32_t eq_size = 1 << log_eq_size;
    const uint32_t pq_size = 1 << log_pq_size;

    const FpExt zero(Fp::zero());
    FpExt local[GKR_S_DEG] = {zero, zero, zero};
    uint32_t idx_base = threadIdx.x + blockIdx.x * blockDim.x;
    // Map phase: compute local sum by striding over grid
    for (uint32_t idx = idx_base; idx < eq_size / 2; idx += blockDim.x * gridDim.x) {
        FpExt eq_even = eq_xi[idx];
        FpExt eq_odd = eq_xi[with_rev_bits(idx, eq_size, 1)];
        FpExt eq_diff = eq_odd - eq_even;

        // \hat p_j({0 or 1}, {even or odd}, ..y)
        // The {even=0, odd=1} evaluations are used to interpolate at 1, 2, 3
        auto const &[p0_even, q0_even] = pq_buffer[idx];
        auto const &[p1_even, q1_even] = pq_buffer[with_rev_bits(idx, pq_size, 1, 0)];
        auto const &[p0_odd, q0_odd] = pq_buffer[with_rev_bits(idx, pq_size, 0, 1)];
        auto const &[p1_odd, q1_odd] = pq_buffer[with_rev_bits(idx, pq_size, 1, 1)];

        FpExt p0_diff = p0_odd - p0_even;
        FpExt q0_diff = q0_odd - q0_even;
        FpExt p1_diff = p1_odd - p1_even;
        FpExt q1_diff = q1_odd - q1_even;

        FpExt eq_val = eq_even;
        FpExt p_j0 = p0_even + lambda * q0_even;
        FpExt q_j0 = q0_even;
        FpExt p_j1 = p1_even;
        FpExt q_j1 = q1_even;

        auto const lambda_times_q0_diff = lambda * q0_diff;

#pragma unroll
        for (int i = 0; i < GKR_S_DEG; ++i) {
            eq_val += eq_diff;
            p_j0 += p0_diff;
            p_j0 += lambda_times_q0_diff;
            q_j0 += q0_diff;
            p_j1 += p1_diff;
            q_j1 += q1_diff;

            // FpExt p_prev = p_j0 * q_j1 + p_j1 * q_j0;
            // FpExt q_prev = q_j0 * q_j1;
            // local[i] += eq_val * (p_prev + lambda * q_prev);
            local[i] += eq_val * (p_j0 * q_j1 + p_j1 * q_j0);
        }
    }
// Reduce phase: reduce all threadIdx.x in the same block, keeping 1,2,3 independent
#pragma unroll
    for (int i = 0; i < GKR_S_DEG; ++i) {
        FpExt reduced = sumcheck::block_reduce_sum(local[i], shared);

        if (threadIdx.x == 0) {
            block_sums[blockIdx.x * 3 + i] = reduced;
        }
        __syncthreads();
    }
}

__global__ void fold_columns_kernel(FpExt *buffer, size_t half, FpExt r) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half) {
        return;
    }

    FpExt &t0 = buffer[idx];
    FpExt const &t1 = buffer[idx | half];
    t0 += r * (t1 - t0);
}

template <bool revert>
__device__ __forceinline__ void one_folding_round(FpExt& lhs, FpExt const& rhs, FpExt const& r_or_r_inv) {
    if constexpr (revert) {
        // z = x + r * (y - x) => x = (z - r * y) / (1 - r) = (z - y + (y - r * y)) / (1 - r)
        lhs = (lhs - rhs) * r_or_r_inv + rhs;
    } else {
        lhs = lhs + r_or_r_inv * (rhs - lhs);
    }
}

template <bool revert>
__global__ void fold_ef_columns_kernel(
    FracExt *__restrict__ buffer,
    uint32_t log_total,
    FpExt r_or_r_inv // is the inverse of 1 - r in case if revert = true
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (1 << (log_total - 2))) {
        return;
    }

    if constexpr (!revert) {
        auto tmp = buffer[idx | (1u << (log_total - 1))];
        buffer[idx | (1u << (log_total - 1))] = buffer[idx | (1u << (log_total - 2))];
        buffer[idx | (1u << (log_total - 2))] = tmp;
    }

#pragma unroll
    for (uint32_t bit : {0u, 1u << (log_total - 2)}) {
        {
            FpExt &t0 = buffer[idx | bit].p;
            FpExt const &t1 = buffer[idx | bit | (1u << (log_total - 1))].p;
            one_folding_round<revert>(t0, t1, r_or_r_inv);
        }
        {
            FpExt &t0 = buffer[idx | bit].q;
            FpExt const &t1 = buffer[idx | bit | (1u << (log_total - 1))].q;
            one_folding_round<revert>(t0, t1, r_or_r_inv);
        }
    }

    if constexpr (revert) {
        auto tmp = buffer[idx | (1u << (log_total - 1))];
        buffer[idx | (1u << (log_total - 1))] = buffer[idx | (1u << (log_total - 2))];
        buffer[idx | (1u << (log_total - 2))] = tmp;
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
    FracExt *layer,
    size_t layer_size,
    bool revert
) {
    if (layer_size == 0) {
        return 0;
    }
    assert(layer_size % 2 == 0);
    layer_size /= 2;

    auto [grid, block] = kernel_launch_params(layer_size);
    if (revert) {
        frac_build_tree_layer_kernel<true>
            <<<grid, block>>>(layer, layer_size);
    } else {
        frac_build_tree_layer_kernel<false>
            <<<grid, block>>>(layer, layer_size);
    }
    return CHECK_KERNEL();
}

inline std::pair<dim3, dim3> frac_compute_round_launch_params(uint32_t stride) {
    return kernel_launch_params(stride >> 1, 256);
}

extern "C" uint32_t _frac_compute_round_temp_buffer_size(uint32_t stride) {
    auto [grid, block] = frac_compute_round_launch_params(stride);
    return grid.x * GKR_S_DEG;
}

extern "C" int _frac_compute_round(
    const FpExt *eq_xi,
    FracExt *pq_buffer,
    size_t eq_size,
    size_t pq_size,
    FpExt lambda,
    FpExt *out,           // Output: [d=3] final results
    FpExt *tmp_block_sums // Temporary buffer: [gridDim.x * d]
) {
    assert(eq_size > 1);
    assert(pq_size == eq_size * 2);

    auto [grid, block] = frac_compute_round_launch_params(eq_size);
    uint32_t num_warps = (block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t shmem_bytes = std::max(1u, num_warps) * sizeof(FpExt);

    // Launch main kernel - writes to tmp_block_sums
    compute_round_block_sum_kernel<<<grid, block, shmem_bytes>>>(
        eq_xi, pq_buffer, __builtin_ctz((uint32_t)eq_size), __builtin_ctz((uint32_t)pq_size), lambda, tmp_block_sums
    );
    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Launch final reduction kernel - reads from tmp_block_sums, writes to output
    auto num_blocks = grid.x;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    sumcheck::static_final_reduce_block_sums<GKR_S_DEG>
        <<<GKR_S_DEG, reduce_block, reduce_shmem>>>(tmp_block_sums, out, num_blocks);

    return CHECK_KERNEL();
}

extern "C" int _frac_fold_columns(FpExt *buffer, size_t size, FpExt r) {
    if (size <= 1) {
        return 0;
    }
    size_t half = size >> 1;
    auto [grid, block] = kernel_launch_params(half);
    fold_columns_kernel<<<grid, block>>>(buffer, half, r);
    return CHECK_KERNEL();
}

extern "C" int _frac_fold_fpext_columns(
    FracExt *__restrict__ buffer,
    size_t size,
    FpExt r_or_r_inv,
    bool revert
) {
    if (size <= 2) {
        return 0;
    }
    uint32_t quarter = size >> 2;
    auto [grid, block] = kernel_launch_params(quarter);
    if (revert) {
        fold_ef_columns_kernel<true>
            <<<grid, block>>>(buffer, __builtin_ctz((uint32_t)size), r_or_r_inv);
    } else {
        fold_ef_columns_kernel<false>
            <<<grid, block>>>(buffer, __builtin_ctz((uint32_t)size), r_or_r_inv);
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
} // namespace fractional_sumcheck_gkr
