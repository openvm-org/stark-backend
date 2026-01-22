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
__global__ void frac_build_tree_layer_kernel(FracExt *__restrict__ layer, uint32_t half) {
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

__device__ __forceinline__ FpExt sqrt_buffer_get(
    const FpExt *__restrict__ eq_xi_low,
    const FpExt *__restrict__ eq_xi_high,
    uint32_t const log_eq_size,
    uint32_t const log_eq_low_cap,
    uint32_t const idx
) {
    return eq_xi_low[idx & ((1u << log_eq_low_cap) - 1)] * eq_xi_high[idx >> log_eq_low_cap];
}

// shared memory size requirement: max(num_warps,1) * sizeof(FpExt)
__global__ void compute_round_block_sum_kernel(
    const FpExt *__restrict__ eq_xi_low,
    const FpExt *__restrict__ eq_xi_high,
    const FracExt *__restrict__ pq_buffer,
    uint32_t log_eq_size,
    uint32_t log_eq_low_cap,
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
        FpExt eq_even = sqrt_buffer_get(eq_xi_low, eq_xi_high, log_eq_size, log_eq_low_cap, idx);
        FpExt eq_odd = sqrt_buffer_get(
            eq_xi_low, eq_xi_high, log_eq_size, log_eq_low_cap, with_rev_bits(idx, eq_size, 1)
        );
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

// Fold kernel operating on FpExt* view of FracExt buffer.
// Since FracExt = {p, q} has p and q folded independently with the same formula,
// we can treat the buffer as FpExt* with 2x the elements.
//
// Pairs (idx, idx+quarter) and (idx+half, idx+3*quarter),
// writes results to dst[idx] and dst[idx+quarter].
// Safe for src == dst (in-place) because each thread reads before writing to the same index.
__global__ void fold_ef_columns_kernel(const FpExt *src, FpExt *dst, uint32_t quarter, FpExt r) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= quarter) {
        return;
    }

    FpExt v0 = src[idx];
    FpExt v1 = src[idx | quarter];
    dst[idx] = v0 + r * (v1 - v0);

    uint32_t half = quarter << 1;

    FpExt v2 = src[idx | half];
    FpExt v3 = src[idx | half | quarter];
    dst[idx | quarter] = v2 + r * (v3 - v2);
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
extern "C" int _frac_build_tree_layer(FracExt *layer, size_t layer_size, bool revert) {
    if (layer_size == 0) {
        return 0;
    }
    assert(layer_size % 2 == 0);
    layer_size /= 2;

    auto [grid, block] = kernel_launch_params(layer_size);
    if (revert) {
        frac_build_tree_layer_kernel<true><<<grid, block>>>(layer, layer_size);
    } else {
        frac_build_tree_layer_kernel<false><<<grid, block>>>(layer, layer_size);
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
    const FpExt *eq_xi_low,
    const FpExt *eq_xi_high,
    const FracExt *pq_buffer,
    size_t eq_size,
    size_t eq_low_cap,
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
        eq_xi_low,
        eq_xi_high,
        pq_buffer,
        __builtin_ctz((uint32_t)eq_size),
        __builtin_ctz((uint32_t)eq_low_cap),
        __builtin_ctz((uint32_t)pq_size),
        lambda,
        tmp_block_sums
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

extern "C" int _frac_fold_fpext_columns(const FracExt *src, FracExt *dst, size_t size, FpExt r) {
    if (size <= 2) {
        return 0;
    }
    // FracExt = {p, q} = 2 FpExt elements.
    // size is in FracExt; quarter_fpext is in FpExt.
    // quarter_fpext = (size / 4) * 2 = size / 2
    uint32_t quarter_fpext = size >> 1;
    auto [grid, block] = kernel_launch_params(quarter_fpext);
    fold_ef_columns_kernel<<<grid, block>>>(
        reinterpret_cast<const FpExt *>(src), reinterpret_cast<FpExt *>(dst), quarter_fpext, r
    );
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
