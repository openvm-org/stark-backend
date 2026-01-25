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

// Helper: Accumulate GKR sumcheck contributions for interpolation at 1, 2, 3.
// Takes the 8 p/q values (even/odd for p0, q0, p1, q1) and accumulates into local accumulators.
__device__ __forceinline__ void accumulate_compute_contributions(
    const FpExt *__restrict__ eq_xi_low,
    const FpExt *__restrict__ eq_xi_high,
    uint32_t idx,
    uint32_t eq_size,
    uint32_t log_eq_size,
    uint32_t log_eq_low_cap,
    FpExt lambda,
    FpExt p0_even, FpExt q0_even, FpExt p0_odd, FpExt q0_odd,
    FpExt p1_even, FpExt q1_even, FpExt p1_odd, FpExt q1_odd,
    FpExt &local0, FpExt &local1, FpExt &local2
) {
    FpExt eq_even = sqrt_buffer_get(eq_xi_low, eq_xi_high, log_eq_size, log_eq_low_cap, idx);
    FpExt eq_odd = sqrt_buffer_get(
        eq_xi_low, eq_xi_high, log_eq_size, log_eq_low_cap, with_rev_bits(idx, eq_size, 1)
    );
    FpExt eq_diff = eq_odd - eq_even;

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

        FpExt contrib = eq_val * (p_j0 * q_j1 + p_j1 * q_j0);
        if (i == 0) local0 += contrib;
        else if (i == 1) local1 += contrib;
        else local2 += contrib;
    }
}

// Helper: Perform block reduction on three accumulators and write to block_sums.
__device__ __forceinline__ void reduce_block_sums(
    FpExt *shared,
    FpExt local0, FpExt local1, FpExt local2,
    FpExt *__restrict__ block_sums
) {
    {
        FpExt reduced = sumcheck::block_reduce_sum(local0, shared);
        if (threadIdx.x == 0) block_sums[blockIdx.x * 3 + 0] = reduced;
    }
    __syncthreads();
    {
        FpExt reduced = sumcheck::block_reduce_sum(local1, shared);
        if (threadIdx.x == 0) block_sums[blockIdx.x * 3 + 1] = reduced;
    }
    __syncthreads();
    {
        FpExt reduced = sumcheck::block_reduce_sum(local2, shared);
        if (threadIdx.x == 0) block_sums[blockIdx.x * 3 + 2] = reduced;
    }
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

// Fused compute round + fold kernel (OUT-OF-PLACE version).
// Reads from pre-fold src_pq buffer (size 2*pq_size), computes sumcheck sums,
// and writes folded output to dst_pq buffer (size pq_size).
//
// This kernel fuses the fold operation into the next round's compute, eliminating
// one kernel launch per inner round and reducing memory traffic.
//
// IMPORTANT: src_pq and dst_pq must NOT alias. Use compute_round_and_fold_inplace_kernel
// when src == dst.
//
// Register-optimized: uses tight scopes for pair loads and writes folded values
// immediately to reduce live register ranges.
//
// shared memory size requirement: max(num_warps,1) * sizeof(FpExt)
__global__ void compute_round_and_fold_kernel(
    const FpExt *__restrict__ eq_xi_low,
    const FpExt *__restrict__ eq_xi_high,
    const FracExt *__restrict__ src_pq,  // Pre-fold buffer (2*pq_size FracExt)
    uint32_t log_eq_size,                // log2(post-fold eq_size)
    uint32_t log_eq_low_cap,
    uint32_t log_pq_size,                // log2(post-fold pq_size)
    FpExt lambda,
    FpExt r_prev,                        // Previous round's challenge for folding
    FpExt *__restrict__ block_sums,      // Output: [gridDim.x * 3]
    FracExt *__restrict__ dst_pq         // Post-fold buffer (pq_size FracExt)
) {
    extern __shared__ FpExt shared[];
    const uint32_t eq_size = 1u << log_eq_size;
    const uint32_t pq_size = 1u << log_pq_size;

    // Index offsets for fold pattern (see detailed comments in inplace kernel)
    const uint32_t half = pq_size;              // src_pq_size / 2
    const uint32_t quarter = pq_size >> 1;      // pq_size / 2
    const uint32_t eighth = pq_size >> 2;       // pq_size / 4
    const uint32_t three_eighths = eighth * 3;  // 3 * pq_size / 4

    // Use scalar accumulators instead of array to help register allocation
    const FpExt zero(Fp::zero());
    FpExt local0 = zero, local1 = zero, local2 = zero;
    uint32_t idx_base = threadIdx.x + blockIdx.x * blockDim.x;

    // Map phase: compute local sum by striding over grid
    for (uint32_t idx = idx_base; idx < eq_size / 2; idx += blockDim.x * gridDim.x) {
        // Scalars for folded values used in compute
        FpExt p0_even, q0_even, p1_even, q1_even;
        FpExt p0_odd, q0_odd, p1_odd, q1_odd;

        // Load pairs in tight scopes, fold, write immediately to reduce register pressure
        // f00 at post-fold idx
        {
            FracExt a = src_pq[idx];
            FracExt b = src_pq[idx + quarter];
            p0_even = a.p + r_prev * (b.p - a.p);
            q0_even = a.q + r_prev * (b.q - a.q);
            dst_pq[idx] = {p0_even, q0_even};
        }
        // f10 at post-fold idx + quarter
        {
            FracExt a = src_pq[idx + half];
            FracExt b = src_pq[idx + half + quarter];
            p1_even = a.p + r_prev * (b.p - a.p);
            q1_even = a.q + r_prev * (b.q - a.q);
            dst_pq[idx + quarter] = {p1_even, q1_even};
        }
        // f01 at post-fold idx + eighth
        {
            FracExt a = src_pq[idx + eighth];
            FracExt b = src_pq[idx + three_eighths];
            p0_odd = a.p + r_prev * (b.p - a.p);
            q0_odd = a.q + r_prev * (b.q - a.q);
            dst_pq[idx + eighth] = {p0_odd, q0_odd};
        }
        // f11 at post-fold idx + three_eighths
        {
            FracExt a = src_pq[idx + half + eighth];
            FracExt b = src_pq[idx + half + three_eighths];
            p1_odd = a.p + r_prev * (b.p - a.p);
            q1_odd = a.q + r_prev * (b.q - a.q);
            dst_pq[idx + three_eighths] = {p1_odd, q1_odd};
        }

        accumulate_compute_contributions(
            eq_xi_low, eq_xi_high, idx, eq_size, log_eq_size, log_eq_low_cap, lambda,
            p0_even, q0_even, p0_odd, q0_odd,
            p1_even, q1_even, p1_odd, q1_odd,
            local0, local1, local2
        );
    }

    reduce_block_sums(shared, local0, local1, local2, block_sums);
}

// Fused compute round + fold kernel (IN-PLACE version).
// Reads from pre-fold pq buffer (size 2*pq_size), computes sumcheck sums,
// and writes folded output back to the same buffer (first pq_size elements).
//
// IN-PLACE SAFETY: Each thread writes only to indices it reads from in the first
// half of the buffer {idx, idx+quarter, idx+eighth, idx+three_eighths}. No cross-thread
// conflicts because different threads have disjoint idx values. Reads from the second
// half [pq_size, 2*pq_size) are never written to.
//
// NOTE: No __restrict__ on pq pointer since we read and write to the same buffer.
//
// shared memory size requirement: max(num_warps,1) * sizeof(FpExt)
__global__ void compute_round_and_fold_inplace_kernel(
    const FpExt *__restrict__ eq_xi_low,
    const FpExt *__restrict__ eq_xi_high,
    FracExt *pq,                         // In-place buffer: reads 2*pq_size, writes pq_size
    uint32_t log_eq_size,                // log2(post-fold eq_size)
    uint32_t log_eq_low_cap,
    uint32_t log_pq_size,                // log2(post-fold pq_size)
    FpExt lambda,
    FpExt r_prev,                        // Previous round's challenge for folding
    FpExt *__restrict__ block_sums       // Output: [gridDim.x * 3]
) {
    extern __shared__ FpExt shared[];
    const uint32_t eq_size = 1u << log_eq_size;
    const uint32_t pq_size = 1u << log_pq_size;

    // Fold pattern analysis:
    // The fold kernel operates on FpExt* view where FracExt[i] = (FpExt[2i], FpExt[2i+1]).
    // For pre-fold size N = 2*pq_size FracExt = 4*pq_size FpExt, the fold kernel uses:
    //   quarter = 2*pq_size / 2 = pq_size (in FpExt)
    //   half = 2 * quarter = 2*pq_size (in FpExt)
    //
    // The fold produces output FracExt with this mapping:
    //   For j in [0, pq_size/2): output[j] = fold(input[j], input[j + pq_size/2])
    //   For j in [pq_size/2, pq_size): output[j] = fold(input[j + pq_size/2], input[j + pq_size])
    //
    // Post-fold indices for compute: idx, idx+quarter, idx+eighth, idx+three_eighths
    // where quarter = pq_size/2, eighth = pq_size/4
    const uint32_t half = pq_size;              // src_pq_size / 2
    const uint32_t quarter = pq_size >> 1;      // pq_size / 2
    const uint32_t eighth = pq_size >> 2;       // pq_size / 4
    const uint32_t three_eighths = eighth * 3;  // 3 * pq_size / 4

    // Use scalar accumulators instead of array to help register allocation
    const FpExt zero(Fp::zero());
    FpExt local0 = zero, local1 = zero, local2 = zero;
    uint32_t idx_base = threadIdx.x + blockIdx.x * blockDim.x;

    // Map phase: compute local sum by striding over grid
    for (uint32_t idx = idx_base; idx < eq_size / 2; idx += blockDim.x * gridDim.x) {
        // Scalars for folded values used in compute
        FpExt p0_even, q0_even, p1_even, q1_even;
        FpExt p0_odd, q0_odd, p1_odd, q1_odd;

        // Load pairs in tight scopes, fold, write immediately to reduce register pressure.
        // For in-place: we must complete ALL reads before ANY writes to avoid data hazards.
        // So we load all 8 source values first, then write all 4 folded values.

        // Load all source pairs first
        FracExt s00_a, s00_b, s01_a, s01_b, s10_a, s10_b, s11_a, s11_b;
        s00_a = pq[idx];
        s00_b = pq[idx + quarter];
        s01_a = pq[idx + eighth];
        s01_b = pq[idx + three_eighths];
        s10_a = pq[idx + half];
        s10_b = pq[idx + half + quarter];
        s11_a = pq[idx + half + eighth];
        s11_b = pq[idx + half + three_eighths];

        // Compute folded values
        p0_even = s00_a.p + r_prev * (s00_b.p - s00_a.p);
        q0_even = s00_a.q + r_prev * (s00_b.q - s00_a.q);
        p1_even = s10_a.p + r_prev * (s10_b.p - s10_a.p);
        q1_even = s10_a.q + r_prev * (s10_b.q - s10_a.q);
        p0_odd = s01_a.p + r_prev * (s01_b.p - s01_a.p);
        q0_odd = s01_a.q + r_prev * (s01_b.q - s01_a.q);
        p1_odd = s11_a.p + r_prev * (s11_b.p - s11_a.p);
        q1_odd = s11_a.q + r_prev * (s11_b.q - s11_a.q);

        // Write all folded values (safe: writes are to first half, reads were from both halves)
        pq[idx] = {p0_even, q0_even};
        pq[idx + quarter] = {p1_even, q1_even};
        pq[idx + eighth] = {p0_odd, q0_odd};
        pq[idx + three_eighths] = {p1_odd, q1_odd};

        accumulate_compute_contributions(
            eq_xi_low, eq_xi_high, idx, eq_size, log_eq_size, log_eq_low_cap, lambda,
            p0_even, q0_even, p0_odd, q0_odd,
            p1_even, q1_even, p1_odd, q1_odd,
            local0, local1, local2
        );
    }

    reduce_block_sums(shared, local0, local1, local2, block_sums);
}

// Fused compute round + tree layer revert kernel.
// Combines frac_build_tree_layer_kernel<true> (revert) with compute_round_block_sum_kernel.
//
// This is used for the FIRST inner round only, where we need to revert the tree layer
// before computing. The revert operation is: layer[i] = layer[i] - layer[i + half] for i < half.
//
// For round 0, the 4 PQ indices accessed per thread are:
//   - idx: in first half, needs revert
//   - idx | quarter: in first half, needs revert
//   - idx | half: in second half, no revert
//   - idx | half | quarter: in second half, no revert
//
// Each index in the first half is written by exactly one thread:
//   - Thread idx writes to indices {idx, idx + quarter}
//   - For idx in [0, eq_size/2), these cover [0, half) completely
//
// shared memory size requirement: max(num_warps,1) * sizeof(FpExt)
__global__ void compute_round_and_revert_kernel(
    const FpExt *__restrict__ eq_xi_low,
    const FpExt *__restrict__ eq_xi_high,
    FracExt *__restrict__ layer,         // Tree layer buffer (modified in-place for revert)
    uint32_t log_eq_size,
    uint32_t log_eq_low_cap,
    uint32_t log_pq_size,                // log2(pq_size), where pq_size = eq_size * 2
    FpExt lambda,
    FpExt *__restrict__ block_sums       // Output: [gridDim.x * 3]
) {
    extern __shared__ FpExt shared[];
    const uint32_t eq_size = 1u << log_eq_size;
    const uint32_t pq_size = 1u << log_pq_size;
    const uint32_t half = pq_size >> 1;     // pq_size / 2
    const uint32_t quarter = pq_size >> 2;  // pq_size / 4

    // Use scalar accumulators instead of array to help register allocation
    const FpExt zero(Fp::zero());
    FpExt local0 = zero, local1 = zero, local2 = zero;
    uint32_t idx_base = threadIdx.x + blockIdx.x * blockDim.x;

    // Map phase: compute local sum by striding over grid
    for (uint32_t idx = idx_base; idx < eq_size / 2; idx += blockDim.x * gridDim.x) {
        // Load second-half values once (these are p1 values AND rhs for frac_unadd)
        // This eliminates redundant loads compared to calling frac_unadd then loading again
        FracExt p1_even_frac = layer[idx + half];
        FracExt p1_odd_frac = layer[idx + half + quarter];

        // Compute reverted values for first half using inline frac_unadd logic
        // frac_unadd: rhs_q_inv = inv(rhs.q); lhs.q *= rhs_q_inv; lhs.p = (lhs.p - lhs.q * rhs.p) * rhs_q_inv
        FracExt p0_even_frac, p0_odd_frac;
        {
            FracExt lhs = layer[idx];
            FpExt rhs_q_inv = inv(p1_even_frac.q);
            p0_even_frac.q = lhs.q * rhs_q_inv;
            p0_even_frac.p = (lhs.p - p0_even_frac.q * p1_even_frac.p) * rhs_q_inv;
            layer[idx] = p0_even_frac;
        }
        {
            FracExt lhs = layer[idx + quarter];
            FpExt rhs_q_inv = inv(p1_odd_frac.q);
            p0_odd_frac.q = lhs.q * rhs_q_inv;
            p0_odd_frac.p = (lhs.p - p0_odd_frac.q * p1_odd_frac.p) * rhs_q_inv;
            layer[idx + quarter] = p0_odd_frac;
        }

        // Extract p and q components
        auto const &[p0_even, q0_even] = p0_even_frac;
        auto const &[p1_even, q1_even] = p1_even_frac;
        auto const &[p0_odd, q0_odd] = p0_odd_frac;
        auto const &[p1_odd, q1_odd] = p1_odd_frac;

        accumulate_compute_contributions(
            eq_xi_low, eq_xi_high, idx, eq_size, log_eq_size, log_eq_low_cap, lambda,
            p0_even, q0_even, p0_odd, q0_odd,
            p1_even, q1_even, p1_odd, q1_odd,
            local0, local1, local2
        );
    }

    reduce_block_sums(shared, local0, local1, local2, block_sums);
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

// Fused compute round + tree layer revert launcher.
// Combines frac_build_tree_layer(revert=true) with compute_round for the first inner round.
extern "C" int _frac_compute_round_and_revert(
    const FpExt *eq_xi_low,
    const FpExt *eq_xi_high,
    FracExt *layer,           // Tree layer buffer (modified in-place for revert)
    size_t eq_size,
    size_t eq_low_cap,
    size_t pq_size,
    FpExt lambda,
    FpExt *out,               // Output: [d=3] final results
    FpExt *tmp_block_sums     // Temporary buffer: [gridDim.x * d]
) {
    assert(eq_size > 1);
    assert(pq_size == eq_size * 2);

    auto [grid, block] = frac_compute_round_launch_params(eq_size);
    uint32_t num_warps = (block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t shmem_bytes = std::max(1u, num_warps) * sizeof(FpExt);

    // Launch fused revert + compute kernel
    compute_round_and_revert_kernel<<<grid, block, shmem_bytes>>>(
        eq_xi_low,
        eq_xi_high,
        layer,
        __builtin_ctz((uint32_t)eq_size),
        __builtin_ctz((uint32_t)eq_low_cap),
        __builtin_ctz((uint32_t)pq_size),
        lambda,
        tmp_block_sums
    );
    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Launch final reduction kernel
    auto num_blocks = grid.x;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    sumcheck::static_final_reduce_block_sums<GKR_S_DEG>
        <<<GKR_S_DEG, reduce_block, reduce_shmem>>>(tmp_block_sums, out, num_blocks);

    return CHECK_KERNEL();
}

// Fused compute round + fold launcher.
// src_pq_size is the pre-fold buffer size (2*pq_size).
// The post-fold eq_size = src_pq_size / 2, pq_size = src_pq_size / 2.
extern "C" int _frac_compute_round_and_fold(
    const FpExt *eq_xi_low,
    const FpExt *eq_xi_high,
    const FracExt *src_pq_buffer,
    FracExt *dst_pq_buffer,
    size_t src_pq_size,           // Pre-fold size in FracExt
    size_t eq_low_cap,
    FpExt lambda,
    FpExt r_prev,
    FpExt *out,                   // Output: [d=3] final results
    FpExt *tmp_block_sums         // Temporary buffer: [gridDim.x * d]
) {
    assert(src_pq_size > 2);
    // Post-fold sizes
    size_t pq_size = src_pq_size >> 1;
    size_t eq_size = pq_size >> 1;
    assert(eq_size > 0);

    auto [grid, block] = frac_compute_round_launch_params(eq_size);
    uint32_t num_warps = (block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t shmem_bytes = std::max(1u, num_warps) * sizeof(FpExt);

    // Launch fused kernel - writes to tmp_block_sums and dst_pq_buffer
    compute_round_and_fold_kernel<<<grid, block, shmem_bytes>>>(
        eq_xi_low,
        eq_xi_high,
        src_pq_buffer,
        __builtin_ctz((uint32_t)eq_size),
        __builtin_ctz((uint32_t)eq_low_cap),
        __builtin_ctz((uint32_t)pq_size),
        lambda,
        r_prev,
        tmp_block_sums,
        dst_pq_buffer
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

// Fused compute round + fold launcher (IN-PLACE version).
// src_pq_size is the pre-fold buffer size (2*pq_size).
// The post-fold eq_size = src_pq_size / 2, pq_size = src_pq_size / 2.
extern "C" int _frac_compute_round_and_fold_inplace(
    const FpExt *eq_xi_low,
    const FpExt *eq_xi_high,
    FracExt *pq_buffer,           // In-place: reads src_pq_size, writes pq_size
    size_t src_pq_size,           // Pre-fold size in FracExt
    size_t eq_low_cap,
    FpExt lambda,
    FpExt r_prev,
    FpExt *out,                   // Output: [d=3] final results
    FpExt *tmp_block_sums         // Temporary buffer: [gridDim.x * d]
) {
    assert(src_pq_size > 2);
    // Post-fold sizes
    size_t pq_size = src_pq_size >> 1;
    size_t eq_size = pq_size >> 1;
    assert(eq_size > 0);

    auto [grid, block] = frac_compute_round_launch_params(eq_size);
    uint32_t num_warps = (block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t shmem_bytes = std::max(1u, num_warps) * sizeof(FpExt);

    // Launch fused in-place kernel - writes to tmp_block_sums and pq_buffer (first half)
    compute_round_and_fold_inplace_kernel<<<grid, block, shmem_bytes>>>(
        eq_xi_low,
        eq_xi_high,
        pq_buffer,
        __builtin_ctz((uint32_t)eq_size),
        __builtin_ctz((uint32_t)eq_low_cap),
        __builtin_ctz((uint32_t)pq_size),
        lambda,
        r_prev,
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
