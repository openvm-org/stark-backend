// GKR fractional sumcheck CUDA kernels. See docs/cuda-backend/gkr-prover.md (repo root)
// for the protocol and implementation details.

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
// Degree of s' polynomial (factored out the first eq term)
constexpr int GKR_SP_DEG = 2;
// ============================================================================
// KERNELS
// ============================================================================
template <bool revert, bool apply_alpha = false>
__global__ void frac_build_tree_layer_kernel(
    FracExt *__restrict__ layer,
    uint32_t half,
    FpExt alpha = FpExt(Fp(0))
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half) {
        return;
    }

    if constexpr (apply_alpha) {
        // When applying alpha, we need to modify both operands before combining
        // Read both values, apply alpha to denominators, then combine
        FracExt lhs_val = layer[idx];
        FracExt rhs_val = layer[idx | half];
        lhs_val.q = lhs_val.q + alpha;
        rhs_val.q = rhs_val.q + alpha;

        // Save alpha-applied rhs_val before combining (for second half write-back)
        FracExt rhs_val_with_alpha = rhs_val;

        if constexpr (revert) {
            frac_unadd_inplace(lhs_val, rhs_val);
        } else {
            frac_add_inplace(lhs_val, rhs_val);
        }

        layer[idx] = lhs_val;
        layer[idx | half] = rhs_val_with_alpha;  // Write back alpha-applied value to second half
    } else {
        // Original fast path: use references for in-place modification
        FracExt &lhs = layer[idx];
        FracExt const &rhs = layer[idx | half];

        if constexpr (revert) {
            frac_unadd_inplace(lhs, rhs);
        } else {
            frac_add_inplace(lhs, rhs);
        }
    }
}

// Reconstruct eq weight from sqrt-decomposed buffers.
// See gkr-prover.md § "Eq buffer sqrt decomposition".
__device__ __forceinline__ FpExt sqrt_buffer_get(
    const FpExt *__restrict__ eq_xi_low,
    const FpExt *__restrict__ eq_xi_high,
    uint32_t const log_eq_low_cap,
    uint32_t const idx
) {
    return eq_xi_low[idx & ((1u << log_eq_low_cap) - 1)] * eq_xi_high[idx >> log_eq_low_cap];
}

// Accumulate GKR sumcheck contributions for s'_t(1) and s'_t(2).
// See gkr-prover.md § "Sumcheck round implementation".
__device__ __forceinline__ void accumulate_compute_contributions(
    const FpExt *__restrict__ eq_xi_low,
    const FpExt *__restrict__ eq_xi_high,
    uint32_t idx,
    uint32_t log_eq_low_cap,
    FpExt lambda,
    FpExt p0_even, FpExt q0_even, FpExt p0_odd, FpExt q0_odd,
    FpExt p1_even, FpExt q1_even, FpExt p1_odd, FpExt q1_odd,
    FpExt &local0, FpExt &local1
) {
    FpExt eq_val = sqrt_buffer_get(eq_xi_low, eq_xi_high, log_eq_low_cap, idx);

    FpExt p0_diff = p0_odd - p0_even;
    FpExt q0_diff = q0_odd - q0_even;
    FpExt p1_diff = p1_odd - p1_even;
    FpExt q1_diff = q1_odd - q1_even;

    FpExt p_j0 = p0_even + lambda * q0_even;
    FpExt q_j0 = q0_even;
    FpExt p_j1 = p1_even;
    FpExt q_j1 = q1_even;

    auto const lambda_times_q0_diff = lambda * q0_diff;

#pragma unroll
    for (int i = 0; i < GKR_SP_DEG; ++i) {
        p_j0 += p0_diff;
        p_j0 += lambda_times_q0_diff;
        q_j0 += q0_diff;
        p_j1 += p1_diff;
        q_j1 += q1_diff;

        FpExt contrib = eq_val * (p_j0 * q_j1 + p_j1 * q_j0);
        if (i == 0) local0 += contrib;
        else local1 += contrib;
    }
}

// Helper: Perform block reduction on two accumulators and write to block_sums.
__device__ __forceinline__ void reduce_block_sums(
    FpExt *shared,
    FpExt local0, FpExt local1,
    FpExt *__restrict__ block_sums
) {
    {
        FpExt reduced = sumcheck::block_reduce_sum(local0, shared);
        if (threadIdx.x == 0) block_sums[blockIdx.x * GKR_SP_DEG + 0] = reduced;
    }
    __syncthreads();
    {
        FpExt reduced = sumcheck::block_reduce_sum(local1, shared);
        if (threadIdx.x == 0) block_sums[blockIdx.x * GKR_SP_DEG + 1] = reduced;
    }
}

// shared memory size requirement: max(num_warps,1) * sizeof(FpExt)
// Computes s' polynomial evaluations at 1 and 2 (first eq term factored out).
__global__ void compute_round_block_sum_kernel(
    const FpExt *__restrict__ eq_xi_low,
    const FpExt *__restrict__ eq_xi_high,
    const FracExt *__restrict__ pq_buffer,
    uint32_t num_x,
    uint32_t log_eq_low_cap,
    FpExt lambda,
    FpExt *__restrict__ block_sums // Output: [gridDim.x][2]
) {
    extern __shared__ FpExt shared[];
    const uint32_t pq_size = 2 * num_x;

    const FpExt zero(Fp::zero());
    FpExt local[GKR_SP_DEG] = {zero, zero};
    uint32_t idx_base = threadIdx.x + blockIdx.x * blockDim.x;
    // Map phase: compute local sum by striding over grid
    for (uint32_t idx = idx_base; idx < num_x / 2; idx += blockDim.x * gridDim.x) {
        FpExt eq_val = sqrt_buffer_get(eq_xi_low, eq_xi_high, log_eq_low_cap, idx);

        // \hat p_j({0 or 1}, {even or odd}, ..y)
        // The {even=0, odd=1} evaluations are used to interpolate at 1, 2
        auto const &[p0_even, q0_even] = pq_buffer[idx];
        auto const &[p1_even, q1_even] = pq_buffer[with_rev_bits(idx, pq_size, 1, 0)];
        auto const &[p0_odd, q0_odd] = pq_buffer[with_rev_bits(idx, pq_size, 0, 1)];
        auto const &[p1_odd, q1_odd] = pq_buffer[with_rev_bits(idx, pq_size, 1, 1)];

        FpExt p0_diff = p0_odd - p0_even;
        FpExt q0_diff = q0_odd - q0_even;
        FpExt p1_diff = p1_odd - p1_even;
        FpExt q1_diff = q1_odd - q1_even;

        FpExt p_j0 = p0_even + lambda * q0_even;
        FpExt q_j0 = q0_even;
        FpExt p_j1 = p1_even;
        FpExt q_j1 = q1_even;

        auto const lambda_times_q0_diff = lambda * q0_diff;

#pragma unroll
        for (int i = 0; i < GKR_SP_DEG; ++i) {
            p_j0 += p0_diff;
            p_j0 += lambda_times_q0_diff;
            q_j0 += q0_diff;
            p_j1 += p1_diff;
            q_j1 += q1_diff;

            local[i] += eq_val * (p_j0 * q_j1 + p_j1 * q_j0);
        }
    }
// Reduce phase: reduce all threadIdx.x in the same block, keeping 1,2 independent
#pragma unroll
    for (int i = 0; i < GKR_SP_DEG; ++i) {
        FpExt reduced = sumcheck::block_reduce_sum(local[i], shared);

        if (threadIdx.x == 0) {
            block_sums[blockIdx.x * GKR_SP_DEG + i] = reduced;
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
// Reads from pre-fold src_pq buffer (size 2*pq_size), computes s' polynomial at 1 and 2,
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
    uint32_t num_x,                      // post-fold num_x (= pq_size / 2)
    uint32_t log_eq_low_cap,
    FpExt lambda,
    FpExt r_prev,                        // Previous round's challenge for folding
    FpExt *__restrict__ block_sums,      // Output: [gridDim.x * 2]
    FracExt *__restrict__ dst_pq         // Post-fold buffer (pq_size FracExt)
) {
    extern __shared__ FpExt shared[];
    const uint32_t pq_size = 2 * num_x;

    // Index offsets for fold pattern (see detailed comments in inplace kernel)
    const uint32_t half = pq_size;              // src_pq_size / 2
    const uint32_t quarter = pq_size >> 1;      // pq_size / 2
    const uint32_t eighth = pq_size >> 2;       // pq_size / 4
    const uint32_t three_eighths = eighth * 3;  // 3 * pq_size / 4

    // Use scalar accumulators instead of array to help register allocation
    const FpExt zero(Fp::zero());
    FpExt local0 = zero, local1 = zero;
    uint32_t idx_base = threadIdx.x + blockIdx.x * blockDim.x;

    // Map phase: compute local sum by striding over grid
    for (uint32_t idx = idx_base; idx < num_x / 2; idx += blockDim.x * gridDim.x) {
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
            eq_xi_low, eq_xi_high, idx, log_eq_low_cap, lambda,
            p0_even, q0_even, p0_odd, q0_odd,
            p1_even, q1_even, p1_odd, q1_odd,
            local0, local1
        );
    }

    reduce_block_sums(shared, local0, local1, block_sums);
}

// Fused compute round + fold kernel (IN-PLACE version).
// Reads from pre-fold pq buffer (size 2*pq_size), computes s' polynomial at 1 and 2,
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
    uint32_t num_x,                      // post-fold num_x (= pq_size / 2)
    uint32_t log_eq_low_cap,
    FpExt lambda,
    FpExt r_prev,                        // Previous round's challenge for folding
    FpExt *__restrict__ block_sums       // Output: [gridDim.x * 2]
) {
    extern __shared__ FpExt shared[];
    const uint32_t pq_size = 2 * num_x;

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
    FpExt local0 = zero, local1 = zero;
    uint32_t idx_base = threadIdx.x + blockIdx.x * blockDim.x;

    // Map phase: compute local sum by striding over grid
    for (uint32_t idx = idx_base; idx < num_x / 2; idx += blockDim.x * gridDim.x) {
        // OPTIMIZATION: Staged load-fold pattern to reduce register pressure.
        // For in-place operation, we must read all 8 source values before writing.
        // BUT we fold and write immediately after each pair to minimize live register count.
        // This reduces peak register usage from ~107 to ~60-70 regs/thread.

        // Declare output scalars for folded values
        FpExt p0_even, q0_even, p1_even, q1_even;
        FpExt p0_odd, q0_odd, p1_odd, q1_odd;

        // Pair 1: f00 (idx -> p0_even, q0_even)
        {
            FracExt a = pq[idx];
            FracExt b = pq[idx + quarter];
            p0_even = a.p + r_prev * (b.p - a.p);
            q0_even = a.q + r_prev * (b.q - a.q);
        }

        // Pair 2: f10 (idx + half -> p1_even, q1_even)
        {
            FracExt a = pq[idx + half];
            FracExt b = pq[idx + half + quarter];
            p1_even = a.p + r_prev * (b.p - a.p);
            q1_even = a.q + r_prev * (b.q - a.q);
        }

        // Pair 3: f01 (idx + eighth -> p0_odd, q0_odd)
        {
            FracExt a = pq[idx + eighth];
            FracExt b = pq[idx + three_eighths];
            p0_odd = a.p + r_prev * (b.p - a.p);
            q0_odd = a.q + r_prev * (b.q - a.q);
        }

        // Pair 4: f11 (idx + half + eighth -> p1_odd, q1_odd)
        {
            FracExt a = pq[idx + half + eighth];
            FracExt b = pq[idx + half + three_eighths];
            p1_odd = a.p + r_prev * (b.p - a.p);
            q1_odd = a.q + r_prev * (b.q - a.q);
        }

        // Write all folded values (safe: writes to first half, reads from both halves completed)
        pq[idx] = {p0_even, q0_even};
        pq[idx + quarter] = {p1_even, q1_even};
        pq[idx + eighth] = {p0_odd, q0_odd};
        pq[idx + three_eighths] = {p1_odd, q1_odd};

        accumulate_compute_contributions(
            eq_xi_low, eq_xi_high, idx, log_eq_low_cap, lambda,
            p0_even, q0_even, p0_odd, q0_odd,
            p1_even, q1_even, p1_odd, q1_odd,
            local0, local1
        );
    }

    reduce_block_sums(shared, local0, local1, block_sums);
}

// Fused compute round + tree layer revert kernel.
// Combines frac_build_tree_layer_kernel<true> (revert) with compute_round_block_sum_kernel.
//
// This is used for the FIRST inner round only, where we need to revert the tree layer
// before computing. Equivalent to fractional subtraction:
//   layer[i] = frac_unadd(layer[i], layer[i + half])
// where rhs = layer[i + half] is kept unchanged.
//
// For round 0, the 4 PQ indices accessed per thread are:
//   - idx: in first half, needs revert
//   - idx | quarter: in first half, needs revert
//   - idx | half: in second half, no revert
//   - idx | half | quarter: in second half, no revert
//
// Each index in the first half is written by exactly one thread:
//   - Thread idx writes to indices {idx, idx + quarter}
//   - For idx in [0, num_x/2), these cover [0, half) completely
//
// shared memory size requirement: max(num_warps,1) * sizeof(FpExt)
__global__ void compute_round_and_revert_kernel(
    const FpExt *__restrict__ eq_xi_low,
    const FpExt *__restrict__ eq_xi_high,
    FracExt *__restrict__ layer,         // Tree layer buffer (modified in-place for revert)
    uint32_t num_x,
    uint32_t log_eq_low_cap,
    FpExt lambda,
    FpExt *__restrict__ block_sums       // Output: [gridDim.x * 2]
) {
    extern __shared__ FpExt shared[];
    const uint32_t pq_size = 2 * num_x;
    const uint32_t half = pq_size >> 1;     // pq_size / 2
    const uint32_t quarter = pq_size >> 2;  // pq_size / 4

    // Use scalar accumulators instead of array to help register allocation
    const FpExt zero(Fp::zero());
    FpExt local0 = zero, local1 = zero;
    uint32_t idx_base = threadIdx.x + blockIdx.x * blockDim.x;

    // Map phase: compute local sum by striding over grid
    for (uint32_t idx = idx_base; idx < num_x / 2; idx += blockDim.x * gridDim.x) {
        // OPTIMIZATION: Extract components immediately, don't keep FracExt structures live.
        // This reduces register pressure by avoiding 4x FracExt (8 FpExt) intermediate storage.

        // Declare output scalars upfront
        FpExt p0_even, q0_even, p1_even, q1_even;
        FpExt p0_odd, q0_odd, p1_odd, q1_odd;

        // Compute p0_even by reverting with p1_even (frac_unadd)
        {
            FracExt lhs = layer[idx];
            FracExt rhs = layer[idx + half];
            FpExt rhs_q_inv = inv(rhs.q);
            q0_even = lhs.q * rhs_q_inv;
            p0_even = (lhs.p - q0_even * rhs.p) * rhs_q_inv;
            p1_even = rhs.p;
            q1_even = rhs.q;
            layer[idx] = {p0_even, q0_even};
        }

        // Compute p0_odd by reverting with p1_odd (frac_unadd)
        {
            FracExt lhs = layer[idx + quarter];
            FracExt rhs = layer[idx + half + quarter];
            FpExt rhs_q_inv = inv(rhs.q);
            q0_odd = lhs.q * rhs_q_inv;
            p0_odd = (lhs.p - q0_odd * rhs.p) * rhs_q_inv;
            p1_odd = rhs.p;
            q1_odd = rhs.q;
            layer[idx + quarter] = {p0_odd, q0_odd};
        }

        accumulate_compute_contributions(
            eq_xi_low, eq_xi_high, idx, log_eq_low_cap, lambda,
            p0_even, q0_even, p0_odd, q0_odd,
            p1_even, q1_even, p1_odd, q1_odd,
            local0, local1
        );
    }

    reduce_block_sums(shared, local0, local1, block_sums);
}

// Number of tail elements loaded into shared memory per loop iteration in
// precompute_m_build_partial_kernel. Each batch allocates:
//   (4 * (2^w + 1) * BATCH + BATCH) * sizeof(FpExt)
// of shared memory (4 data arrays + 1 weight array, with +1 stride for bank
// conflict avoidance). With w=3 (2^3=8) and BATCH=16 this is ~9.25 KB.
// Increasing BATCH improves amortization of __syncthreads() barriers but
// increases shared memory pressure, which can reduce occupancy or prevent launch.
constexpr uint32_t PRECOMPUTE_M_TAIL_BATCH = 16;

// Build partial M blocks by parallelizing over tail points.
// See gkr-prover.md § "Precompute M strategy".
// Each block covers a range of tail points [b_start, b_end) and computes a full M block.
// Uses 2D blocks (m,m) where each thread owns one (u,v) M-matrix entry.
// Shared memory uses +1 padding on the m-dimension stride to avoid bank conflicts.
template <bool inline_fold, uint32_t W>
__global__ void precompute_m_build_partial_kernel(
    const FracExt *__restrict__ pq,
    uint32_t rem_n,          // number of variables AFTER fold (folded rem_n)
    FpExt lambda,
    FpExt r_prev,            // challenge for the inline fold (only used when inline_fold=true)
    const FpExt *__restrict__ eq_tail_low,
    const FpExt *__restrict__ eq_tail_high,
    uint32_t log_eq_tail_low_cap,
    uint32_t tail_tile,
    FpExt *__restrict__ partial_out
) {
    constexpr uint32_t m = 1u << W;
    uint32_t u = threadIdx.y;
    uint32_t v = threadIdx.x;
    if (u >= m || v >= m) {
        return;
    }

    uint32_t tail_n = rem_n - W;
    uint32_t k = 1u << tail_n;
    // When inline_fold: buffer has rem_n+1 variables (unfolded).
    // Otherwise: buffer has rem_n variables (already folded).
    uint32_t fold_half = inline_fold ? (1u << rem_n) : 0;
    uint32_t poly_stride = inline_fold ? (1u << (rem_n + 1)) : (1u << rem_n);

    uint32_t b_start = blockIdx.x * tail_tile;
    uint32_t b_end = min(b_start + tail_tile, k);

    // +1 padding avoids bank conflicts for power-of-2 m (m=8 → stride 9).
    constexpr uint32_t sh_stride = m + 1;
    extern __shared__ FpExt shmem[];
    FpExt *sh_left0 = shmem;
    FpExt *sh_left1 = sh_left0 + sh_stride * PRECOMPUTE_M_TAIL_BATCH;
    FpExt *sh_right0 = sh_left1 + sh_stride * PRECOMPUTE_M_TAIL_BATCH;
    FpExt *sh_right1 = sh_right0 + sh_stride * PRECOMPUTE_M_TAIL_BATCH;
    FpExt *sh_weight = sh_right1 + sh_stride * PRECOMPUTE_M_TAIL_BATCH;

    FpExt zero(Fp::zero());
    FpExt acc = zero;

    uint32_t lane = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t threads = blockDim.x * blockDim.y;

    for (uint32_t b0 = b_start; b0 < b_end; b0 += PRECOMPUTE_M_TAIL_BATCH) {
        uint32_t batch = min(PRECOMPUTE_M_TAIL_BATCH, b_end - b0);
        uint32_t elems = batch * m;

        // Phase 1: Load weights into shared memory.
        for (uint32_t bi = lane; bi < batch; bi += threads) {
            uint32_t b = b0 + bi;
            sh_weight[bi] = sqrt_buffer_get(
                eq_tail_low, eq_tail_high, log_eq_tail_low_cap, b
            );
        }
        __syncthreads();

        // Phase 2: Load pq data, optionally fold inline by r_prev, and premultiply left0/left1 by weight.
        for (uint32_t idx = lane; idx < elems; idx += threads) {
            uint32_t beta = idx / batch;
            uint32_t bi = idx - beta * batch;
            uint32_t b = b0 + bi;
            uint32_t src = beta * k + b;
            FpExt p0, q0, p1, q1;
            if constexpr (inline_fold) {
                // Inline fold: val = lo + r_prev * (hi - lo) for each of {p0, q0, p1, q1}.
                FracExt const &f0_lo = pq[src];
                FracExt const &f0_hi = pq[src + fold_half];
                FracExt const &f1_lo = pq[poly_stride + src];
                FracExt const &f1_hi = pq[poly_stride + src + fold_half];
                p0 = f0_lo.p + r_prev * (f0_hi.p - f0_lo.p);
                q0 = f0_lo.q + r_prev * (f0_hi.q - f0_lo.q);
                p1 = f1_lo.p + r_prev * (f1_hi.p - f1_lo.p);
                q1 = f1_lo.q + r_prev * (f1_hi.q - f1_lo.q);
            } else {
                // Buffer already folded: read directly.
                FracExt const &f0 = pq[src];
                FracExt const &f1 = pq[poly_stride + src];
                p0 = f0.p; q0 = f0.q;
                p1 = f1.p; q1 = f1.q;
            }
            uint32_t sh_idx = bi * sh_stride + beta;
            FpExt wt = sh_weight[bi];
            sh_left0[sh_idx] = wt * (p0 + lambda * q0);
            sh_left1[sh_idx] = wt * p1;
            sh_right0[sh_idx] = q0;
            sh_right1[sh_idx] = q1;
        }
        __syncthreads();

        // Phase 3: Accumulate with 2 muls per iteration (weight already folded in).
        for (uint32_t bi = 0; bi < batch; ++bi) {
            uint32_t row = bi * sh_stride;
            acc += sh_left0[row + u] * sh_right1[row + v]
                 + sh_left1[row + u] * sh_right0[row + v];
        }

        __syncthreads();
    }

    partial_out[blockIdx.x * (m * m) + u * m + v] = acc;
}

template <bool inline_fold, uint32_t W>
inline void launch_precompute_m_build_partial_kernel(
    dim3 grid,
    const FracExt *pq,
    uint32_t rem_n,
    FpExt lambda,
    FpExt r_prev,
    const FpExt *eq_tail_low,
    const FpExt *eq_tail_high,
    uint32_t log_eq_tail_low_cap,
    uint32_t tail_tile,
    FpExt *partial_out
) {
    constexpr uint32_t m = 1u << W;
    dim3 block(m, m);
    constexpr uint32_t sh_stride = m + 1;  // +1 padding to match kernel
    size_t shmem_bytes =
        (4 * sh_stride * PRECOMPUTE_M_TAIL_BATCH + PRECOMPUTE_M_TAIL_BATCH) * sizeof(FpExt);
    precompute_m_build_partial_kernel<inline_fold, W><<<grid, block, shmem_bytes>>>(
        pq,
        rem_n,
        lambda,
        r_prev,
        eq_tail_low,
        eq_tail_high,
        log_eq_tail_low_cap,
        tail_tile,
        partial_out
    );
}

template <bool inline_fold>
inline int launch_precompute_m_build_partial_dispatch(
    uint32_t w,
    dim3 grid,
    const FracExt *pq,
    uint32_t rem_n,
    FpExt lambda,
    FpExt r_prev,
    const FpExt *eq_tail_low,
    const FpExt *eq_tail_high,
    uint32_t log_eq_tail_low_cap,
    uint32_t tail_tile,
    FpExt *partial_out
) {
    using LauncherFn = void (*)(
        dim3, const FracExt *, uint32_t, FpExt, FpExt, const FpExt *, const FpExt *, uint32_t,
        uint32_t, FpExt *
    );
    static constexpr LauncherFn launchers[] = {
        &launch_precompute_m_build_partial_kernel<inline_fold, 1>,
        &launch_precompute_m_build_partial_kernel<inline_fold, 2>,
        &launch_precompute_m_build_partial_kernel<inline_fold, 3>,
        &launch_precompute_m_build_partial_kernel<inline_fold, 4>,
        &launch_precompute_m_build_partial_kernel<inline_fold, 5>,
    };

    constexpr uint32_t min_w = 1;
    constexpr uint32_t max_w = static_cast<uint32_t>(sizeof(launchers) / sizeof(launchers[0]));
    if (w < min_w || w > max_w) {
        // 2D launch uses (2^w, 2^w) threads, so w > 5 exceeds 1024 threads per block.
        return cudaErrorInvalidValue;
    }

    launchers[w - min_w](
        grid,
        pq,
        rem_n,
        lambda,
        r_prev,
        eq_tail_low,
        eq_tail_high,
        log_eq_tail_low_cap,
        tail_tile,
        partial_out
    );
    return 0;
}

// Reduce partial M blocks into a final M matrix.
__global__ void precompute_m_reduce_partials_kernel(
    const FpExt *__restrict__ partial,
    uint32_t num_blocks,
    uint32_t total_entries,
    FpExt *__restrict__ m_total
) {
    uint32_t entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry >= total_entries) {
        return;
    }

    FpExt zero(Fp::zero());
    FpExt acc = zero;
    // Iterate over partial blocks with coalesced loads:
    // for fixed b, neighboring threads read neighboring entries.
    for (uint32_t b = 0; b < num_blocks; ++b) {
        acc += partial[(size_t)b * total_entries + entry];
    }
    m_total[entry] = acc;
}

// Evaluate one round polynomial inside a precompute-M window from precomputed M.
// See gkr-prover.md § "Precompute M strategy".
// Output: out[0] = s'(1), out[1] = s'(2).
__global__ void precompute_m_eval_round_kernel(
    const FpExt *__restrict__ m_total,
    uint32_t w,
    uint32_t t,
    const FpExt *__restrict__ eq_r_prefix,
    const FpExt *__restrict__ eq_suffix,
    FpExt *__restrict__ out
) {
    extern __shared__ FpExt shared[];

    uint32_t m = 1u << w;
    uint32_t prefix_bits = t;
    uint32_t suffix_bits = w - t - 1;
    uint32_t prefix_size = 1u << prefix_bits;
    uint32_t suffix_size = 1u << suffix_bits;

    uint64_t total = (uint64_t)prefix_size * prefix_size * suffix_size;
    uint32_t cur_bit = 1u << suffix_bits;
    FpExt zero(Fp::zero());
    FpExt one(Fp::one());
    FpExt two = one + one;

    FpExt local_s1 = zero;
    FpExt local_s2 = zero;

    for (uint64_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        uint32_t suffix = (uint32_t)(idx % suffix_size);
        uint64_t tmp = idx / suffix_size;
        uint32_t b2 = (uint32_t)(tmp % prefix_size);
        uint32_t b1 = (uint32_t)(tmp / prefix_size);

        FpExt weight = eq_r_prefix[b1] * eq_r_prefix[b2] * eq_suffix[suffix];

        uint32_t prefix_shift = suffix_bits + 1;
        uint32_t beta1_0 = (b1 << prefix_shift) | suffix;
        uint32_t beta1_1 = beta1_0 | cur_bit;
        uint32_t beta2_0 = (b2 << prefix_shift) | suffix;
        uint32_t beta2_1 = beta2_0 | cur_bit;

        FpExt m00 = m_total[beta1_0 * m + beta2_0];
        FpExt m01 = m_total[beta1_0 * m + beta2_1];
        FpExt m10 = m_total[beta1_1 * m + beta2_0];
        FpExt m11 = m_total[beta1_1 * m + beta2_1];

        local_s1 += weight * m11;
        local_s2 += weight * (m00 - two * (m01 + m10 - m11 - m11));
    }

    {
        FpExt reduced = sumcheck::block_reduce_sum(local_s1, shared);
        if (threadIdx.x == 0) out[0] = reduced;
    }
    __syncthreads();
    {
        FpExt reduced = sumcheck::block_reduce_sum(local_s2, shared);
        if (threadIdx.x == 0) out[1] = reduced;
    }
}

// Fold w rounds at once using precomputed eq_r_window.
// One thread computes both poly outputs for a single tail index.
template <uint32_t W>
__global__ void multifold_kernel(
    const FracExt *src,
    FracExt *dst,
    uint32_t tail_size,
    const FpExt *__restrict__ eq_r_window
) {
    constexpr uint32_t beta_size = 1u << W;
    uint32_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= tail_size) {
        return;
    }

    FpExt zero(Fp::zero());
    FpExt acc0_p = zero;
    FpExt acc0_q = zero;
    FpExt acc1_p = zero;
    FpExt acc1_q = zero;
    const FracExt *src_1 = src + ((size_t)tail_size << W);
    for (uint32_t beta = 0; beta < beta_size; ++beta) {
        uint32_t idx = beta * tail_size + out_idx;
        FracExt v0 = src[idx];
        FracExt v1 = src_1[idx];
        FpExt eq_r = eq_r_window[beta];
        acc0_p += eq_r * v0.p;
        acc0_q += eq_r * v0.q;
        acc1_p += eq_r * v1.p;
        acc1_q += eq_r * v1.q;
    }
    dst[out_idx] = {acc0_p, acc0_q};
    dst[tail_size + out_idx] = {acc1_p, acc1_q};
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
template <bool revert, bool apply_alpha>
int launch_frac_build_tree_layer(FracExt *layer, size_t layer_size, FpExt alpha) {
    auto [grid, block] = kernel_launch_params(layer_size);
    frac_build_tree_layer_kernel<revert, apply_alpha><<<grid, block>>>(layer, layer_size, alpha);
    return CHECK_KERNEL();
}

extern "C" int _frac_build_tree_layer(
    FracExt *layer,
    size_t layer_size,
    bool revert,
    FpExt alpha,
    bool apply_alpha
) {
    if (layer_size == 0) {
        return 0;
    }
    assert(layer_size % 2 == 0);
    layer_size /= 2;

    return DISPATCH_BOOL_PAIR(
        launch_frac_build_tree_layer, revert, apply_alpha, layer, layer_size, alpha
    );
}

inline uint32_t min_blocks_target_for_device(uint32_t blocks_per_sm, uint32_t fallback_blocks) {
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess) {
        int sm_count = 0;
        if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) ==
                cudaSuccess &&
            sm_count > 0) {
            return static_cast<uint32_t>(sm_count) * blocks_per_sm;
        }
    }
    return fallback_blocks;
}

inline std::pair<dim3, dim3> frac_compute_round_launch_params(uint32_t num_x) {
    // OPTIMIZATION: Adaptive grid sizing to improve GPU utilization for small workloads.
    // For small num_x, the default sizing creates too few blocks.
    // We reduce block size to create more blocks, ensuring better SM coverage.

    uint32_t elements = num_x >> 1;

    // Target: at least 2 blocks per SM for round-compute kernels. This is a simple default that helps
    // hide latency on memory-heavy kernels without over-fragmenting blocks.
    // For small workloads, we reduce threads/block to increase block count.
    constexpr uint32_t ROUND_COMPUTE_BLOCKS_PER_SM_TARGET = 2;
    constexpr uint32_t ROUND_COMPUTE_FALLBACK_BLOCKS_TARGET = 228;
    uint32_t min_blocks_target =
        min_blocks_target_for_device(
            ROUND_COMPUTE_BLOCKS_PER_SM_TARGET, ROUND_COMPUTE_FALLBACK_BLOCKS_TARGET
        );
    constexpr uint32_t DEFAULT_BLOCK_SIZE = 256;
    constexpr uint32_t MIN_BLOCK_SIZE = 64;      // Minimum for occupancy

    uint32_t block_size = DEFAULT_BLOCK_SIZE;
    uint32_t blocks_needed = (elements + block_size - 1) / block_size;

    // If we'd get too few blocks with default size, try reducing block size
    if (blocks_needed < min_blocks_target && elements >= MIN_BLOCK_SIZE) {
        // Calculate block size needed to reach target
        block_size = (elements + min_blocks_target - 1) / min_blocks_target;
        // Round up to next multiple of 32 (warp size)
        block_size = std::max(MIN_BLOCK_SIZE, ((block_size + 31) / 32) * 32);
        // Recalculate blocks with new size
        blocks_needed = (elements + block_size - 1) / block_size;
    }

    return {dim3(blocks_needed), dim3(block_size)};
}

extern "C" uint32_t _frac_compute_round_temp_buffer_size(uint32_t num_x) {
    auto [grid, block] = frac_compute_round_launch_params(num_x);
    return grid.x * GKR_SP_DEG;
}

inline int final_reduce_block_sums(FpExt *tmp_block_sums, FpExt *out, uint32_t num_blocks) {
    auto [unused_grid, reduce_block] = kernel_launch_params(num_blocks);
    (void)unused_grid;
    unsigned int reduce_warps = div_ceil(reduce_block.x, WARP_SIZE);
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    sumcheck::static_final_reduce_block_sums<GKR_SP_DEG>
        <<<GKR_SP_DEG, reduce_block, reduce_shmem>>>(tmp_block_sums, out, num_blocks);
    return CHECK_KERNEL();
}

extern "C" int _frac_compute_round(
    const FpExt *eq_xi_low,
    const FpExt *eq_xi_high,
    const FracExt *pq_buffer,
    size_t num_x,
    size_t eq_low_cap,
    FpExt lambda,
    FpExt *out,           // Output: [d=2] final results
    FpExt *tmp_block_sums // Temporary buffer: [gridDim.x * d]
) {
    assert(num_x > 1);

    auto [grid, block] = frac_compute_round_launch_params(num_x);
    size_t shmem_bytes = div_ceil(block.x, WARP_SIZE) * sizeof(FpExt);

    // Launch main kernel - writes to tmp_block_sums
    compute_round_block_sum_kernel<<<grid, block, shmem_bytes>>>(
        eq_xi_low,
        eq_xi_high,
        pq_buffer,
        (uint32_t)num_x,
        __builtin_ctz((uint32_t)eq_low_cap),
        lambda,
        tmp_block_sums
    );
    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Launch final reduction kernel - reads from tmp_block_sums, writes to output.
    return final_reduce_block_sums(tmp_block_sums, out, grid.x);
}

// Fused compute round + tree layer revert launcher.
// Combines frac_build_tree_layer(revert=true) with compute_round for the first inner round.
extern "C" int _frac_compute_round_and_revert(
    const FpExt *eq_xi_low,
    const FpExt *eq_xi_high,
    FracExt *layer,           // Tree layer buffer (modified in-place for revert)
    size_t num_x,
    size_t eq_low_cap,
    FpExt lambda,
    FpExt *out,               // Output: [d=2] final results
    FpExt *tmp_block_sums     // Temporary buffer: [gridDim.x * d]
) {
    assert(num_x > 1);

    auto [grid, block] = frac_compute_round_launch_params(num_x);
    size_t shmem_bytes = div_ceil(block.x, WARP_SIZE) * sizeof(FpExt);

    // Launch fused revert + compute kernel
    compute_round_and_revert_kernel<<<grid, block, shmem_bytes>>>(
        eq_xi_low,
        eq_xi_high,
        layer,
        (uint32_t)num_x,
        __builtin_ctz((uint32_t)eq_low_cap),
        lambda,
        tmp_block_sums
    );
    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Launch final reduction kernel.
    return final_reduce_block_sums(tmp_block_sums, out, grid.x);
}

// Fused compute round + fold launcher.
// src_pq_size is the pre-fold buffer size (2*pq_size).
// Post-fold: num_x = pq_size / 2 = src_pq_size / 4.
extern "C" int _frac_compute_round_and_fold(
    const FpExt *eq_xi_low,
    const FpExt *eq_xi_high,
    const FracExt *src_pq_buffer,
    FracExt *dst_pq_buffer,
    size_t src_pq_size,           // Pre-fold size in FracExt
    size_t eq_low_cap,
    FpExt lambda,
    FpExt r_prev,
    FpExt *out,                   // Output: [d=2] final results
    FpExt *tmp_block_sums         // Temporary buffer: [gridDim.x * d]
) {
    assert(src_pq_size > 2);
    // Post-fold sizes
    size_t pq_size = src_pq_size >> 1;
    size_t num_x = pq_size >> 1;
    assert(num_x > 0);

    auto [grid, block] = frac_compute_round_launch_params(num_x);
    size_t shmem_bytes = div_ceil(block.x, WARP_SIZE) * sizeof(FpExt);

    // Launch fused kernel - writes to tmp_block_sums and dst_pq_buffer
    compute_round_and_fold_kernel<<<grid, block, shmem_bytes>>>(
        eq_xi_low,
        eq_xi_high,
        src_pq_buffer,
        (uint32_t)num_x,
        __builtin_ctz((uint32_t)eq_low_cap),
        lambda,
        r_prev,
        tmp_block_sums,
        dst_pq_buffer
    );
    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Launch final reduction kernel - reads from tmp_block_sums, writes to output.
    return final_reduce_block_sums(tmp_block_sums, out, grid.x);
}

// Fused compute round + fold launcher (IN-PLACE version).
// src_pq_size is the pre-fold buffer size (2*pq_size).
// Post-fold: num_x = pq_size / 2 = src_pq_size / 4.
extern "C" int _frac_compute_round_and_fold_inplace(
    const FpExt *eq_xi_low,
    const FpExt *eq_xi_high,
    FracExt *pq_buffer,           // In-place: reads src_pq_size, writes pq_size
    size_t src_pq_size,           // Pre-fold size in FracExt
    size_t eq_low_cap,
    FpExt lambda,
    FpExt r_prev,
    FpExt *out,                   // Output: [d=2] final results
    FpExt *tmp_block_sums         // Temporary buffer: [gridDim.x * d]
) {
    assert(src_pq_size > 2);
    // Post-fold sizes
    size_t pq_size = src_pq_size >> 1;
    size_t num_x = pq_size >> 1;
    assert(num_x > 0);

    auto [grid, block] = frac_compute_round_launch_params(num_x);
    size_t shmem_bytes = div_ceil(block.x, WARP_SIZE) * sizeof(FpExt);

    // Launch fused in-place kernel - writes to tmp_block_sums and pq_buffer (first half)
    compute_round_and_fold_inplace_kernel<<<grid, block, shmem_bytes>>>(
        eq_xi_low,
        eq_xi_high,
        pq_buffer,
        (uint32_t)num_x,
        __builtin_ctz((uint32_t)eq_low_cap),
        lambda,
        r_prev,
        tmp_block_sums
    );
    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Launch final reduction kernel - reads from tmp_block_sums, writes to output.
    return final_reduce_block_sums(tmp_block_sums, out, grid.x);
}

extern "C" int _frac_precompute_m_build(
    const FracExt *pq,
    size_t rem_n,             // folded rem_n
    size_t w,
    FpExt lambda,
    FpExt r_prev,             // challenge for the inline fold (only used when inline_fold=true)
    bool inline_fold,         // true: pq is unfolded (rem_n+1 vars), false: pq is already folded
    const FpExt *eq_tail_low,
    const FpExt *eq_tail_high,
    size_t eq_tail_low_cap,
    size_t tail_tile,
    FpExt *partial_out,
    size_t partial_len,
    FpExt *m_total
) {
    assert(rem_n > 0);
    assert(w > 0 && w <= rem_n);
    assert(tail_tile > 0);

    uint32_t w_u32 = (uint32_t)w;
    uint32_t rem_n_u32 = (uint32_t)rem_n;
    uint32_t m = 1u << w_u32;
    uint32_t tail_n = (uint32_t)rem_n - (uint32_t)w;
    uint32_t k = 1u << tail_n;
    uint32_t tail_tile_u32 = (uint32_t)tail_tile;
    uint32_t num_blocks = div_ceil(k, tail_tile_u32);
    size_t total_entries = (size_t)m * (size_t)m;
    assert(partial_len >= (size_t)num_blocks * total_entries);

    dim3 grid(num_blocks);
    uint32_t log_eq_tail_low_cap = __builtin_ctz((uint32_t)eq_tail_low_cap);
    int launch_err = 0;
    if (inline_fold) {
        launch_err = launch_precompute_m_build_partial_dispatch<true>(
            w_u32,
            grid,
            pq,
            rem_n_u32,
            lambda,
            r_prev,
            eq_tail_low,
            eq_tail_high,
            log_eq_tail_low_cap,
            tail_tile_u32,
            partial_out
        );
    } else {
        launch_err = launch_precompute_m_build_partial_dispatch<false>(
            w_u32,
            grid,
            pq,
            rem_n_u32,
            lambda,
            r_prev,
            eq_tail_low,
            eq_tail_high,
            log_eq_tail_low_cap,
            tail_tile_u32,
            partial_out
        );
    }
    if (launch_err != 0) {
        return launch_err;
    }
    int err = CHECK_KERNEL();
    if (err != 0) {
        return err;
    }

    dim3 reduce_block(128);
    dim3 reduce_grid(div_ceil((uint32_t)total_entries, reduce_block.x));
    precompute_m_reduce_partials_kernel<<<reduce_grid, reduce_block>>>(
        partial_out,
        num_blocks,
        (uint32_t)total_entries,
        m_total
    );
    return CHECK_KERNEL();
}

extern "C" int _frac_precompute_m_eval_round(
    const FpExt *m_total,
    size_t w,
    size_t t,
    const FpExt *eq_r_prefix,
    const FpExt *eq_suffix,
    FpExt *out
) {
    assert(w > 0);
    assert(t < w);

    dim3 block(256);
    size_t shmem_bytes = div_ceil(block.x, WARP_SIZE) * sizeof(FpExt);
    precompute_m_eval_round_kernel<<<1, block, shmem_bytes>>>(
        m_total,
        (uint32_t)w,
        (uint32_t)t,
        eq_r_prefix,
        eq_suffix,
        out
    );
    return CHECK_KERNEL();
}

extern "C" int _frac_multifold(
    const FracExt *src,
    FracExt *dst,
    size_t rem_n,
    size_t w,
    const FpExt *eq_r_window
) {
    assert(rem_n > 0);
    assert(w > 0 && w <= rem_n);

    size_t out_len = 1u << (rem_n - w);
    auto [grid, block] = kernel_launch_params(out_len, 256);

#define DISPATCH_MULTIFOLD(W) \
    multifold_kernel<W><<<grid, block>>>(src, dst, (uint32_t)out_len, eq_r_window); \
    return CHECK_KERNEL();

    switch (w) {
        case 2: { DISPATCH_MULTIFOLD(2) }
        case 3: { DISPATCH_MULTIFOLD(3) }
        case 4: { DISPATCH_MULTIFOLD(4) }
        case 5: { DISPATCH_MULTIFOLD(5) }
        default: assert(false && "unsupported w for multifold"); return -1;
    }
#undef DISPATCH_MULTIFOLD
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
