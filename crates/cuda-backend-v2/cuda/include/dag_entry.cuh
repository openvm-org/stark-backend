#pragma once

#include "codec.cuh"
#include "device_ntt.cuh"
#include "fp.h"
#include <cassert>
#include <cstdint>
#include <vector_types.h>

namespace symbolic_dag {

// Context for evaluating DAG entries using NTT-based coset evaluation.
// iNTT is done once by coset_idx=0 threads, coefficients stored in shared memory,
// then each coset group does shift + forward NTT in parallel.
struct NttEvalContext {
    const Fp *__restrict__ preprocessed;
    const Fp *const *__restrict__ main_parts;
    const Fp *__restrict__ public_values;
    Fp *__restrict__ inter_buffer;
    Fp *__restrict__ ntt_buffer; // shared memory for coset NTT [skip_domain]
    Fp is_first;
    Fp is_last;
    Fp omega_shift;       // g^(coset_idx * z_idx_rev) for coset evaluation
    uint32_t skip_domain; // 2^l_skip
    uint32_t num_cosets;  // number of cosets (d)
    uint32_t num_x;
    uint32_t height; // <= num_x * skip_domain (could be < in lifted case)
    uint32_t buffer_stride;
    uint32_t buffer_size;
    uint32_t ntt_idx;   // 0..skip_domain (thread's position within skip domain)
    uint32_t coset_idx; // 0..num_cosets (thread's coset)
    uint32_t x_int;     // 0..num_x
};

// Evaluates f(g^(coset_idx+1) * omega_skip^ntt_idx, x) using NTT-based coset evaluation.
// Each coset handled in a different block: coset_idx = blockDim.y
//
// Algorithm:
// 1. All threads load trace data into their coset's NTT buffer
// 2. All threads perform iNTT on their own buffer (redundant across cosets, but needed for sync)
// 3. coset_idx=0 stores coefficients to coeffs_buffer
// 4. Sync, all threads read coefficient from coeffs_buffer
// 5. All threads apply coset-specific shift
// 6. All cosets do forward NTT in parallel
//
// Note: The iNTT is done redundantly by all cosets to ensure all threads hit the same
// __syncthreads() calls. Only coset_idx=0's result is used.
template <bool NEEDS_SHMEM>
__device__ __forceinline__ Fp ntt_coset_interpolate(
    const Fp *__restrict__ evals, // must have length height = num_x * skip_domain
    const NttEvalContext &ctx,
    uint8_t offset
) {
    auto const skip_domain = ctx.skip_domain;
    auto const base = ctx.x_int * skip_domain;
    auto const ntt_idx = ctx.ntt_idx;

#ifdef CUDA_DEBUG
    assert(ntt_idx < skip_domain);
    assert(ctx.coset_idx < ctx.num_cosets);
#endif

    uint32_t const l_skip = __ffs(skip_domain) - 1;

    // All threads load trace value from global memory (wrap by height for rotation, height <= lifted_height)
    uint32_t const idx = (base + ntt_idx + offset) % ctx.height;
    Fp this_thread_value = evals[idx];

    // Each coset uses its own NTT buffer slice
    Fp *__restrict__ ntt_buffer = ctx.ntt_buffer;

    // Step 2: iNTT - all cosets perform this redundantly for sync correctness
    if constexpr (NEEDS_SHMEM) {
        ntt_buffer[ntt_idx] = this_thread_value;
        __syncthreads();
    }
    device_ntt::ntt_natural_to_bitrev<true, NEEDS_SHMEM>(
        this_thread_value, ntt_buffer, ntt_idx, l_skip
    );
    // multiply by shift. regardless of whether shmem is needed, the input for forward NTT must be placed in `this_thread_value` for `ntt_bitrev_to_natural`
    this_thread_value *= ctx.omega_shift;
    // Step 6: Forward NTT of shifted coeffs
    device_ntt::ntt_bitrev_to_natural<false, NEEDS_SHMEM>(
        this_thread_value, ntt_buffer, ntt_idx, l_skip
    );

    return this_thread_value;
}

// NTT-based DAG entry evaluation.
// Uses coset NTT instead of barycentric interpolation for ENTRY_PREPROCESSED and ENTRY_MAIN.
// All threads in the block must call this together (for __syncthreads in NTT).
template <bool NEEDS_SHMEM>
__device__ __forceinline__ Fp ntt_eval_dag_entry(const SourceInfo &src, const NttEvalContext &ctx) {
    switch (src.type) {
    case ENTRY_PREPROCESSED: {
        const Fp *col = ctx.preprocessed + ctx.height * src.index;
        return ntt_coset_interpolate<NEEDS_SHMEM>(col, ctx, src.offset);
    }
    case ENTRY_MAIN: {
        auto main_ptr = ctx.main_parts[src.part];
        const Fp *col = main_ptr + ctx.height * src.index;
        return ntt_coset_interpolate<NEEDS_SHMEM>(col, ctx, src.offset);
    }
    case ENTRY_PUBLIC: {
        return ctx.public_values[src.index];
    }
    case SRC_CONSTANT:
        return Fp(src.index);
    case SRC_INTERMEDIATE:
#ifdef CUDA_DEBUG
        assert(ctx.buffer_size > 0);
        assert(src.index < ctx.buffer_size);
#endif
        return ctx.inter_buffer[src.index * ctx.buffer_stride];
    case SRC_IS_FIRST: {
        return ctx.is_first;
    }
    case SRC_IS_LAST: {
        return ctx.is_last;
    }
    case SRC_IS_TRANSITION: {
        // NOTE: we may change this to an unnormalized version
        return Fp::one() - ctx.is_last;
    }
    default:
        assert(false);
    }
    return Fp::zero();
}

} // namespace symbolic_dag
