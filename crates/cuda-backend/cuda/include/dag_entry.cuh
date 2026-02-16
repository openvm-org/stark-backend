#pragma once

#include "codec.cuh"
#include "device_ntt.cuh"
#include "fp.h"
#include <cassert>
#include <cstdint>
#include <vector_types.h>

namespace symbolic_dag {

// Context for evaluating DAG entries using NTT-based coset evaluation.
// Each thread handles ALL cosets in lockstep:
// - iNTT is done once, coefficient saved in register
// - For each coset: apply shift, forward NTT, store result
template <uint32_t NUM_COSETS> struct NttEvalContext {
    const Fp *__restrict__ preprocessed;
    const Fp *const *__restrict__ main_parts;
    const Fp *__restrict__ public_values;
    Fp *__restrict__ inter_buffer; // [buffer_size][NUM_COSETS] per thread
    Fp *__restrict__ ntt_buffer;   // shared memory for NTT scratch (only when NEEDS_SHMEM)
    Fp is_first[NUM_COSETS];       // per-coset selectors
    Fp is_last[NUM_COSETS];
    Fp is_transition[NUM_COSETS];
    Fp omega_shifts[NUM_COSETS]; // precomputed: g^((c+1)*ntt_idx_rev)
    uint32_t skip_domain;        // 2^l_skip
    uint32_t height;             // trace height (could be < num_x * skip_domain in lifted case)
    uint32_t buffer_stride;
    uint32_t buffer_size;
    uint32_t ntt_idx; // 0..skip_domain (thread's position within skip domain)
    uint32_t x_int;   // 0..num_x
};

// Evaluates f(g^c * omega_skip^ntt_idx, x) for all cosets c in [0, NUM_COSETS).
// Single thread handles ALL cosets in lockstep with DAG traversal.
//
// Algorithm:
// 1. Load trace value once
// 2. For identity coset (c=0 with FIRST_COSET_IS_IDENTITY or skip_ntt): use trace value directly
// 3. iNTT once -> save coefficient in register (shared across remaining cosets)
// 4. For each remaining coset c:
//    - Copy coefficient, multiply by omega_shifts[c]
//    - Forward NTT
//    - Store result[c]
//
// For NEEDS_SHMEM=false (l_skip <= 5, skip_domain <= 32):
// - All NTT operations use warp shuffles only
// - No shared memory needed, coefficient stays in register
// - This is the priority case for latency
//
// Template params:
// - FIRST_COSET_IS_IDENTITY: compile-time flag for lockstep kernel when coset 0 has shift=1
// Runtime params:
// - skip_ntt: runtime flag for coset-parallel kernel when processing identity coset
template <uint32_t NUM_COSETS, bool NEEDS_SHMEM, bool FIRST_COSET_IS_IDENTITY = false>
__device__ __forceinline__ void ntt_coset_interpolate(
    Fp *__restrict__ results,     // output [NUM_COSETS]
    const Fp *__restrict__ evals, // must have length height = num_x * skip_domain
    const Fp *omega_shifts,       // [NUM_COSETS] precomputed shifts
    Fp *__restrict__ ntt_buffer,  // shared memory for NTT scratch (unused when !NEEDS_SHMEM)
    uint32_t ntt_idx,
    uint32_t x_int,
    uint32_t skip_domain,
    uint32_t height,
    uint8_t offset,
    bool skip_ntt = false // Runtime flag for identity coset (coset-parallel kernel)
) {
    uint32_t const l_skip = __ffs(skip_domain) - 1;
    uint32_t const base = x_int * skip_domain;

#ifdef CUDA_DEBUG
    assert(ntt_idx < skip_domain);
#endif

    // All threads load trace value from global memory (wrap by height for rotation)
    // NOTE: `height` must be a power of two throughout this code path.
    // Many other computations (e.g. `__ffs(height) - 1`) already rely on this.
    // Use bitmask instead of `%` to avoid the expensive integer remainder in the hot path.
#ifdef CUDA_DEBUG
    assert(height && !(height & (height - 1)));
#endif
    uint32_t const idx = (base + ntt_idx + offset) & (height - 1);
    Fp coeff = evals[idx];

    // Runtime skip path for coset-parallel identity coset
    if (skip_ntt) {
        results[0] = coeff;
        return;
    }

    // Compile-time handling of identity coset for lockstep kernel
    if constexpr (FIRST_COSET_IS_IDENTITY) {
        // Coset 0 with shift=1: result is trace value directly (skip all NTT)
        results[0] = coeff;
    }

    // iNTT for remaining cosets (or all if no identity first)
    constexpr uint32_t start_c = FIRST_COSET_IS_IDENTITY ? 1 : 0;
    if constexpr (start_c < NUM_COSETS) {
        // iNTT once (shared across all cosets that need it)
        if constexpr (NEEDS_SHMEM) {
            ntt_buffer[ntt_idx] = coeff;
            __syncthreads();
        }
        device_ntt::ntt_natural_to_bitrev<true, NEEDS_SHMEM>(coeff, ntt_buffer, ntt_idx, l_skip);

        // Save coefficient in dedicated register - critical for NEEDS_SHMEM=false
        Fp const saved_coeff = coeff;

        // For each coset: shift + forward NTT (sequential, reusing ntt_buffer)
#pragma unroll
        for (uint32_t c = start_c; c < NUM_COSETS; c++) {
            Fp shifted = saved_coeff * omega_shifts[c];
            // For both possibilities of NEEDS_SHMEM, this function starts from the register value `shifted` and then overwrites shared `ntt_buffer` in every location before syncthreads. Hence we don't need a sync between calls to iNTT and cosetNTTs.
            device_ntt::ntt_bitrev_to_natural<false, NEEDS_SHMEM>(shifted, ntt_buffer, ntt_idx, l_skip);
            results[c] = shifted;
        }
    }
}

// NTT-based DAG entry evaluation for all cosets.
// Returns results for all NUM_COSETS cosets simultaneously.
// All threads in the block must call this together (for __syncthreads in NTT when NEEDS_SHMEM).
//
// Template params:
// - FIRST_COSET_IS_IDENTITY: compile-time flag for lockstep kernel when coset 0 has shift=1
// Runtime params:
// - skip_ntt: runtime flag for coset-parallel kernel when processing identity coset
template <uint32_t NUM_COSETS, bool NEEDS_SHMEM, bool FIRST_COSET_IS_IDENTITY = false>
__device__ __forceinline__ void ntt_eval_dag_entry(
    Fp *__restrict__ results, // output [NUM_COSETS]
    const SourceInfo &src,
    const NttEvalContext<NUM_COSETS> &ctx,
    bool skip_ntt = false // Runtime flag for identity coset
) {
    switch (src.type) {
    case ENTRY_PREPROCESSED: {
        const Fp *col = ctx.preprocessed + ctx.height * src.index;
        ntt_coset_interpolate<NUM_COSETS, NEEDS_SHMEM, FIRST_COSET_IS_IDENTITY>(
            results,
            col,
            ctx.omega_shifts,
            ctx.ntt_buffer,
            ctx.ntt_idx,
            ctx.x_int,
            ctx.skip_domain,
            ctx.height,
            src.offset,
            skip_ntt
        );
        return;
    }
    case ENTRY_MAIN: {
        auto main_ptr = ctx.main_parts[src.part];
        const Fp *col = main_ptr + ctx.height * src.index;
        ntt_coset_interpolate<NUM_COSETS, NEEDS_SHMEM, FIRST_COSET_IS_IDENTITY>(
            results,
            col,
            ctx.omega_shifts,
            ctx.ntt_buffer,
            ctx.ntt_idx,
            ctx.x_int,
            ctx.skip_domain,
            ctx.height,
            src.offset,
            skip_ntt
        );
        return;
    }
    case ENTRY_PUBLIC: {
        Fp val = ctx.public_values[src.index];
#pragma unroll
        for (uint32_t c = 0; c < NUM_COSETS; c++) {
            results[c] = val;
        }
        return;
    }
    case SRC_CONSTANT: {
        Fp val = Fp(src.index);
#pragma unroll
        for (uint32_t c = 0; c < NUM_COSETS; c++) {
            results[c] = val;
        }
        return;
    }
    case SRC_INTERMEDIATE:
#ifdef CUDA_DEBUG
        assert(ctx.buffer_size > 0);
        assert(src.index < ctx.buffer_size);
#endif
        // Intermediate buffer layout: [buffer_size][NUM_COSETS] per thread
#pragma unroll
        for (uint32_t c = 0; c < NUM_COSETS; c++) {
            results[c] = ctx.inter_buffer[src.index * ctx.buffer_stride + c];
        }
        return;
    case SRC_IS_FIRST:
#pragma unroll
        for (uint32_t c = 0; c < NUM_COSETS; c++) {
            results[c] = ctx.is_first[c];
        }
        return;
    case SRC_IS_LAST:
#pragma unroll
        for (uint32_t c = 0; c < NUM_COSETS; c++) {
            results[c] = ctx.is_last[c];
        }
        return;
    case SRC_IS_TRANSITION:
#pragma unroll
        for (uint32_t c = 0; c < NUM_COSETS; c++) {
            results[c] = ctx.is_transition[c];
        }
        return;
    default:
        assert(false);
    }
#pragma unroll
    for (uint32_t c = 0; c < NUM_COSETS; c++) {
        results[c] = Fp::zero();
    }
}

} // namespace symbolic_dag
