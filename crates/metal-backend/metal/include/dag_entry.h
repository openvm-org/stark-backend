// DAG node representation and evaluation context for Metal Shading Language.
//
// Based on openvm-org cuda-backend/cuda/include/dag_entry.cuh
//
// The CUDA version is heavily tied to NTT-based coset evaluation with warp
// shuffles and shared memory. This Metal version provides the core types
// and a basic evaluation function that can be used by Metal compute kernels.
// NTT-specific dispatch will be implemented separately in Metal kernel files.

#pragma once

#include <metal_stdlib>

#include "baby_bear.h"
#include "codec.h"

using namespace metal;

namespace symbolic_dag {

// ============================================================================
// Evaluation context
//
// Holds pointers to all the data needed to evaluate a DAG entry (variable
// lookup). The Metal version uses device pointers (passed as kernel arguments)
// rather than raw __restrict__ pointers.
// ============================================================================

struct EvalContext {
    const device Fp *preprocessed;       // Column-major preprocessed trace
    const device Fp *const device *main_parts; // Per-partition main trace pointers
    const device Fp *public_values;      // Public input values
    device Fp *inter_buffer;             // Intermediate value buffer
    uint32_t height;                     // Trace height (rows)
    uint32_t buffer_stride;              // Stride between coset slots in inter_buffer
    uint32_t buffer_size;                // Number of intermediate slots
};

// ============================================================================
// DAG entry evaluation (single point)
//
// Given a SourceInfo (decoded from a Rule), look up or compute the value.
// This is the "single evaluation point" version -- no coset interpolation.
// For coset-based evaluation, Metal kernels will implement their own
// dispatch using threadgroup memory and simdgroup operations.
// ============================================================================

/// Evaluate a single DAG source at a given row index.
/// Returns the Fp value for that source.
inline Fp eval_source(
    const SourceInfo src,
    const device Fp *preprocessed,
    const device Fp *const device *main_parts,
    const device Fp *public_values,
    const device Fp *inter_buffer,
    uint32_t height,
    uint32_t row_idx,
    uint32_t buffer_stride,
    Fp is_first,
    Fp is_last
) {
    switch (src.type) {
    case ENTRY_PREPROCESSED: {
        uint32_t idx = ((row_idx + src.offset) & (height - 1));
        return preprocessed[height * src.index + idx];
    }
    case ENTRY_MAIN: {
        const device Fp *col = main_parts[src.part];
        uint32_t idx = ((row_idx + src.offset) & (height - 1));
        return col[height * src.index + idx];
    }
    case ENTRY_PUBLIC: {
        return public_values[src.index];
    }
    case SRC_CONSTANT: {
        return Fp(src.index);
    }
    case SRC_INTERMEDIATE: {
        return inter_buffer[src.index * buffer_stride];
    }
    case SRC_IS_FIRST: {
        return is_first;
    }
    case SRC_IS_LAST: {
        return is_last;
    }
    case SRC_IS_TRANSITION: {
        return Fp::one() - is_last;
    }
    default:
        return Fp::zero();
    }
}

} // namespace symbolic_dag
