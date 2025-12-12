#pragma once

#include "codec.cuh"
#include "fp.h"
#include <cassert>
#include <cstdint>

namespace constraint_evaluation {
inline constexpr uint32_t BUFFER_THRESHOLD = 16;
inline constexpr uint32_t TASK_SIZE = 65536;

inline uint32_t get_launcher_count(uint32_t buffer_size, uint32_t height) {
    return buffer_size > BUFFER_THRESHOLD ? TASK_SIZE : height;
}
} // namespace constraint_evaluation

namespace interaction_evaluation {
inline constexpr uint32_t BUFFER_THRESHOLD = 10;
inline constexpr uint32_t TASK_SIZE = 65536;

inline uint32_t get_launcher_count(uint32_t buffer_size, uint32_t height) {
    return buffer_size > BUFFER_THRESHOLD ? TASK_SIZE : height;
}
} // namespace interaction_evaluation

namespace symbolic_dag {

// Context for evaluating DAG entries corresponding to (z, \vec x) where `z` is some point in enlarged NTT domain (subgroup of F) and `\vec x` is point on hypercube. Evaluation will directly perform barycentric interpolation to compute value of prismalinear polynomial from its evaluations on univariate skip domain `D`.
struct DagPrismEvalContext {
    const Fp *__restrict__ preprocessed;
    const Fp *const *__restrict__ main_parts;
    const Fp *__restrict__ public_values;
    const Fp *__restrict__ omega_skip_pows;       // [skip_domain]
    const Fp *__restrict__ inv_lagrange_denoms_z; // [skip_domain]
    Fp *__restrict__ inter_buffer;
    Fp is_first;
    Fp is_last;
    uint32_t skip_domain; // 2^l_skip
    uint32_t num_x;
    uint32_t height; // <= num_x * skip_domain (could be < in lifted case)
    uint32_t buffer_stride;
    uint32_t buffer_size;
    uint32_t z_int; // 0..large_domain
    uint32_t x_int; // 0..num_x
    uint32_t expansion_factor;
};

// inv_lagrange_denoms_z[i] = ((z^skip_domain - 1) / skip_domain) * (z - omega_skip_pows[i])^{-1}
// assumes offset < skip_domain
__device__ __forceinline__ Fp barycentric_interpolate(
    const Fp *__restrict__ evals, // must have length height = num_x * skip_domain
    const DagPrismEvalContext &ctx,
    uint8_t offset
) {
    auto skip_domain = ctx.skip_domain;
    auto base = ctx.x_int * skip_domain;
    if (ctx.z_int % ctx.expansion_factor == 0) {
        auto i = ctx.z_int / ctx.expansion_factor;
        uint32_t idx = (base + i + offset) % ctx.height;
        return evals[idx];
    }
    Fp eval = Fp(0);
    for (int i = 0; i < skip_domain; i++) {
        uint32_t idx = (base + i + offset) % ctx.height;
        eval += evals[idx] * ctx.omega_skip_pows[i] * ctx.inv_lagrange_denoms_z[i];
    }
    return eval;
}

__device__ __forceinline__ Fp
bary_eval_dag_entry(const SourceInfo &src, const DagPrismEvalContext &ctx) {
    switch (src.type) {
    case ENTRY_PREPROCESSED: {
        const Fp *col = ctx.preprocessed + ctx.height * src.index;
        return barycentric_interpolate(col, ctx, src.offset);
    }
    case ENTRY_MAIN: {
        auto main_ptr = ctx.main_parts[src.part];
        const Fp *col = main_ptr + ctx.height * src.index;
        return barycentric_interpolate(col, ctx, src.offset);
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
