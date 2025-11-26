#pragma once

#include "codec.cuh"
#include "fp.h"
#include "fpext.h"
#include "matrix.cuh"
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

struct DagEvaluationContext {
    uint32_t row_index;
    const Fp *__restrict__ d_selectors;
    const MainMatrixPtrs<Fp> *__restrict__ d_main;
    uint32_t height;
    uint32_t selectors_width;
    const Fp *__restrict__ d_preprocessed;
    uint32_t preprocessed_air_width;
    const FpExt *__restrict__ d_eq_z;
    const FpExt *__restrict__ d_eq_x;
    const Fp *__restrict__ d_public;
    uint32_t public_len;
    FpExt *__restrict__ inter_buffer;
    uint32_t buffer_stride;
    uint32_t buffer_size;
    uint32_t large_domain;
};

__device__ __forceinline__ FpExt evaluate_dag_entry(
    const SourceInfo &src,
    const DagEvaluationContext &ctx,
    const FpExt *d_challenges = nullptr // Optional: for ENTRY_CHALLENGE in interactions
) {
    (void)ctx.large_domain;
    switch (src.type) {
    case ENTRY_PREPROCESSED: {
        if (ctx.d_preprocessed == nullptr) {
            return FpExt(Fp::zero());
        }
        const auto stride = ctx.height * ctx.preprocessed_air_width;
        const Fp *matrix = ctx.d_preprocessed + stride * src.offset;
        return FpExt(matrix[ctx.height * src.index + ctx.row_index]);
    }
    case ENTRY_MAIN: {
        auto main_ptr = ctx.d_main[src.part];
        const auto stride = ctx.height * main_ptr.air_width;
        const Fp *matrix = main_ptr.data + stride * src.offset;
        return FpExt(matrix[ctx.height * src.index + ctx.row_index]);
    }
    case ENTRY_CHALLENGE:
        assert(d_challenges != nullptr);
        return d_challenges[src.index];
    case ENTRY_PERMUTATION:
    case ENTRY_EXPOSED:
        return FpExt(Fp::zero());
    case ENTRY_PUBLIC: {
        if (src.index >= ctx.public_len || ctx.d_public == nullptr) {
            return FpExt(Fp::zero());
        }
        return FpExt(ctx.d_public[src.index]);
    }
    case SRC_CONSTANT:
        return FpExt(Fp(src.index));
    case SRC_INTERMEDIATE:
        if (ctx.inter_buffer == nullptr || ctx.buffer_size == 0) {
            return FpExt(Fp::zero());
        }
        if (src.index >= ctx.buffer_size) {
            return FpExt(Fp::zero());
        }
        return ctx.inter_buffer[src.index * ctx.buffer_stride];
    case SRC_IS_FIRST: {
        if (ctx.height == 0 || ctx.selectors_width == 0) {
            return FpExt(Fp::zero());
        }
        uint32_t row = ctx.row_index % ctx.height;
        return FpExt(ctx.d_selectors[row]);
    }
    case SRC_IS_LAST: {
        if (ctx.height == 0 || ctx.selectors_width < 3) {
            return FpExt(Fp::zero());
        }
        uint32_t row = ctx.row_index % ctx.height;
        return FpExt(ctx.d_selectors[ctx.height * 2 + row]);
    }
    case SRC_IS_TRANSITION: {
        if (ctx.height == 0 || ctx.selectors_width < 2) {
            return FpExt(Fp::zero());
        }
        uint32_t row = ctx.row_index % ctx.height;
        return FpExt(ctx.d_selectors[ctx.height + row]);
    }
    }
    return FpExt(Fp::zero());
}

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
