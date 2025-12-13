#pragma once

#include "codec.cuh"
#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector_types.h>

// for mle use
namespace constraint_evaluation {
inline constexpr uint32_t BUFFER_THRESHOLD = 16;
inline constexpr uint32_t TASK_SIZE = 65536;

inline uint32_t get_launcher_count(uint32_t buffer_size, uint32_t height) {
    return buffer_size > BUFFER_THRESHOLD ? TASK_SIZE : height;
}
} // namespace constraint_evaluation

// for mle use
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

// Kernel launch parameters and temporary buffer size calculations in the case of no upsampling and where shared memory is used to block reduce in the `x_int` variable.
// There are two variables `z_int, x_int` that threads should vary over, but we linearize them into a single 1d thread dimension for better utilization.
namespace round0_config {

// Returns {zs_per_grid, zs_per_block}
// NOTE: it is best for avoiding warp divergence that `large_domain` is a multiple of `WARP_SIZE` (32).
inline std::pair<uint32_t, uint32_t> get_z_dim(uint32_t large_domain) {
    // Since we are currently not using any shared memory in relation to the `z_int` variable, the only important factor is that every thread in a warp shares the same `x_int`.
    // NOTE: when `large_domain < WARP_SIZE`, there is unavoidable warp divergence, so we just optimize for packing as much into the block as possible.
    uint32_t zs_per_block = std::min(large_domain, static_cast<uint32_t>(WARP_SIZE));
    uint32_t zs_per_grid = div_ceil(large_domain, zs_per_block);
    return {zs_per_grid, zs_per_block};
}

inline std::pair<dim3, dim3> eval_constraints_launch_params(
    uint32_t buffer_size,
    uint32_t large_domain, // range of z_int
    uint32_t num_x,        // range of x_int
    size_t max_temp_bytes,
    uint32_t buffer_threshold, // threshold for switching intermediate buffer to global memory
    uint32_t threads_per_block
) {
    auto [zs_per_grid, zs_per_block] = get_z_dim(large_domain);
    auto xs_per_block =
        std::min(static_cast<size_t>(num_x), div_ceil(threads_per_block, zs_per_block));
    dim3 block(xs_per_block * zs_per_block);
    // We will define grid = xs_per_grid * zs_per_grid, where varying xs_per_grid affects the global memory usage.
    // `temp_sums_buffer_size` is `xs_per_grid * large_domain`
    size_t scale_factor = large_domain;
    if (buffer_size > buffer_threshold) {
        // This mean global memory is used.
        // `intermediates_buffer_size` is `xs_per_grid * xs_per_block * large_domain * buffer_size`
        scale_factor += static_cast<size_t>(xs_per_block) * large_domain * buffer_size;
    }
    size_t xs_per_grid = max_temp_bytes / (scale_factor * sizeof(FpExt));
    // Failsafe to ensure kernel doesn't fail:
    xs_per_grid = std::max(xs_per_grid, static_cast<size_t>(1));
    xs_per_grid = std::min(xs_per_grid, div_ceil(num_x, xs_per_block));
    dim3 grid(xs_per_grid * zs_per_grid);

    return {grid, block};
}

inline uint32_t temp_sums_buffer_size(
    uint32_t buffer_size,
    uint32_t large_domain,
    uint32_t num_x,
    size_t max_temp_bytes,
    uint32_t buffer_threshold,
    uint32_t threads_per_block
) {
    auto [grid, block] = eval_constraints_launch_params(
        buffer_size, large_domain, num_x, max_temp_bytes, buffer_threshold, threads_per_block
    );
    auto [zs_per_grid, zs_per_block] = get_z_dim(large_domain);
    auto xs_per_grid = grid.x / zs_per_grid;
    return large_domain * xs_per_grid;
}

inline uint32_t intermediates_buffer_size(
    uint32_t buffer_size,
    uint32_t large_domain,
    uint32_t num_x,
    size_t max_temp_bytes,
    uint32_t buffer_threshold,
    uint32_t threads_per_block
) {
    if (buffer_size <= buffer_threshold) {
        return 0;
    }
    auto [grid, block] = eval_constraints_launch_params(
        buffer_size, large_domain, num_x, max_temp_bytes, buffer_threshold, threads_per_block
    );
    auto [zs_per_grid, zs_per_block] = get_z_dim(large_domain);
    auto xs_per_block = block.x / zs_per_block;
    auto xs_per_grid = grid.x / zs_per_grid;
    uint32_t task_stride = xs_per_grid * xs_per_block * large_domain;
    return task_stride * buffer_size;
}
} // namespace round0_config
