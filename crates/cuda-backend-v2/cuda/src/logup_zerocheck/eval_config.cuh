#pragma once

#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector_types.h>

// Shared implementation for coset-based round0 launch configs.
// Template parameter COSET_PARALLEL controls grid organization:
// - false (lockstep): grid.y = 1, each thread handles all cosets via NUM_COSETS template
// - true (coset-parallel): grid.y = num_cosets, each block handles one coset
namespace round0_config_impl {

template <bool COSET_PARALLEL>
inline std::pair<dim3, dim3> eval_constraints_launch_params(
    uint32_t buffer_size,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t num_cosets,
    size_t max_temp_bytes,
    uint32_t buffer_threshold,
    uint32_t max_threads_per_block
) {
    // ASSERTION: full skip domain must fit in one block
    assert(skip_domain <= 1024 && "skip_domain exceeds CUDA max threads per block (1024)");
    auto max_threads = std::max(skip_domain, max_threads_per_block);
    auto [grid, block] = kernel_launch_params(skip_domain * num_x, max_threads);

    if constexpr (COSET_PARALLEL) {
        grid.y = num_cosets; // 2D grid for coset-parallel mode
    }
    // grid.y = 1 implicitly for lockstep mode

    // Both modes have the same total buffer size:
    // - Lockstep: grid.x * block.x * num_cosets * buffer_size
    // - Coset-parallel: grid.x * grid.y * block.x * buffer_size (where grid.y = num_cosets)
    size_t desired_intermed_capacity_bytes =
        buffer_size <= buffer_threshold
            ? 0
            : (size_t)grid.x * block.x * num_cosets * buffer_size * sizeof(Fp);

    uint32_t num_x_per_thread = std::max(
        div_ceil(desired_intermed_capacity_bytes, std::max(max_temp_bytes, (size_t)1)), (size_t)1
    );
    if (num_x_per_thread <= grid.x) {
        grid.x = grid.x / num_x_per_thread;
    } else {
        grid.x = 1;
        block.x = std::max(num_x / num_x_per_thread, 1u) * skip_domain;
    }

    return {grid, block};
}

template <bool COSET_PARALLEL>
inline uint32_t temp_sums_buffer_size(
    uint32_t buffer_size,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t num_cosets,
    size_t max_temp_bytes,
    uint32_t buffer_threshold,
    uint32_t threads_per_block
) {
    auto [grid, _] = eval_constraints_launch_params<COSET_PARALLEL>(
        buffer_size,
        skip_domain,
        num_x,
        num_cosets,
        max_temp_bytes,
        buffer_threshold,
        threads_per_block
    );
    // Output layout: [num_blocks][num_cosets * skip_domain]
    return grid.x * num_cosets * skip_domain;
}

template <bool COSET_PARALLEL>
inline uint32_t intermediates_buffer_size(
    uint32_t buffer_size,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t num_cosets,
    size_t max_temp_bytes,
    uint32_t buffer_threshold,
    uint32_t threads_per_block
) {
    if (buffer_size <= buffer_threshold) {
        return 0;
    }
    auto [grid, block] = eval_constraints_launch_params<COSET_PARALLEL>(
        buffer_size,
        skip_domain,
        num_x,
        num_cosets,
        max_temp_bytes,
        buffer_threshold,
        threads_per_block
    );
    // Layout: [buffer_size][num_threads][num_cosets]
    return grid.x * block.x * buffer_size * num_cosets;
}

} // namespace round0_config_impl

// Lockstep mode: 1D grid (grid.x blocks), each thread handles ALL cosets via NUM_COSETS template.
namespace coset_round0_config {

inline std::pair<dim3, dim3> eval_constraints_launch_params(
    uint32_t buffer_size,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t num_cosets,
    size_t max_temp_bytes,
    uint32_t buffer_threshold,
    uint32_t max_threads_per_block
) {
    return round0_config_impl::eval_constraints_launch_params<false>(
        buffer_size,
        skip_domain,
        num_x,
        num_cosets,
        max_temp_bytes,
        buffer_threshold,
        max_threads_per_block
    );
}

} // namespace coset_round0_config

// Coset-parallel mode: 2D grid (grid.x * grid.y where grid.y = num_cosets),
// each block handles ONE coset identified by blockIdx.y.
namespace coset_parallel_round0_config {

inline std::pair<dim3, dim3> eval_constraints_launch_params(
    uint32_t buffer_size,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t num_cosets,
    size_t max_temp_bytes,
    uint32_t buffer_threshold,
    uint32_t max_threads_per_block
) {
    return round0_config_impl::eval_constraints_launch_params<true>(
        buffer_size,
        skip_domain,
        num_x,
        num_cosets,
        max_temp_bytes,
        buffer_threshold,
        max_threads_per_block
    );
}

} // namespace coset_parallel_round0_config

// Launch strategy: linearize (x_int, z_int) into a 1D threadIdx.x to keep warps aligned on x_int.
// We tile z within a block (up to WARP_SIZE) and use the remaining threads for x; grid.x scales x
// based on max_temp_bytes, accounting for both temp sum buffer and (optional) intermediates spill.
namespace align_x_round0_config {

// Returns {zs_per_grid, zs_per_block}
// NOTE: it is best for avoiding warp divergence that `large_domain` is a multiple of `WARP_SIZE` (32).
inline std::pair<uint32_t, uint32_t> get_z_dim(uint32_t large_domain) {
    // Since we are currently not using any shared memory in relation to the `z_int` variable, the only
    // important factor is that every thread in a warp shares the same `x_int`.
    // NOTE: when `large_domain < WARP_SIZE`, there is unavoidable warp divergence, so we just optimize
    // for packing as much into the block as possible.
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
    // We will define grid = xs_per_grid * zs_per_grid, where varying xs_per_grid affects the global
    // memory usage.
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

inline size_t temp_sums_buffer_size(
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
    (void)block;
    auto [zs_per_grid, zs_per_block] = get_z_dim(large_domain);
    (void)zs_per_block;
    auto xs_per_grid = grid.x / zs_per_grid;
    return static_cast<size_t>(large_domain) * xs_per_grid;
}

inline size_t intermediates_buffer_size(
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
    size_t task_stride = static_cast<size_t>(xs_per_grid) * xs_per_block * large_domain;
    return task_stride * buffer_size;
}

} // namespace align_x_round0_config

namespace mle_rounds_config {
inline std::pair<dim3, dim3> eval_constraints_launch_params(
    uint32_t num_x,
    uint32_t num_y,
    size_t max_threads_per_block
) {
    auto threads_per_block = std::min(std::max(WARP_SIZE, (size_t)num_y), max_threads_per_block);
    auto num_blocks_for_y = div_ceil(num_y, threads_per_block);
    dim3 grid = dim3(num_blocks_for_y, num_x);
    dim3 block = dim3(threads_per_block);
    return {grid, block};
}

inline size_t temp_sums_buffer_size(uint32_t num_x, uint32_t num_y, uint32_t threads_per_block) {
    auto [grid, block] = eval_constraints_launch_params(num_x, num_y, threads_per_block);
    (void)block;
    return static_cast<size_t>(grid.x) * grid.y;
}

inline size_t intermediates_buffer_size(
    uint32_t buffer_size,
    uint32_t num_x,
    uint32_t num_y,
    uint32_t buffer_threshold,
    uint32_t threads_per_block
) {
    if (buffer_size <= buffer_threshold) {
        return 0;
    }
    auto [grid, block] = eval_constraints_launch_params(num_x, num_y, threads_per_block);
    return static_cast<size_t>(block.x) * grid.x * grid.y * buffer_size;
}
} // namespace mle_rounds_config
