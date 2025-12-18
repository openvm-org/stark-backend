#pragma once

#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector_types.h>

// Launch strategy: 2D grid (grid.x blocks over x_int, grid.y over z_int), 1D block over x threads.
// Each thread may stride over multiple x_int values to stay within max_temp_bytes for intermediates.
namespace align_z_round0_config {

inline std::pair<dim3, dim3> eval_constraints_launch_params(
    uint32_t buffer_size,
    uint32_t large_domain, // range of z_int
    uint32_t num_x,        // range of x_int
    size_t max_temp_bytes,
    uint32_t buffer_threshold, // threshold for switching intermediate buffer to global memory
    uint32_t max_threads_per_block
) {
    // Decide how many threads each block needs.
    uint32_t threads_per_block =
        std::min(std::max(num_x, static_cast<uint32_t>(WARP_SIZE)), max_threads_per_block);

    // If possible, we would like to have each thread take care of a single row (i.e.
    // x_int). To do this, however, we need to ensure that the intermediate buffer size
    // is less than max_temp_bytes where each thread needs buffer_size Fp slots, which
    // are 4 bytes each.
    uint32_t total_pairs = num_x * large_domain;
    uint32_t desired_intermed_capacity_bytes =
        buffer_size <= buffer_threshold ? 0 : total_pairs * buffer_size * sizeof(Fp);

    uint32_t num_x_per_thread =
        std::max(div_ceil(desired_intermed_capacity_bytes, max_temp_bytes), static_cast<size_t>(1));
    uint32_t grid_x = div_ceil(div_ceil(num_x, num_x_per_thread), threads_per_block);

    dim3 grid(grid_x, large_domain);
    dim3 block(threads_per_block);

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
    (void)block;
    return grid.x * grid.y;
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
    return grid.x * grid.y * block.x * buffer_size;
}

} // namespace align_z_round0_config

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
