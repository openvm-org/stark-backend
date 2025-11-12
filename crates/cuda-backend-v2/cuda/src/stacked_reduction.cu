#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "sumcheck.cuh"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector_types.h>

namespace {

constexpr uint32_t UNI_THREADS = 32;
constexpr uint32_t CUBE_THREADS = 16;
constexpr uint32_t MAX_GRID_DIM = 65535u;
constexpr int S_DEG = 2;

} // namespace

struct UnstackedSlice {
    uint32_t commit_idx;
    uint32_t stacked_row_idx;
    uint32_t stacked_col_idx;
    uint32_t log_height;
};

struct Round0UniPacket {
    FpExt eq_uni;
    FpExt k_rot_0;
    FpExt k_rot_1;
};

__device__ __forceinline__ FpExt get_eq_cube(const FpExt *eq_r_ns, uint32_t cube_size, uint32_t x) {
    return eq_r_ns[cube_size + x];
}

__device__ __forceinline__ uint32_t rot_prev(uint32_t x_int, uint32_t cube_size) {
    return x_int == 0 ? cube_size - 1 : x_int - 1;
}

__device__ __forceinline__ Fp eq1(Fp x, Fp y) { return Fp::one() - x - y + Fp(2) * x * y; }

// Evaluate the `W` polynomial on PLE evaluations on the expanded domain, where `W` is specific to the stacked reduction.
__device__ __forceinline__ FpExt stacked_reduction_round0_eval(
    const Fp *q_upsampled,
    const FpExt *eq_r_ns,
    UnstackedSlice unstacked_slice,
    const FpExt *lambda_pows,
    const Round0UniPacket *z_packets,
    uint32_t upsampled_height,
    uint32_t log_domain_size,
    uint32_t l_skip,
    uint32_t window_idx,
    uint32_t num_x,
    uint32_t z_int,
    uint32_t x_int
) {
    uint32_t col_idx = unstacked_slice.stacked_col_idx;
    // ASSUME: unstacked_slice.stacked_row_idx % (2^l_skip) == 0
    uint32_t row_start = unstacked_slice.stacked_row_idx << (log_domain_size - l_skip);

    FpExt eq_cube = get_eq_cube(eq_r_ns, num_x, x_int);
    FpExt k_rot_cube = get_eq_cube(eq_r_ns, num_x, rot_prev(x_int, num_x));

    uint32_t row_idx = row_start + (x_int << log_domain_size) + z_int;

    FpExt eq_uni = z_packets[z_int].eq_uni;
    FpExt k_rot_0 = z_packets[z_int].k_rot_0;
    FpExt k_rot_1 = z_packets[z_int].k_rot_1;

    FpExt eq = eq_uni * eq_cube;
    FpExt k_rot = k_rot_0 * eq_cube + k_rot_1 * (k_rot_cube - eq_cube);

    uint32_t upsampled_idx = col_idx * upsampled_height + row_idx;
    Fp q = q_upsampled[upsampled_idx];

    return (lambda_pows[2 * window_idx] * eq + lambda_pows[2 * window_idx + 1] * k_rot) * q;
}

// Each block covers a tile of z-values (threadIdx.x) and collaborates across threadIdx.y
// to sum over x for a single (window_idx, z_int) pair. Blocks stride over window_idx via gridDim.y.
//
// See Rust doc comments for more details.
__global__ void stacked_reduction_round0_block_sum_kernel(
    const Fp *const *q_upsampled_ptr,     // pointers to upsampled matrices, one per [commit_idx]
    const FpExt *eq_r_ns,                 // pointer to EqEvalSegments
    const UnstackedSlice *unstacked_cols, // pointer to unstacked_cols at window start
    const FpExt *lambda_pows,             // pointer to lambda_pows at window start
    const Round0UniPacket *z_packets,
    FpExt *
        block_sums, // size = [gridDim.z][gridDim.y][domain_size] for [window_idx_base][blockIdx.y][z_int]
    uint32_t upsampled_height, // height of q_upsampled
    uint32_t l_skip,
    uint32_t log_domain_size,
    uint32_t window_len,
    uint32_t num_x,
    uint32_t domain_size
) {
    extern __shared__ char smem[];
    FpExt *shared = reinterpret_cast<FpExt *>(smem);

    uint32_t z_int = blockIdx.x * blockDim.x + threadIdx.x;
    if (z_int >= domain_size) {
        return;
    }

    // Map phase: compute local sum by striding over window and hypercube
    FpExt local_sum = FpExt(0);
    uint32_t window_idx_base = blockIdx.z;
    uint32_t x_int_base = blockIdx.y * blockDim.y + threadIdx.y;
    // We sum over window at a stride, where stride = gridDim.z is tuned based on compute / memory
    for (uint32_t window_idx = window_idx_base; window_idx < window_len; window_idx += gridDim.z) {
        UnstackedSlice unstacked_slice = unstacked_cols[window_idx];
        const Fp *q_upsampled = q_upsampled_ptr[unstacked_slice.commit_idx];

        // We divide the hypercube (`num_x` points) across the grid y-dimension
        for (uint32_t x_int = x_int_base; x_int < num_x; x_int += blockDim.y * gridDim.y) {
            local_sum += stacked_reduction_round0_eval(
                q_upsampled,
                eq_r_ns,
                unstacked_slice,
                lambda_pows,
                z_packets,
                upsampled_height,
                log_domain_size,
                l_skip,
                window_idx,
                num_x,
                z_int,
                x_int
            );
        }
    }

    // Reduce phase: reduce all threadIdx.y in the same block, keeping z_int independent
    shared[threadIdx.y * blockDim.x + threadIdx.x] = local_sum;
    __syncthreads();

    if (threadIdx.y == 0) {
        FpExt tile_sum = shared[threadIdx.x];
        for (int lane = 1; lane < blockDim.y; ++lane) {
            tile_sum += shared[lane * blockDim.x + threadIdx.x];
        }
        size_t window_offset = window_idx_base * gridDim.y * domain_size;
        block_sums[window_offset + blockIdx.y * domain_size + z_int] = tile_sum;
    }
    __syncthreads();
}

// Triangular sweep
__global__ void initialize_k_rot_from_eq_segments_kernel(
    const FpExt *eq_r_ns,
    FpExt *k_rot_ns,
    FpExt k_rot_uni_0,
    FpExt k_rot_uni_1
) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = blockIdx.y;
    uint32_t num_x = 1 << n;

    if (x >= num_x)
        return;

    auto eq_cube = get_eq_cube(eq_r_ns, num_x, x);
    auto k_rot_cube = get_eq_cube(eq_r_ns, num_x, rot_prev(x, num_x));

    k_rot_ns[num_x + x] = k_rot_uni_0 * eq_cube + k_rot_uni_1 * (k_rot_cube - eq_cube);
}

// Assumes we are not in degenerate case, in particular n = n_lift > 0
__global__ void stacked_reduction_sumcheck_mle_round_kernel(
    const FpExt *const *q_evals, // pointers to matrices of same height, one per [commit_idx]
    const FpExt *eq_r_ns,        // pointer to EqEvalSegments
    const FpExt *k_rot_ns,       // pointer to `k_rot` segments
    const UnstackedSlice *unstacked_cols, // pointer to unstacked_cols at window start
    const FpExt *lambda_pows,             // pointer to lambda_pows at window start
    FpExt *block_sums,
    uint32_t q_height, // height of each matrix in q_evals
    uint32_t window_len,
    uint32_t num_y
) {
    extern __shared__ char smem[];
    FpExt *shared = (FpExt *)smem;

    // Map phase: compute local sum by striding over window
    FpExt local_sums[S_DEG];
// Initialize accumulators
#pragma unroll
    for (int i = 0; i < 2; i++) {
        local_sums[i] = FpExt(0);
    }

    uint32_t window_idx_base = blockIdx.y;
    uint32_t y_int = blockIdx.x * blockDim.x + threadIdx.x;
    if (y_int >= num_y)
        return;
    uint32_t num_evals = num_y * 2;

    // We sum over window at a stride, where stride = gridDim.y is tuned based on compute / memory
    // Currently assumes blockDim.y = 1
    for (uint32_t window_idx = window_idx_base; window_idx < window_len; window_idx += gridDim.y) {
        UnstackedSlice s = unstacked_cols[window_idx];
        const FpExt *q = q_evals[s.commit_idx];

        auto log_height = s.log_height;
        auto col_idx = s.stacked_col_idx;
        auto row_start = (s.stacked_row_idx >> log_height) * num_evals;
        auto q_offset = col_idx * q_height + row_start;

        auto q_0 = q[q_offset + (y_int << 1)];
        auto q_1 = q[q_offset + (y_int << 1) + 1];
        auto q_c1 = q_1 - q_0;

        auto eq_0 = get_eq_cube(eq_r_ns, num_evals, y_int << 1);
        auto eq_1 = get_eq_cube(eq_r_ns, num_evals, (y_int << 1) | 1);
        auto eq_c1 = eq_1 - eq_0;
        auto k_rot_0 = get_eq_cube(k_rot_ns, num_evals, y_int << 1);
        auto k_rot_1 = get_eq_cube(k_rot_ns, num_evals, (y_int << 1) | 1);
        auto k_rot_c1 = k_rot_1 - k_rot_0;

        for (int x_int = 1; x_int <= S_DEG; ++x_int) {
            Fp x = Fp(x_int);
            auto q_x = q_0 + q_c1 * x;
            auto eq = eq_0 + eq_c1 * x;
            auto k_rot = k_rot_0 + k_rot_c1 * x;

            local_sums[x_int - 1] +=
                (lambda_pows[2 * window_idx] * eq + lambda_pows[2 * window_idx + 1] * k_rot) * q_x;
        }
    }

    // Reduce phase: for each x_int
    for (int idx = 0; idx < S_DEG; idx++) {
        FpExt reduced = sumcheck::block_reduce_sum(local_sums[idx], shared);

        if (threadIdx.x == 0) {
            block_sums[(window_idx_base * gridDim.x + blockIdx.x) * S_DEG + idx] = reduced;
        }
        __syncthreads(); // Needed before reusing shared memory
    }
}

// Degenerate case
// PERF: uses single thread. We could parallelize across columns (window_idx) if needed.
__global__ void stacked_reduction_sumcheck_mle_round_degenerate_kernel(
    const FpExt *const *q_evals, // pointers to matrices of same height, one per [commit_idx]
    const FpExt *eq_ub_ptr,      // pointer to `eq_ub_per_trace` at window start
    FpExt eq_r,                  // pointer to EqEvalSegments
    FpExt k_rot_r,               // pointer to `k_rot` segments
    const UnstackedSlice *unstacked_cols, // pointer to unstacked_cols at window start
    const FpExt *lambda_pows,             // pointer to lambda_pows at window start
    FpExt *output,
    uint32_t q_height, // height of each matrix in q_evals
    uint32_t window_len,
    uint32_t shift_factor // = l_skip + round
) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    FpExt local_sums[S_DEG];
#pragma unroll
    for (int i = 0; i < S_DEG; i++) {
        local_sums[i] = FpExt(0);
    }

    for (uint32_t window_idx = 0; window_idx < window_len; ++window_idx) {
        UnstackedSlice s = unstacked_cols[window_idx];
        const FpExt *q = q_evals[s.commit_idx];

        auto col_idx = s.stacked_col_idx;
        auto row_idx = s.stacked_row_idx;
        auto row_start = (row_idx >> shift_factor) << 1;
        auto q_offset = col_idx * q_height + row_start;

        auto q_0 = q[q_offset];
        auto q_1 = q[q_offset + 1];
        auto q_c1 = q_1 - q_0;

        uint32_t b_bool = (row_idx >> (shift_factor - 1)) & 1;
        Fp b = Fp(b_bool);

        auto eq_ub = eq_ub_ptr[window_idx];

        for (int x_int = 1; x_int <= S_DEG; ++x_int) {
            Fp x = Fp(x_int);
            auto eq_ub_x = eq_ub * eq1(x, b);
            auto eq = eq_r * eq_ub_x;
            auto k_rot = k_rot_r * eq_ub_x;

            auto q_x = q_0 + q_c1 * x;

            local_sums[x_int - 1] +=
                (lambda_pows[2 * window_idx] * eq + lambda_pows[2 * window_idx + 1] * k_rot) * q_x;
        }
    }

#pragma unroll
    for (int i = 0; i < S_DEG; i++) {
        output[i] = local_sums[i];
    }
}

// ============================================================================
// LAUNCHERS
// ============================================================================

// (Not a launcher) Utility function to calculate required size of temp buffer.
// Required length of *block_sums in FpExt elements
extern "C" uint32_t _stacked_reduction_r0_required_temp_buffer_size(
    uint32_t domain_size,
    uint32_t num_x,
    uint16_t thread_window_stride
) {
    assert(thread_window_stride <= MAX_GRID_DIM);
    uint32_t desired_grid_y = static_cast<uint32_t>(div_ceil(num_x, CUBE_THREADS));
    uint32_t grid_y = std::min(desired_grid_y, MAX_GRID_DIM);
    uint32_t grid_z = static_cast<uint32_t>(thread_window_stride);
    return grid_z * grid_y * domain_size;
}

extern "C" int _stacked_reduction_sumcheck_round0(
    const Fp *const *q_upsampled_ptr,
    const FpExt *eq_r_ns,
    const UnstackedSlice *unstacked_cols,
    const FpExt *lambda_pows,
    const Round0UniPacket *z_packets,
    FpExt *block_sums,
    FpExt *output, // length should be domain_size
    uint32_t upsampled_height,
    uint32_t log_domain_size,
    uint32_t l_skip,
    uint32_t window_len,
    uint32_t num_x,
    uint16_t thread_window_stride // how many elements in window to sum in one thread
) {
    static_assert(UNI_THREADS * CUBE_THREADS <= 1024, "Threads per block exceeds 1024");
    dim3 block(UNI_THREADS, CUBE_THREADS, 1);

    uint32_t domain_size = 1u << log_domain_size;
    uint32_t grid_x = static_cast<uint32_t>(div_ceil(domain_size, block.x));
    // num_x ~ 2^20, so desired_grid_y may be >= 2^16
    uint32_t desired_grid_y = static_cast<uint32_t>(div_ceil(num_x, block.y));
    uint32_t grid_y = std::min(desired_grid_y, MAX_GRID_DIM);
    uint32_t grid_z = static_cast<uint32_t>(thread_window_stride);
    dim3 grid(grid_x, grid_y, grid_z);

    size_t shmem_bytes = sizeof(FpExt) * block.x * block.y;

    stacked_reduction_round0_block_sum_kernel<<<grid, block, shmem_bytes>>>(
        q_upsampled_ptr,
        eq_r_ns,
        unstacked_cols,
        lambda_pows,
        z_packets,
        block_sums,
        upsampled_height,
        l_skip,
        log_domain_size,
        window_len,
        num_x,
        domain_size
    );

    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Launch final reduction kernel - reads from block_sums, writes to output
    auto num_blocks = grid.y * grid.z;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    // d = domain_size, WD = 1 so we have gridDim.x = domain_size
    sumcheck::final_reduce_block_sums<<<domain_size, reduce_block, reduce_shmem>>>(
        block_sums, output, num_blocks
    );

    return CHECK_KERNEL();
}

extern "C" int _initialize_k_rot_from_eq_segments(
    const FpExt *eq_r_ns,
    FpExt *k_rot_ns,
    FpExt k_rot_uni_0,
    FpExt k_rot_uni_1,
    uint32_t max_n
) {
    auto [grid, block] = kernel_launch_params(1 << max_n);
    grid.y = max_n + 1;

    initialize_k_rot_from_eq_segments_kernel<<<grid, block>>>(
        eq_r_ns, k_rot_ns, k_rot_uni_0, k_rot_uni_1
    );

    return CHECK_KERNEL();
}

extern "C" uint32_t _stacked_reduction_mle_required_temp_buffer_size(
    uint32_t num_y,
    uint16_t thread_window_stride
) {
    assert(thread_window_stride <= MAX_GRID_DIM);
    auto [grid, block] = kernel_launch_params(num_y, 512);
    grid.y = static_cast<uint32_t>(thread_window_stride);
    return grid.y * grid.x * S_DEG;
}

extern "C" int _stacked_reduction_sumcheck_mle_round(
    const FpExt *const *q_evals,
    const FpExt *eq_r_ns,
    const FpExt *k_rot_ns,
    const UnstackedSlice *unstacked_cols,
    const FpExt *lambda_pows,
    FpExt *block_sums,
    FpExt *output,
    uint32_t q_height,
    uint32_t window_len,
    uint32_t num_y,
    uint16_t thread_window_stride // how many elements in window to sum in one thread
) {
    // PERF[jpw]: we could thread over columns too
    // NOTE: update above temp_buffer_size function if launch params change
    auto [grid, block] = kernel_launch_params(num_y, 512);
    grid.y = static_cast<uint32_t>(thread_window_stride);
    unsigned int num_warps = (block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t shmem_bytes = std::max(1u, num_warps) * sizeof(FpExt);

    stacked_reduction_sumcheck_mle_round_kernel<<<grid, block, shmem_bytes>>>(
        q_evals,
        eq_r_ns,
        k_rot_ns,
        unstacked_cols,
        lambda_pows,
        block_sums,
        q_height,
        window_len,
        num_y
    );

    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    auto num_blocks = grid.x * grid.y;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    // There are s_deg = 2 final outputs, so gridDim.x = 2 for exactly 2 blocks
    sumcheck::static_final_reduce_block_sums<S_DEG>
        <<<S_DEG, reduce_block, reduce_shmem>>>(block_sums, output, num_blocks);

    return CHECK_KERNEL();
}

extern "C" int _stacked_reduction_sumcheck_mle_round_degenerate(
    const FpExt *const *q_evals,
    const FpExt *eq_ub_ptr,
    FpExt eq_r,
    FpExt k_rot_r,
    const UnstackedSlice *unstacked_cols,
    const FpExt *lambda_pows,
    FpExt *output,
    uint32_t q_height,
    uint32_t window_len,
    uint32_t l_skip,
    uint32_t round
) {
    auto shift_factor = l_skip + round;
    stacked_reduction_sumcheck_mle_round_degenerate_kernel<<<1, 1>>>(
        q_evals,
        eq_ub_ptr,
        eq_r,
        k_rot_r,
        unstacked_cols,
        lambda_pows,
        output,
        q_height,
        window_len,
        shift_factor
    );

    return CHECK_KERNEL();
}
