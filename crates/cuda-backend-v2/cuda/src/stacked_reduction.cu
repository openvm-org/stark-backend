#include "device_ntt.cuh"
#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "sumcheck.cuh"
#include "utils.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <utility>
#include <vector_types.h>

using namespace device_ntt;

namespace {

constexpr uint32_t MAX_GRID_DIM = 65535u;
constexpr int S_DEG = 2;

} // namespace

struct UnstackedSlice {
    uint32_t commit_idx;
    uint32_t log_height;
    uint32_t stacked_row_idx;
    uint32_t stacked_col_idx;
};

struct Round0UniPacket {
    FpExt eq_uni;
    FpExt k_rot_0;
    FpExt k_rot_1;
};

struct UnstackedPleFoldPacket {
    const Fp *__restrict__ src;
    FpExt *__restrict__ dst;
    uint32_t height;
    uint32_t width;
};

__device__ __forceinline__ FpExt get_eq_cube(const FpExt *eq_r_ns, uint32_t cube_size, uint32_t x) {
    return eq_r_ns[cube_size + x];
}

__device__ __forceinline__ uint32_t rot_prev(uint32_t x_int, uint32_t cube_size) {
    return x_int == 0 ? cube_size - 1 : x_int - 1;
}

__device__ __forceinline__ Fp eq1(Fp x, Fp y) { return Fp::one() - x - y + Fp(2) * x * y; }

__device__ __forceinline__ Fp barycentric_interpolate_strided(
    const Fp *evals,
    uint32_t evals_len,
    uint32_t z,
    uint32_t expansion_factor,     // large_domain / skip_domain
    const Fp *inv_lagrange_denoms, // denoms specifically for z
    const Fp *omega_skip_pows,
    uint32_t skip_domain
) {
    Fp q = Fp::zero();
    uint32_t stride = evals_len >= skip_domain ? 1 : skip_domain / evals_len;
    if (z % expansion_factor == 0) {
        auto i = z / expansion_factor;
        if (i % stride == 0)
            q = evals[i / stride];
    } else {
        for (int idx = 0; idx < (skip_domain / stride); idx++) {
            q += evals[idx] * omega_skip_pows[idx * stride] * inv_lagrange_denoms[idx * stride];
        }
    }
    return q;
}

// Each block covers a tile of z-values (threadIdx.x) and collaborates across threadIdx.y
// to sum over x for a single (window_idx, z_int) pair. Blocks stride over window_idx via gridDim.y.
//
// See Rust doc comments for more details.
__global__ void stacked_reduction_round0_block_sum_kernel(
    const FpExt *__restrict__ eq_r_ns, // pointer to EqEvalSegments
    const Fp *__restrict__ trace_ptr,
    const FpExt *__restrict__ lambda_pows, // pointer to lambda_pows at window start
    const Round0UniPacket
        *__restrict__ z_packets, // pointer to eq_uni, k_rot_0, and k_rot_1 evals for z_int
    FpExt
        *__restrict__ block_sums, // [gridDim.z][gridDim.y][domain_size] for [window_base][blockIdx.y][z_int]
    uint32_t height,              // trace height
    uint32_t width,               // trace width
    uint32_t l_skip,
    uint32_t skip_mask, // 2^l_skip - 1
    uint32_t num_x,     // 1 << n_lift
    uint32_t log_stride
) {
    extern __shared__ char smem[];
    FpExt *shared_sum = reinterpret_cast<FpExt *>(smem); // FpExt[S_DEG][blockDim.x]
    Fp *shared_eval =
        reinterpret_cast<Fp *>(smem + S_DEG * blockDim.x * sizeof(FpExt)); // Fp[blockDim.x]

    bool needs_shmem = l_skip > LOG_WARP_SIZE;
    uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t z_idx = tidx & skip_mask;
    uint32_t x_int = tidx >> l_skip;
    uint32_t col_idx_base = blockIdx.y; // blockDim.y = 1
    // Each thread will handle z_int in S_DEG * z_idx..(S_DEG + 1) * z_idx
    FpExt eq[S_DEG];
    FpExt k_rot[S_DEG];
    // Compute eq, k_rot depending on (z_int, x_int)
    FpExt eq_cube = get_eq_cube(eq_r_ns, num_x, x_int);
    FpExt k_rot_cube = get_eq_cube(eq_r_ns, num_x, rot_prev(x_int, num_x));
    auto const z_idx_rev = rev_len(z_idx, l_skip);
    Fp omega_shift = pow(TWO_ADIC_GENERATORS[l_skip + 1], z_idx_rev); // 1 = log2(S_DEG)

#pragma unroll
    for (uint32_t i = 0; i < S_DEG; i++) {
        uint32_t z_int = S_DEG * z_idx + i;
        FpExt eq_uni = z_packets[z_int].eq_uni;
        FpExt k_rot_0 = z_packets[z_int].k_rot_0;
        FpExt k_rot_1 = z_packets[z_int].k_rot_1;

        eq[i] = eq_uni * eq_cube;
        k_rot[i] = k_rot_0 * eq_cube + k_rot_1 * (k_rot_cube - eq_cube);
    }
    FpExt local_sum[S_DEG];
#pragma unroll
    for (uint32_t i = 0; i < S_DEG; i++) {
        local_sum[i] = FpExt(Fp::zero());
    }

    auto stride_mask = (1u << log_stride) - 1;
    for (uint32_t col_idx = col_idx_base; col_idx < width; col_idx += gridDim.y) {

        // First, each thread reads trace_ptr and loads to shared_eval
        auto evals = trace_ptr + col_idx * height + (x_int << (l_skip - log_stride));
        Fp this_thread_value = (z_idx & stride_mask) == 0 ? evals[z_idx >> log_stride] : Fp::zero();
        // We make special use of the fact S_DEG=2:
        {
            // i = 0: it's just the eval
            auto q = this_thread_value;
            local_sum[0] +=
                (lambda_pows[2 * col_idx] * eq[0] + lambda_pows[2 * col_idx + 1] * k_rot[0]) * q;
        }
        {
            // i = 1: iNTT, shift, forward NTT
            // Note: we configured the block so all threads are active
            Fp *__restrict__ sbuf = shared_eval + ((threadIdx.x >> l_skip) << l_skip);
            if (needs_shmem) {
                sbuf[z_idx] = this_thread_value;
                __syncthreads();
            }
            // iNTT, bitrev output stored to this_thread_value
            DISPATCH_BOOL_PAIR(
                ntt_natural_to_bitrev, true, needs_shmem, this_thread_value, sbuf, z_idx, l_skip
            );
            // multiply by shift. regardless of whether shmem is needed, the input for forward NTT must be placed in `this_thread_value` for `ntt_bitrev_to_natural`
            this_thread_value *= omega_shift;
            // forward NTT of shifted coeffs
            DISPATCH_BOOL_PAIR(
                ntt_bitrev_to_natural, false, needs_shmem, this_thread_value, sbuf, z_idx, l_skip
            );
            auto q = this_thread_value;
            local_sum[1] +=
                (lambda_pows[2 * col_idx] * eq[1] + lambda_pows[2 * col_idx + 1] * k_rot[1]) * q;
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < S_DEG; i++) {
        shared_sum[i * blockDim.x + threadIdx.x] = local_sum[i];
    }
    __syncthreads();

    if ((threadIdx.x >> l_skip) == 0) {
#pragma unroll
        for (uint32_t i = 0; i < S_DEG; i++) {
            FpExt tile_sum = shared_sum[i * blockDim.x + z_idx];
            for (int lane = 1; lane < (blockDim.x >> l_skip); ++lane) {
                tile_sum += shared_sum[i * blockDim.x + (lane << l_skip) + z_idx];
            }
            block_sums
                [(col_idx_base * gridDim.x + blockIdx.x) * (S_DEG << l_skip) + S_DEG * z_idx + i] =
                    tile_sum;
        }
    }
}

// ASSUMPTION: no trace has width exceeding u16::MAX
// PERF[jpw]: the barycentric interpolation is non-coalesced since each thread reads a different memory slice
__global__ void stacked_reduction_fold_ple_kernel(
    const UnstackedPleFoldPacket *__restrict__ trace_packets,
    const Fp *__restrict__ omega_skip_pows,
    const FpExt *__restrict__ inv_lagrange_denoms,
    uint32_t skip_domain, // 2^l_skip
    uint32_t l_skip,      // log2(domain_size)
    uint16_t num_packets
) {
    uint32_t trace_idx = blockIdx.z;
    if (trace_idx >= num_packets)
        return;
    auto packet = trace_packets[trace_idx];
    uint32_t height = packet.height;
    uint32_t new_height;
    uint32_t stride;
    if (height >= skip_domain) {
        new_height = height / skip_domain;
        stride = 1;
    } else {
        new_height = 1;
        stride = skip_domain / height;
    }
    uint32_t width = packet.width;

    uint32_t new_row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t col_idx = blockIdx.y;
    if (new_row_idx >= new_height || col_idx >= width)
        return;

    auto src_len = std::min(height, skip_domain);
    auto src = packet.src + (col_idx * height + new_row_idx * src_len);
    auto dst = packet.dst + (col_idx * new_height + new_row_idx);

    // Barycentric interpolation:
    // NOTE: this is evaluation at a random out of domain point, so we should not have divide by zero in inv_lagrange_denoms
    FpExt result(Fp::zero());
    for (int idx = 0; idx < (skip_domain / stride); idx++) {
        result += src[idx] * omega_skip_pows[idx * stride] * inv_lagrange_denoms[idx * stride];
    }
    dst[0] = result;
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
// Uses warp-aggregated atomics for reduction - no shared memory or __syncthreads() needed.
__global__ void stacked_reduction_sumcheck_mle_round_kernel(
    const FpExt *__restrict__ const
        *__restrict__ q_evals,          // pointers to matrices of same height, one per [commit_idx]
    const FpExt *__restrict__ eq_r_ns,  // pointer to EqEvalSegments
    const FpExt *__restrict__ k_rot_ns, // pointer to `k_rot` segments
    const UnstackedSlice *__restrict__ unstacked_cols, // pointer to unstacked_cols at window start
    const FpExt *__restrict__ lambda_pows,             // pointer to lambda_pows at window start
    uint64_t *__restrict__ output, // [S_DEG * 4] - atomic accumulator, reduced on CPU
    uint32_t q_height,             // height of each matrix in q_evals
    uint32_t window_len,
    uint32_t num_y
) {
    extern __shared__ char smem[];
    FpExt *shared = (FpExt *)smem;
    // Map phase: compute local sum by striding over window
    FpExt local_sums[S_DEG];
// Initialize accumulators
#pragma unroll
    for (int i = 0; i < S_DEG; i++) {
        local_sums[i] = FpExt(0);
    }

    uint32_t window_idx_base = blockIdx.y;
    uint32_t y_int = blockIdx.x * blockDim.x + threadIdx.x;
    bool const active_thread = (y_int < num_y);

    if (active_thread) {
        uint32_t num_evals = num_y * 2;

        auto eq_0 = get_eq_cube(eq_r_ns, num_evals, y_int << 1);
        auto eq_1 = get_eq_cube(eq_r_ns, num_evals, (y_int << 1) | 1);
        auto eq_c1 = eq_1 - eq_0;
        auto k_rot_0 = get_eq_cube(k_rot_ns, num_evals, y_int << 1);
        auto k_rot_1 = get_eq_cube(k_rot_ns, num_evals, (y_int << 1) | 1);
        auto k_rot_c1 = k_rot_1 - k_rot_0;

        // We sum over window at a stride, where stride = gridDim.y is tuned based on compute / memory
        // Currently assumes blockDim.y = 1
        for (uint32_t window_idx = window_idx_base; window_idx < window_len;
             window_idx += gridDim.y) {
            UnstackedSlice s = unstacked_cols[window_idx];
            const FpExt *__restrict__ q = q_evals[s.commit_idx];

            auto log_height = s.log_height;
            auto col_idx = s.stacked_col_idx;
            auto row_start = (s.stacked_row_idx >> log_height) * num_evals;
            auto q_offset = col_idx * q_height + row_start;

            auto q_0 = q[q_offset + (y_int << 1)];
            auto q_1 = q[q_offset + (y_int << 1) + 1];
            auto q_c1 = q_1 - q_0;

#pragma unroll
            for (int x_int = 1; x_int <= S_DEG; ++x_int) {
                Fp x = Fp(x_int);
                auto q_x = q_0 + q_c1 * x;
                auto eq = eq_0 + eq_c1 * x;
                auto k_rot = k_rot_0 + k_rot_c1 * x;

                local_sums[x_int - 1] +=
                    (lambda_pows[2 * window_idx] * eq + lambda_pows[2 * window_idx + 1] * k_rot) *
                    q_x;
            }
        }
    }

#pragma unroll
    for (int idx = 0; idx < S_DEG; idx++) {
        FpExt reduced = sumcheck::block_reduce_sum(local_sums[idx], shared);
        if (threadIdx.x == 0) {
            sumcheck::atomic_add_fpext_to_u64(output + idx * 4, reduced);
        }
        __syncthreads();
    }
}

// Degenerate case
// Uses warp-aggregated atomics for reduction - no shared memory or __syncthreads() needed.
__global__ void stacked_reduction_sumcheck_mle_round_degenerate_kernel(
    const FpExt *__restrict__ const
        *__restrict__ q_evals, // pointers to matrices of same height, one per [commit_idx]
    const FpExt *__restrict__ eq_ub_ptr, // pointer to `eq_ub_per_trace` at window start
    FpExt eq_r,                          // pointer to EqEvalSegments
    FpExt k_rot_r,                       // pointer to `k_rot` segments
    const UnstackedSlice *__restrict__ unstacked_cols, // pointer to unstacked_cols at window start
    const FpExt *__restrict__ lambda_pows,             // pointer to lambda_pows at window start
    uint64_t *__restrict__ output, // [S_DEG * 4] - atomic accumulator, reduced on CPU
    uint32_t q_height,             // height of each matrix in q_evals
    uint32_t window_len,
    uint32_t shift_factor // = l_skip + round
) {
    extern __shared__ char smem[];
    FpExt *shared = (FpExt *)smem;

    FpExt local_sums[S_DEG];
#pragma unroll
    for (int i = 0; i < S_DEG; i++) {
        local_sums[i] = FpExt(0);
    }

    for (uint32_t window_idx = threadIdx.x; window_idx < window_len; window_idx += blockDim.x) {
        UnstackedSlice s = unstacked_cols[window_idx];
        const FpExt *__restrict__ q = q_evals[s.commit_idx];

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

#pragma unroll
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
    for (int idx = 0; idx < S_DEG; idx++) {
        FpExt reduced = sumcheck::block_reduce_sum(local_sums[idx], shared);
        if (threadIdx.x == 0) {
            sumcheck::atomic_add_fpext_to_u64(output + idx * 4, reduced);
        }
        __syncthreads();
    }
}

// ============================================================================
// LAUNCHERS
// ============================================================================

inline std::pair<dim3, dim3> stacked_reduction_round0_launch_params(
    uint32_t trace_height,
    uint32_t l_skip,
    uint16_t col_stride
) {
    uint32_t skip_domain = 1u << l_skip;
    auto lifted_height = std::max(trace_height, skip_domain);
    // Entire skip domain must fit within a block
    auto max_threads = std::max(skip_domain, 256u);
    auto [grid, block] = kernel_launch_params(lifted_height, max_threads);
    // NOTE: lifted_height = skip_domain * num_x will always be divisible by block.x since num_x is a power of 2. Therefore all threads in a block will be active.
    grid.y = static_cast<uint32_t>(col_stride);
    return {grid, block};
}

// (Not a launcher) Utility function to calculate required size of temp buffer.
// Required length of *block_sums in FpExt elements
extern "C" uint32_t _stacked_reduction_r0_required_temp_buffer_size(
    uint32_t trace_height,
    uint32_t l_skip,
    uint16_t col_stride
) {
    auto [grid, block] = stacked_reduction_round0_launch_params(trace_height, l_skip, col_stride);
    return ((grid.x * grid.y) << l_skip) * S_DEG;
}

extern "C" int _stacked_reduction_sumcheck_round0(
    const FpExt *eq_r_ns,
    const Fp *trace_ptr,
    const FpExt *lambda_pows,
    const Round0UniPacket *z_packets,
    FpExt *block_sums,
    FpExt *output, // length should be domain_size
    uint32_t trace_height,
    uint32_t trace_width,
    uint32_t l_skip,
    uint32_t num_x,
    uint16_t col_stride // how many elements in window to sum in one thread
) {
    // We currently use this hardcoded value to optimize the coset NTT
    static_assert(S_DEG == 2);
    uint32_t skip_domain = 1u << l_skip;
    uint32_t stride = std::max(skip_domain / trace_height, 1u);
    auto [grid, block] = stacked_reduction_round0_launch_params(trace_height, l_skip, col_stride);

    size_t shmem_sum_size = sizeof(FpExt) * block.x * S_DEG;
    size_t shmem_eval_size = sizeof(Fp) * block.x;
    size_t shmem_bytes = shmem_sum_size + shmem_eval_size;

    stacked_reduction_round0_block_sum_kernel<<<grid, block, shmem_bytes>>>(
        eq_r_ns,
        trace_ptr,
        lambda_pows,
        z_packets,
        block_sums,
        trace_height,
        trace_width,
        l_skip,
        (1u << l_skip) - 1,
        num_x,
        31 - __builtin_clz(stride) // log_stride
    );

    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Launch final reduction kernel - reads from block_sums, writes to output
    auto num_blocks = grid.x * grid.y;
    auto domain_size = S_DEG << l_skip;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    size_t reduce_shmem = div_ceil(reduce_block.x, WARP_SIZE) * sizeof(FpExt);
    sumcheck::final_reduce_block_sums<<<domain_size, reduce_block, reduce_shmem>>>(
        block_sums, output, num_blocks
    );

    return CHECK_KERNEL();
}

extern "C" int _stacked_reduction_fold_ple(
    const UnstackedPleFoldPacket *trace_packets,
    const Fp *omega_skip_pows,
    const FpExt *inv_lagrange_denoms,
    uint32_t l_skip, // log2(domain_size)
    uint16_t num_packets,
    uint32_t max_new_height,
    uint16_t max_trace_width
) {
    auto [grid, block] = kernel_launch_params(max_new_height);
    grid.y = max_trace_width;
    grid.z = num_packets;

    uint32_t skip_domain = 1 << l_skip;
    stacked_reduction_fold_ple_kernel<<<grid, block>>>(
        trace_packets, omega_skip_pows, inv_lagrange_denoms, skip_domain, l_skip, num_packets
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

extern "C" int _stacked_reduction_sumcheck_mle_round(
    const FpExt *const *q_evals,
    const FpExt *eq_r_ns,
    const FpExt *k_rot_ns,
    const UnstackedSlice *unstacked_cols,
    const FpExt *lambda_pows,
    uint64_t *output, // [S_DEG * D_EF] - atomic accumulator, reduced on CPU
    uint32_t q_height,
    uint32_t window_len,
    uint32_t num_y,
    uint32_t sm_count
) {
    // Smaller block size for more eligible warps to hide latency
    auto [grid, block] = kernel_launch_params(num_y, 256);
    assert(sm_count);

    // Heuristic auto-tuning for window-stride (grid.y):
    // - Increase work per thread (reduce atomics) by targeting multiple window iterations/thread
    // - Ensure enough total blocks to keep SMs busy when num_y is small
    constexpr uint32_t WAVES_TARGET = 4; // blocks/SM target
    constexpr uint32_t ITERS_MIN = 4;    // prefer >= this many window iterations/thread
    constexpr uint32_t ITERS_MAX = 16;   // prefer <= this many window iterations/thread

    uint32_t stride_occ = div_ceil(sm_count * WAVES_TARGET, grid.x);
    uint32_t stride_loop_lo = div_ceil(window_len, ITERS_MAX);
    uint32_t stride_loop_hi = div_ceil(window_len, ITERS_MIN);

    uint32_t lo = std::max(1u, std::max(stride_occ, stride_loop_lo));
    uint32_t hi = std::min(std::min(window_len, MAX_GRID_DIM), stride_loop_hi);

    uint32_t tuned_stride = (lo <= hi) ? lo : std::min(lo, std::min(window_len, MAX_GRID_DIM));
    grid.y = tuned_stride;

    // Ensure that atomic add does not overflow u64
    assert((size_t)num_y * grid.y < (size_t)1u << 32);
    size_t shmem_bytes = div_ceil(block.x, WARP_SIZE) * sizeof(FpExt);

    stacked_reduction_sumcheck_mle_round_kernel<<<grid, block, shmem_bytes>>>(
        q_evals, eq_r_ns, k_rot_ns, unstacked_cols, lambda_pows, output, q_height, window_len, num_y
    );

    return CHECK_KERNEL();
}

extern "C" int _stacked_reduction_sumcheck_mle_round_degenerate(
    const FpExt *const *q_evals,
    const FpExt *eq_ub_ptr,
    FpExt eq_r,
    FpExt k_rot_r,
    const UnstackedSlice *unstacked_cols,
    const FpExt *lambda_pows,
    uint64_t *output, // [S_DEG * 4] - atomic accumulator, reduced on CPU
    uint32_t q_height,
    uint32_t window_len,
    uint32_t l_skip,
    uint32_t round
) {
    auto shift_factor = l_skip + round;
    dim3 block(std::min(window_len, 256u));
    dim3 grid(1);
    // block.x <= 512 < 2^32 so atomic u64 will not overflow
    size_t shmem_bytes = div_ceil(block.x, WARP_SIZE) * sizeof(FpExt);

    stacked_reduction_sumcheck_mle_round_degenerate_kernel<<<grid, block, shmem_bytes>>>(
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
