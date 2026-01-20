#include "device_ntt.cuh"
#include "fp.h"
#include "fpext.h"
#include "frac_ext.cuh"
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
template <bool NEEDS_SHMEM>
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
    // Use blockDim.x + 1 stride to avoid shared memory bank conflicts
    // when accessing shared_sum[i * stride + ...] for different i values
    const uint32_t PADDED_X = blockDim.x + 1;
    FpExt *shared_sum = reinterpret_cast<FpExt *>(smem); // FpExt[S_DEG][PADDED_X]

    uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t z_idx = tidx & skip_mask;
    uint32_t x_int = tidx >> l_skip;
    uint32_t col_idx = blockIdx.y;
    // Each thread will handle z_int in S_DEG * z_idx..(S_DEG + 1) * z_idx
    FpExt eq_k_rot[S_DEG];
    // Compute eq, k_rot depending on (z_int, x_int)
    FpExt eq_cube = get_eq_cube(eq_r_ns, num_x, x_int);
    FpExt k_rot_cube = get_eq_cube(eq_r_ns, num_x, rot_prev(x_int, num_x)) - eq_cube;
    FpExt eq_lambda = lambda_pows[2 * col_idx] * eq_cube;
    FpExt k_rot_lambda = lambda_pows[2 * col_idx + 1];

#pragma unroll
    for (uint32_t i = 0; i < S_DEG; i++) {
        uint32_t z_int = (i << l_skip) + z_idx;
        auto packet = z_packets[z_int];
        eq_k_rot[i] = packet.eq_uni * eq_lambda +
                      (packet.k_rot_0 * eq_cube + packet.k_rot_1 * k_rot_cube) * k_rot_lambda;
    }

    // First, each thread reads trace_ptr and loads to shared_eval
    auto evals = trace_ptr + col_idx * height + (x_int << (l_skip - log_stride));
    auto stride_mask = (1u << log_stride) - 1;
    Fp this_thread_value = (z_idx & stride_mask) == 0 ? evals[z_idx >> log_stride] : Fp::zero();

    // We make special use of the fact S_DEG=2:
    {
        // i = 0: it's just the eval
        auto q = this_thread_value;
        shared_sum[threadIdx.x] = eq_k_rot[0] * q;
    }
    {
        // i = 1: iNTT, shift, forward NTT
        // Note: we configured the block so all threads are active
        Fp *__restrict__ sbuf = nullptr;
        if constexpr (NEEDS_SHMEM) {
            Fp *shared_eval =
                reinterpret_cast<Fp *>(smem + S_DEG * PADDED_X * sizeof(FpExt)); // Fp[blockDim.x]
            sbuf = shared_eval + ((threadIdx.x >> l_skip) << l_skip);
            sbuf[z_idx] = this_thread_value;
            __syncthreads();
        }
        // iNTT, bitrev output stored to this_thread_value
        ntt_natural_to_bitrev<true, NEEDS_SHMEM>(this_thread_value, sbuf, z_idx, l_skip);
        // multiply by shift. regardless of whether shmem is needed, the input for forward NTT must be placed in `this_thread_value` for `ntt_bitrev_to_natural`
        auto const z_idx_rev = rev_len(z_idx, l_skip);
        Fp omega_shift = pow(TWO_ADIC_GENERATORS[l_skip + 1], z_idx_rev); // 1 = log2(S_DEG)
        this_thread_value *= omega_shift;
        // forward NTT of shifted coeffs
        ntt_bitrev_to_natural<false, NEEDS_SHMEM>(this_thread_value, sbuf, z_idx, l_skip);
        auto q = this_thread_value;
        shared_sum[PADDED_X + threadIdx.x] = eq_k_rot[1] * q;
    }

    __syncthreads();
    if ((threadIdx.x >> l_skip) == 0) {
        // Compute both tile sums (i=0 and i=1) in lockstep. Note we are using
        // FracExt as a contiguous FpExt pair (instead of a fraction), as we
        // use S_DEG = 2.
        FracExt out{shared_sum[0 * PADDED_X + z_idx], shared_sum[1 * PADDED_X + z_idx]};
        for (int lane = 1; lane < (blockDim.x >> l_skip); ++lane) {
            out.p += shared_sum[0 * PADDED_X + (lane << l_skip) + z_idx];
            out.q += shared_sum[1 * PADDED_X + (lane << l_skip) + z_idx];
        }

        // Write both tile sums with a single contiguous store
        FpExt *out_ptr =
            block_sums + (col_idx * gridDim.x + blockIdx.x) * (S_DEG << l_skip) + (S_DEG * z_idx);
        *reinterpret_cast<FracExt *>(out_ptr) = out;
    }
}

// Single-trace kernel: each chunk of skip_domain threads handles one output cell
// Parallelizes barycentric interpolation across threads for better performance
// Grid: (num_row_blocks, trace_width) where blockIdx.y selects the column
__global__ void stacked_reduction_fold_ple_kernel(
    const Fp *__restrict__ src,
    FpExt *__restrict__ dst,
    const Fp *__restrict__ omega_skip_pows,
    const FpExt *__restrict__ inv_lagrange_denoms,
    uint32_t trace_height,
    uint32_t new_height,
    uint32_t skip_domain
) {
    extern __shared__ char smem_raw[];
    FpExt *smem = reinterpret_cast<FpExt *>(smem_raw);

    uint32_t col_idx = blockIdx.y;
    uint32_t chunks_per_block = blockDim.x / skip_domain;
    uint32_t chunk_in_block = threadIdx.x / skip_domain;
    uint32_t tid_in_chunk = threadIdx.x % skip_domain;
    uint32_t row_idx = blockIdx.x * chunks_per_block + chunk_in_block;

    // Cannot early-return: all threads must participate in chunk_reduce_sum (uses __syncthreads)
    bool const active_chunk = (row_idx < new_height);

    // Barycentric interpolation: each thread handles exactly ONE term
    // (since src_len <= skip_domain, each thread contributes at most one term)
    FpExt local_val(Fp::zero());
    if (active_chunk) {
        uint32_t src_len = std::min(trace_height, skip_domain);
        uint32_t stride = skip_domain / src_len;
        const Fp *cell_src = src + col_idx * trace_height + row_idx * src_len;

        if (tid_in_chunk < src_len) {
            uint32_t idx = tid_in_chunk;
            local_val =
                cell_src[idx] * omega_skip_pows[idx * stride] * inv_lagrange_denoms[idx * stride];
        }
    }

    // Reduce within chunk using the helper from sumcheck.cuh
    // All threads must participate; inactive chunks contribute zero
    FpExt result =
        sumcheck::chunk_reduce_sum(local_val, smem, tid_in_chunk, skip_domain, chunk_in_block);

    // No __syncthreads() needed: each chunk writes to a distinct dst location
    // and kernel exits immediately after this write
    if (active_chunk && tid_in_chunk == 0) {
        dst[col_idx * new_height + row_idx] = result;
    }
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
    uint32_t trace_width,
    uint32_t l_skip
) {
    uint32_t skip_domain = 1u << l_skip;
    auto lifted_height = std::max(trace_height, skip_domain);
    // Entire skip domain must fit within a block
    auto max_threads = std::max(skip_domain, 256u);
    auto [grid, block] = kernel_launch_params(lifted_height, max_threads);
    // NOTE: lifted_height = skip_domain * num_x will always be divisible by block.x since num_x is a power of 2. Therefore all threads in a block will be active.
    grid.y = trace_width;
    return {grid, block};
}

// (Not a launcher) Utility function to calculate required size of temp buffer.
// Required length of *block_sums in FpExt elements
extern "C" uint32_t _stacked_reduction_r0_required_temp_buffer_size(
    uint32_t trace_height,
    uint32_t trace_width,
    uint32_t l_skip
) {
    auto [grid, block] = stacked_reduction_round0_launch_params(trace_height, trace_width, l_skip);
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
    uint32_t num_x
) {
    // We currently use this hardcoded value to optimize the coset NTT
    static_assert(S_DEG == 2);
    uint32_t skip_domain = 1u << l_skip;
    uint32_t stride = std::max(skip_domain / trace_height, 1u);
    auto [grid, block] = stacked_reduction_round0_launch_params(trace_height, trace_width, l_skip);

    // Use block.x + 1 stride for shared_sum to avoid bank conflicts
    size_t shmem_sum_size = sizeof(FpExt) * (block.x + 1) * S_DEG;

#define ARGUMENTS                                                                                  \
    eq_r_ns, trace_ptr, lambda_pows, z_packets, block_sums, trace_height, trace_width, l_skip,     \
        (1u << l_skip) - 1, num_x, 31 - __builtin_clz(stride)

    if (l_skip > LOG_WARP_SIZE) {
        size_t shmem_eval_size = sizeof(Fp) * block.x;
        size_t shmem_bytes = shmem_sum_size + shmem_eval_size;
        stacked_reduction_round0_block_sum_kernel<true><<<grid, block, shmem_bytes>>>(ARGUMENTS);
    } else {
        size_t shmem_bytes = shmem_sum_size;
        stacked_reduction_round0_block_sum_kernel<false><<<grid, block, shmem_bytes>>>(ARGUMENTS);
    }

    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Launch final reduction kernel - reads from block_sums, writes to output
    auto num_blocks = grid.x * grid.y;
    auto domain_size = S_DEG << l_skip;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    size_t reduce_shmem = div_ceil(reduce_block.x, WARP_SIZE) * sizeof(FpExt);
    sumcheck::final_reduce_block_sums<true>
        <<<domain_size, reduce_block, reduce_shmem>>>(block_sums, output, num_blocks);

    return CHECK_KERNEL();
}

// Parallelizes barycentric interpolation across 2^l_skip threads per output cell
extern "C" int _stacked_reduction_fold_ple(
    const Fp *src,
    FpExt *dst,
    const Fp *omega_skip_pows,
    const FpExt *inv_lagrange_denoms,
    uint32_t trace_height,
    uint32_t trace_width,
    uint32_t l_skip
) {
    uint32_t skip_domain = 1u << l_skip;
    uint32_t new_height = std::max(trace_height, skip_domain) / skip_domain;

    // Block size: at least skip_domain, prefer 256 for occupancy
    uint32_t block_size = std::max(256u, skip_domain);
    uint32_t chunks_per_block = block_size / skip_domain;

    // 2D grid: x-dim for rows, y-dim for columns
    dim3 grid(div_ceil(new_height, chunks_per_block), trace_width);
    dim3 block(block_size);

    // Shared memory for cross-warp reduction (needed when skip_domain > 32)
    // Each warp in the block needs one FpExt slot for cross-warp reduction
    uint32_t total_warps_in_block = block_size / WARP_SIZE;
    size_t smem_bytes = (skip_domain > WARP_SIZE) ? total_warps_in_block * sizeof(FpExt) : 0;

    stacked_reduction_fold_ple_kernel<<<grid, block, smem_bytes>>>(
        src, dst, omega_skip_pows, inv_lagrange_denoms, trace_height, new_height, skip_domain
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
