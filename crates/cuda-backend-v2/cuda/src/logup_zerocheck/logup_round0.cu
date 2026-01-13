#include "codec.cuh"
#include "dag_entry.cuh"
#include "eval_config.cuh"
#include "fp.h"
#include "fpext.h"
#include "frac_ext.cuh"
#include "launcher.cuh"
#include "sumcheck.cuh"
#include "utils.cuh"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector_types.h>

using namespace symbolic_dag;

namespace logup_round0 {

// NOTE[jpw]: keeping separate from zerocheck so it can be tuned separately
constexpr uint32_t BUFFER_THRESHOLD = 16;

// Device function equivalent to helper.eval_interactions without eq_* parts
// This computes the interaction numerator and denominator sums (weighted by eq_3b)
// The eq_* multiplication is done separately in the kernel
template <bool NEEDS_SHMEM>
__device__ __forceinline__ void acc_interactions(
    const NttEvalContext &eval_ctx,
    const FpExt *__restrict__ numer_weights,
    const FpExt *__restrict__ denom_weights,
    const Rule *__restrict__ d_rules,
    size_t rules_len,
    FpExt &numer_sum,
    FpExt &denom_sum
) {
    for (size_t node = 0; node < rules_len; ++node) {
        Rule rule = d_rules[node];
        DecodedRule decoded = decode_rule(rule);

        Fp x_val = ntt_eval_dag_entry<NEEDS_SHMEM>(decoded.x, eval_ctx);
        Fp result;
        switch (decoded.op) {
        case OP_ADD: {
            Fp y_val = ntt_eval_dag_entry<NEEDS_SHMEM>(decoded.y, eval_ctx);
            result = x_val + y_val;
            break;
        }
        case OP_SUB: {
            Fp y_val = ntt_eval_dag_entry<NEEDS_SHMEM>(decoded.y, eval_ctx);
            result = x_val - y_val;
            break;
        }
        case OP_MUL: {
            Fp y_val = ntt_eval_dag_entry<NEEDS_SHMEM>(decoded.y, eval_ctx);
            result = x_val * y_val;
            break;
        }
        case OP_NEG:
            result = -x_val;
            break;
        case OP_VAR:
            result = x_val;
            break;
        case OP_INV:
            result = inv(x_val);
            break;
        }

        if (decoded.buffer_result && eval_ctx.buffer_size > 0) {
#ifdef CUDA_DEBUG
            assert(decoded.z_index < eval_ctx.buffer_size);
#endif
            eval_ctx.inter_buffer[decoded.z_index * eval_ctx.buffer_stride] = result;
        }

        if (decoded.is_constraint) {
            numer_sum += numer_weights[node] * result;
            denom_sum += denom_weights[node] * result;
        }
    }
}

// ============================================================================
// KERNELS
// ============================================================================

// Round0 phase interactions kernel (for sumcheck round0)
template <bool GLOBAL, bool NEEDS_SHMEM>
__global__ void logup_r0_ntt_eval_interactions_kernel(
    FracExt *__restrict__ tmp_sums_buffer, // [gridDim.x][num_cosets * skip_domain]
    const Fp *__restrict__ selectors_cube, // [3][num_x]
    const Fp *__restrict__ preprocessed,
    const Fp *const *__restrict__ main_parts,
    const FpExt *__restrict__ eq_cube, // [num_x]
    const Fp *__restrict__ public_values,
    const FpExt *__restrict__ numer_weights,
    const FpExt *__restrict__ denom_weights,
    FpExt denom_sum_init,
    const Rule *__restrict__ d_rules,
    size_t rules_len,
    uint32_t buffer_size,
    Fp *__restrict__ d_intermediates,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t height,
    Fp g_shift
) {
    extern __shared__ char smem[];
    // Shared memory layout:
    // - FracExt[blockDim.x]: shared_sum for reduction
    // - Fp[blockDim.x]: ntt_buffers (one skip_domain-sized buffer per x_int group in block)
    FracExt *shared_sum = reinterpret_cast<FracExt *>(smem);
    Fp *ntt_buffers_base = reinterpret_cast<Fp *>(smem + blockDim.x * sizeof(FracExt));

    uint32_t l_skip = __ffs(skip_domain) - 1;
    uint32_t x_int_in_block = threadIdx.x >> l_skip;
    Fp *ntt_buffer = ntt_buffers_base + x_int_in_block * skip_domain;

    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t ntt_idx = tidx & (skip_domain - 1);
    uint32_t x_int_base = tidx >> l_skip;
    uint32_t coset_idx = blockIdx.y;

    auto num_cosets = gridDim.y;

    uint32_t const ntt_idx_rev = rev_len(ntt_idx, l_skip);
    Fp const g_coset = pow(g_shift, coset_idx + 1);
    Fp const omega_shift = pow(g_coset, ntt_idx_rev);

    uint32_t const log_height_total = __ffs(height) - 1;
    uint32_t const log_segment = std::min(l_skip, log_height_total);
    uint32_t const segment_size = 1u << log_segment;
    uint32_t const log_stride = l_skip - log_segment;

    Fp const omega_skip = TWO_ADIC_GENERATORS[l_skip];
    Fp const eval_point = g_coset * pow(omega_skip, ntt_idx);
    Fp const omega = pow(eval_point, 1u << log_stride);
    Fp const eta = TWO_ADIC_GENERATORS[l_skip - log_stride];
    Fp const is_first_mult = avg_gp(omega, segment_size);
    Fp const is_last_mult = avg_gp(omega * eta, segment_size);

    Fp local_buffer[BUFFER_THRESHOLD];
    Fp *inter_buffer;
    uint32_t buffer_stride;
    if constexpr (GLOBAL) {
        uint32_t task_offset = coset_idx * gridDim.x * blockDim.x + tidx;
        uint32_t task_stride = gridDim.y * gridDim.x * blockDim.x;
        inter_buffer = d_intermediates + task_offset;
        buffer_stride = task_stride;
    } else {
        inter_buffer = local_buffer;
        buffer_stride = 1;
    }

    FracExt sum = {FpExt(Fp::zero()), FpExt(Fp::zero())};
    uint32_t x_int_stride = (gridDim.x * blockDim.x) >> l_skip;
    for (uint32_t x_int = x_int_base; x_int < num_x; x_int += x_int_stride) {
        Fp const is_first = is_first_mult * selectors_cube[x_int];
        Fp const is_last = is_last_mult * selectors_cube[2 * num_x + x_int];

        NttEvalContext eval_ctx{
            preprocessed,
            main_parts,
            public_values,
            inter_buffer,
            ntt_buffer,
            is_first,
            is_last,
            omega_shift,
            skip_domain,
            num_cosets,
            num_x,
            height,
            buffer_stride,
            buffer_size,
            ntt_idx,
            coset_idx,
            x_int
        };

        FpExt numer = FpExt(Fp::zero());
        FpExt denom = denom_sum_init;
        acc_interactions<NEEDS_SHMEM>(eval_ctx, numer_weights, denom_weights, d_rules, rules_len, numer, denom);

        FpExt eq = eq_cube[x_int];
        sum.p += eq * numer;
        sum.q += eq * denom;
    }

    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    if (threadIdx.x < skip_domain) {
        FracExt tile_sum = shared_sum[threadIdx.x];
        for (int lane = 1; lane < (blockDim.x >> l_skip); ++lane) {
            auto lane_offset = (lane << l_skip) + threadIdx.x;
            tile_sum.p += shared_sum[lane_offset].p;
            tile_sum.q += shared_sum[lane_offset].q;
        }
        tmp_sums_buffer[blockIdx.x * num_cosets * skip_domain + (coset_idx << l_skip) + ntt_idx] =
            tile_sum;
    }
}

// ============================================================================
// LAUNCHERS
// ============================================================================

constexpr uint32_t MAX_THREADS = 256;

// (Not a launcher) Utility function to calculate required size of temp sum buffer.
// Required length of *temp_sum_buffer in FracExt elements
extern "C" size_t _logup_r0_temp_sums_buffer_size(
    uint32_t buffer_size,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t num_cosets,
    size_t max_temp_bytes
) {
    return coset_round0_config::temp_sums_buffer_size(
        buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes, BUFFER_THRESHOLD, MAX_THREADS
    );
}

extern "C" size_t _logup_r0_intermediates_buffer_size(
    uint32_t buffer_size,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t num_cosets,
    size_t max_temp_bytes
) {
    return coset_round0_config::intermediates_buffer_size(
        buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes, BUFFER_THRESHOLD, MAX_THREADS
    );
}

template <bool GLOBAL, bool NEEDS_SHMEM>
int launch_logup_ntt_eval_interactions(
    FracExt *tmp_sums_buffer,
    FracExt *output,
    const Fp *selectors_cube,
    const Fp *preprocessed,
    const Fp *const *main_parts,
    const FpExt *eq_cube,
    const Fp *public_values,
    const FpExt *numer_weights,
    const FpExt *denom_weights,
    FpExt denom_sum_init,
    const Rule *d_rules,
    size_t rules_len,
    uint32_t buffer_size,
    Fp *d_intermediates,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t height,
    uint32_t num_cosets,
    Fp g_shift,
    size_t max_temp_bytes
) {
    auto [grid, block] = coset_round0_config::eval_constraints_launch_params(
        buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes, BUFFER_THRESHOLD, MAX_THREADS
    );

    size_t shared_sum_size = sizeof(FracExt) * block.x;
    size_t ntt_buffers_size = NEEDS_SHMEM ? sizeof(Fp) * block.x : 0;
    size_t shmem_bytes = shared_sum_size + ntt_buffers_size;

    logup_r0_ntt_eval_interactions_kernel<GLOBAL, NEEDS_SHMEM><<<grid, block, shmem_bytes>>>(
        tmp_sums_buffer,
        selectors_cube,
        preprocessed,
        main_parts,
        eq_cube,
        public_values,
        numer_weights,
        denom_weights,
        denom_sum_init,
        d_rules,
        rules_len,
        buffer_size,
        d_intermediates,
        skip_domain,
        num_x,
        height,
        g_shift
    );

    int err = CHECK_KERNEL();
    if (err != 0) {
        return err;
    }

    auto num_blocks = grid.x;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = div_ceil(reduce_block.x, WARP_SIZE);
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    auto large_domain = num_cosets * skip_domain;
    sumcheck::final_reduce_block_sums<<<2 * large_domain, reduce_block, reduce_shmem>>>(
        reinterpret_cast<FpExt *>(tmp_sums_buffer), reinterpret_cast<FpExt *>(output), num_blocks
    );
    return CHECK_KERNEL();
}

extern "C" int _logup_bary_eval_interactions_round0(
    FracExt *tmp_sums_buffer, // [num_x][large_domain]
    FracExt *output,          // [large_domain]
    const Fp *selectors_cube, // [3][num_x]
    const Fp *preprocessed,
    const Fp *const *main_parts,
    const FpExt *eq_cube,          // [num_x]
    const Fp *public_values,
    const FpExt *numer_weights,
    const FpExt *denom_weights,
    FpExt denom_sum_init,
    const Rule *d_rules,
    size_t rules_len,
    uint32_t buffer_size,
    Fp *d_intermediates,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t height,
    uint32_t num_cosets,
    Fp g_shift,
    size_t max_temp_bytes
) {
    bool is_global = buffer_size > BUFFER_THRESHOLD;
    bool needs_shmem = skip_domain > WARP_SIZE;

    return DISPATCH_BOOL_PAIR(
        launch_logup_ntt_eval_interactions,
        is_global,
        needs_shmem,
        tmp_sums_buffer,
        output,
        selectors_cube,
        preprocessed,
        main_parts,
        eq_cube,
        public_values,
        numer_weights,
        denom_weights,
        denom_sum_init,
        d_rules,
        rules_len,
        buffer_size,
        d_intermediates,
        skip_domain,
        num_x,
        height,
        num_cosets,
        g_shift,
        max_temp_bytes
    );
}

} // namespace logup_round0
