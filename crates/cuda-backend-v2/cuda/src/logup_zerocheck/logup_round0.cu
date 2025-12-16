#include "codec.cuh"
#include "dag_entry.cuh"
#include "fp.h"
#include "fpext.h"
#include "frac_ext.cuh"
#include "launcher.cuh"
#include "round0_config.cuh"
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
template <bool GLOBAL>
__device__ __forceinline__ void acc_interactions(
    const DagPrismEvalContext &eval_ctx,
    const FpExt *__restrict__ numer_weights,
    const FpExt *__restrict__ denom_weights,
    const Rule *__restrict__ d_rules,
    size_t rules_len,
    Fp *__restrict__ local_buffer,
    FpExt &numer_sum,
    FpExt &denom_sum
) {
    // TODO: duplicate from bary_acc_constraints, make into function
    for (size_t node = 0; node < rules_len; ++node) {
        Rule rule = d_rules[node];
        DecodedRule decoded = decode_rule(rule);

        Fp x_val = bary_eval_dag_entry(decoded.x, eval_ctx);
        Fp result;
        switch (decoded.op) {
        case OP_ADD: {
            Fp y_val = bary_eval_dag_entry(decoded.y, eval_ctx);
            result = x_val + y_val;
            break;
        }
        case OP_SUB: {
            Fp y_val = bary_eval_dag_entry(decoded.y, eval_ctx);
            result = x_val - y_val;
            break;
        }
        case OP_MUL: {
            Fp y_val = bary_eval_dag_entry(decoded.y, eval_ctx);
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
            if constexpr (GLOBAL) {
                // Note: decoded.z_index refers to a decoding index, it doesn't have to do with prism coordinate
                eval_ctx.inter_buffer[decoded.z_index * eval_ctx.buffer_stride] = result;
            } else {
                local_buffer[decoded.z_index] = result;
            }
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
template <bool GLOBAL>
__global__ void logup_r0_bary_eval_interactions_kernel(
    FracExt *__restrict__ tmp_sums_buffer, // [blockDim.y * gridDim.y][large_domain]
    const Fp *__restrict__ selectors_cube, // [3][num_x]
    const Fp *__restrict__ preprocessed,
    const Fp *const *__restrict__ main_parts,
    const Fp *__restrict__ omega_skip_pows,     // [skip_domain]
    const Fp *__restrict__ inv_lagrange_denoms, // [large_domain][skip_domain]
    const FpExt *__restrict__ eq_sharp_uni,     // [large_domain]
    const FpExt *__restrict__ eq_cube,          // [num_x]
    const Fp *__restrict__ public_values,
    const FpExt *__restrict__ numer_weights,
    const FpExt *__restrict__ denom_weights,
    FpExt denom_sum_init,
    const Rule *__restrict__ d_rules,
    size_t rules_len,
    uint32_t buffer_size,
    Fp *__restrict__ d_intermediates,
    uint32_t large_domain,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t height,
    uint32_t
        expansion_factor, // large_domain.next_power_of_two() / skip_domain: determines when to skip barycentric eval
    std::pair<uint32_t, uint32_t> z_dim
) {
    extern __shared__ char smem[];
    FracExt *shared = reinterpret_cast<FracExt *>(smem);

    // We unlinearize the thread index
    auto [zs_per_grid, zs_per_block] = z_dim;
    uint32_t xs_per_block = blockDim.x / zs_per_block;
    uint32_t tidx_z = threadIdx.x % zs_per_block;
    uint32_t tidx_x = threadIdx.x / zs_per_block;
    uint32_t bidx_z = blockIdx.x % zs_per_grid;
    uint32_t bidx_x = blockIdx.x / zs_per_grid;
    uint32_t z_int = tidx_z + bidx_z * zs_per_block;
    bool const active_thread = (z_int < large_domain);

    if (active_thread) {
        // The hypercube coordinates: we want to sum over these.
        uint32_t x_int_base = tidx_x + bidx_x * xs_per_block;
        uint32_t x_int_stride = (gridDim.x / zs_per_grid) * xs_per_block;

        uint32_t task_offset = x_int_base * large_domain + z_int;
        // The maximal amount of rows done by different threads; hence we need enough global memory [buffer_size * task_stride] for intermediates
        uint32_t task_stride = x_int_stride * large_domain;

        Fp local_buffer[BUFFER_THRESHOLD];
        Fp *inter_buffer;
        uint32_t buffer_stride;
        if constexpr (GLOBAL) {
            inter_buffer = d_intermediates + task_offset;
            buffer_stride = task_stride;
        } else {
            inter_buffer = local_buffer;
            buffer_stride = 1;
        }

        uint32_t log_skip = __ffs(skip_domain) - 1;
        uint32_t log_height_total = __ffs(height) - 1;
        uint32_t log_segment = std::min(log_skip, log_height_total);
        uint32_t segment_size = 1u << log_segment;
        uint32_t log_stride = log_skip - log_segment;

        Fp omega_root = TWO_ADIC_GENERATORS[__ffs(expansion_factor * skip_domain) - 1];
        Fp omega = pow(omega_root, z_int << log_stride);

        Fp eta = TWO_ADIC_GENERATORS[log_skip - log_stride];

        Fp is_first_mult = avg_gp(omega, segment_size);
        Fp is_last_mult = avg_gp(omega * eta, segment_size);

        FracExt sum = {FpExt(Fp::zero()), FpExt(Fp::zero())};
        // See zerocheck_bary_evaluate_constraints_kernel for comments
        for (uint32_t x_int = x_int_base; x_int < num_x; x_int += x_int_stride) {
            Fp is_first = is_first_mult * selectors_cube[x_int];
            Fp is_last = is_last_mult * selectors_cube[2 * num_x + x_int];
            const Fp *inv_lagrange_denoms_z = inv_lagrange_denoms + z_int * skip_domain;

            DagPrismEvalContext eval_ctx{
                preprocessed,
                main_parts,
                public_values,
                omega_skip_pows,
                inv_lagrange_denoms_z,
                inter_buffer,
                is_first,
                is_last,
                skip_domain,
                num_x,
                height,
                buffer_stride,
                buffer_size,
                z_int,
                x_int,
                expansion_factor
            };

            FpExt numer = FpExt(Fp::zero());
            FpExt denom = denom_sum_init;
            // Compute interaction sums (without eq_* multiplication)
            acc_interactions<GLOBAL>(
                eval_ctx,
                numer_weights,
                denom_weights,
                d_rules,
                rules_len,
                local_buffer,
                numer,
                denom
            );
            FpExt eq_sharp = eq_sharp_uni[z_int] * eq_cube[x_int];

            // Apply eq_val multiplier to both sums
            sum.p += eq_sharp * numer;
            sum.q += eq_sharp * denom;
        }

        // Reduce phase: reduce all threadIdx.y in the same block, keeping z_int independent
        shared[threadIdx.x] = sum;
    }
    __syncthreads();

    if (active_thread && tidx_x == 0) {
        FracExt tile_sum = shared[tidx_z];
        for (int lane = 1; lane < xs_per_block; ++lane) {
            auto lane_offset = lane * zs_per_block + tidx_z;
            tile_sum.p += shared[lane_offset].p;
            tile_sum.q += shared[lane_offset].q;
        }
        tmp_sums_buffer[bidx_x * large_domain + z_int] = tile_sum;
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
    uint32_t large_domain,
    uint32_t num_x,
    size_t max_temp_bytes
) {
    return align_x_round0_config::temp_sums_buffer_size(
        buffer_size, large_domain, num_x, max_temp_bytes, BUFFER_THRESHOLD, MAX_THREADS
    );
}

extern "C" size_t _logup_r0_intermediates_buffer_size(
    uint32_t buffer_size,
    uint32_t large_domain,
    uint32_t num_x,
    size_t max_temp_bytes
) {
    return align_x_round0_config::intermediates_buffer_size(
        buffer_size, large_domain, num_x, max_temp_bytes, BUFFER_THRESHOLD, MAX_THREADS
    );
}

extern "C" int _logup_bary_eval_interactions_round0(
    FracExt *tmp_sums_buffer, // [blockDim.y * gridDim.y][large_domain]
    FracExt *output,          // [large_domain]
    const Fp *selectors_cube, // [3][num_x]
    const Fp *preprocessed,
    const Fp *const *main_parts,
    const Fp *omega_skip_pows,     // [skip_domain]
    const Fp *inv_lagrange_denoms, // [large_domain][skip_domain]
    const FpExt *eq_sharp_uni,     // [large_domain]
    const FpExt *eq_cube,          // [num_x]
    const Fp *public_values,
    const FpExt *numer_weights,
    const FpExt *denom_weights,
    FpExt denom_sum_init,
    const Rule *d_rules,
    size_t rules_len,
    uint32_t buffer_size,
    Fp *d_intermediates,
    uint32_t large_domain,
    uint32_t skip_domain,
    uint32_t num_x,
    uint32_t height,
    uint32_t expansion_factor,
    size_t max_temp_bytes
) {
    auto [grid, block] = align_x_round0_config::eval_constraints_launch_params(
        buffer_size, large_domain, num_x, max_temp_bytes, BUFFER_THRESHOLD, MAX_THREADS
    );
    auto z_dim = align_x_round0_config::get_z_dim(large_domain);
    auto xs_per_grid = grid.x / z_dim.first;
    size_t shmem_bytes = sizeof(FracExt) * block.x;

#define ARGUMENTS                                                                                  \
    tmp_sums_buffer, selectors_cube, preprocessed, main_parts, omega_skip_pows,                    \
        inv_lagrange_denoms, eq_sharp_uni, eq_cube, public_values, numer_weights, denom_weights,   \
        denom_sum_init, d_rules, rules_len, buffer_size, d_intermediates, large_domain,            \
        skip_domain, num_x, height, expansion_factor, z_dim

    if (buffer_size > BUFFER_THRESHOLD) {
        logup_r0_bary_eval_interactions_kernel<true><<<grid, block, shmem_bytes>>>(ARGUMENTS);
    } else {
        logup_r0_bary_eval_interactions_kernel<false><<<grid, block, shmem_bytes>>>(ARGUMENTS);
    }
    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Launch final reduction kernel - reads from block_sums, writes to output
    auto num_blocks = xs_per_grid;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    // FracExt = (FpExt, FpExt) so we set block = 2 * large_domain
    sumcheck::final_reduce_block_sums<<<2 * large_domain, reduce_block, reduce_shmem>>>(
        reinterpret_cast<FpExt *>(tmp_sums_buffer), reinterpret_cast<FpExt *>(output), num_blocks
    );
    return CHECK_KERNEL();
}

} // namespace logup_round0
