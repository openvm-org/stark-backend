#include "codec.cuh"
#include "eval_config.cuh"
#include "fp.h"
#include "fpext.h"
#include "frac_ext.cuh"
#include "launcher.cuh"
#include "matrix.cuh"
#include "sumcheck.cuh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdio.h>
#include <vector_types.h>

namespace logup_zerocheck_mle {

struct BlockCtx {
    uint32_t local_block_idx_x;
    uint32_t air_idx;
};

struct EvalCtx {
    const FpExt *__restrict__ d_selectors;
    const MainMatrixPtrs<FpExt> d_preprocessed;
    const MainMatrixPtrs<FpExt> *__restrict__ d_main;
    const Fp *__restrict__ d_public;
    FpExt *__restrict__ d_intermediates;
    uint32_t height;
};

struct ZerocheckCtx {
    EvalCtx eval_ctx;
    uint32_t num_y;
    const FpExt *__restrict__ d_eq_xi;
    const uint32_t *__restrict__ d_lambda_indices;
    const Rule *__restrict__ d_rules;
    size_t rules_len;
    const size_t *__restrict__ d_used_nodes;
    size_t used_nodes_len;
    uint32_t buffer_size;
};

struct LogupCtx {
    EvalCtx eval_ctx;
    uint32_t num_y;
    const FpExt *__restrict__ d_eq_sharp;
    const FpExt *__restrict__ d_challenges;
    const FpExt *__restrict__ d_eq_3bs;
    const Rule *__restrict__ d_rules;
    size_t rules_len;
    const size_t *__restrict__ d_used_nodes;
    size_t used_nodes_len;
    uint32_t buffer_size;
};

__device__ __forceinline__ FpExt evaluate_mle_entry(
    const SourceInfo &src,
    uint32_t row,
    const EvalCtx &ctx,
    uint32_t buffer_stride,
    const FpExt *__restrict__ d_challenges = nullptr
) {
    switch (src.type) {
    case ENTRY_PREPROCESSED: {
#ifdef CUDA_DEBUG
        assert(ctx.d_preprocessed.data);
#endif
        const auto stride = ctx.height * ctx.d_preprocessed.air_width;
        const FpExt *__restrict__ matrix = ctx.d_preprocessed.data + stride * src.offset;
        const FpExt *__restrict__ column = matrix + ctx.height * src.index;
        return column[row];
    }
    case ENTRY_MAIN: {
        auto main_ptr = ctx.d_main[src.part];
        const auto stride = ctx.height * main_ptr.air_width;
        const FpExt *__restrict__ matrix = main_ptr.data + stride * src.offset;
        const FpExt *__restrict__ column = matrix + ctx.height * src.index;
        return column[row];
    }
    case SRC_INTERMEDIATE:
        return ctx.d_intermediates[src.index * buffer_stride];
    case SRC_IS_FIRST: {
        const FpExt *__restrict__ column = ctx.d_selectors;
        return FpExt(column[row]);
    }
    case SRC_IS_LAST: {
        const FpExt *__restrict__ column = ctx.d_selectors + 2 * ctx.height;
        return FpExt(column[row]);
    }
    case SRC_IS_TRANSITION: {
        const FpExt *__restrict__ column = ctx.d_selectors + ctx.height;
        return FpExt(column[row]);
    }
    case ENTRY_CHALLENGE:
#ifdef CUDA_DEBUG
        assert(d_challenges);
#endif
        return d_challenges[src.index];
    case ENTRY_PUBLIC:
        return FpExt(ctx.d_public[src.index]);
    case SRC_CONSTANT:
        return FpExt(Fp(src.index));
    default:
        assert(false);
    }
    return FpExt(Fp::zero());
}

// ============================================================================
// KERNELS
// ============================================================================

__global__ void zerocheck_batch_mle_kernel(
    FpExt *__restrict__ tmp_sums_buffer,
    const BlockCtx *__restrict__ d_block_ctxs,
    const ZerocheckCtx *__restrict__ d_zc_ctxs,
    const FpExt *__restrict__ d_lambda_pows,
    size_t lambda_len
) {
    extern __shared__ char smem[];
    FpExt *shared = (FpExt *)smem;

    BlockCtx block_ctx = d_block_ctxs[blockIdx.x];
    ZerocheckCtx zc_ctx = d_zc_ctxs[block_ctx.air_idx];

    uint32_t num_x = gridDim.y;
    uint32_t x_int = blockIdx.y;
    uint32_t y_int = threadIdx.x + block_ctx.local_block_idx_x * blockDim.x;
    bool const active_thread = (y_int < zc_ctx.num_y);

    FpExt sum(Fp::zero());

    if (active_thread) {
        EvalCtx eval_ctx = zc_ctx.eval_ctx;
        uint32_t buffer_stride = 0;
        uint32_t row = x_int * zc_ctx.num_y + y_int;
        if (zc_ctx.buffer_size > 0) {
            assert(eval_ctx.d_intermediates != nullptr);
            // Match non-batch GLOBAL layout: SoA over tasks with stride = task_stride.
            uint32_t y_int_stride = ((zc_ctx.num_y + blockDim.x - 1) / blockDim.x) * blockDim.x;
            uint32_t task_offset = x_int * y_int_stride + y_int;
            uint32_t task_stride = y_int_stride * num_x;
            eval_ctx.d_intermediates = eval_ctx.d_intermediates + task_offset;
            buffer_stride = task_stride;
        } else {
            eval_ctx.d_intermediates = nullptr;
            buffer_stride = 0;
        }
        uint32_t lambda_idx = 0;

        for (size_t node = 0; node < zc_ctx.rules_len; ++node) {
            Rule rule = zc_ctx.d_rules[node];
            DecodedRule decoded = decode_rule(rule);

            FpExt x_val = evaluate_mle_entry(decoded.x, row, eval_ctx, buffer_stride);
            FpExt result;
            switch (decoded.op) {
            case OP_ADD: {
                FpExt y_val = evaluate_mle_entry(decoded.y, row, eval_ctx, buffer_stride);
                result = x_val + y_val;
                break;
            }
            case OP_SUB: {
                FpExt y_val = evaluate_mle_entry(decoded.y, row, eval_ctx, buffer_stride);
                result = x_val - y_val;
                break;
            }
            case OP_MUL: {
                FpExt y_val = evaluate_mle_entry(decoded.y, row, eval_ctx, buffer_stride);
                result = x_val * y_val;
                break;
            }
            case OP_NEG:
                result = -x_val;
                break;
            case OP_VAR:
                result = x_val;
                break;
            default:
                assert(false);
            }

            if (decoded.buffer_result && zc_ctx.buffer_size > 0) {
                eval_ctx.d_intermediates[decoded.z_index * buffer_stride] = result;
            }

            if (decoded.is_constraint) {
                while (lambda_idx < lambda_len && lambda_idx < zc_ctx.used_nodes_len &&
                       zc_ctx.d_used_nodes[lambda_idx] == node) {
                    uint32_t mapped_idx =
                        zc_ctx.d_lambda_indices ? zc_ctx.d_lambda_indices[lambda_idx] : lambda_idx;
                    FpExt lambda = d_lambda_pows[mapped_idx];
                    lambda_idx++;
                    sum += lambda * result;
                }
            }
        }
        sum *= zc_ctx.d_eq_xi[row];
    }

    FpExt reduced = sumcheck::block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) {
        tmp_sums_buffer[blockIdx.x * num_x + x_int] = reduced;
    }
}

__global__ void logup_batch_mle_kernel(
    FracExt *__restrict__ tmp_sums_buffer,
    const BlockCtx *__restrict__ d_block_ctxs,
    const LogupCtx *__restrict__ d_logup_ctxs
) {
    extern __shared__ char smem[];
    FpExt *shared = reinterpret_cast<FpExt *>(smem);

    BlockCtx block_ctx = d_block_ctxs[blockIdx.x];
    LogupCtx logup_ctx = d_logup_ctxs[block_ctx.air_idx];

    uint32_t num_x = gridDim.y;
    uint32_t x_int = blockIdx.y;
    uint32_t y_int = threadIdx.x + block_ctx.local_block_idx_x * blockDim.x;
    bool const active_thread = (y_int < logup_ctx.num_y);

    FpExt numer_sum = FpExt(Fp::zero());
    FpExt denom_sum = FpExt(Fp::zero());

    if (active_thread) {
        // Match non-batch GLOBAL layout: SoA over tasks with stride = task_stride.
        FpExt *intermediates = nullptr;
        uint32_t buffer_stride = 0;
        if (logup_ctx.buffer_size > 0) {
            assert(logup_ctx.eval_ctx.d_intermediates != nullptr);
            // Unique task slot per (x_int, y_int) for this AIR.
            // y_int_stride is rounded up to blockDim.x so the mapping matches `BlockCtx.local_block_idx_x`.
            uint32_t y_int_stride = ((logup_ctx.num_y + blockDim.x - 1) / blockDim.x) * blockDim.x;
            uint32_t task_offset = x_int * y_int_stride + y_int;
            uint32_t task_stride = y_int_stride * num_x;
            intermediates = logup_ctx.eval_ctx.d_intermediates + task_offset;
            buffer_stride = task_stride;
        }

        // Build local EvalCtx with correct intermediates backing.
        // NOTE: `eval_ctx.d_intermediates` points at the current task's base, so indexing uses
        // `decoded.z_index * buffer_stride`.
        EvalCtx eval_ctx = logup_ctx.eval_ctx;
        eval_ctx.d_intermediates = intermediates;

        uint32_t row = x_int * logup_ctx.num_y + y_int;
        size_t rules_evaluated = 0;

        // Iterate through used_nodes (alternates: numer, denom, numer, denom, ...)
        for (size_t used_idx = 0; used_idx < logup_ctx.used_nodes_len; ++used_idx) {
            size_t node_idx = logup_ctx.d_used_nodes[used_idx];
            FpExt result(Fp::zero());

            if (node_idx < rules_evaluated) {
                // Node already evaluated, get from buffer
                Rule rule = logup_ctx.d_rules[node_idx];
                DecodedRule decoded = decode_rule(rule);
                if (decoded.op == OP_VAR) {
                    result = evaluate_mle_entry(
                        decoded.x, row, eval_ctx, buffer_stride, logup_ctx.d_challenges
                    );
                } else if (logup_ctx.buffer_size > 0 && decoded.buffer_result) {
                    result = eval_ctx.d_intermediates[decoded.z_index * buffer_stride];
                }
            } else {
                // Need to evaluate this node (and all nodes up to it)
                for (; rules_evaluated <= node_idx; ++rules_evaluated) {
                    Rule rule = logup_ctx.d_rules[rules_evaluated];
                    DecodedRule decoded = decode_rule(rule);

                    FpExt x_val = evaluate_mle_entry(
                        decoded.x, row, eval_ctx, buffer_stride, logup_ctx.d_challenges
                    );
                    FpExt node_result;
                    switch (decoded.op) {
                    case OP_ADD: {
                        FpExt y_val = evaluate_mle_entry(
                            decoded.y, row, eval_ctx, buffer_stride, logup_ctx.d_challenges
                        );
                        node_result = x_val + y_val;
                        break;
                    }
                    case OP_SUB: {
                        FpExt y_val = evaluate_mle_entry(
                            decoded.y, row, eval_ctx, buffer_stride, logup_ctx.d_challenges
                        );
                        node_result = x_val - y_val;
                        break;
                    }
                    case OP_MUL: {
                        FpExt y_val = evaluate_mle_entry(
                            decoded.y, row, eval_ctx, buffer_stride, logup_ctx.d_challenges
                        );
                        node_result = x_val * y_val;
                        break;
                    }
                    case OP_NEG:
                        node_result = -x_val;
                        break;
                    case OP_VAR:
                        node_result = x_val;
                        break;
                    default:
                        assert(false);
                    }

                    if (decoded.buffer_result && logup_ctx.buffer_size > 0) {
                        eval_ctx.d_intermediates[decoded.z_index * buffer_stride] = node_result;
                    }

                    // If this is the node we're looking for, use it
                    if (rules_evaluated == node_idx) {
                        result = node_result;
                    }
                }
            }

            // Accumulate to numer or denom based on alternation
            // Weight by eq_3bs[used_idx / 2] (each interaction has numer and denom)
            result *= logup_ctx.d_eq_3bs[used_idx >> 1];

            if (used_idx & 1) {
                denom_sum += result;
            } else {
                numer_sum += result;
            }
        }

        FpExt eq_sharp_val = logup_ctx.d_eq_sharp[row];
        numer_sum *= eq_sharp_val;
        denom_sum *= eq_sharp_val;
    }

    FpExt numer_reduced = sumcheck::block_reduce_sum(numer_sum, shared);
    __syncthreads(); // Ensure first reduction completes before reusing shared memory
    FpExt denom_reduced = sumcheck::block_reduce_sum(denom_sum, shared);
    if (threadIdx.x == 0) {
        tmp_sums_buffer[blockIdx.x * num_x + x_int] = {numer_reduced, denom_reduced};
    }
}

// ============================================================================
// LAUNCHERS
// ============================================================================
//
constexpr uint32_t MAX_THREADS = 128;

// ============================================================================
// SIZE HELPERS (batch kernels)
// ============================================================================
//
// Batch kernels always use global intermediates when buffer_size > 0 (no local-threshold path),
// so we need dedicated sizing helpers instead of `_zerocheck_mle_intermediates_buffer_size` /
// `_logup_mle_intermediates_buffer_size` from `mle.cu`, which return 0 under local thresholds.

// In FpExt elements.
extern "C" size_t _zerocheck_batch_mle_intermediates_buffer_size(
    uint32_t buffer_size,
    uint32_t num_x,
    uint32_t num_y
) {
    if (buffer_size == 0)
        return 0;
    // Must match `zerocheck_batch_mle_kernel`'s y-stride computation.
    uint32_t y_int_stride = ((num_y + MAX_THREADS - 1) / MAX_THREADS) * MAX_THREADS;
    size_t task_stride = static_cast<size_t>(y_int_stride) * static_cast<size_t>(num_x);
    return static_cast<size_t>(buffer_size) * task_stride;
}

// In FpExt elements.
extern "C" size_t _logup_batch_mle_intermediates_buffer_size(
    uint32_t buffer_size,
    uint32_t num_x,
    uint32_t num_y
) {
    if (buffer_size == 0)
        return 0;
    // Must match `logup_batch_mle_kernel`'s y-stride computation.
    uint32_t y_int_stride = ((num_y + MAX_THREADS - 1) / MAX_THREADS) * MAX_THREADS;
    size_t task_stride = static_cast<size_t>(y_int_stride) * static_cast<size_t>(num_x);
    return static_cast<size_t>(buffer_size) * task_stride;
}

extern "C" int _zerocheck_batch_eval_mle(
    FpExt *tmp_sums_buffer,
    FpExt *output,
    const BlockCtx *block_ctxs,
    const ZerocheckCtx *zc_ctxs,
    const uint32_t *air_block_offsets, // size = num_airs + 1, grouped by air_idx
    const FpExt *lambda_pows,
    size_t lambda_len,
    uint32_t num_blocks,
    uint32_t num_x,
    uint32_t num_airs
) {
    dim3 grid(num_blocks, num_x);
    dim3 block(MAX_THREADS);
    size_t shmem_bytes = div_ceil(block.x, WARP_SIZE) * sizeof(FpExt);

    zerocheck_batch_mle_kernel<<<grid, block, shmem_bytes>>>(
        tmp_sums_buffer, block_ctxs, zc_ctxs, lambda_pows, lambda_len
    );
    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Reduce per AIR. block_ctxs are grouped by air_idx using air_block_offsets.
    for (uint32_t air = 0; air < num_airs; ++air) {
        uint32_t start = air_block_offsets[air];
        uint32_t end = air_block_offsets[air + 1];
        uint32_t blocks_for_air = end - start;
        if (blocks_for_air == 0)
            continue;

        auto [reduce_grid, reduce_block] = kernel_launch_params(blocks_for_air);
        unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
        size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
        sumcheck::final_reduce_block_sums<<<num_x, reduce_block, reduce_shmem>>>(
            tmp_sums_buffer + start * num_x, output + air * num_x, blocks_for_air
        );
        err = CHECK_KERNEL();
        if (err != 0)
            return err;
    }

    return 0;
}

extern "C" uint32_t _mle_eval_num_blocks(uint32_t num_x, uint32_t num_y) {
    auto [grid, block] =
        mle_rounds_config::eval_constraints_launch_params(num_x, num_y, MAX_THREADS);
    (void)block;
    return grid.x;
}

extern "C" int _logup_batch_eval_mle(
    FracExt *tmp_sums_buffer,
    FracExt *output,
    const BlockCtx *block_ctxs,
    const LogupCtx *logup_ctxs,
    const uint32_t *air_block_offsets, // size = num_airs + 1, grouped by air_idx
    uint32_t num_blocks,
    uint32_t num_x,
    uint32_t num_airs
) {
    dim3 grid(num_blocks, num_x);
    dim3 block(MAX_THREADS);
    size_t shmem_bytes = div_ceil(block.x, WARP_SIZE) * sizeof(FpExt);

    logup_batch_mle_kernel<<<grid, block, shmem_bytes>>>(tmp_sums_buffer, block_ctxs, logup_ctxs);
    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Launch final reduction kernel per AIR. block_ctxs are grouped by air_idx with offsets provided.
    for (uint32_t air = 0; air < num_airs; ++air) {
        uint32_t start = air_block_offsets[air];
        uint32_t end = air_block_offsets[air + 1];
        uint32_t blocks_for_air = end - start;
        if (blocks_for_air == 0)
            continue;

        auto [reduce_grid, reduce_block] = kernel_launch_params(blocks_for_air);
        unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
        size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);

        // FracExt = (FpExt, FpExt) so we set block = 2 * num_x
        sumcheck::final_reduce_block_sums<<<2 * num_x, reduce_block, reduce_shmem>>>(
            reinterpret_cast<FpExt *>(tmp_sums_buffer + start * num_x),
            reinterpret_cast<FpExt *>(output + air * num_x),
            blocks_for_air
        );
        err = CHECK_KERNEL();
        if (err != 0)
            return err;
    }

    return 0;
}

} // namespace logup_zerocheck_mle
