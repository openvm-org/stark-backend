#include "eval_ctx.cuh"
#include "frac_ext.cuh"
#include "launcher.cuh"
#include "monomial.cuh"
#include "sumcheck.cuh"

#include <cassert>
#include <cstdint>

namespace logup_zerocheck_mle {

__device__ __forceinline__ FpExt
eval_variable(PackedVar var, uint32_t row, const EvalCoreCtx &ctx, uint32_t height) {
    uint8_t entry_type = var.entry_type();
    uint8_t offset = var.offset();

    switch (entry_type) {
    case 1: { // MAIN
        auto main_ptr = ctx.d_main[var.part_index()];
        const auto stride = height * main_ptr.air_width;
        const FpExt *__restrict__ matrix = main_ptr.data + stride * offset;
        const FpExt *__restrict__ column = matrix + height * var.col_index();
        return column[row];
    }
    case 0: { // PREPROCESSED
        const auto stride = height * ctx.d_preprocessed.air_width;
        const FpExt *__restrict__ matrix = ctx.d_preprocessed.data + stride * offset;
        const FpExt *__restrict__ column = matrix + height * var.col_index();
        return column[row];
    }
    case 3: // PUBLIC
        return FpExt(ctx.d_public[var.col_index()]);
    case 8: // IS_FIRST
        return ctx.d_selectors[row];
    case 9: // IS_LAST
        return ctx.d_selectors[2 * height + row];
    case 10: // IS_TRANSITION
        return ctx.d_selectors[height + row];
    default:
        assert(false);
    }
    return FpExt(Fp::zero());
}

// For batched monomial evaluation across multiple AIRs
struct MonomialAirCtx {
    const MonomialHeader *__restrict__ d_headers;
    const PackedVar *__restrict__ d_variables;
    const FpExt *__restrict__ d_lambda_combinations; // Precomputed per-monomial
    uint32_t num_monomials;
    EvalCoreCtx eval_ctx;
    const FpExt *__restrict__ d_eq_xi;
    uint32_t num_y;
};

// ============================================================================
// PRECOMPUTE LAMBDA COMBINATIONS (once per AIR after lambda is sampled)
// ============================================================================

__global__ void precompute_lambda_combinations_kernel(
    FpExt *__restrict__ out,
    const MonomialHeader *__restrict__ headers,
    const LambdaTerm *__restrict__ lambda_terms,
    const FpExt *__restrict__ lambda_pows,
    uint32_t num_monomials
) {
    uint32_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_monomials)
        return;

    MonomialHeader hdr = headers[m];
    FpExt sum(Fp::zero());
    for (uint16_t l = 0; l < hdr.num_terms; ++l) {
        LambdaTerm term = lambda_terms[hdr.term_offset + l];
        sum += FpExt(term.coefficient) * lambda_pows[term.constraint_idx];
    }
    out[m] = sum;
}

extern "C" int _precompute_lambda_combinations(
    FpExt *out,
    const MonomialHeader *headers,
    const LambdaTerm *lambda_terms,
    const FpExt *lambda_pows,
    uint32_t num_monomials
) {
    if (num_monomials == 0)
        return 0;

    constexpr uint32_t threads = 256;
    uint32_t blocks = div_ceil(num_monomials, threads);
    precompute_lambda_combinations_kernel<<<blocks, threads>>>(
        out, headers, lambda_terms, lambda_pows, num_monomials
    );
    return CHECK_KERNEL();
}

// ============================================================================
// BATCHED KERNEL (multiple AIRs in single launch)
// ============================================================================

__global__ void zerocheck_monomial_kernel(
    FpExt *__restrict__ tmp_sums,
    const BlockCtx *__restrict__ block_ctxs,
    const MonomialAirCtx *__restrict__ air_ctxs,
    uint32_t threads_per_block
) {
    extern __shared__ char smem[];
    FpExt *shared = (FpExt *)smem;

    BlockCtx bctx = block_ctxs[blockIdx.x];
    MonomialAirCtx actx = air_ctxs[bctx.air_idx];

    uint32_t num_x = gridDim.y;
    uint32_t x_int = blockIdx.y;

    // Decode y_int and mono_block from local_block_idx_x
    uint32_t mono_blocks = (actx.num_monomials + threads_per_block - 1) / threads_per_block;
    uint32_t y_int = bctx.local_block_idx_x / mono_blocks;
    uint32_t mono_block = bctx.local_block_idx_x % mono_blocks;

    uint32_t height = num_x * actx.num_y;
    uint32_t row = x_int * actx.num_y + y_int;

    FpExt sum(Fp::zero());

    // Each thread evaluates one monomial
    uint32_t m = mono_block * threads_per_block + threadIdx.x;
    if (m < actx.num_monomials) {
        MonomialHeader hdr = actx.d_headers[m];

        // Evaluate product of variables
        FpExt product(Fp::one());
        for (uint16_t v = 0; v < hdr.num_vars; ++v) {
            PackedVar var = actx.d_variables[hdr.var_offset + v];
            product *= eval_variable(var, row, actx.eval_ctx, height);
        }

        sum = product * actx.d_lambda_combinations[m];
    }

    // Block reduction
    FpExt reduced = sumcheck::block_reduce_sum(sum, shared);

    if (threadIdx.x == 0) {
        // Apply eq_xi here (eliminates separate phase 2)
        // eq_xi is eq(xi, y) which only depends on y, not x
        reduced *= actx.d_eq_xi[y_int];
        tmp_sums[blockIdx.x * num_x + x_int] = reduced;
    }
}

extern "C" int _zerocheck_monomial_batched(
    FpExt *tmp_sums,
    FpExt *output,
    const BlockCtx *block_ctxs,
    const MonomialAirCtx *air_ctxs,
    const uint32_t *air_block_offsets,
    uint32_t num_blocks,
    uint32_t num_x,
    uint32_t num_airs,
    uint32_t threads_per_block
) {
    if (num_blocks == 0) {
        return 0;
    }

    dim3 grid(num_blocks, num_x);
    dim3 block(threads_per_block);
    size_t shmem = div_ceil(block.x, WARP_SIZE) * sizeof(FpExt);

    // Phase 1: Main monomial evaluation kernel
    zerocheck_monomial_kernel<<<grid, block, shmem>>>(
        tmp_sums, block_ctxs, air_ctxs, threads_per_block
    );
    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Phase 2: Batched reduction for all AIRs in single launch
    // Grid: (num_airs, num_x) - each block handles one (air, x) pair
    auto [_, reduce_block] = kernel_launch_params(num_blocks / num_airs + 1);
    unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);

    dim3 reduce_grid(num_airs, num_x);
    sumcheck::batched_final_reduce_block_sums<<<reduce_grid, reduce_block, reduce_shmem>>>(
        tmp_sums, output, air_block_offsets, num_x
    );

    return CHECK_KERNEL();
}

// ============================================================================
// PAR-Y KERNEL (parallelizes over y_int, tiles monomials)
// ============================================================================

__global__ void zerocheck_monomial_par_y_kernel(
    FpExt *__restrict__ tmp_sums,
    const BlockCtx *__restrict__ block_ctxs,
    const MonomialAirCtx *__restrict__ air_ctxs,
    uint32_t threads_per_block,
    uint32_t chunk_size // monomials per mono_chunk
) {
    extern __shared__ char smem[];
    FpExt *shared = (FpExt *)smem;

    BlockCtx bctx = block_ctxs[blockIdx.x];
    MonomialAirCtx actx = air_ctxs[bctx.air_idx];

    uint32_t num_x = gridDim.y;
    uint32_t x_int = blockIdx.y;

    // Decode y_block and mono_chunk from local_block_idx_x
    // Encoding: local_block_idx_x = y_block * air_mono_chunks + mono_chunk
    uint32_t air_mono_chunks = (actx.num_monomials + chunk_size - 1) / chunk_size;
    uint32_t y_block = bctx.local_block_idx_x / air_mono_chunks;
    uint32_t mono_chunk = bctx.local_block_idx_x % air_mono_chunks;

    // Decode y_int from thread position within y_block
    uint32_t y_int = threadIdx.x + y_block * threads_per_block;
    bool active = (y_int < actx.num_y);

    uint32_t height = num_x * actx.num_y;
    uint32_t row = x_int * actx.num_y + y_int;

    // Contiguous chunk of monomials for this mono_chunk
    uint32_t mono_start = mono_chunk * chunk_size;
    uint32_t mono_end = min(mono_start + chunk_size, actx.num_monomials);

    FpExt sum(Fp::zero());
    if (active) {
        // Each thread loops over its chunk of monomials
        for (uint32_t m = mono_start; m < mono_end; ++m) {
            MonomialHeader hdr = actx.d_headers[m];

            FpExt product(Fp::one());
            for (uint16_t v = 0; v < hdr.num_vars; ++v) {
                PackedVar var = actx.d_variables[hdr.var_offset + v];
                product *= eval_variable(var, row, actx.eval_ctx, height);
            }

            sum += product * actx.d_lambda_combinations[m];
        }

        // Apply eq_xi
        sum *= actx.d_eq_xi[y_int];
    }

    // Block reduction sums across y_int values in this block
    FpExt reduced = sumcheck::block_reduce_sum(sum, shared);

    if (threadIdx.x == 0) {
        tmp_sums[blockIdx.x * num_x + x_int] = reduced;
    }
}

extern "C" int _zerocheck_monomial_par_y_batched(
    FpExt *tmp_sums,
    FpExt *output,
    const BlockCtx *block_ctxs,
    const MonomialAirCtx *air_ctxs,
    const uint32_t *air_block_offsets,
    uint32_t num_blocks,
    uint32_t num_x,
    uint32_t num_airs,
    uint32_t chunk_size,
    uint32_t threads_per_block
) {
    if (num_blocks == 0) {
        return 0;
    }

    dim3 grid(num_blocks, num_x);
    dim3 block(threads_per_block);
    size_t shmem = div_ceil(block.x, WARP_SIZE) * sizeof(FpExt);

    // Phase 1: Main par-y kernel
    zerocheck_monomial_par_y_kernel<<<grid, block, shmem>>>(
        tmp_sums, block_ctxs, air_ctxs, threads_per_block, chunk_size
    );
    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Phase 2: Batched reduction for all AIRs in single launch
    auto [_, reduce_block] = kernel_launch_params(num_blocks / num_airs + 1);
    unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);

    dim3 reduce_grid(num_airs, num_x);
    sumcheck::batched_final_reduce_block_sums<<<reduce_grid, reduce_block, reduce_shmem>>>(
        tmp_sums, output, air_block_offsets, num_x
    );

    return CHECK_KERNEL();
}

// ============================================================================
// LOGUP MONOMIAL KERNELS
// ============================================================================

template <bool IS_DENOM>
__global__ void precompute_logup_combinations_kernel(
    FpExt *__restrict__ out,
    const MonomialHeader *__restrict__ headers,
    const InteractionMonomialTerm *__restrict__ terms,
    const FpExt *__restrict__ beta_pows,
    const FpExt *__restrict__ eq_3bs,
    uint32_t num_monomials
) {
    uint32_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_monomials)
        return;

    MonomialHeader hdr = headers[m];
    FpExt sum(Fp::zero());
    for (uint16_t t = 0; t < hdr.num_terms; ++t) {
        InteractionMonomialTerm term = terms[hdr.term_offset + t];
        FpExt value = FpExt(term.coefficient) * eq_3bs[term.interaction_idx];
        if constexpr (IS_DENOM) {
            value *= beta_pows[term.field_idx];
        }
        sum += value;
    }
    out[m] = sum;
}

extern "C" int _precompute_logup_numer_combinations(
    FpExt *out,
    const MonomialHeader *headers,
    const InteractionMonomialTerm *terms,
    const FpExt *eq_3bs,
    uint32_t num_monomials
) {
    if (num_monomials == 0)
        return 0;

    constexpr uint32_t threads = 256;
    uint32_t blocks = div_ceil(num_monomials, threads);
    precompute_logup_combinations_kernel<false><<<blocks, threads>>>(
        out, headers, terms, nullptr, eq_3bs, num_monomials
    );
    return CHECK_KERNEL();
}

extern "C" int _precompute_logup_denom_combinations(
    FpExt *out,
    const MonomialHeader *headers,
    const InteractionMonomialTerm *terms,
    const FpExt *beta_pows,
    const FpExt *eq_3bs,
    uint32_t num_monomials
) {
    if (num_monomials == 0)
        return 0;

    constexpr uint32_t threads = 256;
    uint32_t blocks = div_ceil(num_monomials, threads);
    precompute_logup_combinations_kernel<true><<<blocks, threads>>>(
        out, headers, terms, beta_pows, eq_3bs, num_monomials
    );
    return CHECK_KERNEL();
}

// AIR context for logup monomial MLE evaluation
// Note: These structures match the Rust LogupMonomial*Ctx types
struct LogupMonomialCommonCtx {
    EvalCoreCtx eval_ctx;
    const FpExt *__restrict__ d_eq_xi;
    FpExt bus_term_sum; // Precomputed sum_i(beta[message_len_i] * (bus_idx[i]+1) * eq_3bs[i])
    uint32_t num_y;
    uint32_t mono_blocks;
};

struct LogupMonomialCtx {
    const MonomialHeader *__restrict__ d_headers;
    const PackedVar *__restrict__ d_variables;
    const FpExt *__restrict__ d_combinations;
    uint32_t num_monomials;
};

// Logup monomial MLE kernel - evaluates numerator or denominator monomials per block.
// This produces FracExt output compatible with the standard reduction pattern.
//
// Block assignment: one block per (y_int, mono_block).
// Each block evaluates a chunk of monomials across threads.
//
// Template parameter IS_DENOM:
//   - false: numerator kernel, writes to tmp_sums[...].p
//   - true: denominator kernel, writes to tmp_sums[...].q and adds bus_term_sum when mono_block == 0
template <bool IS_DENOM>
__global__ void logup_monomial_kernel(
    FracExt *__restrict__ tmp_sums,
    const BlockCtx *__restrict__ block_ctxs,
    const LogupMonomialCommonCtx *__restrict__ common_ctxs,
    const LogupMonomialCtx *__restrict__ ctxs
) {
    extern __shared__ char smem[];
    FpExt *shared = (FpExt *)smem;

    BlockCtx bctx = block_ctxs[blockIdx.x];
    LogupMonomialCommonCtx common_ctx = common_ctxs[bctx.air_idx];
    LogupMonomialCtx ctx = ctxs[bctx.air_idx];

    uint32_t num_x = gridDim.y;
    uint32_t x_int = blockIdx.y;
    uint32_t height = num_x * common_ctx.num_y;

    uint32_t mono_blocks = common_ctx.mono_blocks;
#ifdef CUDA_DEBUG
    assert(mono_blocks > 0);
#endif
    uint32_t y_int = bctx.local_block_idx_x / mono_blocks;
    uint32_t mono_block = bctx.local_block_idx_x % mono_blocks;
    uint32_t row = x_int * common_ctx.num_y + y_int;
    uint32_t m = mono_block * blockDim.x + threadIdx.x;

    FpExt sum(Fp::zero());
    if (y_int < common_ctx.num_y && m < ctx.num_monomials) {
        MonomialHeader hdr = ctx.d_headers[m];
        FpExt monomial = ctx.d_combinations[m];
        for (uint16_t v = 0; v < hdr.num_vars; ++v) {
            PackedVar var = ctx.d_variables[hdr.var_offset + v];
            monomial *= eval_variable(var, row, common_ctx.eval_ctx, height);
        }
        sum = monomial * common_ctx.d_eq_xi[y_int];
    }

    FpExt reduced = sumcheck::block_reduce_sum(sum, shared);

    if (threadIdx.x == 0) {
        if constexpr (IS_DENOM) {
            if (mono_block == 0 && y_int < common_ctx.num_y) {
                reduced += common_ctx.bus_term_sum * common_ctx.d_eq_xi[y_int];
            }
            tmp_sums[blockIdx.x * num_x + x_int].q = reduced;
        } else {
            tmp_sums[blockIdx.x * num_x + x_int].p = reduced;
        }
    }
}

extern "C" int _logup_monomial_batched(
    FracExt *tmp_sums,
    FracExt *output,
    const BlockCtx *block_ctxs,
    const LogupMonomialCommonCtx *common_ctxs,
    const LogupMonomialCtx *numer_ctxs,
    const LogupMonomialCtx *denom_ctxs,
    const uint32_t *air_block_offsets,
    uint32_t num_blocks,
    uint32_t num_x,
    uint32_t num_airs,
    uint32_t threads_per_block
) {
    if (num_blocks == 0)
        return 0;

    dim3 grid(num_blocks, num_x);
    dim3 block(threads_per_block);
    size_t shmem = div_ceil(threads_per_block, WARP_SIZE) * sizeof(FpExt);

    // Phase 1: Evaluate numerator monomials
    logup_monomial_kernel<false><<<grid, block, shmem>>>(
        tmp_sums, block_ctxs, common_ctxs, numer_ctxs
    );
    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Phase 1b: Evaluate denominator monomials
    logup_monomial_kernel<true><<<grid, block, shmem>>>(
        tmp_sums, block_ctxs, common_ctxs, denom_ctxs
    );
    err = CHECK_KERNEL();
    if (err != 0)
        return err;

    // Phase 2: Batched reduction
    // FracExt = (FpExt, FpExt) so treat as 2 * num_x elements per AIR
    auto [_, reduce_block] = kernel_launch_params(num_blocks / num_airs + 1);
    unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);

    dim3 reduce_grid(num_airs, 2 * num_x);
    sumcheck::batched_final_reduce_block_sums<<<reduce_grid, reduce_block, reduce_shmem>>>(
        reinterpret_cast<FpExt *>(tmp_sums),
        reinterpret_cast<FpExt *>(output),
        air_block_offsets,
        2 * num_x
    );

    return CHECK_KERNEL();
}

} // namespace logup_zerocheck_mle
