// logup_zerocheck/batch_mle_monomial - Monomial-based MLE evaluation kernels
// Ported from CUDA: cuda-backend/cuda/src/logup_zerocheck/batch_mle_monomial.cu
#include <metal_stdlib>
using namespace metal;

#include "baby_bear.h"
#include "baby_bear_ext.h"
#include "eval_ctx.h"
#include "frac_ext.h"
#include "monomial.h"
#include "sumcheck.h"

inline const device MonomialHeader *as_monomial_headers(uint64_t ptr) {
    return reinterpret_cast<const device MonomialHeader *>(ptr);
}

inline const device PackedVar *as_packed_vars(uint64_t ptr) {
    return reinterpret_cast<const device PackedVar *>(ptr);
}

// Evaluate a variable from packed representation
inline FpExt eval_variable_monomial(
    PackedVar var,
    uint32_t row,
    EvalCoreCtx ctx,
    uint32_t height
) {
    const device FpExt *d_selectors = as_fpext_ptr(ctx.d_selectors);
    const device MainMatrixPtrsExt *d_main = as_main_matrix_ptrs_ext(ctx.d_main);
    const device Fp *d_public = as_fp_ptr(ctx.d_public);
    const device FpExt *d_preprocessed = as_fpext_ptr(ctx.d_preprocessed.data);
    uint8_t entry_type = var.entry_type();
    uint8_t offset = var.offset();

    switch (entry_type) {
    case 1: { // MAIN
        MainMatrixPtrsExt main_ptr = d_main[var.part_index()];
        uint32_t stride = height * main_ptr.air_width;
        const device FpExt *matrix = as_fpext_ptr(main_ptr.data) + stride * offset;
        const device FpExt *column = matrix + height * var.col_index();
        return column[row];
    }
    case 0: { // PREPROCESSED
        uint32_t stride = height * ctx.d_preprocessed.air_width;
        const device FpExt *matrix = d_preprocessed + stride * offset;
        const device FpExt *column = matrix + height * var.col_index();
        return column[row];
    }
    case 3: // PUBLIC
        return FpExt{d_public[var.col_index()], Fp(0u), Fp(0u), Fp(0u)};
    case 8: // IS_FIRST
        return d_selectors[row];
    case 9: // IS_LAST
        return d_selectors[2 * height + row];
    case 10: // IS_TRANSITION
        return d_selectors[height + row];
    default:
        break;
    }
    return FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
}

// Per-AIR context for batched monomial evaluation
struct MonomialAirCtx {
    uint64_t d_headers;
    uint64_t d_variables;
    uint64_t d_lambda_combinations;
    uint32_t num_monomials;
    EvalCoreCtx eval_ctx;
    uint64_t d_eq_xi;
    uint32_t num_y;
};

// Logup monomial common context
struct LogupMonomialCommonCtx {
    EvalCoreCtx eval_ctx;
    uint64_t d_eq_xi;
    FpExt bus_term_sum;
    uint32_t num_y;
    uint32_t mono_blocks;
};

struct LogupMonomialCtx {
    uint64_t d_headers;
    uint64_t d_variables;
    uint64_t d_combinations;
    uint32_t num_monomials;
};

// Precompute lambda combinations: for each monomial, sum(coefficient * lambda_pows[constraint_idx])
kernel void precompute_lambda_combinations_kernel(
    device FpExt *out [[buffer(0)]],
    const device MonomialHeader *headers [[buffer(1)]],
    const device LambdaTerm *lambda_terms [[buffer(2)]],
    const device FpExt *lambda_pows [[buffer(3)]],
    constant uint32_t &num_monomials [[buffer(4)]],
    uint tidx [[thread_position_in_grid]]
) {
    if (tidx >= num_monomials) return;

    MonomialHeader hdr = headers[tidx];
    FpExt sum = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    for (uint16_t l = 0; l < hdr.num_terms; ++l) {
        LambdaTerm term = lambda_terms[hdr.term_offset + l];
        sum = sum + lambda_pows[term.constraint_idx] * term.coefficient;
    }
    out[tidx] = sum;
}

// Precompute logup numerator combinations
kernel void precompute_logup_numer_combinations_kernel(
    device FpExt *out [[buffer(0)]],
    const device MonomialHeader *headers [[buffer(1)]],
    const device InteractionMonomialTerm *terms [[buffer(2)]],
    const device FpExt *eq_3bs [[buffer(3)]],
    constant uint32_t &num_monomials [[buffer(4)]],
    uint tidx [[thread_position_in_grid]]
) {
    if (tidx >= num_monomials) return;

    MonomialHeader hdr = headers[tidx];
    FpExt sum = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    for (uint16_t t = 0; t < hdr.num_terms; ++t) {
        InteractionMonomialTerm term = terms[hdr.term_offset + t];
        sum = sum + eq_3bs[term.interaction_idx] * term.coefficient;
    }
    out[tidx] = sum;
}

// Precompute logup denominator combinations
kernel void precompute_logup_denom_combinations_kernel(
    device FpExt *out [[buffer(0)]],
    const device MonomialHeader *headers [[buffer(1)]],
    const device InteractionMonomialTerm *terms [[buffer(2)]],
    const device FpExt *beta_pows [[buffer(3)]],
    const device FpExt *eq_3bs [[buffer(4)]],
    constant uint32_t &num_monomials [[buffer(5)]],
    uint tidx [[thread_position_in_grid]]
) {
    if (tidx >= num_monomials) return;

    MonomialHeader hdr = headers[tidx];
    FpExt sum = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    for (uint16_t t = 0; t < hdr.num_terms; ++t) {
        InteractionMonomialTerm term = terms[hdr.term_offset + t];
        FpExt value = eq_3bs[term.interaction_idx] * term.coefficient;
        value = value * beta_pows[term.field_idx];
        sum = sum + value;
    }
    out[tidx] = sum;
}

// Zerocheck monomial kernel: each thread evaluates one monomial
kernel void zerocheck_monomial_kernel(
    device FpExt *tmp_sums [[buffer(0)]],
    const device BlockCtx *block_ctxs [[buffer(1)]],
    const device MonomialAirCtx *air_ctxs [[buffer(2)]],
    constant uint32_t &threads_per_block [[buffer(3)]],
    constant uint32_t &num_x [[buffer(4)]],
    threadgroup FpExt *shared [[threadgroup(0)]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tg_size2 [[threads_per_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    BlockCtx bctx = block_ctxs[gid.x];
    MonomialAirCtx actx = air_ctxs[bctx.air_idx];

    uint32_t x_int = gid.y;

    uint32_t mono_blocks = (actx.num_monomials + threads_per_block - 1) / threads_per_block;
    uint32_t y_int = bctx.local_block_idx_x / mono_blocks;
    uint32_t mono_block = bctx.local_block_idx_x % mono_blocks;

    uint32_t height = num_x * actx.num_y;
    uint32_t row = x_int * actx.num_y + y_int;

    const device MonomialHeader *d_headers = as_monomial_headers(actx.d_headers);
    const device PackedVar *d_variables = as_packed_vars(actx.d_variables);
    const device FpExt *d_lambda_combinations = as_fpext_ptr(actx.d_lambda_combinations);
    const device FpExt *d_eq_xi = as_fpext_ptr(actx.d_eq_xi);

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt sum = zero;

    uint32_t m = mono_block * threads_per_block + tid;
    if (m < actx.num_monomials) {
        MonomialHeader hdr = d_headers[m];
        FpExt product = FpExt(Fp(1u));
        for (uint16_t v = 0; v < hdr.num_vars; ++v) {
            PackedVar var = d_variables[hdr.var_offset + v];
            product = product * eval_variable_monomial(var, row, actx.eval_ctx, height);
        }
        sum = product * d_lambda_combinations[m];
    }

    FpExt reduced = block_reduce_sum(sum, shared, tid, tg_size);

    if (tid == 0) {
        reduced = reduced * d_eq_xi[y_int];
        tmp_sums[gid.x * num_x + x_int] = reduced;
    }
}

// Zerocheck monomial par-y kernel: parallelizes over y_int, tiles monomials
kernel void zerocheck_monomial_par_y_kernel(
    device FpExt *tmp_sums [[buffer(0)]],
    const device BlockCtx *block_ctxs [[buffer(1)]],
    const device MonomialAirCtx *air_ctxs [[buffer(2)]],
    constant uint32_t &threads_per_block [[buffer(3)]],
    constant uint32_t &chunk_size [[buffer(4)]],
    constant uint32_t &num_x [[buffer(5)]],
    threadgroup FpExt *shared [[threadgroup(0)]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tg_size2 [[threads_per_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    BlockCtx bctx = block_ctxs[gid.x];
    MonomialAirCtx actx = air_ctxs[bctx.air_idx];

    uint32_t x_int = gid.y;

    uint32_t air_mono_chunks = (actx.num_monomials + chunk_size - 1) / chunk_size;
    uint32_t y_block = bctx.local_block_idx_x / air_mono_chunks;
    uint32_t mono_chunk = bctx.local_block_idx_x % air_mono_chunks;

    uint32_t y_int = tid + y_block * threads_per_block;
    bool active = (y_int < actx.num_y);

    uint32_t height = num_x * actx.num_y;
    uint32_t row = x_int * actx.num_y + y_int;

    uint32_t mono_start = mono_chunk * chunk_size;
    uint32_t mono_end = min(mono_start + chunk_size, actx.num_monomials);

    const device MonomialHeader *d_headers = as_monomial_headers(actx.d_headers);
    const device PackedVar *d_variables = as_packed_vars(actx.d_variables);
    const device FpExt *d_lambda_combinations = as_fpext_ptr(actx.d_lambda_combinations);
    const device FpExt *d_eq_xi = as_fpext_ptr(actx.d_eq_xi);

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt sum = zero;
    if (active) {
        for (uint32_t m = mono_start; m < mono_end; ++m) {
            MonomialHeader hdr = d_headers[m];

            FpExt product = FpExt(Fp(1u));
            for (uint16_t v = 0; v < hdr.num_vars; ++v) {
                PackedVar var = d_variables[hdr.var_offset + v];
                product = product * eval_variable_monomial(var, row, actx.eval_ctx, height);
            }

            sum = sum + product * d_lambda_combinations[m];
        }
        sum = sum * d_eq_xi[y_int];
    }

    FpExt reduced = block_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) {
        tmp_sums[gid.x * num_x + x_int] = reduced;
    }
}

// Logup monomial numerator kernel
kernel void logup_monomial_numer_kernel(
    device FpExt *tmp_sums_p [[buffer(0)]],
    const device BlockCtx *block_ctxs [[buffer(1)]],
    const device LogupMonomialCommonCtx *common_ctxs [[buffer(2)]],
    const device LogupMonomialCtx *ctxs [[buffer(3)]],
    constant uint32_t &num_x [[buffer(4)]],
    threadgroup FpExt *shared [[threadgroup(0)]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tg_size2 [[threads_per_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    BlockCtx bctx = block_ctxs[gid.x];
    LogupMonomialCommonCtx common_ctx = common_ctxs[bctx.air_idx];
    LogupMonomialCtx ctx = ctxs[bctx.air_idx];

    uint32_t x_int = gid.y;
    uint32_t height = num_x * common_ctx.num_y;

    uint32_t mono_blocks = common_ctx.mono_blocks;
    uint32_t y_int = bctx.local_block_idx_x / mono_blocks;
    uint32_t mono_block = bctx.local_block_idx_x % mono_blocks;
    uint32_t row = x_int * common_ctx.num_y + y_int;
    uint32_t m = mono_block * tg_size + tid;

    const device MonomialHeader *d_headers = as_monomial_headers(ctx.d_headers);
    const device PackedVar *d_variables = as_packed_vars(ctx.d_variables);
    const device FpExt *d_combinations = as_fpext_ptr(ctx.d_combinations);
    const device FpExt *d_eq_xi = as_fpext_ptr(common_ctx.d_eq_xi);

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt sum = zero;
    if (y_int < common_ctx.num_y && m < ctx.num_monomials) {
        MonomialHeader hdr = d_headers[m];
        FpExt monomial = d_combinations[m];
        for (uint16_t v = 0; v < hdr.num_vars; ++v) {
            PackedVar var = d_variables[hdr.var_offset + v];
            monomial = monomial * eval_variable_monomial(var, row, common_ctx.eval_ctx, height);
        }
        sum = monomial * d_eq_xi[y_int];
    }

    FpExt reduced = block_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) {
        tmp_sums_p[gid.x * num_x + x_int] = reduced;
    }
}

// Logup monomial denominator kernel
kernel void logup_monomial_denom_kernel(
    device FpExt *tmp_sums_q [[buffer(0)]],
    const device BlockCtx *block_ctxs [[buffer(1)]],
    const device LogupMonomialCommonCtx *common_ctxs [[buffer(2)]],
    const device LogupMonomialCtx *ctxs [[buffer(3)]],
    constant uint32_t &num_x [[buffer(4)]],
    threadgroup FpExt *shared [[threadgroup(0)]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tg_size2 [[threads_per_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    BlockCtx bctx = block_ctxs[gid.x];
    LogupMonomialCommonCtx common_ctx = common_ctxs[bctx.air_idx];
    LogupMonomialCtx ctx = ctxs[bctx.air_idx];

    uint32_t x_int = gid.y;
    uint32_t height = num_x * common_ctx.num_y;

    uint32_t mono_blocks = common_ctx.mono_blocks;
    uint32_t y_int = bctx.local_block_idx_x / mono_blocks;
    uint32_t mono_block = bctx.local_block_idx_x % mono_blocks;
    uint32_t row = x_int * common_ctx.num_y + y_int;
    uint32_t m = mono_block * tg_size + tid;

    const device MonomialHeader *d_headers = as_monomial_headers(ctx.d_headers);
    const device PackedVar *d_variables = as_packed_vars(ctx.d_variables);
    const device FpExt *d_combinations = as_fpext_ptr(ctx.d_combinations);
    const device FpExt *d_eq_xi = as_fpext_ptr(common_ctx.d_eq_xi);

    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    FpExt sum = zero;
    if (y_int < common_ctx.num_y && m < ctx.num_monomials) {
        MonomialHeader hdr = d_headers[m];
        FpExt monomial = d_combinations[m];
        for (uint16_t v = 0; v < hdr.num_vars; ++v) {
            PackedVar var = d_variables[hdr.var_offset + v];
            monomial = monomial * eval_variable_monomial(var, row, common_ctx.eval_ctx, height);
        }
        sum = monomial * d_eq_xi[y_int];
    }

    FpExt reduced = block_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) {
        if (mono_block == 0 && y_int < common_ctx.num_y) {
            reduced = reduced + common_ctx.bus_term_sum * d_eq_xi[y_int];
        }
        tmp_sums_q[gid.x * num_x + x_int] = reduced;
    }
}
