// Evaluation context for Metal
// Translated from CUDA: cuda-backend/cuda/include/eval_ctx.cuh
#pragma once

#include "baby_bear.h"
#include "baby_bear_ext.h"

// Matrix pointer with width information
// Metal equivalent of MainMatrixPtrs<T>
struct MainMatrixPtrsExt {
    uint64_t data;
    uint32_t air_width;
};

struct MainMatrixPtrsFp {
    uint64_t data;
    uint32_t air_width;
};

struct EvalCoreCtx {
    uint64_t d_selectors;
    MainMatrixPtrsExt d_preprocessed;
    uint64_t d_main;
    uint64_t d_public;
};

inline const device FpExt *as_fpext_ptr(uint64_t ptr) {
    return reinterpret_cast<const device FpExt *>(ptr);
}

inline const device Fp *as_fp_ptr(uint64_t ptr) {
    return reinterpret_cast<const device Fp *>(ptr);
}

inline const device MainMatrixPtrsExt *as_main_matrix_ptrs_ext(uint64_t ptr) {
    return reinterpret_cast<const device MainMatrixPtrsExt *>(ptr);
}

struct BlockCtx {
    uint32_t local_block_idx_x;
    uint32_t air_idx;
};
