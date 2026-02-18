// Evaluation context for Metal
// Translated from CUDA: cuda-backend/cuda/include/eval_ctx.cuh
#pragma once

#include "baby_bear.h"
#include "baby_bear_ext.h"

// Matrix pointer with width information
// Metal equivalent of MainMatrixPtrs<T>
struct MainMatrixPtrsExt {
    const device FpExt *data;
    uint32_t air_width;
};

struct MainMatrixPtrsFp {
    const device Fp *data;
    uint32_t air_width;
};

struct EvalCoreCtx {
    const device FpExt *d_selectors;
    MainMatrixPtrsExt d_preprocessed;
    const device MainMatrixPtrsExt *d_main;
    const device Fp *d_public;
};

struct BlockCtx {
    uint32_t local_block_idx_x;
    uint32_t air_idx;
};
