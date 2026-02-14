#pragma once

#include "fp.h"
#include "fpext.h"
#include "matrix.cuh"

#include <cstdint>

namespace logup_zerocheck_mle {

struct EvalCoreCtx {
    const FpExt *__restrict__ d_selectors;
    const MainMatrixPtrs<FpExt> d_preprocessed;
    const MainMatrixPtrs<FpExt> *__restrict__ d_main;
    const Fp *__restrict__ d_public;
};

struct BlockCtx {
    uint32_t local_block_idx_x;
    uint32_t air_idx;
};

} // namespace logup_zerocheck_mle
