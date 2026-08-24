#pragma once

#include "base_off.cuh"
#include "block_ctx.cuh"
#include "fp.h"
#include "fpext.h"
#include "matrix.cuh"

#include <cstdint>

namespace logup_zerocheck_mle {

/// The core evaluator inputs shared by every batched context struct, in the
/// base+offset ABI: every device pointer is a `BaseOff` into the pool whose
/// base the launcher receives. See `base_off.cuh`.
///
/// `d_main` addresses an array of [`MainMatrixDesc`] -- descriptors, not
/// pointers -- so no level of this structure embeds a device address.
struct EvalCoreCtx {
    BaseOff d_selectors;
    MainMatrixDesc d_preprocessed;
    BaseOff d_main;
    BaseOff d_public;
};

/// The decoded, device-local form of [`EvalCoreCtx`].
///
/// `d_main` stays a descriptor array: its element count is only known from the
/// rule stream, so its entries are resolved on demand against `base`.
struct EvalCoreRT {
    const FpExt *__restrict__ d_selectors;
    MainMatrixPtrs<FpExt> d_preprocessed;
    const MainMatrixDesc *__restrict__ d_main;
    const Fp *__restrict__ d_public;
    const uint8_t *base;
};

__host__ __device__ __forceinline__ EvalCoreRT
resolve_eval_core(const EvalCoreCtx &c, const uint8_t *base) {
    return EvalCoreRT{
        base_off_ptr<const FpExt>(base, c.d_selectors),
        resolve_main_matrix<FpExt>(c.d_preprocessed, base),
        base_off_ptr<const MainMatrixDesc>(base, c.d_main),
        base_off_ptr<const Fp>(base, c.d_public),
        base
    };
}

} // namespace logup_zerocheck_mle
