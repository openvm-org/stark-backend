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

/// The raw-pointer twin of [`EvalCoreCtx`] -- the ORIGINAL, pre-base+offset
/// ABI, kept for the eager prover.
///
/// It shares no field type, no sentinel and no decode function with
/// [`EvalCoreCtx`]; `resolve_eval_core` below is a plain field copy. That is
/// deliberate. Every equality test in the graph-IR port checks a graph result
/// against an eager one, so if both encoded and decoded `BaseOff` a wrong null
/// sentinel or a Rust/C++ layout drift would be applied identically to both
/// sides and no test could see it. See `base_off.cuh` for the graph ABI.
struct EvalCoreCtxRaw {
    const FpExt *d_selectors;
    MainMatrixPtrs<FpExt> d_preprocessed;
    const MainMatrixPtrs<FpExt> *d_main;
    const Fp *d_public;
};

/// The decoded, device-local form of [`EvalCoreCtx`] / [`EvalCoreCtxRaw`].
///
/// `d_main` stays an array whose element count is only known from the rule
/// stream, so its entries are resolved on demand: `MainT = MainMatrixDesc`
/// resolves against `base`, `MainT = MainMatrixPtrs<FpExt>` is already
/// resolved. Both go through `resolve_main_matrix`, which is overloaded on the
/// element type (`matrix.cuh`).
template <typename MainT> struct EvalCoreRTT {
    /// The main-matrix array's element type, so a kernel holding an `auto`
    /// resolved context can name it.
    using MainType = MainT;

    const FpExt *__restrict__ d_selectors;
    MainMatrixPtrs<FpExt> d_preprocessed;
    const MainT *__restrict__ d_main;
    const Fp *__restrict__ d_public;
    const uint8_t *base;
};

/// The base+offset (graph-IR) decoded form.
using EvalCoreRT = EvalCoreRTT<MainMatrixDesc>;
/// The raw-pointer (eager) form.
using EvalCoreRawRT = EvalCoreRTT<MainMatrixPtrs<FpExt>>;

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

/// The eager path's "decode": a field copy. No arithmetic, no sentinel.
__host__ __device__ __forceinline__ EvalCoreRawRT
resolve_eval_core(const EvalCoreCtxRaw &c, const uint8_t * /*base*/) {
    return EvalCoreRawRT{c.d_selectors, c.d_preprocessed, c.d_main, c.d_public, nullptr};
}

} // namespace logup_zerocheck_mle
