#pragma once

#include <cstdint>

// ============================================================================
// The base+offset device-pointer ABI.
// ============================================================================
//
// The batched logup-zerocheck entry points are driven by arrays of `#[repr(C)]`
// context structs (`ZerocheckCtx`, `LogupCtx`, `MonomialAirCtx`, ...). Those
// structs used to embed raw device pointers, which makes them opaque to the
// graph-IR compiler: a struct of host-baked addresses is a leaf and alias
// analysis through it is impossible.
//
// Instead every device-pointer field is a `BaseOff` -- a byte offset from a
// single `pool_base` pointer that the *launcher* takes as an argument. The
// arrays therefore hold integers, not pointers.
//
// Two producers encode into the same ABI:
//
//   * graph-IR path: `pool_base` is the base of the `GraphExe`'s unified device
//     pool and `off` is `GraphExe::plan().offsets[b]` (+ an intra-buffer byte
//     offset). Both terms are constant for the exe's lifetime, which is the
//     same invariant that makes CUDA-graph capture legal.
//   * eager path: `pool_base` is `nullptr` and `off` is the absolute device
//     address. `base + off` then reproduces the original pointer exactly, so
//     the eager path's behaviour is unchanged.
//
// `BASE_OFF_NULL` is the "absent" encoding (`d_preprocessed.data` when the AIR
// has no preprocessed trace, `d_intermediates` when `buffer_size == 0`). Offset
// 0 cannot serve as the null sentinel: it is a perfectly valid pool offset --
// the first packed buffer lives there.

struct BaseOff {
    uint64_t off;
};

/// The "absent" offset. Decodes to `nullptr` for every base.
static constexpr uint64_t BASE_OFF_NULL = ~static_cast<uint64_t>(0);

/// Decode one `BaseOff` against the pool base.
///
/// Address arithmetic goes through `uintptr_t` rather than `const uint8_t *`
/// so the eager encoding (`base == nullptr`, `off ==` absolute address) is not
/// pointer arithmetic on a null pointer.
template <typename T>
__host__ __device__ __forceinline__ T *base_off_ptr(const uint8_t *base, BaseOff o) {
    if (o.off == BASE_OFF_NULL) {
        return nullptr;
    }
    return reinterpret_cast<T *>(
        reinterpret_cast<uintptr_t>(base) + static_cast<uintptr_t>(o.off)
    );
}
