#pragma once

#include "base_off.cuh"

#include <cstdint>

/// A matrix's device base pointer plus its padded AIR width.
///
/// This is the *runtime* (decoded) form. It is still what device code holds in
/// registers, and it is still the ABI of the single-AIR `mle.cu` entry points,
/// which take it as a plain kernel argument rather than through a context
/// array.
template <typename T> struct MainMatrixPtrs {
    const T *__restrict__ data;
    uint32_t air_width;
};

/// The base+offset form of [`MainMatrixPtrs`] -- what the batched context
/// arrays actually store, so that those arrays hold integers instead of
/// embedded device pointers. See `base_off.cuh`.
struct MainMatrixDesc {
    BaseOff data;
    uint32_t air_width;
};

/// Decode one descriptor against the pool base.
template <typename T>
__host__ __device__ __forceinline__ MainMatrixPtrs<T>
resolve_main_matrix(const MainMatrixDesc &d, const uint8_t *base) {
    return MainMatrixPtrs<T>{base_off_ptr<const T>(base, d.data), d.air_width};
}
