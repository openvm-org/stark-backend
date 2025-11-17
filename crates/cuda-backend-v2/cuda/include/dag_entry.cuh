#pragma once

#include "codec.cuh"
#include "fp.h"
#include "fpext.h"
#include "matrix.cuh"
#include <cassert>
#include <cstdint>

namespace constraint_evaluation {
inline constexpr uint32_t BUFFER_THRESHOLD = 16;
inline constexpr uint32_t TASK_SIZE = 65536;

inline uint32_t get_launcher_count(uint32_t buffer_size, uint32_t height) {
    return buffer_size > BUFFER_THRESHOLD ? TASK_SIZE : height;
}
} // namespace constraint_evaluation

namespace interaction_evaluation {
inline constexpr uint32_t BUFFER_THRESHOLD = 10;
inline constexpr uint32_t TASK_SIZE = 65536;

inline uint32_t get_launcher_count(uint32_t buffer_size, uint32_t height) {
    return buffer_size > BUFFER_THRESHOLD ? TASK_SIZE : height;
}
} // namespace interaction_evaluation

namespace symbolic_dag {

__device__ __forceinline__ FpExt evaluate_dag_entry(
    const SourceInfo &src,
    uint32_t row_index,
    const Fp *d_selectors,
    const MainMatrixPtrs<Fp> *__restrict__ d_main,
    uint32_t height,
    uint32_t selectors_width,
    const Fp *__restrict__ d_preprocessed,
    uint32_t preprocessed_air_width,
    const FpExt *d_eq_z,
    const FpExt *d_eq_x,
    const Fp *__restrict__ d_public,
    uint32_t public_len,
    const FpExt *inter_buffer,
    uint32_t buffer_stride,
    uint32_t buffer_size,
    uint32_t large_domain,
    const FpExt *d_challenges = nullptr // Optional: for ENTRY_CHALLENGE in interactions
) {
    (void)large_domain;
    switch (src.type) {
    case ENTRY_PREPROCESSED: {
        if (d_preprocessed == nullptr) {
            return FpExt(Fp::zero());
        }
        const auto stride = height * preprocessed_air_width;
        const Fp *matrix = d_preprocessed + stride * src.offset;
        return FpExt(matrix[height * src.index + row_index]);
    }
    case ENTRY_MAIN: {
        auto main_ptr = d_main[src.part];
        const auto stride = height * main_ptr.air_width;
        const Fp *matrix = main_ptr.data + stride * src.offset;
        return FpExt(matrix[height * src.index + row_index]);
    }
    case ENTRY_CHALLENGE:
        assert(d_challenges != nullptr);
        return d_challenges[src.index];
    case ENTRY_PERMUTATION:
    case ENTRY_EXPOSED:
        return FpExt(Fp::zero());
    case ENTRY_PUBLIC: {
        if (src.index >= public_len || d_public == nullptr) {
            return FpExt(Fp::zero());
        }
        return FpExt(d_public[src.index]);
    }
    case SRC_CONSTANT:
        return FpExt(Fp(src.index));
    case SRC_INTERMEDIATE:
        if (inter_buffer == nullptr || buffer_size == 0) {
            return FpExt(Fp::zero());
        }
        if (src.index >= buffer_size) {
            return FpExt(Fp::zero());
        }
        return inter_buffer[src.index * buffer_stride];
    case SRC_IS_FIRST: {
        if (height == 0 || selectors_width == 0) {
            return FpExt(Fp::zero());
        }
        uint32_t row = row_index % height;
        return FpExt(d_selectors[row]);
    }
    case SRC_IS_LAST: {
        if (height == 0 || selectors_width < 3) {
            return FpExt(Fp::zero());
        }
        uint32_t row = row_index % height;
        return FpExt(d_selectors[height * 2 + row]);
    }
    case SRC_IS_TRANSITION: {
        if (height == 0 || selectors_width < 2) {
            return FpExt(Fp::zero());
        }
        uint32_t row = row_index % height;
        return FpExt(d_selectors[height + row]);
    }
    }
    return FpExt(Fp::zero());
}

} // namespace symbolic_dag
