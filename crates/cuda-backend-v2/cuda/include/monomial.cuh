#pragma once

#include "fp.h"

// Packed variable: 4 bytes
// Bits 0-3:   entry_type
// Bits 4-11:  part_index
// Bits 12-15: offset
// Bits 16-31: column_index
struct PackedVar {
    uint32_t data;
    __device__ __forceinline__ uint8_t entry_type() const { return data & 0xF; }
    __device__ __forceinline__ uint8_t part_index() const { return (data >> 4) & 0xFF; }
    __device__ __forceinline__ uint8_t offset() const { return (data >> 12) & 0xF; }
    __device__ __forceinline__ uint16_t col_index() const { return data >> 16; }
};

// Monomial metadata (12 bytes, fields ordered to avoid padding)
struct MonomialHeader {
    uint32_t var_offset;
    uint32_t lambda_offset;
    uint16_t num_vars;
    uint16_t num_lambdas;
};

// Lambda term: (constraint_idx, coefficient) pair (8 bytes)
// On GPU we compute: sum_i (coefficient_i * lambda_pows[constraint_idx_i])
struct LambdaTerm {
    uint32_t constraint_idx;
    Fp coefficient;
};
