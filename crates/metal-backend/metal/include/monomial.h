// Monomial computation structures for Metal
// Translated from CUDA: cuda-backend/cuda/include/monomial.cuh
#pragma once

#include "baby_bear.h"

// Packed variable: 4 bytes
// Bits 0-3:   entry_type
// Bits 4-11:  part_index
// Bits 12-15: offset
// Bits 16-31: column_index
struct PackedVar {
    uint32_t data;
    inline uint8_t entry_type() const { return data & 0xF; }
    inline uint8_t part_index() const { return (data >> 4) & 0xFF; }
    inline uint8_t offset() const { return (data >> 12) & 0xF; }
    inline uint16_t col_index() const { return data >> 16; }
};

// Monomial metadata (12 bytes, fields ordered to avoid padding)
struct MonomialHeader {
    uint32_t var_offset;
    uint32_t term_offset;
    uint16_t num_vars;
    uint16_t num_terms;
};

// Lambda term: (constraint_idx, coefficient) pair (8 bytes)
// On GPU we compute: sum_i (coefficient_i * lambda_pows[constraint_idx_i])
struct LambdaTerm {
    uint32_t constraint_idx;
    Fp coefficient;
};

// Interaction monomial term: maps a monomial to interaction context
// On GPU we compute:
//   For numerator:   sum_i (coefficient_i * eq_3bs[interaction_idx_i])
//   For denominator: sum_i (coefficient_i * beta_pows[field_idx_i] * eq_3bs[interaction_idx_i])
struct InteractionMonomialTerm {
    Fp coefficient;
    uint16_t interaction_idx;
    uint16_t field_idx; // For denom: index into message for beta_pows. For numer: unused.
};
