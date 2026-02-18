// Monomial computation structures for Metal Shading Language.
//
// Based on openvm-org cuda-backend/cuda/include/monomial.cuh
//
// These structs describe the layout of pre-compiled monomial evaluation data
// that is uploaded to the GPU for constraint evaluation.

#pragma once

#include <metal_stdlib>

#include "baby_bear.h"

using namespace metal;

// ============================================================================
// PackedVar: 4 bytes encoding a variable reference
// ============================================================================
//
// Bit layout:
//   Bits  0-3:   entry_type   (which table: preprocessed, main, etc.)
//   Bits  4-11:  part_index   (which partition of the table)
//   Bits 12-15:  offset       (row offset for rotation: 0 = current, 1 = next)
//   Bits 16-31:  column_index (which column within the table)

struct PackedVar {
    uint32_t data;

    inline uint8_t entry_type() const { return data & 0xF; }
    inline uint8_t part_index() const { return (data >> 4) & 0xFF; }
    inline uint8_t offset()     const { return (data >> 12) & 0xF; }
    inline uint16_t col_index() const { return data >> 16; }
};

// ============================================================================
// MonomialHeader: 12 bytes of metadata per monomial
// ============================================================================
//
// Describes one monomial: where its variables and terms start in the
// flat arrays, and how many of each there are.

struct MonomialHeader {
    uint32_t var_offset;    // Index into the PackedVar array
    uint32_t term_offset;   // Index into the LambdaTerm / InteractionMonomialTerm array
    uint16_t num_vars;      // Number of variables in this monomial
    uint16_t num_terms;     // Number of constraint terms using this monomial
};

// ============================================================================
// LambdaTerm: 8 bytes -- (constraint_idx, coefficient) pair
// ============================================================================
//
// Used in the "standard" (non-interaction) constraint evaluation path.
// On GPU we compute: sum_i (coefficient_i * lambda_pows[constraint_idx_i])

struct LambdaTerm {
    uint32_t constraint_idx;
    Fp coefficient;
};

// ============================================================================
// InteractionMonomialTerm: maps a monomial to interaction context
// ============================================================================
//
// Used in the logup/interaction constraint evaluation path.
// On GPU we compute:
//   For numerator:   sum_i (coefficient_i * eq_3bs[interaction_idx_i])
//   For denominator: sum_i (coefficient_i * beta_pows[field_idx_i] * eq_3bs[interaction_idx_i])

struct InteractionMonomialTerm {
    Fp coefficient;
    uint16_t interaction_idx;
    uint16_t field_idx; // For denom: index into message for beta_pows. For numer: unused.
};
