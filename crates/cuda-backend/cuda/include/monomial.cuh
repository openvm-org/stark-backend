/*
 * Monomial format for batch MLE evaluation
 *
 * This header defines the serialized monomial format used for efficient
 * GPU evaluation of constraint polynomials when num_y is small.
 */
#pragma once

#include <cstdint>
#include "codec.cuh"

// Lambda term: coefficient * λ^constraint_idx
// Serialized as 6 bytes: u16 constraint_idx + u32 coefficient
struct LambdaTerm {
    uint16_t constraint_idx;
    uint32_t coefficient;  // BabyBear field element
};

// Monomial variable: encoded using same format as Source in codec.cuh
// Serialized as 8 bytes (u64)
// Uses EntryType: ENTRY_PREPROCESSED, ENTRY_MAIN, SRC_IS_FIRST, SRC_IS_LAST, SRC_IS_TRANSITION

// Monomial format (variable-length):
// - num_lambda_terms: u8
// - num_vars: u8
// - lambda_terms: [LambdaTerm; num_lambda_terms]  (6 bytes each)
// - variables: [u64; num_vars]                    (8 bytes each, encoded Source)
//
// Total size: 2 + 6*num_lambda_terms + 8*num_vars bytes

// Read a monomial header and return pointers to lambda terms and variables
__device__ __forceinline__ void parse_monomial(
    const uint8_t* data,
    uint8_t& num_lambda_terms,
    uint8_t& num_vars,
    const LambdaTerm*& lambda_terms,
    const uint64_t*& variables
) {
    num_lambda_terms = data[0];
    num_vars = data[1];
    lambda_terms = (const LambdaTerm*)(data + 2);
    variables = (const uint64_t*)(data + 2 + 6 * num_lambda_terms);
}

// Evaluate lambda-polynomial coefficient: Σ_i coeff_i * λ^{idx_i}
__device__ __forceinline__ FpExt eval_lambda_coeff(
    const LambdaTerm* terms,
    uint8_t num_terms,
    const FpExt* lambda_pows
) {
    FpExt result = FpExt(0);
    for (uint8_t i = 0; i < num_terms; ++i) {
        result = result + FpExt(Fp(terms[i].coefficient)) * lambda_pows[terms[i].constraint_idx];
    }
    return result;
}

// Evaluate variable at given evaluation point
// This is for the MLE sumcheck evaluation context
__device__ __forceinline__ FpExt eval_monomial_var(
    uint64_t encoded_var,
    uint32_t x_int,
    uint32_t y_int,
    uint32_t num_y,
    const FpExt* d_mat_evals,      // Flattened matrix evaluations [part][var_idx][x_int * num_y + y_int]
    const uint32_t* d_mat_widths,  // Width of each matrix part
    const uint32_t* d_mat_offsets, // Offset into d_mat_evals for each part
    const FpExt* d_sels,           // Selector evaluations [sel_type][x_int * num_y + y_int]
    uint32_t num_x
) {
    SourceInfo src = decode_source(encoded_var);
    uint32_t xy_idx = x_int * num_y + y_int;

    switch (src.type) {
    case ENTRY_PREPROCESSED: {
        // Preprocessed is part 0 with special offset in d_mat_evals
        // Note: preprocessed doesn't have row offset in MLE context
        uint32_t col = src.index;
        // Assume preprocessed is stored at a specific location
        // This may need adjustment based on actual layout
        return d_mat_evals[d_mat_offsets[0] + col * (num_x * num_y) + xy_idx];
    }
    case ENTRY_MAIN: {
        uint32_t part = src.part;
        uint32_t col = src.index;
        // Row offset handled by caller providing different mat_evals for local/next
        uint32_t base = d_mat_offsets[part + 1]; // +1 to skip preprocessed
        return d_mat_evals[base + col * (num_x * num_y) + xy_idx];
    }
    case SRC_IS_FIRST:
        return d_sels[0 * (num_x * num_y) + xy_idx];
    case SRC_IS_LAST:
        return d_sels[1 * (num_x * num_y) + xy_idx];
    case SRC_IS_TRANSITION:
        return d_sels[2 * (num_x * num_y) + xy_idx];
    default:
        // PUBLIC, CHALLENGE, etc. should not appear in monomials
        // as they are absorbed into the coefficient during expansion
        return FpExt(0);
    }
}
