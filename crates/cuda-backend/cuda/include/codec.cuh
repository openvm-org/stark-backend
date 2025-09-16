/*
 * Source: https://github.com/scroll-tech/plonky3-gpu (private repo)
 * Status: BASED ON plonky3-gpu/gpu-backend/src/cuda/kernels/codec.h
 * Imported: 2025-01-25 by @gaxiom
 */

// A very simple custom codec for constraints wroten in the AIR/RAP frontend language.
#pragma once

#include <cstdint>

// Constraint is encoded in 128-bit little-endian, but uin128_t is not supported in CUDA.
// Therefore we use two `uint64_t`s to represent it.
typedef struct {
    uint64_t low;
    uint64_t high;
} Rule;

// 7-bit op
typedef enum { OP_ADD, OP_SUB, OP_MUL, OP_NEG, OP_VAR, OP_INV } OperationType;

// Source enum
typedef enum {
    // sources in the category of SRC_VAR: 0..=5
    ENTRY_PREPROCESSED, // 0
    ENTRY_MAIN,         // 1
    ENTRY_PERMUTATION,  // 2
    ENTRY_PUBLIC,       // 3
    ENTRY_CHALLENGE,    // 4
    ENTRY_EXPOSED,      // 5
    SRC_INTERMEDIATE,   // 6
    SRC_CONSTANT,       // 7
    SRC_IS_FIRST,       // 8
    SRC_IS_LAST,        // 9
    SRC_IS_TRANSITION,  // 10
    // TERMINAL,           // 11
    // BUFF_PREPROCESSED,  // 12
    // BUFF_MAIN,          // 13
    // BUFF_PERMUTATION    // 14
} EntryType;

// Source info
typedef struct {
    EntryType type; // 4-bit
    uint8_t part;   // 8-bit
    // In practice, offset is often set to be {0, 1} to refer to current and next row.
    uint8_t offset; // 4-bit row
    // In most case, the index is less than 8192, so we can just use 13 bits for index.
    // But for the `Constant` variant, the index encodes the constant field element.
    // And the native field that we support is BabyBear whose modulus has 31 bits.
    uint32_t index;      // 16-bit col for variables, 20-bit for intermediates, 32-bit for constants
    uint32_t buffer_idx; // 16-bit buffer index for buffered variables
} SourceInfo;

typedef struct {
    bool is_constraint; // 1-bit, we need to accumulate the constraint's value if it's true
    OperationType op;   // 7-bit
    SourceInfo x;       // 48-bit
    SourceInfo y;       // 48-bit
    SourceInfo z;       // 24-bit
} DecodedRule;

// decode source from 48-bit little-endian integer
__host__ __device__ __forceinline__ SourceInfo decode_source(uint64_t encoded);
// decode rule from 128-bit little-endian integer
__host__ __device__ __forceinline__ DecodedRule decode_rule(Rule encoded);

// 0. There are 11 variants of source that can be encoded in 4 bits:
//
//    Preprocessed,  0, Field
//    Main,          1, Field
//    Permutation,   2, ExtensionField
//    Public,        3, Field
//    Challenge,     4, ExtensionField
//    Exposed,       5, ExtensionField
//    Intermediate,  6, Field
//    Constant,      7, Field
//    IsFirst,       8, Field
//    IsLast,        9, Field
//    IsTransition, 10, Field
//
//  For sources like `Preprocessed, Main, Permutation` they can have multiple
//  rotations/offsets and multiple parts. Therefore we allocate 4 more bits for
//  offset, and 8 bits for the part.
//
//  Entry is encode in 16-bit little-endian:
//     4-bit src | 8-bit part | 4-bit offset(row)
//
// 1. Source info is encoded in 48-bit little-endian
//   16-bit entry | 32-bit index
//
// 2. Constraint is encoded in 128-bit little-endian
//   48-bit x | 48-bit y | 24-bit z | 7-bit op | 1-bit is_constraint
//
//   Since the `z` operand always refers to `Intermediate` variant of Source,
//   and it doesn't have `part/offset` fields, and the index never exceed 2^20.
//   Therefore, we only need 24-bit for encoding. z:  4-bit src | 20-bit index

// Bit masks and shifts
// 16-bit entry: 4-bit src | 8-bit part | 4-bit offset(row)
static const uint64_t ENTRY_SRC_MASK = 0xF; // 4bit
static const uint64_t ENTRY_PART_SHIFT = 4;
static const uint64_t ENTRY_PART_MASK = 0xFF; // 8bit
static const uint64_t ENTRY_OFFSET_SHIFT = 12;
static const uint64_t ENTRY_OFFSET_MASK = 0xF; // 4bit
// 48-bit source: 16-bit entry | 16-bit index | 16-bit buffer index
static const uint64_t ENTRY_INDEX_SHIFT = 16;
static const uint64_t ENTRY_BUFFER_INDEX_SHIFT = 32;
static const uint64_t ENTRY_INDEX_MASK = 0xFFFF; // 32-bit
// 24bit Z: 4-bit src | 20-bit index
static const uint64_t SOURCE_INTERMEDIATE_SHIFT = 4;
static const uint64_t SOURCE_INTERMEDIATE_MASK = 0xFFFFF;
// 48bit Constant: 16-bit src | 32-bit base field
static const uint64_t SOURCE_CONSTANT_SHIFT = 16;
static const uint64_t SOURCE_CONSTANT_MASK = 0xFFFFFFFF; // 32bit

// constraint: 48-bit x | 48-bit y | 24-bit z | 7-bit op | 1-bit is_constraint
static const uint64_t LOW_48_BITS_MASK = 0xFFFFFFFFFFFF;
static const int Y_HIGH_SHIFT = 16;
static const uint64_t Y_HIGH_MASK = 0xFFFFFFFF; // 32-bit
static const int Z_LOW_SHIFT = 32;
static const uint64_t Z_LOW_MASK = 0xFFFFFF; // 24-bit
static const uint64_t OP_MASK = 0x7F;        // 7-bit
static const int OP_SHIFT = 56;
static const uint64_t IS_CONSTRAINT_MASK = 0x8000000000000000; // 127th bit

const uint64_t one = 1;
static_assert(LOW_48_BITS_MASK == (one << 48) - 1, "LOW_48_BITS_MASK must be (1 << 48) - 1");
static_assert(Y_HIGH_MASK == (one << 32) - 1, "Y_HIGH_MASK must be (1 << 32) - 1");
static_assert(Z_LOW_MASK == (one << 24) - 1, "Z_LOW_MASK must be (1 << 24) - 1");
static_assert(OP_MASK == (one << 7) - 1, "OP_MASK must be (1 << 7) - 1");
static_assert(IS_CONSTRAINT_MASK == one << 63, "IS_CONSTRAINT_MASK must be (1 << 63)");

// big-endian: 4-bit src | 12-bit col | 31-bit row | 1-bit reserved
__host__ __device__ __forceinline__ SourceInfo decode_source(uint64_t encoded) {
    // common
    SourceInfo src;
    src.type = (EntryType)(encoded & ENTRY_SRC_MASK);                          // 4-bit
    src.part = (encoded >> ENTRY_PART_SHIFT) & ENTRY_PART_MASK;                // 8-bit
    src.offset = (encoded >> ENTRY_OFFSET_SHIFT) & ENTRY_OFFSET_MASK;          // 4-bit
    src.index = (encoded >> ENTRY_INDEX_SHIFT) & ENTRY_INDEX_MASK;             // 16-bit
    src.buffer_idx = (encoded >> ENTRY_BUFFER_INDEX_SHIFT) & ENTRY_INDEX_MASK; // 16-bit

    if (src.type == SRC_INTERMEDIATE) {
        // 24bit: 4-bit src | 20-bit index
        src.part = 0;
        src.offset = 0;
        src.index = (encoded >> SOURCE_INTERMEDIATE_SHIFT) & SOURCE_INTERMEDIATE_MASK;
    } else if (src.type == SRC_CONSTANT) {
        // 48bit: 16-bit src | 32-bit base field
        src.index = (encoded >> SOURCE_CONSTANT_SHIFT) & SOURCE_CONSTANT_MASK;
    }
    return src;
}

__host__ __device__ __forceinline__ DecodedRule decode_rule(Rule encoded) {
    DecodedRule rule;

    // Extract x (48 bits from the right)
    uint64_t x_encoded = (encoded.low & LOW_48_BITS_MASK);
    rule.x = decode_source(x_encoded);

    // Extract y (next 48 bits)
    uint64_t y_encoded = ((encoded.low >> 48) | ((encoded.high & Y_HIGH_MASK) << Y_HIGH_SHIFT));
    rule.y = decode_source(y_encoded);

    // Extract z (next 24 bits)
    uint64_t z_encoded = (encoded.high >> Z_LOW_SHIFT) & Z_LOW_MASK;
    rule.z = decode_source(z_encoded);

    // Extract op (next 7 bits)
    rule.op = (OperationType)((encoded.high >> OP_SHIFT) & OP_MASK);

    // Extract is_constraint (highest bit)
    rule.is_constraint = (encoded.high & IS_CONSTRAINT_MASK) != 0;

    return rule;
}