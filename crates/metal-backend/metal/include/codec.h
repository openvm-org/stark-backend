// Instruction encoding for DAG eval - Metal translation
// Translated from CUDA: cuda-backend/cuda/include/codec.cuh
#pragma once

#include <metal_stdlib>
using namespace metal;

struct Rule {
    uint64_t low;
    uint64_t high;
};

enum OperationType : uint32_t {
    OP_ADD = 0,
    OP_SUB = 1,
    OP_MUL = 2,
    OP_NEG = 3,
    OP_VAR = 4,
    OP_INV = 5
};

enum EntryType : uint32_t {
    ENTRY_PREPROCESSED = 0,
    ENTRY_MAIN = 1,
    ENTRY_PERMUTATION = 2,
    ENTRY_PUBLIC = 3,
    ENTRY_CHALLENGE = 4,
    ENTRY_EXPOSED = 5,
    SRC_INTERMEDIATE = 6,
    SRC_CONSTANT = 7,
    SRC_IS_FIRST = 8,
    SRC_IS_LAST = 9,
    SRC_IS_TRANSITION = 10
};

struct SourceInfo {
    EntryType type;
    uint8_t part;
    uint8_t offset;
    uint32_t index;
};

struct DecodedRule {
    bool is_constraint;
    bool buffer_result;
    OperationType op;
    SourceInfo x;
    SourceInfo y;
    uint32_t z_index;
};

struct RuleHeader {
    bool is_constraint;
    bool buffer_result;
    OperationType op;
    SourceInfo x;
};

constant uint64_t ENTRY_SRC_MASK = 0xF;
constant uint64_t ENTRY_PART_SHIFT = 4;
constant uint64_t ENTRY_PART_MASK = 0xFF;
constant uint64_t ENTRY_OFFSET_SHIFT = 12;
constant uint64_t ENTRY_OFFSET_MASK = 0xF;
constant uint64_t ENTRY_INDEX_SHIFT = 16;
constant uint64_t ENTRY_INDEX_MASK = 0xFFFFFFFF;
constant uint64_t SOURCE_INTERMEDIATE_SHIFT = 4;
constant uint64_t SOURCE_INTERMEDIATE_MASK = 0xFFFFF;
constant uint64_t SOURCE_CONSTANT_SHIFT = 16;
constant uint64_t SOURCE_CONSTANT_MASK = 0xFFFFFFFF;
constant uint64_t LOW_48_BITS_MASK = 0xFFFFFFFFFFFF;
constant int Y_HIGH_SHIFT = 16;
constant uint64_t Y_HIGH_MASK = 0xFFFFFFFF;
constant int Z_LOW_SHIFT = 32;
constant uint64_t Z_LOW_MASK = 0xFFFFFF;
constant uint64_t OP_MASK = 0x3F;
constant int OP_SHIFT = 56;
constant uint64_t BUFFER_RESULT_MASK = 0x4000000000000000ULL;
constant uint64_t IS_CONSTRAINT_MASK = 0x8000000000000000ULL;

inline SourceInfo decode_source(uint64_t encoded) {
    SourceInfo src;
    src.type = static_cast<EntryType>(encoded & ENTRY_SRC_MASK);
    src.part = (encoded >> ENTRY_PART_SHIFT) & ENTRY_PART_MASK;
    src.offset = (encoded >> ENTRY_OFFSET_SHIFT) & ENTRY_OFFSET_MASK;
    src.index = (encoded >> ENTRY_INDEX_SHIFT) & ENTRY_INDEX_MASK;

    if (src.type == SRC_INTERMEDIATE) {
        src.part = 0;
        src.offset = 0;
        src.index = (encoded >> SOURCE_INTERMEDIATE_SHIFT) & SOURCE_INTERMEDIATE_MASK;
    } else if (src.type == SRC_CONSTANT) {
        src.index = (encoded >> SOURCE_CONSTANT_SHIFT) & SOURCE_CONSTANT_MASK;
    }
    return src;
}

inline DecodedRule decode_rule(Rule encoded) {
    DecodedRule rule;

    uint64_t x_encoded = (encoded.low & LOW_48_BITS_MASK);
    rule.x = decode_source(x_encoded);

    uint64_t y_encoded = ((encoded.low >> 48) | ((encoded.high & Y_HIGH_MASK) << Y_HIGH_SHIFT));
    rule.y = decode_source(y_encoded);

    uint64_t z_encoded = (encoded.high >> Z_LOW_SHIFT) & Z_LOW_MASK;
    rule.z_index = (z_encoded >> SOURCE_INTERMEDIATE_SHIFT) & SOURCE_INTERMEDIATE_MASK;

    rule.op = static_cast<OperationType>((encoded.high >> OP_SHIFT) & OP_MASK);

    rule.buffer_result = (encoded.high & BUFFER_RESULT_MASK) != 0;
    rule.is_constraint = (encoded.high & IS_CONSTRAINT_MASK) != 0;

    return rule;
}

// Decode only header (op, flags, x) - for lazy decoding pattern
inline RuleHeader decode_rule_header(Rule encoded) {
    RuleHeader header;

    uint64_t x_encoded = (encoded.low & LOW_48_BITS_MASK);
    header.x = decode_source(x_encoded);

    header.op = static_cast<OperationType>((encoded.high >> OP_SHIFT) & OP_MASK);
    header.buffer_result = (encoded.high & BUFFER_RESULT_MASK) != 0;
    header.is_constraint = (encoded.high & IS_CONSTRAINT_MASK) != 0;

    return header;
}

// Decode y operand on demand (only needed for binary ops: ADD, SUB, MUL)
inline SourceInfo decode_y(Rule encoded) {
    uint64_t y_encoded = ((encoded.low >> 48) | ((encoded.high & Y_HIGH_MASK) << Y_HIGH_SHIFT));
    return decode_source(y_encoded);
}

// Decode z_index on demand (only needed when buffer_result is true)
inline uint32_t decode_z_index(Rule encoded) {
    uint64_t z_encoded = (encoded.high >> Z_LOW_SHIFT) & Z_LOW_MASK;
    return (z_encoded >> SOURCE_INTERMEDIATE_SHIFT) & SOURCE_INTERMEDIATE_MASK;
}
