#pragma once

#include "fpext.h"
#include <cstdint>

const uint32_t NUM_REGISTERS = 16;

__forceinline__ __host__ __device__ FpExt read_from_buffer(
    const FpExt *global_buffer,
    const FpExt *register_buffer,
    const uint32_t idx,
    const uint64_t task_stride
) {
    const FpExt *buffer = (idx < NUM_REGISTERS) ? register_buffer : global_buffer;
    uint32_t index = (idx < NUM_REGISTERS) ? idx : (idx - NUM_REGISTERS) * task_stride;
    return buffer[index];
}

__forceinline__ __host__ __device__ void write_to_buffer(
    FpExt *global_buffer,
    FpExt *register_buffer,
    uint32_t idx,
    const uint64_t task_stride,
    FpExt value
) {
    FpExt *buffer = (idx < NUM_REGISTERS) ? register_buffer : global_buffer;
    uint32_t index = (idx < NUM_REGISTERS) ? idx : (idx - NUM_REGISTERS) * task_stride;
    buffer[index] = value;
}
