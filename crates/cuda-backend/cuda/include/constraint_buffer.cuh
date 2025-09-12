#pragma once

#include "fpext.h"
#include <cstdint>

const uint32_t NUM_REGISTERS = 2;

struct ConstraintBuffer {
    FpExt *global_buffer;
    FpExt r0;
    FpExt r1;
    // FpExt r2;
    // FpExt r3;
    // FpExt r4;
    // FpExt r5;
    // FpExt r6;
    // FpExt r7;
    // FpExt r8;
    // FpExt r9;
    // FpExt r10;
    // FpExt r11;
    // FpExt r12;
    // FpExt r13;
    // FpExt r14;
    // FpExt r15;

    __forceinline__ __host__ __device__ ConstraintBuffer(FpExt *global_buffer)
        : global_buffer(global_buffer) {}

    __forceinline__ __host__ __device__ FpExt read(uint32_t idx, const uint32_t task_stride) {
        if (idx == 0) {
            return r0;
        } else if (idx == 1) {
            return r1;
            // switch (idx) {
            // case 0:
            //     return r0;
            // case 1:
            //     return r1;
            // case 2:
            //     return r2;
            // case 3:
            //     return r3;
            // case 4:
            //     return r4;
            // case 5:
            //     return r5;
            // case 6:
            //     return r6;
            // case 7:
            //     return r7;
            //     case 8:
            //         return r8;
            //     case 9:
            //         return r9;
            //     case 10:
            //         return r10;
            //     case 11:
            //         return r11;
            //     case 12:
            //         return r12;
            //     case 13:
            //         return r13;
            //     case 14:
            //         return r14;
            //     case 15:
            //         return r15;
            // }
        }
        return global_buffer[(idx - NUM_REGISTERS) * task_stride];
    }

    __forceinline__ __host__ __device__ void write(
        uint32_t idx,
        FpExt value,
        const uint32_t task_stride
    ) {
        if (idx == 0) {
            r0 = value;
        } else if (idx == 1) {
            r1 = value;

            // switch (idx) {
            // case 0:
            //     r0 = value;
            //     break;
            // case 1:
            //     r1 = value;
            //     break;
            // case 2:
            //     r2 = value;
            //     break;
            // case 3:
            //     r3 = value;
            //     break;
            // case 4:
            //     r4 = value;
            //     break;
            // case 5:
            //     r5 = value;
            //     break;
            // case 6:
            //     r6 = value;
            //     break;
            // case 7:
            //     r7 = value;
            //     break;
            //     case 8:
            //         r8 = value;
            //         break;
            //     case 9:
            //         r9 = value;
            //         break;
            //     case 10:
            //         r10 = value;
            //         break;
            //     case 11:
            //         r11 = value;
            //         break;
            //     case 12:
            //         r12 = value;
            //         break;
            //     case 13:
            //         r13 = value;
            //         break;
            //     case 14:
            //         r14 = value;
            //         break;
            //     case 15:
            //         r15 = value;
            //         break;
            // }
        } else {
            global_buffer[(idx - NUM_REGISTERS) * task_stride] = value;
        }
    }
};
