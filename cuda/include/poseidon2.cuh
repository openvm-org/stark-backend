#pragma once

#include "fp.h"
#include <cuda_runtime.h>

__constant__ Fp INITIAL_ROUND_CONSTANTS[64];
__constant__ Fp TERMINAL_ROUND_CONSTANTS[64];
__constant__ Fp INTERNAL_ROUND_CONSTANTS[13];

static bool poseidon2_initialized = false;

extern "C" int _init_poseidon2_constants(
    const Fp *initial_round_constants,
    const Fp *terminal_round_constants,
    const Fp *internal_round_constants
) {
    if (poseidon2_initialized) {
        return cudaSuccess;
    }
    cudaError_t error;

    error = cudaMemcpyToSymbol(INITIAL_ROUND_CONSTANTS, initial_round_constants, 64 * sizeof(Fp));
    if (error != cudaSuccess)
        return error;

    error = cudaMemcpyToSymbol(TERMINAL_ROUND_CONSTANTS, terminal_round_constants, 64 * sizeof(Fp));
    if (error != cudaSuccess)
        return error;

    error = cudaMemcpyToSymbol(INTERNAL_ROUND_CONSTANTS, internal_round_constants, 13 * sizeof(Fp));
    if (error != cudaSuccess)
        return error;

    poseidon2_initialized = true;

    return cudaDeviceSynchronize();
}

