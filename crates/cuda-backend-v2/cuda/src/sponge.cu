/**
 * GPU-accelerated Poseidon2 duplex sponge grinding kernel.
 * 
 * This implements proof-of-work grinding on GPU for the Fiat-Shamir transcript.
 */

#include "fp.h"
#include "launcher.cuh"
#include "poseidon2.cuh"
#include <cstdint>

// Must match the Rust DeviceSpongeState struct layout
struct DeviceSpongeState {
    Fp state[CELLS];      // WIDTH = 16
    uint32_t absorb_idx;
    uint32_t sample_idx;
};

static_assert(sizeof(DeviceSpongeState) == CELLS * sizeof(Fp) + 2 * sizeof(uint32_t),
              "DeviceSpongeState size mismatch with Rust");

// Sponge operations matching DuplexSponge behavior

__device__ void sponge_observe(DeviceSpongeState& sponge, Fp value) {
    sponge.state[sponge.absorb_idx] = value;
    sponge.absorb_idx += 1;
    if (sponge.absorb_idx == CELLS_RATE) {
        poseidon2::poseidon2_mix(sponge.state);
        sponge.absorb_idx = 0;
        sponge.sample_idx = CELLS_RATE;
    }
}

__device__ Fp sponge_sample(DeviceSpongeState& sponge) {
    if (sponge.absorb_idx != 0 || sponge.sample_idx == 0) {
        poseidon2::poseidon2_mix(sponge.state);
        sponge.absorb_idx = 0;
        sponge.sample_idx = CELLS_RATE;
    }
    sponge.sample_idx -= 1;
    return sponge.state[sponge.sample_idx];
}

__device__ uint32_t sponge_sample_bits(DeviceSpongeState& sponge, uint32_t bits) {
    Fp rand_f = sponge_sample(sponge);
    uint32_t rand_u32 = rand_f.asUInt32();
    return rand_u32 & ((1u << bits) - 1);
}

__device__ bool sponge_check_witness(DeviceSpongeState& sponge, uint32_t bits, Fp witness) {
    sponge_observe(sponge, witness);
    return sponge_sample_bits(sponge, bits) == 0;
}

/**
 * Grinding kernel - finds a witness value such that check_witness(bits, witness) returns true.
 * 
 * Each thread searches a strided subset of the search space.
 * When a valid witness is found, it's atomically stored to g_grind_result.
 * 
 * @param init_state Initial sponge state (before observing the witness)
 * @param bits Number of bits that must be zero in the sampled value
 * @param max_witness Maximum witness value to try (usually F::ORDER - 1)
 */
__global__ void grind_kernel(
    const DeviceSpongeState* init_state,
    uint32_t bits,
    uint32_t min_witness,
    uint32_t max_witness,
    uint32_t* result
) {
    uint32_t w = min_witness + blockIdx.x * blockDim.x + threadIdx.x;
    if (w > max_witness || *result < w) {
        return;
    }

    // Clone the initial state to local registers
    DeviceSpongeState local_state = *init_state;

    // Check if this witness value works
    Fp witness = Fp(w);
    if (sponge_check_witness(local_state, bits, witness)) {
        // Found a valid witness - record it atomically
        atomicMin(result, w);
        return;
    }
}

// Launcher function callable from Rust

extern "C" int _sponge_grind(
    const DeviceSpongeState* init_state,
    uint32_t bits,
    uint32_t min_witness,
    uint32_t max_witness,
    uint32_t* result  // Output: device pointer where the found witness value will be written.
    // Must be set to `UINT32_MAX` before this function call
) {
    auto const [grid, block] = kernel_launch_params(1 << bits);
    grind_kernel<<<grid, block>>>(init_state, bits, min_witness, max_witness, result);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    // Synchronize and copy result back
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return err;
    }

    return CHECK_KERNEL();
}

