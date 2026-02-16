// Sponge (Poseidon2 duplex) grinding kernel for Metal
// Translated from CUDA: cuda-backend/cuda/src/sponge.cu
#include <metal_stdlib>
using namespace metal;

#include "../include/baby_bear.h"
#include "../include/poseidon2.h"

// Must match the Rust DeviceSpongeState struct layout
struct DeviceSpongeState {
    Fp state[CELLS];      // WIDTH = 16
    uint32_t absorb_idx;
    uint32_t sample_idx;
};

inline void sponge_observe(thread DeviceSpongeState &sponge, Fp value) {
    sponge.state[sponge.absorb_idx] = value;
    sponge.absorb_idx += 1;
    if (sponge.absorb_idx == CELLS_RATE) {
        poseidon2_mix(sponge.state);
        sponge.absorb_idx = 0;
        sponge.sample_idx = CELLS_RATE;
    }
}

inline Fp sponge_sample(thread DeviceSpongeState &sponge) {
    if (sponge.absorb_idx != 0 || sponge.sample_idx == 0) {
        poseidon2_mix(sponge.state);
        sponge.absorb_idx = 0;
        sponge.sample_idx = CELLS_RATE;
    }
    sponge.sample_idx -= 1;
    return sponge.state[sponge.sample_idx];
}

inline uint32_t sponge_sample_bits(thread DeviceSpongeState &sponge, uint32_t bits) {
    Fp rand_f = sponge_sample(sponge);
    uint32_t rand_u32 = fp_to_uint(rand_f);
    return rand_u32 & ((1u << bits) - 1);
}

inline bool sponge_check_witness(thread DeviceSpongeState &sponge, uint32_t bits, Fp witness) {
    sponge_observe(sponge, witness);
    return sponge_sample_bits(sponge, bits) == 0;
}

// Grinding kernel - finds a witness value such that check_witness(bits, witness) returns true.
// Each thread searches a single candidate witness value.
// When a valid witness is found, it's atomically stored to result.
kernel void grind_kernel(
    const device DeviceSpongeState *init_state   [[buffer(0)]],
    constant uint32_t &bits                      [[buffer(1)]],
    constant uint32_t &min_witness               [[buffer(2)]],
    constant uint32_t &max_witness               [[buffer(3)]],
    device atomic_uint *result                   [[buffer(4)]],
    uint gid                                     [[thread_position_in_grid]]
) {
    uint32_t w = min_witness + gid;
    if (w > max_witness) {
        return;
    }

    // Early exit if someone already found a smaller witness
    uint32_t current_best = atomic_load_explicit(result, memory_order_relaxed);
    if (current_best < w) {
        return;
    }

    // Clone the initial state to local registers
    DeviceSpongeState local_state = *init_state;

    // Check if this witness value works
    Fp witness = Fp(w);
    if (sponge_check_witness(local_state, bits, witness)) {
        // Found a valid witness - record it atomically (min)
        atomic_fetch_min_explicit(result, w, memory_order_relaxed);
    }
}
