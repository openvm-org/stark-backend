/// Single owner of the BN254 Poseidon2 round-constant `__device__ __constant__`
/// storage and the two host-side upload launchers.
///
/// Two consumers reference these symbols via `extern __device__ __constant__`:
///   - bn254_poseidon2_grind.cu    (b64 sponge transcript / grinding kernel)
///   - bn254_poseidon2_row_hash.cu (b32 Merkle row-hash + compress kernels;
///                                  reads the same bytes reinterpreted as
///                                  Bn254Fr32 via pointer cast).
///
/// Keeping the storage in a separate TU avoids multiple-definition link
/// errors and is the only place that needs to grow if a new caller is added.

#include "poseidon2_bn254_common.cuh"

// ---------------------------------------------------------------------------
// Round constant device memory (filled by _init_bn254_poseidon2_rc)
// ---------------------------------------------------------------------------

/// External initial round constants: 4 rounds × 3 elements
__device__ __constant__ Bn254Fr g_initial_rc[4][3];

/// Internal (partial) round constants: 56 rounds × 1 element (for state[0] only)
__device__ __constant__ Bn254Fr g_partial_rc[56];

/// External terminal round constants: 4 rounds × 3 elements
__device__ __constant__ Bn254Fr g_terminal_rc[4][3];

// --- Width-2 permutation (rF=6, rP=50) for Merkle compression ---
// Matches Poseidon2Bn254Width2 (gnark-crypto constants).

/// Width-2 external initial round constants: 3 rounds × 2 elements
__device__ __constant__ Bn254Fr g_initial_rc_w2[3][2];

/// Width-2 internal (partial) round constants: 50 rounds × 1 element
__device__ __constant__ Bn254Fr g_partial_rc_w2[50];

/// Width-2 external terminal round constants: 3 rounds × 2 elements
__device__ __constant__ Bn254Fr g_terminal_rc_w2[3][2];

// ---------------------------------------------------------------------------
// Extern "C" launchers
// ---------------------------------------------------------------------------

/// Upload BN254 Poseidon2 round constants (in Montgomery form) to device constant memory.
///
/// @param initial_rc   Flat array of 4*3*4 = 48 uint64s  (initial external rounds)
/// @param partial_rc   Flat array of 56*4 = 224 uint64s  (internal/partial rounds)
/// @param terminal_rc  Flat array of 4*3*4 = 48 uint64s  (terminal external rounds)
extern "C" int _init_bn254_poseidon2_rc(
    const uint64_t *initial_rc,
    const uint64_t *partial_rc,
    const uint64_t *terminal_rc,
    cudaStream_t stream
) {
    cudaError_t err;
    err = cudaMemcpyToSymbol(g_initial_rc, initial_rc, 4 * 3 * 4 * sizeof(uint64_t));
    if (err != cudaSuccess)
        return (int)err;
    err = cudaMemcpyToSymbol(g_partial_rc, partial_rc, 56 * 4 * sizeof(uint64_t));
    if (err != cudaSuccess)
        return (int)err;
    err = cudaMemcpyToSymbol(g_terminal_rc, terminal_rc, 4 * 3 * 4 * sizeof(uint64_t));
    if (err != cudaSuccess)
        return (int)err;
    return (int)cudaSuccess;
}

/// Upload width-2 BN254 Poseidon2 round constants (in Montgomery form).
///
/// @param initial_rc   Flat array of 3*2*4 = 24 uint64s  (initial external rounds)
/// @param partial_rc   Flat array of 50*4  = 200 uint64s  (internal/partial rounds)
/// @param terminal_rc  Flat array of 3*2*4 = 24 uint64s  (terminal external rounds)
extern "C" int _init_bn254_poseidon2_rc_w2(
    const uint64_t *initial_rc,
    const uint64_t *partial_rc,
    const uint64_t *terminal_rc,
    cudaStream_t stream
) {
    cudaError_t err;
    err = cudaMemcpyToSymbol(g_initial_rc_w2, initial_rc, 3 * 2 * 4 * sizeof(uint64_t));
    if (err != cudaSuccess)
        return (int)err;
    err = cudaMemcpyToSymbol(g_partial_rc_w2, partial_rc, 50 * 4 * sizeof(uint64_t));
    if (err != cudaSuccess)
        return (int)err;
    err = cudaMemcpyToSymbol(g_terminal_rc_w2, terminal_rc, 3 * 2 * 4 * sizeof(uint64_t));
    if (err != cudaSuccess)
        return (int)err;
    return (int)cudaSuccess;
}
