/// BN254 sponge transcript + PoW grinding kernel.
///
/// The BN254 CUDA backend is split across several translation units to keep
/// inlining and register pressure manageable:
///
///   bn254_constants.cu          - The actual `__device__ __constant__`
///                                 storage for the width-3 and width-2 round
///                                 constants, plus the `_init_bn254_poseidon2_rc`
///                                 / `_init_bn254_poseidon2_rc_w2` launchers
///                                 that upload them from the host.
///   bn254_poseidon2_row_hash.cu - The b32-limb Merkle row-hash kernels
///                                 (`_bn254_poseidon2_compressing_row_hashes{,_ext}`)
///                                 and the adjacent-compress launcher.
///   bn254_poseidon2_grind.cu  (this)  - The b64 width-3 Poseidon2 permutation used
///                                 by the BN254 sponge transcript, the sponge
///                                 itself, and the grinding kernel
///                                 (`_bn254_sponge_grind`).
///
/// Round constants are owned by `bn254_constants.cu` and referenced here via
/// `extern __device__ __constant__` declarations. The b64 permutation body
/// uses the `bn254_noinline::*` helpers (see poseidon2_bn254_noinline.cuh) so
/// the per-round arithmetic stays out-of-line — that keeps the grinding
/// kernel's hot loop compact and register-light.

#include "fp.h"
#include "launcher.cuh"
#include "poseidon2_bn254_noinline.cuh"
#include <cstdint>

// ---------------------------------------------------------------------------
// Round constant device memory (filled by _init_bn254_poseidon2_rc)
// ---------------------------------------------------------------------------

/// External initial round constants: 4 rounds × 3 elements
extern __device__ __constant__ Bn254Fr g_initial_rc[4][3];

/// Internal (partial) round constants: 56 rounds × 1 element (for state[0] only)
extern __device__ __constant__ Bn254Fr g_partial_rc[56];

/// External terminal round constants: 4 rounds × 3 elements
extern __device__ __constant__ Bn254Fr g_terminal_rc[4][3];

struct Bn254PoseidonPermShared {
    Bn254Fr *initial_rc;
    Bn254Fr *partial_rc;
    Bn254Fr *terminal_rc;
};

// make sure to __syncthreads() before reading
static __device__ Bn254PoseidonPermShared load_shared() {
    __shared__ uint64_t buf[(4 * 3 + 56 + 4 * 3) * 4];
    for (int i = threadIdx.x; i < 12 * 4; i += blockDim.x) {
        buf[i] = ((uint64_t *)g_initial_rc)[i];
    }

    for (int i = threadIdx.x; i < 56 * 4; i += blockDim.x) {
        buf[i + 12 * 4] = ((uint64_t *)g_partial_rc)[i];
    }

    for (int i = threadIdx.x; i < 12 * 4; i += blockDim.x) {
        buf[i + 12 * 4 + 56 * 4] = ((uint64_t *)g_terminal_rc)[i];
    }
    auto ptr = (Bn254Fr *)buf;

    return {ptr, ptr + 12, ptr + 12 + 56};
}

template <int WIDTH, int HALF_F, int ROUNDS_P>
static __device__ __noinline__ void bn254_poseidon2_permute_implv2(
    Bn254Fr state[WIDTH],
    const Bn254Fr *initial_rc,
    const Bn254Fr *partial_rc,
    const Bn254Fr *terminal_rc
) {
    // --- Initial external layer ---
    bn254_noinline::bn254_mds_external<WIDTH>(state);
    for (int r = 0; r < HALF_F; r++) {
        for (int i = 0; i < WIDTH; i++) {
            state[i] = bn254_noinline::bn254_add(state[i], initial_rc[r * WIDTH + i]);
            state[i] = bn254_noinline::bn254_sbox(state[i]);
        }
        bn254_noinline::bn254_mds_external<WIDTH>(state);
    }

    // --- Internal (partial) layer ---
#pragma unroll 1
    for (int r = 0; r < ROUNDS_P; r++) {
        state[0] = bn254_noinline::bn254_add(state[0], partial_rc[r]);
        state[0] = bn254_noinline::bn254_sbox(state[0]);
        bn254_noinline::bn254_mds_internal<WIDTH>(state);
    }

    // --- Terminal external layer ---
    for (int r = 0; r < HALF_F; r++) {
        for (int i = 0; i < WIDTH; i++) {
            state[i] = bn254_noinline::bn254_add(state[i], terminal_rc[r * WIDTH + i]);
            state[i] = bn254_noinline::bn254_sbox(state[i]);
        }
        bn254_noinline::bn254_mds_external<WIDTH>(state);
    }
}

// --- Width-3 permutation (rF=8, rP=56) for leaf hashing and transcript sponge ---
// Matches p3-bn254's Poseidon2Bn254<3> (HorizenLabs constants).

static __device__ void bn254_poseidon2_permute_v2(
    Bn254Fr state[3],
    Bn254PoseidonPermShared shared_states
) {
    bn254_poseidon2_permute_implv2<3, 4, 56>(
        state, shared_states.initial_rc, shared_states.partial_rc, shared_states.terminal_rc
    );
}

// --- Width-2 permutation (rF=6, rP=50) for Merkle compression ---
// Matches Poseidon2Bn254Width2 (gnark-crypto constants).

/// Width-2 external initial round constants: 3 rounds × 2 elements
extern __device__ __constant__ Bn254Fr g_initial_rc_w2[3][2];

/// Width-2 internal (partial) round constants: 50 rounds × 1 element
extern __device__ __constant__ Bn254Fr g_partial_rc_w2[50];

/// Width-2 external terminal round constants: 3 rounds × 2 elements
extern __device__ __constant__ Bn254Fr g_terminal_rc_w2[3][2];

// ---------------------------------------------------------------------------
// BN254 Merkle digest: a single Bn254Fr element (32 bytes)
// Matches Digest = [Bn254Scalar; 1] on the Rust side.
// ---------------------------------------------------------------------------

struct bn254_digest_t {
    Bn254Fr elem;
};

// ---------------------------------------------------------------------------
// Helper: zero-initialize a Bn254Fr
// ---------------------------------------------------------------------------

static __device__ Bn254Fr bn254_zero_init() {
    Bn254Fr z;
    for (int i = 0; i < 4; i++)
        z.limbs[i] = 0;
    return z;
}

// ---------------------------------------------------------------------------
// Sponge constants for Merkle hashing
//
// Matches MultiFieldHasher<BabyBear, Bn254Scalar, Perm, 3, 16, 1>:
//   BABY_BEAR_RATE = 16  BabyBear values absorbed per permutation
//   NUM_F_ELMS = 8       BabyBear values packed per Bn254Fr (floor(254/31) = 8)
// ---------------------------------------------------------------------------

static const int BN254_BABY_BEAR_RATE = 16;
static const int BN254_NUM_F_ELMS = 8;

// ---------------------------------------------------------------------------
// BN254 sponge state for GPU grinding
//
// Matches MultiFieldTranscript<BabyBear, Bn254Scalar, Perm, WIDTH=3, RATE=2>:
//   num_obs_per_word = SF::bits() / CF::bits() = 254/31 = 8
//   num_samples_per_word = 5  (base-p decomposition, ≥100 bits bias slack)
//
// The sponge uses overwrite-mode duplex with absorb_idx/sample_idx tracking.
// Rust DeviceBn254SpongeState must have identical layout (verified by size assert).
// ---------------------------------------------------------------------------

static const uint32_t BN254_NUM_OBS_PER_WORD = 8;
static const uint32_t BN254_SPONGE_RATE = 2;

struct DeviceBn254SpongeState {
    Bn254Fr sponge_state[3];  // 96 bytes
    uint32_t absorb_idx;      //  4 bytes
    uint32_t sample_idx;      //  4 bytes
    uint32_t observe_buf[8];  // 32 bytes
    uint32_t observe_buf_len; //  4 bytes
    // total = 140 + 4 padding = 144 bytes (aligned to 8)
    // Note: sample_buf is not needed on device — observe() clears it before grinding.
};

static_assert(
    sizeof(DeviceBn254SpongeState) == 144,
    "DeviceBn254SpongeState size mismatch with Rust"
);

// --- Low-level sponge (absorb/squeeze matching DuplexSponge) ---

/// Absorb one BN254 word into the sponge (overwrite mode).
__device__ void bn254_sponge_absorb(
    DeviceBn254SpongeState &s,
    Bn254Fr value,
    Bn254PoseidonPermShared shared_state
) {
    s.sponge_state[s.absorb_idx] = value;
    s.absorb_idx++;
    if (s.absorb_idx == BN254_SPONGE_RATE) {
        bn254_poseidon2_permute_v2(s.sponge_state, shared_state);
        s.absorb_idx = 0;
        s.sample_idx = BN254_SPONGE_RATE;
    }
}

/// Squeeze one BN254 word from the sponge.
__device__ Bn254Fr
bn254_sponge_squeeze(DeviceBn254SpongeState &s, Bn254PoseidonPermShared shared_state) {
    if (s.absorb_idx != 0 || s.sample_idx == 0) {
        bn254_poseidon2_permute_v2(s.sponge_state, shared_state);
        s.absorb_idx = 0;
        s.sample_idx = BN254_SPONGE_RATE;
    }
    s.sample_idx--;
    return s.sponge_state[s.sample_idx];
}

// --- MultiFieldTranscript operations ---

/// Flush observe_buf: pack and absorb into sponge.
__device__ void bn254_transcript_flush_observe(
    DeviceBn254SpongeState &s,
    Bn254PoseidonPermShared shared_state
) {
    if (s.observe_buf_len > 0) {
        Bn254Fr packed = bn254_noinline::bn254_pack_base_2_31(s.observe_buf, s.observe_buf_len);
        bn254_sponge_absorb(s, packed, shared_state);
        s.observe_buf_len = 0;
    }
}

/// Observe a canonical BabyBear u32 value.
/// Matches MultiFieldTranscript::observe().
/// Note: sample_buf clearing is a no-op on device (not present in device state).
__device__ void bn254_transcript_observe(
    DeviceBn254SpongeState &s,
    uint32_t value,
    Bn254PoseidonPermShared shared_state
) {
    s.observe_buf[s.observe_buf_len++] = value;
    if (s.observe_buf_len == BN254_NUM_OBS_PER_WORD) {
        Bn254Fr packed =
            bn254_noinline::bn254_pack_base_2_31(s.observe_buf, BN254_NUM_OBS_PER_WORD);
        bn254_sponge_absorb(s, packed, shared_state);
        s.observe_buf_len = 0;
    }
}

/// Sample a single canonical BabyBear u32 value (first base-p digit).
///
/// During grinding, sample_buf is always empty (observe clears it), so we
/// always squeeze fresh. The first base-p digit is simply `canonical_value % p`.
__device__ uint32_t
bn254_transcript_sample(DeviceBn254SpongeState &s, Bn254PoseidonPermShared shared_state) {
    bn254_transcript_flush_observe(s, shared_state);
    Bn254Fr squeezed = bn254_sponge_squeeze(s, shared_state);
    uint64_t canonical[4];
    bn254_noinline::bn254_to_canonical(canonical, squeezed);
    return bn254_noinline::u256_mod_u32(canonical, (uint32_t)BABYBEAR_PRIME);
}

/// Returns true if check_witness(bits, witness) passes.
__device__ bool bn254_sponge_check_witness(
    DeviceBn254SpongeState &s,
    uint32_t bits,
    uint32_t witness,
    Bn254PoseidonPermShared shared_state
) {
    bn254_transcript_observe(s, witness, shared_state);
    uint32_t sample = bn254_transcript_sample(s, shared_state);
    return (sample & ((1u << bits) - 1)) == 0;
}

static const uint32_t BN254_GRIND_BLOCK_SIZE = 32;

/// Grinding kernel: find any w in [min_witness, max_witness] with check_witness(bits,w)==true.
__launch_bounds__(BN254_GRIND_BLOCK_SIZE) __global__ void bn254_grind_kernel(
    const DeviceBn254SpongeState *init_state,
    uint32_t bits,
    uint32_t min_witness,
    uint32_t max_witness,
    volatile uint32_t *result
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    uint32_t w = min_witness + tid;
    __shared__ DeviceBn254SpongeState s_local_state[1];
    for (int i = threadIdx.x; i < sizeof(DeviceBn254SpongeState) / sizeof(uint32_t);
         i += blockDim.x) {
        ((uint32_t *)s_local_state)[i] = ((uint32_t *)init_state)[i];
    }

    Bn254PoseidonPermShared shared_state = load_shared();
    __syncthreads();
    if (w > max_witness || *result != UINT32_MAX)
        return;

    while (w <= max_witness) {
        if (*result != UINT32_MAX)
            return;

        DeviceBn254SpongeState local_state = s_local_state[0];
        if (bn254_sponge_check_witness(local_state, bits, w, shared_state)) {
            atomicCAS((uint32_t *)result, UINT32_MAX, w);
            return;
        }

        if (max_witness - w < stride)
            return;
        w += stride;
    }
}

extern "C" int _bn254_sponge_grind(
    const DeviceBn254SpongeState *init_state,
    uint32_t bits,
    uint32_t min_witness,
    uint32_t max_witness,
    uint32_t *result,
    cudaStream_t stream
) {
    if (bits >= 32 || (uint64_t{1} << bits) >= Fp::P) {
        return cudaErrorInvalidValue;
    }
    const size_t block_size = BN254_GRIND_BLOCK_SIZE;
    size_t total_threads = size_t{1} << bits;
    size_t grid_size = div_ceil(total_threads, block_size);

    bn254_grind_kernel<<<grid_size, block_size, 0, stream>>>(
        init_state, bits, min_witness, max_witness, result
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return (int)err;

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
        return (int)err;

    return CHECK_KERNEL();
}
