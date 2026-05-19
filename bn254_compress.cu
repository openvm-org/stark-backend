/// BN254 Poseidon2 row-hash kernels, extracted from
/// crates/cuda-backend/cuda/src/bn254_poseidon2.cu for standalone iteration.
///
/// Contains only what's needed to build and benchmark
/// `bn254_compressing_row_hashes_kernel{,_v2,_v3}` and their launchers:
///   _init_bn254_poseidon2_rc, _init_bn254_poseidon2_rc_w2,
///   _bn254_poseidon2_compressing_row_hashes{,_v2,_v3}.
///
/// The adjacent-compress kernel and the BN254 sponge grinding kernel are
/// deliberately not included here.

#include "bn254_u32_utils.cu" // bn254_b32::*
#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "poseidon2_bn254.cuh"
#include "poseidon2_bn254_noinline.cuh"
#include <cassert>
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// Round constant device memory (filled by _init_bn254_poseidon2_rc)
// ---------------------------------------------------------------------------

/// External initial round constants: 4 rounds × 3 elements
__device__ __constant__ Bn254Fr g_initial_rc[4][3];

/// Internal (partial) round constants: 56 rounds × 1 element (for state[0] only)
__device__ __constant__ Bn254Fr g_partial_rc[56];

/// External terminal round constants: 4 rounds × 3 elements
__device__ __constant__ Bn254Fr g_terminal_rc[4][3];

struct Bn254PoseidonPermShared {
    Bn254Fr *initial_rc;
    Bn254Fr *partial_rc;
    Bn254Fr *terminal_rc;
};
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

// ---------------------------------------------------------------------------
// Generic Poseidon2 permutation over BN254
//
// Parameterized by WIDTH, HALF_F (rF/2), and ROUNDS_P.
// Round constants are passed as pointers (to __constant__ memory).
//
//   external_initial_permute_state: MDS first, then HALF_F × (add_RC + sbox_all + MDS)
//   internal_permute_state:         ROUNDS_P × (add_RC[0] + sbox[0] + matmul_internal)
//   external_terminal_permute_state: HALF_F × (add_RC + sbox_all + MDS)
// ---------------------------------------------------------------------------

template <int WIDTH, int HALF_F, int ROUNDS_P>
static __device__ void bn254_poseidon2_permute_impl(
    Bn254Fr state[WIDTH],
    const Bn254Fr initial_rc[][WIDTH],
    const Bn254Fr *partial_rc,
    const Bn254Fr terminal_rc[][WIDTH]
) {
    // --- Initial external layer ---
    bn254::bn254_mds_external<WIDTH>(state);
    for (int r = 0; r < HALF_F; r++) {
        for (int i = 0; i < WIDTH; i++) {
            state[i] = bn254::bn254_add(state[i], initial_rc[r][i]);
            state[i] = bn254::bn254_sbox(state[i]);
        }
        bn254::bn254_mds_external<WIDTH>(state);
    }

    // --- Internal (partial) layer ---
    for (int r = 0; r < ROUNDS_P; r++) {
        state[0] = bn254::bn254_add(state[0], partial_rc[r]);
        state[0] = bn254::bn254_sbox(state[0]);
        bn254::bn254_mds_internal<WIDTH>(state);
    }

    // --- Terminal external layer ---
    for (int r = 0; r < HALF_F; r++) {
        for (int i = 0; i < WIDTH; i++) {
            state[i] = bn254::bn254_add(state[i], terminal_rc[r][i]);
            state[i] = bn254::bn254_sbox(state[i]);
        }
        bn254::bn254_mds_external<WIDTH>(state);
    }
}

// --- Width-3 permutation (rF=8, rP=56) for leaf hashing and transcript sponge ---
// Matches p3-bn254's Poseidon2Bn254<3> (HorizenLabs constants).

static __device__ void bn254_poseidon2_permute(Bn254Fr state[3]) {
    bn254_poseidon2_permute_impl<3, 4, 56>(state, g_initial_rc, g_partial_rc, g_terminal_rc);
}

// --- Width-2 permutation (rF=6, rP=50) for Merkle compression ---
// Matches Poseidon2Bn254Width2 (gnark-crypto constants).

/// Width-2 external initial round constants: 3 rounds × 2 elements
__device__ __constant__ Bn254Fr g_initial_rc_w2[3][2];

/// Width-2 internal (partial) round constants: 50 rounds × 1 element
__device__ __constant__ Bn254Fr g_partial_rc_w2[50];

/// Width-2 external terminal round constants: 3 rounds × 2 elements
__device__ __constant__ Bn254Fr g_terminal_rc_w2[3][2];

static __device__ Bn254PoseidonPermShared load_shared_w2() {
    __shared__ uint64_t buf2[(3 * 2 + 50 + 3 * 2) * 4];
    for (int i = threadIdx.x; i < 6 * 4; i += blockDim.x) {
        buf2[i] = ((uint64_t *)g_initial_rc_w2)[i];
    }

    for (int i = threadIdx.x; i < 50 * 4; i += blockDim.x) {
        buf2[i + 6 * 4] = ((uint64_t *)g_partial_rc_w2)[i];
    }

    for (int i = threadIdx.x; i < 6 * 4; i += blockDim.x) {
        buf2[i + 6 * 4 + 50 * 4] = ((uint64_t *)g_terminal_rc_w2)[i];
    }
    auto ptr = (Bn254Fr *)buf2;

    return {ptr, ptr + 6, ptr + 6 + 50};
}

static __device__ void bn254_poseidon2_permute_w2(Bn254Fr state[2]) {
    bn254_poseidon2_permute_impl<2, 3, 50>(
        state, g_initial_rc_w2, g_partial_rc_w2, g_terminal_rc_w2
    );
}

// ---------------------------------------------------------------------------
// 32-bit-limb mirrors of the Poseidon2 round constants. Bn254Fr32 has the
// same 32-byte layout as Bn254Fr, so the host fills both copies with the
// same source bytes in _init_bn254_poseidon2_rc{,_w2} below.
// ---------------------------------------------------------------------------

__device__ __constant__ Bn254Fr32 g_initial_rc_b32[4][3];
__device__ __constant__ Bn254Fr32 g_partial_rc_b32[56];
__device__ __constant__ Bn254Fr32 g_terminal_rc_b32[4][3];

__device__ __constant__ Bn254Fr32 g_initial_rc_w2_b32[3][2];
__device__ __constant__ Bn254Fr32 g_partial_rc_w2_b32[50];
__device__ __constant__ Bn254Fr32 g_terminal_rc_w2_b32[3][2];

// b32 Poseidon2 permutation impl (parameterized over WIDTH/HALF_F/ROUNDS_P).
// Built on bn254_b32::bn254_add / bn254_sbox / bn254_mds_external /
// bn254_mds_internal — all of which lower to IMAD.WIDE / IADD3.X chains via
// inline PTX.
template <int WIDTH, int HALF_F, int ROUNDS_P>
static __device__ void bn254_poseidon2_permute_impl_b32(
    Bn254Fr32 state[WIDTH],
    const Bn254Fr32 initial_rc[][WIDTH],
    const Bn254Fr32 *partial_rc,
    const Bn254Fr32 terminal_rc[][WIDTH]
) {
    bn254_b32::bn254_mds_external<WIDTH>(state);
    for (int r = 0; r < HALF_F; r++) {
        for (int i = 0; i < WIDTH; i++) {
            state[i] = bn254_b32::bn254_add(state[i], initial_rc[r][i]);
            state[i] = bn254_b32::bn254_sbox(state[i]);
        }
        bn254_b32::bn254_mds_external<WIDTH>(state);
    }
    for (int r = 0; r < ROUNDS_P; r++) {
        state[0] = bn254_b32::bn254_add(state[0], partial_rc[r]);
        state[0] = bn254_b32::bn254_sbox(state[0]);
        bn254_b32::bn254_mds_internal<WIDTH>(state);
    }
    for (int r = 0; r < HALF_F; r++) {
        for (int i = 0; i < WIDTH; i++) {
            state[i] = bn254_b32::bn254_add(state[i], terminal_rc[r][i]);
            state[i] = bn254_b32::bn254_sbox(state[i]);
        }
        bn254_b32::bn254_mds_external<WIDTH>(state);
    }
}

static __device__ void bn254_poseidon2_permute_b32(Bn254Fr32 state[3]) {
    bn254_poseidon2_permute_impl_b32<3, 4, 56>(
        state, g_initial_rc_b32, g_partial_rc_b32, g_terminal_rc_b32
    );
}

static __device__ void bn254_poseidon2_permute_w2_b32(Bn254Fr32 state[2]) {
    bn254_poseidon2_permute_impl_b32<2, 3, 50>(
        state, g_initial_rc_w2_b32, g_partial_rc_w2_b32, g_terminal_rc_w2_b32
    );
}

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
// Row hash helpers
// ---------------------------------------------------------------------------

/// Row hash for a base-field (Fp / BabyBear) matrix row.
static __device__ Bn254Fr
bn254_row_hash(const Fp *matrix, size_t width, size_t height, size_t row) {
    Bn254Fr state[3];
    state[0] = bn254_zero_init();
    state[1] = bn254_zero_init();
    state[2] = bn254_zero_init();

    uint32_t buf[BN254_BABY_BEAR_RATE];
    int cnt = 0;

    for (size_t col = 0; col < width; col++) {
        buf[cnt++] = matrix[col * height + row].asUInt32();
        if (cnt == BN254_BABY_BEAR_RATE) {
            state[0] = bn254::bn254_pack_base_2_31(buf, BN254_NUM_F_ELMS);
            state[1] = bn254::bn254_pack_base_2_31(buf + BN254_NUM_F_ELMS, BN254_NUM_F_ELMS);
            bn254_poseidon2_permute(state);
            cnt = 0;
        }
    }
    if (cnt > 0) {
        state[0] = bn254::bn254_pack_base_2_31(buf, min(BN254_NUM_F_ELMS, cnt));
        if (cnt > BN254_NUM_F_ELMS)
            state[1] = bn254::bn254_pack_base_2_31(
                buf + BN254_NUM_F_ELMS, min(BN254_NUM_F_ELMS, cnt - BN254_NUM_F_ELMS)
            );
        bn254_poseidon2_permute(state);
    }
    return state[0];
}

static_assert(
    BN254_BABY_BEAR_RATE % 4 == 0,
    "BN254_BABY_BEAR_RATE must be a multiple of FpExt degree (4)"
);

/// Row hash for an extension-field (FpExt / BinomialExtensionField<BabyBear,4>)
/// matrix row. Each matrix entry contributes 4 BabyBear u32 values to the
/// sponge buffer.
static __device__ Bn254Fr
bn254_row_hash_ext(const FpExt *matrix, size_t width, size_t height, size_t row) {
    Bn254Fr state[3];
    state[0] = bn254_zero_init();
    state[1] = bn254_zero_init();
    state[2] = bn254_zero_init();

    uint32_t buf[BN254_BABY_BEAR_RATE];
    int cnt = 0;

    for (size_t col = 0; col < width; col++) {
        FpExt elem = matrix[col * height + row];
        for (int d = 0; d < 4; d++) {
            buf[cnt++] = elem.elems[d].asUInt32();
            if (cnt == BN254_BABY_BEAR_RATE) {
                state[0] = bn254::bn254_pack_base_2_31(buf, BN254_NUM_F_ELMS);
                state[1] =
                    bn254::bn254_pack_base_2_31(buf + BN254_NUM_F_ELMS, BN254_NUM_F_ELMS);
                bn254_poseidon2_permute(state);
                cnt = 0;
            }
        }
    }
    if (cnt > 0) {
        state[0] = bn254::bn254_pack_base_2_31(buf, min(BN254_NUM_F_ELMS, cnt));
        if (cnt > BN254_NUM_F_ELMS)
            state[1] = bn254::bn254_pack_base_2_31(
                buf + BN254_NUM_F_ELMS, min(BN254_NUM_F_ELMS, cnt - BN254_NUM_F_ELMS)
            );
        bn254_poseidon2_permute(state);
    }
    return state[0];
}

/// b32 mirror of bn254_row_hash. Uses bn254_b32::bn254_pack_base_2_31 and the
/// b32 width-3 permutation. The 16-u32 sponge `buf` lives in shared memory
/// with a strided layout to avoid spilling it to per-thread local memory:
///
///   thread tid's buf[i] is at buf_shared[i * BLOCK_SIZE + tid]
///
/// Since BLOCK_SIZE is a multiple of the bank count (32), the warp accesses
/// `buf_shared[i * BLOCK_SIZE + t]` for t in 0..31 fall on distinct banks for
/// every i — no bank conflicts.
template <int BLOCK_SIZE>
static __device__ Bn254Fr32 bn254_row_hash_b32(
    const Fp *matrix,
    size_t width,
    size_t height,
    size_t row,
    uint32_t *buf_shared,
    int tid_lin
) {
    Bn254Fr32 state[3];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        state[0].limbs[i] = 0;
        state[1].limbs[i] = 0;
        state[2].limbs[i] = 0;
    }

    auto buf_set = [&](int i, uint32_t v) { buf_shared[i * BLOCK_SIZE + tid_lin] = v; };
    auto buf_get = [&](int i) -> uint32_t { return buf_shared[i * BLOCK_SIZE + tid_lin]; };

    // Equivalent of bn254_b32::bn254_pack_base_2_31 but reading buf via
    // strided shared accesses instead of from a contiguous u32 array.
    auto pack_strided = [&](int start, int count) -> Bn254Fr32 {
        uint32_t canonical[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        for (int i = 0; i < count; i++) {
            uint32_t v = buf_get(start + i);
            int bit_pos = i * 31;
            int limb = bit_pos >> 5;
            int shift = bit_pos & 31;
            canonical[limb] |= v << shift;
            if (shift > 1 && limb < 7) {
                canonical[limb + 1] |= v >> (32 - shift);
            }
        }
        return bn254_b32::bn254_from_canonical(canonical);
    };

    int cnt = 0;
    for (size_t col = 0; col < width; col++) {
        buf_set(cnt, matrix[col * height + row].asUInt32());
        cnt++;
        if (cnt == BN254_BABY_BEAR_RATE) {
            state[0] = pack_strided(0, BN254_NUM_F_ELMS);
            state[1] = pack_strided(BN254_NUM_F_ELMS, BN254_NUM_F_ELMS);
            bn254_poseidon2_permute_b32(state);
            cnt = 0;
        }
    }
    if (cnt > 0) {
        state[0] = pack_strided(0, min(BN254_NUM_F_ELMS, cnt));
        if (cnt > BN254_NUM_F_ELMS)
            state[1] =
                pack_strided(BN254_NUM_F_ELMS, min(BN254_NUM_F_ELMS, cnt - BN254_NUM_F_ELMS));
        bn254_poseidon2_permute_b32(state);
    }
    return state[0];
}

/// b32 mirror of bn254_row_hash_ext. Same shared-buf strided layout as
/// bn254_row_hash_b32; each matrix entry contributes 4 BabyBear u32 limbs
/// (the FpExt's 4 base-field components) to the sponge buffer.
template <int BLOCK_SIZE>
static __device__ Bn254Fr32 bn254_row_hash_ext_b32(
    const FpExt *matrix,
    size_t width,
    size_t height,
    size_t row,
    uint32_t *buf_shared,
    int tid_lin
) {
    Bn254Fr32 state[3];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        state[0].limbs[i] = 0;
        state[1].limbs[i] = 0;
        state[2].limbs[i] = 0;
    }

    auto buf_set = [&](int i, uint32_t v) { buf_shared[i * BLOCK_SIZE + tid_lin] = v; };
    auto buf_get = [&](int i) -> uint32_t { return buf_shared[i * BLOCK_SIZE + tid_lin]; };

    auto pack_strided = [&](int start, int count) -> Bn254Fr32 {
        uint32_t canonical[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        for (int i = 0; i < count; i++) {
            uint32_t v = buf_get(start + i);
            int bit_pos = i * 31;
            int limb = bit_pos >> 5;
            int shift = bit_pos & 31;
            canonical[limb] |= v << shift;
            if (shift > 1 && limb < 7) {
                canonical[limb + 1] |= v >> (32 - shift);
            }
        }
        return bn254_b32::bn254_from_canonical(canonical);
    };

    int cnt = 0;
    for (size_t col = 0; col < width; col++) {
        FpExt elem = matrix[col * height + row];
        for (int d = 0; d < 4; d++) {
            buf_set(cnt, elem.elems[d].asUInt32());
            cnt++;
            if (cnt == BN254_BABY_BEAR_RATE) {
                state[0] = pack_strided(0, BN254_NUM_F_ELMS);
                state[1] = pack_strided(BN254_NUM_F_ELMS, BN254_NUM_F_ELMS);
                bn254_poseidon2_permute_b32(state);
                cnt = 0;
            }
        }
    }
    if (cnt > 0) {
        state[0] = pack_strided(0, min(BN254_NUM_F_ELMS, cnt));
        if (cnt > BN254_NUM_F_ELMS)
            state[1] =
                pack_strided(BN254_NUM_F_ELMS, min(BN254_NUM_F_ELMS, cnt - BN254_NUM_F_ELMS));
        bn254_poseidon2_permute_b32(state);
    }
    return state[0];
}

/// TruncatedPermutation compress: (left, right) → permute_w2([left, right])[0].
static __device__ Bn254Fr bn254_compress(Bn254Fr left, Bn254Fr right) {
    Bn254Fr state[2];
    state[0] = left;
    state[1] = right;
    bn254_poseidon2_permute_w2(state);
    return state[0];
}

/// b32 mirror of bn254_compress.
static __device__ Bn254Fr32 bn254_compress_b32(Bn254Fr32 left, Bn254Fr32 right) {
    Bn254Fr32 state[2];
    state[0] = left;
    state[1] = right;
    bn254_poseidon2_permute_w2_b32(state);
    return state[0];
}

// ---------------------------------------------------------------------------
// Merkle row-hash kernel (F / BabyBear matrix)
// ---------------------------------------------------------------------------

__global__ void bn254_compressing_row_hashes_kernel(
    bn254_digest_t *out,
    const Fp *matrix,
    size_t width,
    size_t height,
    size_t query_stride,
    size_t log_rows_per_query
) {
    extern __shared__ char smem[]; // Bn254Fr[blockDim.x * (blockDim.y/2)]
    Bn254Fr *shared = reinterpret_cast<Bn254Fr *>(smem);

    const uint32_t stride_idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t leaf_idx = threadIdx.y;
    const size_t row = leaf_idx * query_stride + stride_idx;

    Bn254Fr digest = bn254_zero_init();

    if (stride_idx < query_stride) {
        digest = bn254_row_hash(matrix, width, height, row);
    }

    // Tree reduction (same structure as the BabyBear kernel)
    for (int layer = 0; layer < (int)log_rows_per_query; ++layer) {
        uint32_t mask = (1 << (layer + 1)) - 1;
        uint32_t shared_offset = ((leaf_idx >> (layer + 1)) << layer) * blockDim.x + threadIdx.x;

        if ((leaf_idx & mask) == (1u << layer)) {
            shared[shared_offset] = digest;
        }
        __syncthreads();
        if ((leaf_idx & mask) == 0) {
            Bn254Fr sibling = shared[shared_offset];
            digest = bn254_compress(digest, sibling);
        }
        __syncthreads();
    }

    if (leaf_idx == 0 && stride_idx < query_stride) {
        out[stride_idx].elem = digest;
    }
}

template <int NThreads, int LogRowsPerQuery>
__global__ __launch_bounds__(NThreads) void bn254_compressing_row_hashes_kernel_v3(
    bn254_digest_t *out,
    const Fp *matrix,
    size_t width,
    size_t height,
    size_t query_stride
) {
    // 32-bit-limb digests through the row-hash + tree reduction. The shared
    // array serves two time-disjoint purposes:
    //
    //   - during row_hash: scratch for each thread's 16-u32 sponge `buf`,
    //     laid out so thread t's buf[i] lives at ((u32*)shared)[i*NThreads + t].
    //     NThreads is a multiple of 32 → no bank conflicts within a warp.
    //     Size needed: NThreads × BN254_BABY_BEAR_RATE u32 = NThreads × 2 Bn254Fr32.
    //   - during tree reduction: Bn254Fr32 digest exchange, only the first
    //     NThreads/2 elements are used.
    //
    // A __syncthreads() separates the two uses so writes from row_hash are
    // visible (and finished) before tree reduction overwrites them.
    static_assert(NThreads % 32 == 0, "NThreads must be a multiple of warp size");
    static_assert(BN254_BABY_BEAR_RATE == 16, "buf-in-shared sizing assumes RATE=16");
    __shared__ Bn254Fr32 shared[NThreads * 2];

    const uint32_t stride_idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t leaf_idx = threadIdx.y;
    const size_t row = leaf_idx * query_stride + stride_idx;
    const int tid_lin = threadIdx.y * blockDim.x + threadIdx.x;

    Bn254Fr32 digest{};
    if (stride_idx < query_stride) {
        digest =
            bn254_row_hash_b32<NThreads>(matrix, width, height, row, (uint32_t *)shared, tid_lin);
    }
    // Repurpose `shared` for tree-reduction digest exchange; ensure all of
    // row_hash's buf writes are done across the block.
    __syncthreads();

    // Tree reduction (same structure as the BabyBear kernel)
#pragma unroll 1
    for (int layer = 0; layer < (int)LogRowsPerQuery; ++layer) {
        uint32_t mask = (1 << (layer + 1)) - 1;
        uint32_t shared_offset = ((leaf_idx >> (layer + 1)) << layer) * blockDim.x + threadIdx.x;

        if ((leaf_idx & mask) == (1u << layer)) {
            shared[shared_offset] = digest;
        }
        __syncthreads();
        if ((leaf_idx & mask) == 0) {
            Bn254Fr32 sibling = shared[shared_offset];
            digest = bn254_compress_b32(digest, sibling);
        }
        __syncthreads();
    }

    if (leaf_idx == 0 && stride_idx < query_stride) {
        ((Bn254Fr32 *)out)[stride_idx] = digest;
    }
}

// ---------------------------------------------------------------------------
// Merkle row-hash kernel (FpExt matrix). Mirror of the F kernel above but
// reading from a degree-4 extension matrix (4 BabyBear u32s per entry).
// ---------------------------------------------------------------------------

__global__ void bn254_compressing_row_hashes_ext_kernel(
    bn254_digest_t *out,
    const FpExt *matrix,
    size_t width,
    size_t height,
    size_t query_stride,
    size_t log_rows_per_query
) {
    extern __shared__ char smem[];
    Bn254Fr *shared = reinterpret_cast<Bn254Fr *>(smem);

    const uint32_t stride_idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t leaf_idx = threadIdx.y;
    const size_t row = leaf_idx * query_stride + stride_idx;

    Bn254Fr digest = bn254_zero_init();
    if (stride_idx < query_stride) {
        digest = bn254_row_hash_ext(matrix, width, height, row);
    }

    for (int layer = 0; layer < (int)log_rows_per_query; ++layer) {
        uint32_t mask = (1 << (layer + 1)) - 1;
        uint32_t shared_offset = ((leaf_idx >> (layer + 1)) << layer) * blockDim.x + threadIdx.x;

        if ((leaf_idx & mask) == (1u << layer)) {
            shared[shared_offset] = digest;
        }
        __syncthreads();
        if ((leaf_idx & mask) == 0) {
            Bn254Fr sibling = shared[shared_offset];
            digest = bn254_compress(digest, sibling);
        }
        __syncthreads();
    }

    if (leaf_idx == 0 && stride_idx < query_stride) {
        out[stride_idx].elem = digest;
    }
}

// b32 mirror of the ext row-hash kernel — same structure as v3 (F-field) but
// uses bn254_row_hash_ext_b32, which iterates over FpExt rows.
template <int NThreads, int LogRowsPerQuery>
__global__ __launch_bounds__(NThreads) void bn254_compressing_row_hashes_ext_kernel_v3(
    bn254_digest_t *out,
    const FpExt *matrix,
    size_t width,
    size_t height,
    size_t query_stride
) {
    static_assert(NThreads % 32 == 0, "NThreads must be a multiple of warp size");
    static_assert(BN254_BABY_BEAR_RATE == 16, "buf-in-shared sizing assumes RATE=16");
    __shared__ Bn254Fr32 shared[NThreads * 2];

    const uint32_t stride_idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t leaf_idx = threadIdx.y;
    const size_t row = leaf_idx * query_stride + stride_idx;
    const int tid_lin = threadIdx.y * blockDim.x + threadIdx.x;

    Bn254Fr32 digest{};
    if (stride_idx < query_stride) {
        digest = bn254_row_hash_ext_b32<NThreads>(
            matrix, width, height, row, (uint32_t *)shared, tid_lin
        );
    }
    __syncthreads();

#pragma unroll 1
    for (int layer = 0; layer < (int)LogRowsPerQuery; ++layer) {
        uint32_t mask = (1 << (layer + 1)) - 1;
        uint32_t shared_offset = ((leaf_idx >> (layer + 1)) << layer) * blockDim.x + threadIdx.x;

        if ((leaf_idx & mask) == (1u << layer)) {
            shared[shared_offset] = digest;
        }
        __syncthreads();
        if ((leaf_idx & mask) == 0) {
            Bn254Fr32 sibling = shared[shared_offset];
            digest = bn254_compress_b32(digest, sibling);
        }
        __syncthreads();
    }

    if (leaf_idx == 0 && stride_idx < query_stride) {
        ((Bn254Fr32 *)out)[stride_idx] = digest;
    }
}

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
    // Mirror into b32 constants. Bn254Fr32 (8 × u32) has the same 32-byte
    // little-endian layout as Bn254Fr (4 × u64), so the source bytes copy
    // through directly — matches the (lo, hi) pair convention used by
    // BN254_P_32 / BN254_R2_32 in the header.
    err = cudaMemcpyToSymbol(g_initial_rc_b32, initial_rc, 4 * 3 * 4 * sizeof(uint64_t));
    if (err != cudaSuccess)
        return (int)err;
    err = cudaMemcpyToSymbol(g_partial_rc_b32, partial_rc, 56 * 4 * sizeof(uint64_t));
    if (err != cudaSuccess)
        return (int)err;
    err = cudaMemcpyToSymbol(g_terminal_rc_b32, terminal_rc, 4 * 3 * 4 * sizeof(uint64_t));
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
    // Mirror into b32 width-2 constants (same byte layout reasoning).
    err = cudaMemcpyToSymbol(g_initial_rc_w2_b32, initial_rc, 3 * 2 * 4 * sizeof(uint64_t));
    if (err != cudaSuccess)
        return (int)err;
    err = cudaMemcpyToSymbol(g_partial_rc_w2_b32, partial_rc, 50 * 4 * sizeof(uint64_t));
    if (err != cudaSuccess)
        return (int)err;
    err = cudaMemcpyToSymbol(g_terminal_rc_w2_b32, terminal_rc, 3 * 2 * 4 * sizeof(uint64_t));
    if (err != cudaSuccess)
        return (int)err;
    return (int)cudaSuccess;
}

extern "C" int _bn254_poseidon2_compressing_row_hashes(
    bn254_digest_t *out,
    const Fp *matrix,
    size_t width,
    size_t query_stride,
    size_t log_rows_per_query,
    cudaStream_t stream
) {
    if (log_rows_per_query > 10) {
        return cudaErrorInvalidValue;
    }
    size_t block_y = size_t{1} << log_rows_per_query;
    size_t threads_x = std::max<size_t>(1, size_t{512} / block_y);
    auto [grid, block] = kernel_launch_params(query_stride, threads_x);
    block.y = block_y;
    size_t shared_stride = block.x * div_ceil(block.y, 2);
    size_t shmem_bytes = shared_stride * sizeof(Bn254Fr);
    auto height = query_stride << log_rows_per_query;

    bn254_compressing_row_hashes_kernel<<<grid, block, shmem_bytes, stream>>>(
        out, matrix, width, height, query_stride, log_rows_per_query
    );
    return CHECK_KERNEL();
}

extern "C" int _bn254_poseidon2_compressing_row_hashes_v3(
    bn254_digest_t *out,
    const Fp *matrix,
    size_t width,
    size_t query_stride,
    size_t log_rows_per_query,
    cudaStream_t stream
) {
    if (log_rows_per_query > 10) {
        return cudaErrorInvalidValue;
    }
    size_t block_y = size_t{1} << log_rows_per_query;
    size_t threads_x = std::max<size_t>(1, size_t{512} / block_y);
    auto [grid, block] = kernel_launch_params(query_stride, threads_x);
    block.y = block_y;
    auto height = query_stride << log_rows_per_query;

    bn254_compressing_row_hashes_kernel_v3<512, 4>
        <<<grid, block, 0, stream>>>(out, matrix, width, height, query_stride);
    return CHECK_KERNEL();
}

extern "C" int _bn254_poseidon2_compressing_row_hashes_ext(
    bn254_digest_t *out,
    const FpExt *matrix,
    size_t width,
    size_t query_stride,
    size_t log_rows_per_query,
    cudaStream_t stream
) {
    if (log_rows_per_query > 10) {
        return cudaErrorInvalidValue;
    }
    size_t block_y = size_t{1} << log_rows_per_query;
    size_t threads_x = std::max<size_t>(1, size_t{512} / block_y);
    auto [grid, block] = kernel_launch_params(query_stride, threads_x);
    block.y = block_y;
    size_t shared_stride = block.x * div_ceil(block.y, 2);
    size_t shmem_bytes = shared_stride * sizeof(Bn254Fr);
    auto height = query_stride << log_rows_per_query;

    bn254_compressing_row_hashes_ext_kernel<<<grid, block, shmem_bytes, stream>>>(
        out, matrix, width, height, query_stride, log_rows_per_query
    );
    return CHECK_KERNEL();
}

extern "C" int _bn254_poseidon2_compressing_row_hashes_ext_v3(
    bn254_digest_t *out,
    const FpExt *matrix,
    size_t width,
    size_t query_stride,
    size_t log_rows_per_query,
    cudaStream_t stream
) {
    if (log_rows_per_query > 10) {
        return cudaErrorInvalidValue;
    }
    size_t block_y = size_t{1} << log_rows_per_query;
    size_t threads_x = std::max<size_t>(1, size_t{512} / block_y);
    auto [grid, block] = kernel_launch_params(query_stride, threads_x);
    block.y = block_y;
    auto height = query_stride << log_rows_per_query;

    bn254_compressing_row_hashes_ext_kernel_v3<512, 4>
        <<<grid, block, 0, stream>>>(out, matrix, width, height, query_stride);
    return CHECK_KERNEL();
}
