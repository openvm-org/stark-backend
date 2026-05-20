/// BN254 Poseidon2 Merkle row-hash and adjacent-compress kernels (b32 limbs).
///
/// Implements the entry points called by the Rust Merkle layer:
///   _bn254_poseidon2_compressing_row_hashes     - Merkle leaf hashes, F matrix
///   _bn254_poseidon2_compressing_row_hashes_ext - Merkle leaf hashes, EF matrix
///   _bn254_poseidon2_adjacent_compress_layer    - Merkle internal compress
///
/// All three are built on the 32-bit-limb Poseidon2 primitives in `poseidon2_bn254_b32.cuh`. The
/// width-3 and width-2 round constants are owned by `bn254_constants.cu`;
/// this TU just declares them `extern` and reinterprets the storage as
/// `Bn254Fr32 *` (the layouts are byte-identical: 4 × u64 ↔ 8 × u32).
///
/// The grinding kernel and its b64 sponge state live in `bn254_poseidon2_grind.cu`.
/// The round-constant `cudaMemcpyToSymbol` launchers live in
/// `bn254_constants.cu`.

#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "poseidon2_bn254_b32.cuh" // bn254_b32::*
#include <cstdint>

// ---------------------------------------------------------------------------
// Round constant device memory.
//
// Storage is defined in bn254_constants.cu as Bn254Fr (4 × u64). The b32
// permutation below reinterprets the same bytes as Bn254Fr32 (8 × u32) via
// pointer cast — Bn254Fr32 has identical 32-byte little-endian layout to
// Bn254Fr, with adjacent (lo, hi) u32 pairs matching one u64 limb (see the
// BN254_P_32 / BN254_R2_32 layout in poseidon2_bn254_common.cuh).
// ---------------------------------------------------------------------------

/// External initial round constants: 4 rounds × 3 elements
extern __device__ __constant__ Bn254Fr g_initial_rc[4][3];

/// Internal (partial) round constants: 56 rounds × 1 element (for state[0] only)
extern __device__ __constant__ Bn254Fr g_partial_rc[56];

/// External terminal round constants: 4 rounds × 3 elements
extern __device__ __constant__ Bn254Fr g_terminal_rc[4][3];

// b32 Poseidon2 permutation impl (parameterized over WIDTH/HALF_F/ROUNDS_P).
// Built on bn254_b32::bn254_add / bn254_sbox / bn254_mds_external /
// bn254_mds_internal — all of which lower to IMAD.WIDE / IADD3.X chains via
// inline PTX.
template <int WIDTH, int HALF_F, int ROUNDS_P>
static __device__ void bn254_poseidon2_permute_impl_b32(
    Bn254Fr32 state[WIDTH],
    const Bn254Fr32 *initial_rc,
    const Bn254Fr32 *partial_rc,
    const Bn254Fr32 *terminal_rc
) {
    bn254_b32::bn254_mds_external<WIDTH>(state);
    for (int r = 0; r < HALF_F; r++) {
        for (int i = 0; i < WIDTH; i++) {
            state[i] = bn254_b32::bn254_add(state[i], initial_rc[r * WIDTH + i]);
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
            state[i] = bn254_b32::bn254_add(state[i], terminal_rc[r * WIDTH + i]);
            state[i] = bn254_b32::bn254_sbox(state[i]);
        }
        bn254_b32::bn254_mds_external<WIDTH>(state);
    }
}

static __device__ void bn254_poseidon2_permute_b32(Bn254Fr32 state[3]) {
    bn254_poseidon2_permute_impl_b32<3, 4, 56>(
        state, (Bn254Fr32 *)g_initial_rc, (Bn254Fr32 *)g_partial_rc, (Bn254Fr32 *)g_terminal_rc
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
static __device__ void bn254_poseidon2_permute_w2_b32(Bn254Fr32 state[2]) {
    bn254_poseidon2_permute_impl_b32<2, 3, 50>(
        state,
        (Bn254Fr32 *)g_initial_rc_w2,
        (Bn254Fr32 *)g_partial_rc_w2,
        (Bn254Fr32 *)g_terminal_rc_w2
    );
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

/// 2-to-1 truncated permutation compress: (left, right) → permute_w2([l, r])[0].
static __device__ Bn254Fr32 bn254_compress_b32(Bn254Fr32 left, Bn254Fr32 right) {
    Bn254Fr32 state[2];
    state[0] = left;
    state[1] = right;
    bn254_poseidon2_permute_w2_b32(state);
    return state[0];
}

// ---------------------------------------------------------------------------
// Adjacent-compress kernel
//   output[gid] = compress(prev_layer[2*gid], prev_layer[2*gid + 1])
// for one internal layer of the Merkle tree.
// ---------------------------------------------------------------------------

__global__ void bn254_adjacent_compress_layer_kernel(
    Bn254Fr32 *output,
    const Bn254Fr32 *prev_layer,
    size_t output_size
) {
    uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= output_size)
        return;

    Bn254Fr32 left = prev_layer[2 * gid];
    Bn254Fr32 right = prev_layer[2 * gid + 1];
    output[gid] = bn254_compress_b32(left, right);
}

template <int NThreads>
__global__ __launch_bounds__(NThreads) void bn254_compressing_row_hashes_kernel_v3(
    Bn254Fr32 *out,
    const Fp *matrix,
    size_t width,
    size_t height,
    size_t query_stride,
    size_t log_rows_per_query
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
    for (int layer = 0; layer < (int)log_rows_per_query; ++layer) {
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

// b32 mirror of the ext row-hash kernel — same structure as v3 (F-field) but
// uses bn254_row_hash_ext_b32, which iterates over FpExt rows.
template <int NThreads>
__global__ __launch_bounds__(NThreads) void bn254_compressing_row_hashes_ext_kernel_v3(
    bn254_digest_t *out,
    const FpExt *matrix,
    size_t width,
    size_t height,
    size_t query_stride,
    size_t log_rows_per_query
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
    for (int layer = 0; layer < (int)log_rows_per_query; ++layer) {
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

extern "C" int _bn254_poseidon2_adjacent_compress_layer(
    bn254_digest_t *output,
    const bn254_digest_t *prev_layer,
    size_t output_size,
    cudaStream_t stream
) {
    auto [grid, block] = kernel_launch_params(output_size);
    bn254_adjacent_compress_layer_kernel<<<grid, block, 0, stream>>>(
        (Bn254Fr32 *)output, (Bn254Fr32 *)prev_layer, output_size
    );
    return CHECK_KERNEL();
}

// Both row-hash launchers below dispatch on `log_rows_per_query` to one of
// 5 instantiations of the v3 kernel (LogRowsPerQuery ∈ {0, 1, 2, 3, 4}). The
// kernel's tree-reduction loop is bounded by that template parameter, so it
// must match the runtime block.y = (1 << log_rows_per_query) — every other
// caller knob (block.x, total threads = 512, static shared = 32 KB) is the
// same across the four instantiations.

extern "C" int _bn254_poseidon2_compressing_row_hashes(
    bn254_digest_t *out,
    const Fp *matrix,
    size_t width,
    size_t query_stride,
    size_t log_rows_per_query,
    cudaStream_t stream
) {
    // make sure to change line 36 of merkle_tree.rs to adjust the constants
    if (log_rows_per_query > 9) {
        return cudaErrorInvalidValue;
    }
    size_t block_y = size_t{1} << log_rows_per_query;
    size_t threads_x = std::max<size_t>(1, size_t{512} / block_y);
    auto [grid, block] = kernel_launch_params(query_stride, threads_x);
    block.y = block_y;
    auto height = query_stride << log_rows_per_query;

    bn254_compressing_row_hashes_kernel_v3<512>
        <<<grid, block, 0, stream>>>((Bn254Fr32 *)out, matrix, width, height, query_stride, log_rows_per_query);
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
    if (log_rows_per_query > 9) {
        return cudaErrorInvalidValue;
    }
    size_t block_y = size_t{1} << log_rows_per_query;
    size_t threads_x = std::max<size_t>(1, size_t{512} / block_y);
    auto [grid, block] = kernel_launch_params(query_stride, threads_x);
    block.y = block_y;
    auto height = query_stride << log_rows_per_query;

    bn254_compressing_row_hashes_ext_kernel_v3<512>
        <<<grid, block, 0, stream>>>(out, matrix, width, height, query_stride, log_rows_per_query);
    return CHECK_KERNEL();
}
