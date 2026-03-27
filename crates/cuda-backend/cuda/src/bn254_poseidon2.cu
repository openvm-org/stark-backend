/// BN254 Poseidon2 CUDA kernels.
///
/// All BN254 code lives in this single translation unit to avoid device-side
/// linkage issues with `static __device__` variables.
///
/// Entry points (called from Rust via FFI):
///   _init_bn254_poseidon2_rc         - Upload round constants from Rust
///   _bn254_poseidon2_compressing_row_hashes     - Merkle leaf hashes (F matrix)
///   _bn254_poseidon2_compressing_row_hashes_ext - Merkle leaf hashes (EF matrix)
///   _bn254_poseidon2_adjacent_compress_layer    - Merkle internal compress
///   _bn254_sponge_grind              - PoW grinding for BN254 transcript

#include "poseidon2_bn254.cuh"
#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include <cstdint>

// ---------------------------------------------------------------------------
// Round constant device memory (filled by _init_bn254_poseidon2_rc)
// ---------------------------------------------------------------------------

/// External initial round constants: 4 rounds × 3 elements
__device__ __constant__ Bn254Fr g_initial_rc[4][3];

/// Internal (partial) round constants: 56 rounds × 1 element (for state[0] only)
__device__ __constant__ Bn254Fr g_partial_rc[56];

/// External terminal round constants: 4 rounds × 3 elements
__device__ __constant__ Bn254Fr g_terminal_rc[4][3];

// ---------------------------------------------------------------------------
// BN254 Poseidon2 permutation (WIDTH=3, 4 + 56 + 4 rounds)
//
// Matches p3-bn254's Poseidon2Bn254<3> permutation exactly:
//   external_initial_permute_state:  MDS first, then 4×(add_RC + sbox_all + MDS)
//   internal_permute_state:          56×(add_RC[0] + sbox[0] + matmul_internal)
//   external_terminal_permute_state: 4×(add_RC + sbox_all + MDS)
// ---------------------------------------------------------------------------

static __device__ __forceinline__
void bn254_poseidon2_permute(Bn254Fr state[3]) {
    // --- Initial external layer ---
    bn254_mds_external(state); // MDS without round constants
    for (int r = 0; r < 4; r++) {
        for (int i = 0; i < 3; i++) {
            state[i] = bn254_add(state[i], g_initial_rc[r][i]);
            state[i] = bn254_sbox(state[i]);
        }
        bn254_mds_external(state);
    }

    // --- Internal (partial) layer ---
    for (int r = 0; r < 56; r++) {
        state[0] = bn254_add(state[0], g_partial_rc[r]);
        state[0] = bn254_sbox(state[0]);
        bn254_mds_internal(state);
    }

    // --- Terminal external layer ---
    for (int r = 0; r < 4; r++) {
        for (int i = 0; i < 3; i++) {
            state[i] = bn254_add(state[i], g_terminal_rc[r][i]);
            state[i] = bn254_sbox(state[i]);
        }
        bn254_mds_external(state);
    }
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

static __device__ __forceinline__
Bn254Fr bn254_zero_init() {
    Bn254Fr z;
    for (int i = 0; i < 4; i++) z.limbs[i] = 0;
    return z;
}

// ---------------------------------------------------------------------------
// Sponge constants for Merkle hashing
//
// Matches MultiField32PaddingFreeSponge<BabyBear, Bn254Scalar, Perm, 3, 16, 1>:
//   BABY_BEAR_RATE = 16  BabyBear values absorbed per permutation
//   NUM_F_ELMS = 8       BabyBear values packed per Bn254Fr (floor(254/31) = 8)
// ---------------------------------------------------------------------------

static const int BN254_BABY_BEAR_RATE = 16;
static const int BN254_NUM_F_ELMS    = 8;

// ---------------------------------------------------------------------------
// Row hash helpers
// ---------------------------------------------------------------------------

/// Row hash for a base-field (Fp / BabyBear) matrix row.
static __device__ __forceinline__
Bn254Fr bn254_row_hash(const Fp* matrix, int width, int height, int row) {
    Bn254Fr state[3];
    state[0] = bn254_zero_init();
    state[1] = bn254_zero_init();
    state[2] = bn254_zero_init();

    uint32_t buf[BN254_BABY_BEAR_RATE];
    int cnt = 0;

    for (int col = 0; col < width; col++) {
        buf[cnt++] = matrix[col * height + row].asUInt32();
        if (cnt == BN254_BABY_BEAR_RATE) {
            state[0] = bn254_reduce_32(buf, BN254_NUM_F_ELMS);
            state[1] = bn254_reduce_32(buf + BN254_NUM_F_ELMS, BN254_NUM_F_ELMS);
            bn254_poseidon2_permute(state);
            cnt = 0;
        }
    }
    if (cnt > 0) {
        state[0] = bn254_reduce_32(buf, min(BN254_NUM_F_ELMS, cnt));
        if (cnt > BN254_NUM_F_ELMS)
            state[1] = bn254_reduce_32(buf + BN254_NUM_F_ELMS,
                                       min(BN254_NUM_F_ELMS, cnt - BN254_NUM_F_ELMS));
        bn254_poseidon2_permute(state);
    }
    return state[0];
}

/// Row hash for an extension-field (FpExt / BinomialExtensionField<BabyBear,4>) matrix row.
static __device__ __forceinline__
Bn254Fr bn254_row_hash_ext(const FpExt* matrix, int width, int height, int row) {
    Bn254Fr state[3];
    state[0] = bn254_zero_init();
    state[1] = bn254_zero_init();
    state[2] = bn254_zero_init();

    uint32_t buf[BN254_BABY_BEAR_RATE];
    int cnt = 0;

    for (int col = 0; col < width; col++) {
        FpExt elem = matrix[col * height + row];
        for (int d = 0; d < 4; d++) {
            buf[cnt++] = elem.elems[d].asUInt32();
            if (cnt == BN254_BABY_BEAR_RATE) {
                state[0] = bn254_reduce_32(buf, BN254_NUM_F_ELMS);
                state[1] = bn254_reduce_32(buf + BN254_NUM_F_ELMS, BN254_NUM_F_ELMS);
                bn254_poseidon2_permute(state);
                cnt = 0;
            }
        }
    }
    if (cnt > 0) {
        state[0] = bn254_reduce_32(buf, min(BN254_NUM_F_ELMS, cnt));
        if (cnt > BN254_NUM_F_ELMS)
            state[1] = bn254_reduce_32(buf + BN254_NUM_F_ELMS,
                                       min(BN254_NUM_F_ELMS, cnt - BN254_NUM_F_ELMS));
        bn254_poseidon2_permute(state);
    }
    return state[0];
}

/// TruncatedPermutation compress: (left, right) → permute([left, right, 0])[0].
static __device__ __forceinline__
Bn254Fr bn254_compress(Bn254Fr left, Bn254Fr right) {
    Bn254Fr state[3];
    state[0] = left;
    state[1] = right;
    state[2] = bn254_zero_init();
    bn254_poseidon2_permute(state);
    return state[0];
}

// ---------------------------------------------------------------------------
// Merkle row-hash kernel (F / BabyBear matrix)
// ---------------------------------------------------------------------------

__global__ void bn254_compressing_row_hashes_kernel(
    bn254_digest_t* out,
    const Fp*       matrix,
    size_t          width,
    size_t          height,
    size_t          query_stride,
    size_t          log_rows_per_query
) {
    extern __shared__ char smem[]; // Bn254Fr[blockDim.x * (blockDim.y/2)]
    Bn254Fr* shared = reinterpret_cast<Bn254Fr*>(smem);

    const uint32_t stride_idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t       leaf_idx   = threadIdx.y;
    const uint32_t row        = leaf_idx * query_stride + stride_idx;

    Bn254Fr digest = bn254_zero_init();

    if (stride_idx < query_stride) {
        digest = bn254_row_hash(matrix, (int)width, (int)height, (int)row);
    }

    // Tree reduction (same structure as the BabyBear kernel)
    for (int layer = 0; layer < (int)log_rows_per_query; ++layer) {
        uint32_t mask          = (1 << (layer + 1)) - 1;
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

// ---------------------------------------------------------------------------
// Merkle row-hash kernel (EF / BinomialExtensionField<BabyBear,4> matrix)
// ---------------------------------------------------------------------------

__global__ void bn254_compressing_row_hashes_ext_kernel(
    bn254_digest_t* out,
    const FpExt*    matrix,
    size_t          width,
    size_t          height,
    size_t          query_stride,
    size_t          log_rows_per_query
) {
    extern __shared__ char smem[];
    Bn254Fr* shared = reinterpret_cast<Bn254Fr*>(smem);

    const uint32_t stride_idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t       leaf_idx   = threadIdx.y;
    const uint32_t row        = leaf_idx * query_stride + stride_idx;

    Bn254Fr digest = bn254_zero_init();

    if (stride_idx < query_stride) {
        digest = bn254_row_hash_ext(matrix, (int)width, (int)height, (int)row);
    }

    for (int layer = 0; layer < (int)log_rows_per_query; ++layer) {
        uint32_t mask          = (1 << (layer + 1)) - 1;
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

static_assert(BN254_BABY_BEAR_RATE % 4 == 0,
              "BN254_BABY_BEAR_RATE must be a multiple of FpExt degree (4)");

// ---------------------------------------------------------------------------
// Adjacent compress layer kernel
// ---------------------------------------------------------------------------

__global__ void bn254_adjacent_compress_layer_kernel(
    bn254_digest_t*       output,
    const bn254_digest_t* prev_layer,
    size_t                output_size
) {
    uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= output_size) return;

    Bn254Fr left  = prev_layer[2 * gid    ].elem;
    Bn254Fr right = prev_layer[2 * gid + 1].elem;
    output[gid].elem = bn254_compress(left, right);
}

// ---------------------------------------------------------------------------
// BN254 sponge state for GPU grinding
//
// Matches MultiFieldTranscript<BabyBear, Bn254Scalar, Perm, WIDTH=3, RATE=2>:
//   num_obs_per_word = SF::bits() / CF::bits() = 254/31 = 8
//   num_samples_per_word = 7  (base-p decomposition)
//
// The sponge uses overwrite-mode duplex with absorb_idx/sample_idx tracking.
// Rust DeviceBn254SpongeState must have identical layout (verified by size assert).
// ---------------------------------------------------------------------------

static const uint32_t BN254_NUM_OBS_PER_WORD = 8;
static const uint32_t BN254_SPONGE_RATE      = 2;

struct DeviceBn254SpongeState {
    Bn254Fr  sponge_state[3];    // 96 bytes
    uint32_t absorb_idx;         //  4 bytes
    uint32_t sample_idx;         //  4 bytes
    uint32_t observe_buf[8];     // 32 bytes
    uint32_t observe_buf_len;    //  4 bytes
    // total = 140 + 4 padding = 144 bytes (aligned to 8)
    // Note: sample_buf is not needed on device — observe() clears it before grinding.
};

static_assert(sizeof(DeviceBn254SpongeState) == 144,
              "DeviceBn254SpongeState size mismatch with Rust");

// --- Low-level sponge (absorb/squeeze matching DuplexSponge) ---

/// Absorb one BN254 word into the sponge (overwrite mode).
__device__ void bn254_sponge_absorb(DeviceBn254SpongeState& s, Bn254Fr value) {
    s.sponge_state[s.absorb_idx] = value;
    s.absorb_idx++;
    if (s.absorb_idx == BN254_SPONGE_RATE) {
        bn254_poseidon2_permute(s.sponge_state);
        s.absorb_idx = 0;
        s.sample_idx = BN254_SPONGE_RATE;
    }
}

/// Squeeze one BN254 word from the sponge.
__device__ Bn254Fr bn254_sponge_squeeze(DeviceBn254SpongeState& s) {
    if (s.absorb_idx != 0 || s.sample_idx == 0) {
        bn254_poseidon2_permute(s.sponge_state);
        s.absorb_idx = 0;
        s.sample_idx = BN254_SPONGE_RATE;
    }
    s.sample_idx--;
    return s.sponge_state[s.sample_idx];
}

// --- MultiFieldTranscript operations ---

/// Flush observe_buf: pack and absorb into sponge.
__device__ void bn254_transcript_flush_observe(DeviceBn254SpongeState& s) {
    if (s.observe_buf_len > 0) {
        Bn254Fr packed = bn254_pack_base_2_31(s.observe_buf, s.observe_buf_len);
        bn254_sponge_absorb(s, packed);
        s.observe_buf_len = 0;
    }
}

/// Observe a canonical BabyBear u32 value.
/// Matches MultiFieldTranscript::observe().
/// Note: sample_buf clearing is a no-op on device (not present in device state).
__device__ void bn254_transcript_observe(DeviceBn254SpongeState& s, uint32_t value) {
    s.observe_buf[s.observe_buf_len++] = value;
    if (s.observe_buf_len == BN254_NUM_OBS_PER_WORD) {
        Bn254Fr packed = bn254_pack_base_2_31(s.observe_buf, BN254_NUM_OBS_PER_WORD);
        bn254_sponge_absorb(s, packed);
        s.observe_buf_len = 0;
    }
}

/// Sample a single canonical BabyBear u32 value (first base-p digit).
///
/// During grinding, sample_buf is always empty (observe clears it), so we
/// always squeeze fresh. The first base-p digit is simply `canonical_value % p`.
__device__ uint32_t bn254_transcript_sample(DeviceBn254SpongeState& s) {
    bn254_transcript_flush_observe(s);
    Bn254Fr squeezed = bn254_sponge_squeeze(s);
    uint64_t canonical[4];
    bn254_to_canonical(canonical, squeezed);
    return u256_mod_u32(canonical, (uint32_t)BABYBEAR_PRIME);
}

/// Returns true if check_witness(bits, witness) passes.
__device__ bool bn254_sponge_check_witness(DeviceBn254SpongeState& s,
                                           uint32_t bits, uint32_t witness) {
    bn254_transcript_observe(s, witness);
    uint32_t sample = bn254_transcript_sample(s);
    return (sample & ((1u << bits) - 1)) == 0;
}

static const uint32_t BN254_GRIND_BLOCK_SIZE = 32;

/// Grinding kernel: find any w in [min_witness, max_witness] with check_witness(bits,w)==true.
__launch_bounds__(BN254_GRIND_BLOCK_SIZE) __global__ void bn254_grind_kernel(
    const DeviceBn254SpongeState* init_state,
    uint32_t bits,
    uint32_t min_witness,
    uint32_t max_witness,
    uint32_t* result
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    uint32_t w = min_witness + tid;

    if (w > max_witness || *result != UINT32_MAX) return;

    while (w <= max_witness) {
        if (*result != UINT32_MAX) return;

        DeviceBn254SpongeState local_state = *init_state;
        if (bn254_sponge_check_witness(local_state, bits, w)) {
            atomicCAS(result, UINT32_MAX, w);
            return;
        }

        if (max_witness - w < stride) return;
        w += stride;
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
    const uint64_t* initial_rc,
    const uint64_t* partial_rc,
    const uint64_t* terminal_rc
) {
    cudaError_t err;
    err = cudaMemcpyToSymbol(g_initial_rc,  initial_rc,  4 * 3 * 4 * sizeof(uint64_t));
    if (err != cudaSuccess) return (int)err;
    err = cudaMemcpyToSymbol(g_partial_rc,  partial_rc,  56  * 4 * sizeof(uint64_t));
    if (err != cudaSuccess) return (int)err;
    err = cudaMemcpyToSymbol(g_terminal_rc, terminal_rc, 4 * 3 * 4 * sizeof(uint64_t));
    if (err != cudaSuccess) return (int)err;
    return (int)cudaSuccess;
}

extern "C" int _bn254_poseidon2_compressing_row_hashes(
    bn254_digest_t* out,
    const Fp*       matrix,
    size_t          width,
    size_t          query_stride,
    size_t          log_rows_per_query
) {
    auto [grid, block] = kernel_launch_params(query_stride, 512 >> log_rows_per_query);
    block.y = 1 << log_rows_per_query;
    size_t shared_stride = block.x * div_ceil(block.y, 2);
    size_t shmem_bytes   = shared_stride * sizeof(Bn254Fr);
    auto   height        = query_stride << log_rows_per_query;

    bn254_compressing_row_hashes_kernel<<<grid, block, shmem_bytes>>>(
        out, matrix, width, height, query_stride, log_rows_per_query
    );
    return CHECK_KERNEL();
}

extern "C" int _bn254_poseidon2_compressing_row_hashes_ext(
    bn254_digest_t* out,
    const FpExt*    matrix,
    size_t          width,
    size_t          query_stride,
    size_t          log_rows_per_query
) {
    auto [grid, block] = kernel_launch_params(query_stride, 512 >> log_rows_per_query);
    block.y = 1 << log_rows_per_query;
    size_t shared_stride = block.x * div_ceil(block.y, 2);
    size_t shmem_bytes   = shared_stride * sizeof(Bn254Fr);
    auto   height        = query_stride << log_rows_per_query;

    bn254_compressing_row_hashes_ext_kernel<<<grid, block, shmem_bytes>>>(
        out, matrix, width, height, query_stride, log_rows_per_query
    );
    return CHECK_KERNEL();
}

extern "C" int _bn254_poseidon2_adjacent_compress_layer(
    bn254_digest_t*       output,
    const bn254_digest_t* prev_layer,
    size_t                output_size
) {
    auto [grid, block] = kernel_launch_params(output_size);
    bn254_adjacent_compress_layer_kernel<<<grid, block>>>(output, prev_layer, output_size);
    return CHECK_KERNEL();
}

extern "C" int _bn254_sponge_grind(
    const DeviceBn254SpongeState* init_state,
    uint32_t bits,
    uint32_t min_witness,
    uint32_t max_witness,
    uint32_t* result
) {
    const size_t block_size = BN254_GRIND_BLOCK_SIZE;
    size_t total_threads = size_t{1} << bits;
    size_t grid_size = div_ceil(total_threads, block_size);

    bn254_grind_kernel<<<grid_size, block_size>>>(init_state, bits, min_witness, max_witness, result);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return (int)err;

    return CHECK_KERNEL();
}
