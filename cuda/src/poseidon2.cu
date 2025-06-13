// FROM https://github.com/scroll-tech/plonky3-gpu/blob/openvm-v2/gpu-backend/src/cuda/kernels/poseidon2.cu

// Copyright *
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// #include <stdio.h>
#include "fp.h"
#include "launcher.cuh"

#define CELLS 16
#define CELLS_RATE 8
#define CELLS_OUT 8

#define ROUNDS_FULL 8
#define ROUNDS_HALF_FULL (ROUNDS_FULL / 2)
#define ROUNDS_PARTIAL 13

#define PRINT_STATE(msg, cells, size)                                                              \
    if (threadIdx.x == 0) {                                                                        \
        printf("thread d: %s=[%d", threadIdx.x, msg, cells[0].asUInt32());                         \
        for (uint i = 1; i < size; i++) {                                                          \
            printf(", %d", cells[i].asUInt32());                                                   \
        }                                                                                          \
        printf("]\r\n");                                                                           \
    }

__constant__ Fp INITIAL_ROUND_CONSTANTS[64];
__constant__ Fp TERMINAL_ROUND_CONSTANTS[64];
__constant__ Fp INTERNAL_ROUND_CONSTANTS[13];

namespace poseidon2 {

__device__ Fp sbox_d7(Fp x) {
    Fp x2 = x * x;
    Fp x4 = x2 * x2;
    Fp x6 = x4 * x2;
    return x6 * x;
}

__device__ void do_full_sboxes(Fp *cells) {
    for (uint i = 0; i < CELLS; i++) {
        cells[i] = sbox_d7(cells[i]);
    }
}

__device__ void do_partial_sboxes(Fp *cells) { cells[0] = sbox_d7(cells[0]); }

// Plonky3 version
// Multiply a 4-element vector x by:
// [ 2 3 1 1 ]
// [ 1 2 3 1 ]
// [ 1 1 2 3 ]
// [ 3 1 1 2 ].
__device__ void multiply_by_4x4_circulant(Fp *x) {
    Fp t01 = x[0] + x[1];
    Fp t23 = x[2] + x[3];
    Fp t0123 = t01 + t23;
    Fp t01123 = t0123 + x[1];
    Fp t01233 = t0123 + x[3];

    // The order here is important.
    // Need to overwrite x[0] and x[2] after x[1] and x[3].
    x[3] = t01233 + Fp(2) * x[0];
    x[1] = t01123 + Fp(2) * x[2];
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

__device__ void multiply_by_m_ext(Fp *old_cells) {
    // Optimized method for multiplication by M_EXT.
    // See appendix B of Poseidon2 paper for additional details.
    Fp cells[CELLS];
    for (uint i = 0; i < CELLS; i++) {
        cells[0] = 0;
    }
    Fp tmp_sums[4];
    for (uint i = 0; i < 4; i++) {
        tmp_sums[i] = 0;
    }
    for (uint i = 0; i < CELLS / 4; i++) {
        multiply_by_4x4_circulant(old_cells + i * 4);
        for (uint j = 0; j < 4; j++) {
            Fp to_add = old_cells[i * 4 + j];
            tmp_sums[j] += to_add;
            cells[i * 4 + j] += to_add;
        }
    }
    for (uint i = 0; i < CELLS; i++) {
        old_cells[i] = cells[i] + tmp_sums[i % 4];
    }
}

__device__ void add_round_constants_full(const Fp *ROUND_CONSTANTS_PLONKY3, Fp *cells, uint round) {
    for (uint i = 0; i < CELLS; i++) {
        cells[i] += ROUND_CONSTANTS_PLONKY3[round * CELLS + i];
    }
}

__device__ void add_round_constants_partial(
    const Fp *PARTIAL_ROUND_CONSTANTS_PLONKY3,
    Fp *cells,
    uint round
) {
    cells[0] += PARTIAL_ROUND_CONSTANTS_PLONKY3[round];
}

// baby-bear/src/poseidon2.rs
// InternalLayerBaseParameters<BabyBearParameters, 16> for BabyBearInternalLayerParameters
// `fn internal_layer_mat_mul`
// Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
// We ignore `state[0]` as it is handled separately.
__device__ void internal_layer_mat_mul(Fp *cells, Fp sum) {
    // The diagonal matrix is defined by the vector:
    // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8, -1/16, -1/2^27]
    cells[1] += sum;
    cells[2] = cells[2].doubled() + sum;
    cells[3] = cells[3].halve() + sum;
    cells[4] = sum + cells[4].doubled() + cells[4];
    cells[5] = sum + cells[5].doubled().doubled();
    cells[6] = sum - cells[6].halve();
    cells[7] = sum - (cells[7].doubled() + cells[7]);
    cells[8] = sum - cells[8].doubled().doubled();
    cells[9] = cells[9].mul_2exp_neg_n(8);
    cells[9] += sum;
    cells[10] = cells[10].mul_2exp_neg_n(2);
    cells[10] += sum;
    cells[11] = cells[11].mul_2exp_neg_n(3);
    cells[11] += sum;
    cells[12] = cells[12].mul_2exp_neg_n(27);
    cells[12] += sum;
    cells[13] = cells[13].mul_2exp_neg_n(8);
    cells[13] = sum - cells[13];
    cells[14] = cells[14].mul_2exp_neg_n(4);
    cells[14] = sum - cells[14];
    cells[15] = cells[15].mul_2exp_neg_n(27);
    cells[15] = sum - cells[15];
}

__device__ void full_round_half(const Fp *ROUND_CONSTANTS, Fp *cells, uint round) {
    add_round_constants_full(ROUND_CONSTANTS, cells, round);
    do_full_sboxes(cells);
    multiply_by_m_ext(cells);
}

__device__ void partial_round(const Fp *PARTIAL_ROUND_CONSTANTS, Fp *cells, uint round) {
    add_round_constants_partial(PARTIAL_ROUND_CONSTANTS, cells, round);
    do_partial_sboxes(cells);
    Fp part_sum = Fp(0);
    for (uint i = 1; i < CELLS; i++) {
        part_sum += cells[i];
    }
    Fp full_sum = part_sum + cells[0];
    cells[0] = part_sum - cells[0];
    internal_layer_mat_mul(cells, full_sum);
}

__device__ void poseidon2_mix(Fp *cells) {
    // PRINT_STATE("gpu state(input)", cells, CELLS);

    // First linear layer.
    multiply_by_m_ext(cells);
    // PRINT_STATE("gpu state(init m_ext)", cells, CELLS);

    // perform initial full rounds (external)
    for (uint i = 0; i < ROUNDS_HALF_FULL; i++) {
        full_round_half(INITIAL_ROUND_CONSTANTS, cells, i);
    }
    // PRINT_STATE("gpu state(initial full rounds)", cells, CELLS);

    // perform partial rounds (internal)
    for (uint i = 0; i < ROUNDS_PARTIAL; i++) {
        partial_round(INTERNAL_ROUND_CONSTANTS, cells, i);
    }
    // PRINT_STATE("gpu state(partial rounds)", cells, CELLS);

    // perform terminal full rounds (external)
    for (uint r = 0; r < ROUNDS_HALF_FULL; r++) {
        full_round_half(TERMINAL_ROUND_CONSTANTS, cells, r);
    }
    // PRINT_STATE("gpu state(terminal full rounds)", cells, CELLS);
}

} // namespace poseidon2

// all matrices are on natural order, so we need to bit_rev row_idx when write
__global__ void poseidon2_rows_p3_multi_kernel(
    Fp *out,
    const uint64_t
        *matrices_ptr, // matrices[0] is the first matrix, matrices[1] is the second matrix, etc.
    const uint64_t *matrices_col,
    const uint64_t *matrices_row,
    uint64_t row_size,
    uint64_t matrix_num
) {
    uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= row_size) {
        return;
    }

    uint used = 0;
    Fp cells[CELLS];
    for (int i = 0; i < CELLS; i++) {
        cells[i] = Fp(0);
    }

    for (uint m = 0; m < matrix_num; m++) {
        uint64_t col_size = matrices_col[m];
        Fp *matrix = (Fp *)(matrices_ptr[m]);
        for (uint i = 0; i < col_size; i++) {
            cells[used++] = matrix[i * row_size + gid];
            if (used == CELLS_RATE) {
                poseidon2::poseidon2_mix(cells);
                used = 0;
            }
        }
    }

    if (used != 0 || row_size == 0) {
        poseidon2::poseidon2_mix(cells);
    }

    gid = __brev(gid) >> (__clz(row_size) + 1);
    for (uint i = 0; i < CELLS_OUT; i++) {
        out[CELLS_OUT * gid + i] = cells[i];
    }
}

__global__ void poseidon2_compress_kernel(
    Fp *output,
    const Fp *input,
    uint32_t output_size,
    bool is_inject
) {
    uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= output_size) {
        return;
    }

    Fp cells[CELLS];
    for (size_t i = 0; i < CELLS_OUT; i++) {
        cells[i] = input[(2 * gid + 0) * CELLS_OUT + i];
        cells[i + CELLS_OUT] = input[(2 * gid + 1) * CELLS_OUT + i];
    }

    poseidon2::poseidon2_mix(cells);
    if (is_inject) {
        // hash_pair(&res, &cur)
        for (uint i = 0; i < CELLS_OUT; i++) {
            cells[i + CELLS_OUT] = output[gid * CELLS_OUT + i];
        }
        poseidon2::poseidon2_mix(cells);
    }

    for (uint i = 0; i < CELLS_OUT; i++) {
        output[gid * CELLS_OUT + i] = cells[i];
    }
}

__global__ void babybear_encode_mont_form_kernel(Fp *inout, uint32_t size) {
    uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= size) {
        return;
    }

    for (uint i = 0; i < CELLS_OUT; i++) {
        inout[gid * CELLS_OUT + i] = Fp(inout[gid * CELLS_OUT + i].get()); // encode
    }
}

/*
query[0][0,...layers-1]
query[1][0,...layers-1]
...
query[k][0,...layers-1]
*/
__global__ void cukernel_query_digest_layers(
    Fp *d_digest_matrix,          // Fp*, also Digest: CELLS_OUT=8 Fp elements
    const uint64_t *d_layers_ptr, // array of Digest layers
    uint64_t *d_indices,          // uint64_t*, indices to query, size = num_query * num_layer
    uint64_t num_query,           // e.g. 100
    uint64_t num_layer
) // e.g. 23
{
    const uint32_t ELEM_PER_DIGEST = CELLS_OUT; // 8 * Fp
    uint64_t gidx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t layer_idx = gidx / ELEM_PER_DIGEST;
    uint64_t elem_offset = gidx % ELEM_PER_DIGEST; // thread group: [0,..7]
    uint64_t query_idx = blockIdx.y;               // [0, num_query -1]
    if (layer_idx >= num_layer) {                  // [0, layers - 1]
        return;
    }

    Fp *d_layer = (Fp *)d_layers_ptr[layer_idx];
    uint64_t digest_offset = d_indices[query_idx * num_layer + layer_idx];
    Fp digest_elem = d_layer[digest_offset * ELEM_PER_DIGEST + elem_offset];
    // now each thread get 1/ELEM_PER_DIGEST of the digest

    uint64_t output_query_offset = query_idx * num_layer * ELEM_PER_DIGEST;
    uint64_t output_layer_offset = layer_idx * ELEM_PER_DIGEST + elem_offset;
    d_digest_matrix[output_query_offset + output_layer_offset] = digest_elem;
}

// END OF FILE gpu-backend/src/cuda/kernels/poseidon2.cu

static bool poseidon2_initialized = false;

extern "C" int _poseidon2_rows_p3_multi(
    Fp *out,
    const uint64_t *matrices_ptr,
    const uint64_t *matrices_col,
    const uint64_t *matrices_row,
    const uint64_t row_size,
    uint64_t matrix_num
) {
    if (!poseidon2_initialized) {
        return cudaErrorNotReady;
    }
    auto [grid, block] = kernel_launch_params(row_size);
    poseidon2_rows_p3_multi_kernel<<<grid, block>>>(
        out, matrices_ptr, matrices_col, matrices_row, row_size, matrix_num
    );
    return cudaGetLastError();
}

extern "C" int _poseidon2_compress(
    Fp *output,
    const Fp *input,
    uint32_t output_size,
    bool is_inject
) {
    if (!poseidon2_initialized) {
        return cudaErrorNotReady;
    }
    auto [grid, block] = kernel_launch_params(output_size);
    poseidon2_compress_kernel<<<grid, block>>>(output, input, output_size, is_inject);
    return cudaGetLastError();
}

extern "C" int _babybear_encode_mont_form(Fp *inout, uint32_t size) {
    auto [grid, block] = kernel_launch_params(size);
    babybear_encode_mont_form_kernel<<<grid, block>>>(inout, size);
    return cudaGetLastError();
}

extern "C" int _init_poseidon2_constants(
    const Fp *initial_round_constants,
    const Fp *terminal_round_constants,
    const Fp *internal_round_constants
) {
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

static const size_t QUERY_DIGEST_THREADS = 128;
static const size_t DIGEST_WIDTH = 8;

extern "C" int _query_digest_layers(
    Fp *d_digest_matrix,
    const uint64_t *d_layers_ptr,
    uint64_t *d_indices,
    uint64_t num_query,
    uint64_t num_layer
) {
    auto block = QUERY_DIGEST_THREADS;
    dim3 grid = dim3(div_ceil(num_layer * DIGEST_WIDTH, block), num_query);
    cukernel_query_digest_layers<<<grid, block>>>(
        d_digest_matrix, d_layers_ptr, d_indices, num_query, num_layer
    );
    return cudaGetLastError();
}