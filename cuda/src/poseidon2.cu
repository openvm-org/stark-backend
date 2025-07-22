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
#include "poseidon2.cuh"
#include "launcher.cuh"

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

extern "C" int _poseidon2_rows_p3_multi(
    Fp *out,
    const uint64_t *matrices_ptr,
    const uint64_t *matrices_col,
    const uint64_t *matrices_row,
    const uint64_t row_size,
    uint64_t matrix_num
) {
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
    auto [grid, block] = kernel_launch_params(output_size);
    poseidon2_compress_kernel<<<grid, block>>>(output, input, output_size, is_inject);
    return cudaGetLastError();
}

extern "C" int _babybear_encode_mont_form(Fp *inout, uint32_t size) {
    auto [grid, block] = kernel_launch_params(size);
    babybear_encode_mont_form_kernel<<<grid, block>>>(inout, size);
    return cudaGetLastError();
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