/*
 * Source: https://github.com/supranational/sppark (tag=v0.1.12)
 * Status: MODIFIED from sppark/ntt/kernels/ct_mixed_radix_narrow.cu
 * Imported: 2025-08-13 by @gaxiom
 * 
 * LOCAL CHANGES (high level):
 * - 2025-08-13: support multiple rows in _CT_NTT
 * - 2025-08-13: avoid using sppark's stream_t
 * - 2025-08-26: no need to sync default stream
 * - 2025-09-10: delete CT_launcher class - extern "C" launcher instead
 */

// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include "ntt/ntt.cuh"

template<int z_count, bool coalesced = false, class fr_t>
__launch_bounds__(768, 1) __global__
void _CT_NTT(const unsigned int radix, const unsigned int lg_domain_size,
             const unsigned int stage, const unsigned int iterations,
             fr_t* d_inout, const unsigned int padded_poly_size,
             const fr_t (*d_partial_twiddles)[WINDOW_SIZE],
             const fr_t* d_radix6_twiddles, const fr_t* d_radixX_twiddles,
             bool is_intt, const fr_t d_domain_size_inverse)
{
#if (__CUDACC_VER_MAJOR__-0) >= 11
    __builtin_assume(lg_domain_size <= MAX_LG_DOMAIN_SIZE);
    __builtin_assume(radix <= 10);
    __builtin_assume(iterations <= radix);
    __builtin_assume(stage <= lg_domain_size - iterations);
#endif
    extern __shared__ fr_t shared_exchange[];

    index_t tid = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
    d_inout += blockIdx.y * padded_poly_size;   // [DIFF]: move in/out ptr to another row

    const index_t diff_mask = (1 << (iterations - 1)) - 1;
    const index_t inp_mask = ((index_t)1 << stage) - 1;
    const index_t out_mask = ((index_t)1 << (stage + iterations - 1)) - 1;

    const index_t tiz = (tid & ~diff_mask) * z_count + (tid & diff_mask);
    const index_t thread_ntt_pos = (tiz >> (iterations - 1)) & inp_mask;

    // rearrange |tiz|'s bits
    index_t idx0 = (tiz & ~out_mask) | ((tiz << stage) & out_mask);
    idx0 = idx0 * 2 + thread_ntt_pos;
    index_t idx1 = idx0 + ((index_t)1 << stage);

    fr_t r[2][z_count];

    if (coalesced) {
        coalesced_load<z_count>(r[0], d_inout, idx0, stage + 1);
        coalesced_load<z_count>(r[1], d_inout, idx1, stage + 1);
        transpose<z_count>(r[0]);
        __syncwarp();
        transpose<z_count>(r[1]);
    } else {
        unsigned int z_shift = inp_mask==0 ? iterations : 0;
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            r[0][z] = d_inout[idx0 + (z << z_shift)];
            r[1][z] = d_inout[idx1 + (z << z_shift)];
        }
    }

    if (stage != 0) {
        unsigned int thread_ntt_idx = (tiz & diff_mask) * 2;
        unsigned int nbits = MAX_LG_DOMAIN_SIZE - stage;
        index_t idx0 = bit_rev(thread_ntt_idx, nbits);
        index_t root_idx0 = idx0 * thread_ntt_pos;
        index_t root_idx1 = root_idx0 + (thread_ntt_pos << (nbits - 1));

        fr_t first_root, second_root;
        get_intermediate_roots(first_root, second_root,
                               root_idx0, root_idx1, d_partial_twiddles);
        r[0][0] = r[0][0] * first_root;
        r[1][0] = r[1][0] * second_root;

        if (z_count > 1) {
            fr_t first_root_z = get_intermediate_root(idx0, d_partial_twiddles);
            unsigned int off = (nbits - 1) / LG_WINDOW_SIZE;
            unsigned int win = off * LG_WINDOW_SIZE;
            fr_t second_root_z = d_partial_twiddles[off][1 << (nbits - 1 - win)];

            second_root_z *= first_root_z;
            #pragma unroll
            for (int z = 1; z < z_count; z++) {
                first_root *= first_root_z;
                second_root *= second_root_z;
                r[0][z] = r[0][z] * first_root;
                r[1][z] = r[1][z] * second_root;
            }
        }
    }

    #pragma unroll
    for (int z = 0; z < z_count; z++) {
        fr_t t = r[1][z];
        r[1][z] = r[0][z] - t;
        r[0][z] = r[0][z] + t;
    }

    #pragma unroll 1
    for (unsigned int s = 1; s < min(iterations, 6u); s++) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        fr_t root = d_radix6_twiddles[rank << (6 - (s + 1))];

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            fr_t t = fr_t::csel(r[1][z], r[0][z], pos);

            t.shfl_bfly(laneMask);

            r[0][z] = fr_t::csel(r[0][z], t, pos);
            r[1][z] = fr_t::csel(t, r[1][z], pos);

            t = root * r[1][z];
            r[1][z] = r[0][z] - t;
            r[0][z] = r[0][z] + t;
        }
    }

    #pragma unroll 1
    for (unsigned int s = 6; s < iterations; s++) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        fr_t root = d_radixX_twiddles[rank << (radix - (s + 1))];

        fr_t (*xchg)[z_count] = reinterpret_cast<decltype(xchg)>(shared_exchange);

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            fr_t t = fr_t::csel(r[1][z], r[0][z], pos);
            xchg[threadIdx.x][z] = t;
        }

        __syncthreads();

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            fr_t t = xchg[threadIdx.x ^ laneMask][z];

            r[0][z] = fr_t::csel(r[0][z], t, pos);
            r[1][z] = fr_t::csel(t, r[1][z], pos);

            t = root * r[1][z];
            r[1][z] = r[0][z] - t;
            r[0][z] = t + r[0][z];
        }

        __syncthreads();
    }

    if (is_intt && (stage + iterations) == lg_domain_size) {
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            r[0][z] = r[0][z] * d_domain_size_inverse;
            r[1][z] = r[1][z] * d_domain_size_inverse;
        }
    }

    // rotate "iterations" bits in indices
    index_t mask = (index_t)((1 << iterations) - 1) << stage;
    index_t rotw = idx0 & mask;
    rotw = (rotw >> 1) | (rotw << (iterations - 1));
    idx0 = (idx0 & ~mask) | (rotw & mask);
    rotw = idx1 & mask;
    rotw = (rotw >> 1) | (rotw << (iterations - 1));
    idx1 = (idx1 & ~mask) | (rotw & mask);

    if (coalesced) {
        transpose<z_count>(r[0]);
        __syncwarp();
        transpose<z_count>(r[1]);
        coalesced_store<z_count>(d_inout, idx0, r[0], stage);
        coalesced_store<z_count>(d_inout, idx1, r[1], stage);
    } else {
        unsigned int z_shift = inp_mask==0 ? iterations : 0;
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            d_inout[idx0 + (z << z_shift)] = r[0][z];
            d_inout[idx1 + (z << z_shift)] = r[1][z];
        }
    }
}

extern "C" int _ct_mixed_radix_narrow(
    fr_t* d_inout,
    uint32_t radix,
    uint32_t lg_domain_size,
    uint32_t stage,
    uint32_t iterations,
    uint32_t padded_poly_size,
    uint32_t poly_count,
    const fr_t (*d_partial_twiddles)[WINDOW_SIZE],
    const fr_t* d_radix_twiddles,
    const uint32_t twiddles_offset,
    bool is_intt
) {
    index_t num_threads = (index_t)1 << (lg_domain_size - 1);
    index_t block_size = 1 << (radix - 1);
    index_t num_blocks;

    block_size = (num_threads <= block_size) ? num_threads : block_size;
    num_blocks = (num_threads + block_size - 1) / block_size;

    assert(num_blocks == (unsigned int)num_blocks);

    const int Z_COUNT = 256/8/sizeof(fr_t);
    size_t shared_sz = sizeof(fr_t) << (radix - 1);
    auto d_radixX_twiddles = d_radix_twiddles + twiddles_offset;

    #define NTT_ARGUMENTS radix, lg_domain_size, stage, iterations, \
            d_inout, padded_poly_size, d_partial_twiddles, \
            d_radix_twiddles, d_radixX_twiddles, \
            is_intt, domain_size_inverse[lg_domain_size]

    // [DIFF]: N -> dim3(N, poly_count) in grid_size; stream -> cudaStreamPerThread
    if (num_blocks < Z_COUNT)
        _CT_NTT<1><<<dim3(num_blocks, poly_count), block_size, shared_sz>>>(NTT_ARGUMENTS);
    else if (stage == 0 || lg_domain_size < 12)
        _CT_NTT<Z_COUNT><<<dim3(num_blocks/Z_COUNT, poly_count), block_size, Z_COUNT*shared_sz>>>(NTT_ARGUMENTS);
    else if (lg_domain_size < MAX_LG_DOMAIN_SIZE)
        _CT_NTT<Z_COUNT, true><<<dim3(num_blocks/Z_COUNT, poly_count), block_size, Z_COUNT*shared_sz>>>(NTT_ARGUMENTS);
    else
        assert(lg_domain_size < MAX_LG_DOMAIN_SIZE);
            
    #undef NTT_ARGUMENTS

    return cudaGetLastError();
}
