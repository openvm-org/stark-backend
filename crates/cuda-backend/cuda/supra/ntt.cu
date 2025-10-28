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

#include <cstdint>

#include "launcher.cuh"
#include "ntt/ntt.cuh"

namespace {
uint32_t max_grid_dim_y() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        return 65535u;

    int attr = 0;
    if (cudaDeviceGetAttribute(&attr, cudaDevAttrMaxGridDimY, device) != cudaSuccess)
        return 65535u;

    return attr > 0 ? static_cast<uint32_t>(attr) : 65535u;
}
} // namespace

template<int z_count, bool coalesced = false, class fr_t>
__launch_bounds__(768, 1) __global__
void _CT_NTT(const unsigned int radix, const unsigned int lg_domain_size,
             const unsigned int stage, const unsigned int iterations,
             fr_t* d_inout, const unsigned int padded_poly_size,
             const uint32_t poly_count, bool is_intt,
             const fr_t d_domain_size_inverse)
{
#if (__CUDACC_VER_MAJOR__-0) >= 11
    __builtin_assume(lg_domain_size <= MAX_LG_DOMAIN_SIZE);
    __builtin_assume(radix <= 10);
    __builtin_assume(iterations <= radix);
    __builtin_assume(stage <= lg_domain_size - iterations);
#endif
    extern __shared__ fr_t shared_exchange[];

    const fr_t (*d_partial_twiddles)[WINDOW_SIZE] = is_intt ? INVERSE_PARTIAL_TWIDDLES : FORWARD_PARTIAL_TWIDDLES;
    const fr_t* d_radix6_twiddles = is_intt ? INVERSE_TWIDDLES : FORWARD_TWIDDLES;
    auto twiddles_offset = (1 << (radix - 1)) - 32;
    const fr_t* d_radixX_twiddles = d_radix6_twiddles + twiddles_offset;

    index_t tid = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
    const uint32_t poly_idx = blockIdx.y + blockIdx.z * gridDim.y; // [DIFF]: use gridDim.y to calculate poly_idx
    if (poly_idx >= poly_count)
        return;
    d_inout += static_cast<size_t>(poly_idx) * padded_poly_size;   // [DIFF]: move in/out ptr to another row

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
    bool is_intt
) {
    index_t num_threads = (index_t)1 << (lg_domain_size - 1);
    index_t block_size = 1 << (radix - 1);
    index_t num_blocks;

    if (poly_count == 0)
        return cudaSuccess;

    block_size = (num_threads <= block_size) ? num_threads : block_size;
    num_blocks = (num_threads + block_size - 1) / block_size;

    assert(num_blocks == (unsigned int)num_blocks);

    const int Z_COUNT = 256/8/sizeof(fr_t);
    size_t shared_sz = sizeof(fr_t) << (radix - 1);

    // [DIFF]: calculate grid_z from poly_count
    const uint32_t max_y = max_grid_dim_y();
    const uint64_t total_polys = poly_count;
    const uint64_t max_y_64 = max_y == 0 ? 1 : static_cast<uint64_t>(max_y);
    uint32_t grid_z = static_cast<uint32_t>((total_polys + max_y_64 - 1) / max_y_64);
    if (grid_z == 0)
        grid_z = 1;
    uint64_t grid_y_64 = (total_polys + grid_z - 1) / grid_z;
    uint32_t grid_y = static_cast<uint32_t>(grid_y_64);
    if (grid_y > max_y)
        grid_y = max_y == 0 ? 1 : max_y;

    #define NTT_ARGUMENTS radix, lg_domain_size, stage, iterations, \
            d_inout, padded_poly_size, poly_count, is_intt, domain_size_inverse[lg_domain_size]

    // [DIFF]: N -> dim3(N, poly_count) in grid_size; stream -> cudaStreamPerThread
    if (num_blocks < Z_COUNT)
        _CT_NTT<1><<<dim3(static_cast<unsigned int>(num_blocks), grid_y, grid_z), block_size, shared_sz>>>(NTT_ARGUMENTS);
    else if (stage == 0 || lg_domain_size < 12)
        _CT_NTT<Z_COUNT><<<dim3(static_cast<unsigned int>(num_blocks/Z_COUNT), grid_y, grid_z), block_size, Z_COUNT*shared_sz>>>(NTT_ARGUMENTS);
    else if (lg_domain_size < MAX_LG_DOMAIN_SIZE)
        _CT_NTT<Z_COUNT, true><<<dim3(static_cast<unsigned int>(num_blocks/Z_COUNT), grid_y, grid_z), block_size, Z_COUNT*shared_sz>>>(NTT_ARGUMENTS);
    else
        assert(lg_domain_size < MAX_LG_DOMAIN_SIZE);
            
    #undef NTT_ARGUMENTS

    return CHECK_KERNEL();
}
