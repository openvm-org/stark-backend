/*
 * Source: https://github.com/supranational/sppark (tag=v0.1.12)
 * Status: MODIFIED from sppark/ntt/ntt.cuh
 * Imported: 2025-08-13 by @gaxiom
 * 
 * LOCAL CHANGES (high level):
 * - 2025-08-13: support only NTT with CT algorithm
 * - 2025-08-13: support multiple rows on bit_rev
 */

// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_NTT_NTT_CUH__
#define __SPPARK_NTT_NTT_CUH__

#include <cassert>
#include <stdio.h>

#include "launcher.cuh"
#include "parameters.cuh"
#include "kernels/common.cu"

class NTT {
public:
    // [DIFF]: delete InputOutputOrder, Type & Algorithm enums
    enum class Direction { forward, inverse };

protected:
    static void bit_rev(fr_t* d_out, const fr_t* d_inp, 
        uint32_t lg_domain_size, uint32_t padded_poly_size, uint32_t poly_count)
    {
        assert(lg_domain_size <= MAX_LG_DOMAIN_SIZE);

        size_t domain_size = (size_t)1 << lg_domain_size;
        // aim to read 4 cache lines of consecutive data per read
        const uint32_t Z_COUNT = 256 / sizeof(fr_t);
        const uint32_t bsize = Z_COUNT > WARP_SIZE ? Z_COUNT : WARP_SIZE;

        // [DIFF]: N -> dim3(N, poly_count) in grid_size; stream -> cudaStreamPerThread
        if (domain_size <= 1024)
            bit_rev_permutation<<<dim3(1, poly_count), domain_size>>>
                                (d_out, d_inp, lg_domain_size, padded_poly_size);
        else if (domain_size < bsize * Z_COUNT)
            bit_rev_permutation<<<dim3(domain_size / WARP_SIZE, poly_count), WARP_SIZE>>>
                                (d_out, d_inp, lg_domain_size, padded_poly_size);
        else if (Z_COUNT > WARP_SIZE || lg_domain_size <= 32)
            bit_rev_permutation_z<Z_COUNT><<<dim3(domain_size / Z_COUNT / bsize, poly_count), bsize,
                                             bsize * Z_COUNT * sizeof(fr_t)>>>
                                (d_out, d_inp, lg_domain_size, padded_poly_size);
        else {
            // Those GPUs that can reserve 96KB of shared memory can
            // schedule 2 blocks to each SM...
            int device;
            CUDA_OK(cudaGetDevice(&device));
            int sm_count;
            CUDA_OK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

            bit_rev_permutation_z<Z_COUNT><<<dim3(sm_count * 2, poly_count), 192,
                                             192 * Z_COUNT * sizeof(fr_t)>>>
                                 (d_out, d_inp, lg_domain_size, padded_poly_size);
        }

        CUDA_OK(cudaGetLastError());
    }

private:
    static void CT_NTT(fr_t* d_inout, const uint32_t lg_domain_size, 
                       const uint32_t padded_poly_size, const uint32_t poly_count,
                       bool intt, const NTTParameters& ntt_parameters)
    {
        CT_launcher params{d_inout, lg_domain_size, padded_poly_size, poly_count, intt, ntt_parameters};

        if (lg_domain_size <= 10) {
            params.step(lg_domain_size);
        } else if (lg_domain_size <= 17) {
            int step = lg_domain_size / 2;
            params.step(step + lg_domain_size % 2);
            params.step(step);
        } else if (lg_domain_size <= 30) {
            int step = lg_domain_size / 3;
            int rem = lg_domain_size % 3;
            params.step(step);
            params.step(step + (lg_domain_size == 29 ? 1 : 0));
            params.step(step + (lg_domain_size == 29 ? 1 : rem));
        } else if (lg_domain_size <= 40) {
            int step = lg_domain_size / 4;
            int rem = lg_domain_size % 4;
            params.step(step);
            params.step(step + (rem > 2));
            params.step(step + (rem > 1));
            params.step(step + (rem > 0));
        } else {
            assert(false);
        }
    }

public:
    // [DIFF]: protected NTT_internal -> public NTT_RUN
    static void NTT_RUN(fr_t* d_inout, uint32_t lg_domain_size, 
                    uint32_t padded_poly_size, uint32_t poly_count,
                    bool bit_reverse, Direction direction)
    {
        const bool intt = direction == Direction::inverse;
        const auto& ntt_parameters = NTTParametersHolder::all(intt);

        if (bit_reverse) {
            bit_rev(d_inout, d_inout, lg_domain_size, padded_poly_size, poly_count);
        }

        CT_NTT(d_inout, lg_domain_size, padded_poly_size, poly_count, intt, ntt_parameters);
    }
};
#endif
