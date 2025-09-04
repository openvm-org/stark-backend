/*
 * Source: https://github.com/supranational/sppark (tag=v0.1.12)
 * Status: MODIFIED from sppark/ntt/parameters.cuh
 * Imported: 2025-08-13 by @gaxiom
 * 
 * LOCAL CHANGES (high level):
 * - 2025-08-13: BABY_BEAR_CANONICAL constants copy from sppark/ntt/parameters/baby_bear.h
 * - 2025-08-13: #if !defined FEATURE_BABY_BEAR -> delete
 * - 2025-08-13: NTTParameters constructor async on custom stream
 * - 2025-08-13: static params holder all_params() -> NTTParametersHolder
 * - 2025-08-26: NTTParameters constructor on cudaStreamPerThread
 * - 2025-09-05: Stop using __constant__ for twiddles[0]
 */

// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_NTT_PARAMETERS_CUH__
#define __SPPARK_NTT_PARAMETERS_CUH__

#define MAX_LG_DOMAIN_SIZE 27
#define LG_WINDOW_SIZE ((MAX_LG_DOMAIN_SIZE + 4) / 5)
#define WINDOW_SIZE (1 << LG_WINDOW_SIZE)
#define WINDOW_NUM ((MAX_LG_DOMAIN_SIZE + LG_WINDOW_SIZE - 1) / LG_WINDOW_SIZE)

const fr_t group_gen = fr_t(0x1f);  // primitive_root(0x78000001)
const fr_t group_gen_inverse = fr_t(0x03def7be);

// Values in Montgomery form
const fr_t forward_roots_of_unity[MAX_LG_DOMAIN_SIZE + 1] = {
    fr_t(0x0ffffffeu),
    fr_t(0x68000003u),
    fr_t(0x1c38d511u),
    fr_t(0x3d85298fu),
    fr_t(0x5f06e481u),
    fr_t(0x3f5c39ecu),
    fr_t(0x5516a97au),
    fr_t(0x3d6be592u),
    fr_t(0x5bb04149u),
    fr_t(0x4907f9abu),
    fr_t(0x548b8e90u),
    fr_t(0x1d8ca617u),
    fr_t(0x2ce7f0e6u),
    fr_t(0x621b371fu),
    fr_t(0x6d4d2d78u),
    fr_t(0x18716fcdu),
    fr_t(0x3b30a682u),
    fr_t(0x1c6f4728u),
    fr_t(0x59b01f7cu),
    fr_t(0x1a7f97acu),
    fr_t(0x0732561cu),
    fr_t(0x2b5a1cd4u),
    fr_t(0x6f7d26f9u),
    fr_t(0x16e2f919u),
    fr_t(0x285ab85bu),
    fr_t(0x0dd5a9ecu),
    fr_t(0x43f13568u),
    fr_t(0x57fab6eeu)
};

const fr_t inverse_roots_of_unity[MAX_LG_DOMAIN_SIZE + 1] = {
    fr_t(0x0ffffffeu),
    fr_t(0x68000003u),
    fr_t(0x5bc72af0u),
    fr_t(0x02ec07f3u),
    fr_t(0x67e027cau),
    fr_t(0x5e1a0700u),
    fr_t(0x4bcc008cu),
    fr_t(0x0bed94d1u),
    fr_t(0x330b2e00u),
    fr_t(0x6b469805u),
    fr_t(0x0d83fad2u),
    fr_t(0x26e64394u),
    fr_t(0x0855523bu),
    fr_t(0x5c9f0045u),
    fr_t(0x5a7ba8c3u),
    fr_t(0x3c8b04e2u),
    fr_t(0x0c0f2066u),
    fr_t(0x1b51d34cu),
    fr_t(0x59f9bc12u),
    fr_t(0x3511f012u),
    fr_t(0x061ec85fu),
    fr_t(0x5fd09c6bu),
    fr_t(0x26bdc06cu),
    fr_t(0x1272832eu),
    fr_t(0x052ce2e8u),
    fr_t(0x02ff110du),
    fr_t(0x216ce204u),
    fr_t(0x5e12c8e9u)
};

const fr_t domain_size_inverse[MAX_LG_DOMAIN_SIZE + 1] = {
    fr_t(0x0ffffffeu),
    fr_t(0x07ffffffu),
    fr_t(0x40000000u),
    fr_t(0x20000000u),
    fr_t(0x10000000u),
    fr_t(0x08000000u),
    fr_t(0x04000000u),
    fr_t(0x02000000u),
    fr_t(0x01000000u),
    fr_t(0x00800000u),
    fr_t(0x00400000u),
    fr_t(0x00200000u),
    fr_t(0x00100000u),
    fr_t(0x00080000u),
    fr_t(0x00040000u),
    fr_t(0x00020000u),
    fr_t(0x00010000u),
    fr_t(0x00008000u),
    fr_t(0x00004000u),
    fr_t(0x00002000u),
    fr_t(0x00001000u),
    fr_t(0x00000800u),
    fr_t(0x00000400u),
    fr_t(0x00000200u),
    fr_t(0x00000100u),
    fr_t(0x00000080u),
    fr_t(0x00000040u),
    fr_t(0x00000020u)
};

typedef unsigned int index_t;

template<class fr_t> __global__
void generate_partial_twiddles(fr_t (*roots)[WINDOW_SIZE],
                               const fr_t root_of_unity)
{
    const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    assert(tid < WINDOW_SIZE);
    fr_t root;

    root = root_of_unity^tid;

    roots[0][tid] = root;

    for (int off = 1; off < WINDOW_NUM; off++) {
        for (int i = 0; i < LG_WINDOW_SIZE; i++)
            root.sqr();
        roots[off][tid] = root;
    }
}

// [DIFF]: change order to natural one: 7,8,9,10,6 -> 6,7,8,9,10
template<class fr_t> __global__
void generate_all_twiddles(fr_t* d_radixX_twiddles, 
                           const fr_t root6,
                           const fr_t root7,
                           const fr_t root8,
                           const fr_t root9,
                           const fr_t root10)
{
    const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int pow = 0;
    fr_t root_of_unity;

    if (tid < 32) {
        pow = tid;
        root_of_unity = root6;
    } else if (tid < 32 + 64) {
        pow = tid - 32;
        root_of_unity = root7;
    } else if (tid < 32 + 64 + 128) {
        pow = tid - 32 - 64;
        root_of_unity = root8;
    } else if (tid < 32 + 64 + 128 + 256) {
        pow = tid - 32 - 64 - 128;
        root_of_unity = root9;
    } else if (tid < 32 + 64 + 128 + 256 + 512) {
        pow = tid - 32 - 64 - 128 - 256;
        root_of_unity = root10;
    } else {
        assert(false);
    }

    d_radixX_twiddles[tid] = root_of_unity^pow;
}

class NTTParameters {
private:
     // [DIFF]: const gpu_t& -> cudaStreamPerThread
    bool inverse;
public:
    fr_t (*partial_twiddles)[WINDOW_SIZE];
    fr_t* twiddles[5];

public:
    // [DIFF]: avoid !FEATURE_BABY_BEAR branches & use stream
    NTTParameters(const bool _inverse) : inverse(_inverse)
    {
        const fr_t* roots = inverse ? inverse_roots_of_unity
                                    : forward_roots_of_unity;

        const size_t blob_sz = 32 + 64 + 128 + 256 + 512;

        fr_t* blob;
        CUDA_OK(cudaMallocAsync(&blob, blob_sz * sizeof(fr_t), cudaStreamPerThread));

        twiddles[0] = blob;                 /* radix6_twiddles */
        twiddles[1] = twiddles[0] + 32;     /* radix7_twiddles */
        twiddles[2] = twiddles[1] + 64;     /* radix8_twiddles */
        twiddles[3] = twiddles[2] + 128;    /* radix9_twiddles */
        twiddles[4] = twiddles[3] + 256;    /* radix10_twiddles */

        generate_all_twiddles<<<blob_sz/32, 32>>>(
            blob, roots[6], roots[7], roots[8], roots[9], roots[10]);
        CUDA_OK(cudaGetLastError());

        const size_t partial_sz = WINDOW_NUM * WINDOW_SIZE;
        // [DIFF]: fix allocation size for partial_twiddles
        CUDA_OK(cudaMallocAsync(&partial_twiddles, partial_sz * sizeof(fr_t), cudaStreamPerThread));

        generate_partial_twiddles<<<WINDOW_SIZE/32, 32>>>(
            partial_twiddles, roots[MAX_LG_DOMAIN_SIZE]);
        CUDA_OK(cudaGetLastError());
    }
    NTTParameters(const NTTParameters&) = delete;
    NTTParameters(NTTParameters&&) = default;

    ~NTTParameters()
    {
        cudaFreeAsync(partial_twiddles, cudaStreamPerThread);
        cudaFreeAsync(twiddles[0], cudaStreamPerThread);
    }
};

// [DIFF]: introducing new class to hold a pair of static params
class NTTParametersHolder {
public:
    NTTParameters forward;
    NTTParameters inverse;

    NTTParametersHolder()
        : forward(false), inverse(true) {}

    static const NTTParameters& all(bool inverse = false)
    {
        static NTTParametersHolder params;
        return inverse ? params.inverse : params.forward;
    }
};
#endif /* __SPPARK_NTT_PARAMETERS_CUH__ */
