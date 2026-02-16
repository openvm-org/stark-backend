/// NTT kernels for Metal.
/// Translated from cuda-backend/cuda/supra/ntt.cu and ntt_bitrev.cu.
///
/// Major differences from CUDA:
/// - No constant memory: twiddle tables passed as buffer arguments
/// - SIMD shuffle (simd_shuffle_xor) replaces __shfl_xor_sync
/// - threadgroup_barrier replaces __syncthreads
/// - thread_position_in_grid replaces blockIdx*blockDim+threadIdx

#include <metal_stdlib>
using namespace metal;

#include "baby_bear.h"
#include "baby_bear_ext.h"
#include "utils.h"

// ============================================================================
// Twiddle Generation
// ============================================================================

/// Generate twiddle factors for all NTT levels 1..max_level.
/// Layout: twiddles for level L start at offset (2^L - 2) and have 2^L entries.
/// Each entry is TWO_ADIC_GENERATORS[level]^index.
kernel void generate_all_twiddles(
    device Fp *d_twiddles [[buffer(0)]],
    constant uint32_t &max_level [[buffer(1)]],
    constant uint32_t &total_size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_size) return;

    // Find which level this tid belongs to
    uint32_t level = 1;
    uint32_t offset = 0;
    while (level <= max_level) {
        uint32_t level_size = 1u << level;
        if (tid < offset + level_size) break;
        offset += level_size;
        level++;
    }

    uint32_t index = tid - offset;
    d_twiddles[tid] = pow(TWO_ADIC_GENERATORS[level], index);
}

/// Generate partial twiddle factors for a specific window.
/// Used by the sppark-style NTT for intermediate root computation.
kernel void generate_partial_twiddles(
    device Fp *d_twiddles [[buffer(0)]],
    constant uint32_t &level [[buffer(1)]],
    constant uint32_t &window_size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= window_size) return;
    d_twiddles[tid] = pow(TWO_ADIC_GENERATORS[level], tid);
}

// ============================================================================
// Bit Reversal Permutation
// ============================================================================

/// Out-of-place or in-place bit reversal permutation for Fp.
/// Each thread swaps data[i] with data[bit_reverse(i)] where i < rev(i).
kernel void bit_reverse(
    device Fp *d_out [[buffer(0)]],
    const device Fp *d_in [[buffer(1)]],
    constant uint32_t &lg_domain_size [[buffer(2)]],
    constant uint32_t &padded_poly_size [[buffer(3)]],
    constant uint32_t &poly_count [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]] // (idx_within_domain, poly_idx)
) {
    uint32_t poly_idx = gid.y;
    if (poly_idx >= poly_count) return;

    uint32_t domain_size = 1u << lg_domain_size;
    uint32_t idx = gid.x;
    if (idx >= domain_size) return;

    uint32_t rev_idx = rev_len(idx, lg_domain_size);
    size_t base = size_t(poly_idx) * padded_poly_size;

    if (d_out == d_in) {
        // In-place: only swap if idx < rev_idx to avoid double-swap
        if (idx < rev_idx) {
            Fp tmp = d_out[base + idx];
            d_out[base + idx] = d_out[base + rev_idx];
            d_out[base + rev_idx] = tmp;
        }
    } else {
        // Out-of-place: each thread writes its value to the reversed position
        d_out[base + rev_idx] = d_in[base + idx];
    }
}

/// Bit reversal permutation for FpExt (extension field elements).
kernel void bit_reverse_ext(
    device FpExt *d_out [[buffer(0)]],
    const device FpExt *d_in [[buffer(1)]],
    constant uint32_t &lg_domain_size [[buffer(2)]],
    constant uint32_t &padded_poly_size [[buffer(3)]],
    constant uint32_t &poly_count [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint32_t poly_idx = gid.y;
    if (poly_idx >= poly_count) return;

    uint32_t domain_size = 1u << lg_domain_size;
    uint32_t idx = gid.x;
    if (idx >= domain_size) return;

    uint32_t rev_idx = rev_len(idx, lg_domain_size);
    size_t base = size_t(poly_idx) * padded_poly_size;

    if (d_out == d_in) {
        if (idx < rev_idx) {
            FpExt tmp = d_out[base + idx];
            d_out[base + idx] = d_out[base + rev_idx];
            d_out[base + rev_idx] = tmp;
        }
    } else {
        d_out[base + rev_idx] = d_in[base + idx];
    }
}

// ============================================================================
// NTT Butterfly Kernels
// ============================================================================

/// Forward NTT step: Cooley-Tukey butterfly.
/// Translated from risc0's multi_ntt_fwd_step.
/// Processes one butterfly stage of the NTT.
kernel void ntt_forward_step(
    device Fp *io [[buffer(0)]],
    const device Fp *rou [[buffer(1)]],  // roots of unity table
    constant uint32_t &nBits [[buffer(2)]],
    constant uint32_t &sBits [[buffer(3)]],
    constant uint32_t &cSize [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 t_pos [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    uint3 tg_count [[threadgroups_per_grid]]
) {
    uint32_t gSize = 1u << (nBits - sBits);
    uint32_t sSize = 1u << (sBits - 1);
    uint32_t nSize = 1u << nBits;
    uint32_t sOff = t_pos.x + tg_pos.x * tg_size.x;
    uint32_t sStep = tg_size.x * tg_count.x;
    uint32_t gOff = t_pos.y + tg_pos.y * tg_size.y;
    uint32_t gStep = tg_size.y * tg_count.y;
    uint32_t cOff = t_pos.z + tg_pos.z * tg_size.z;
    uint32_t cStep = tg_size.z * tg_count.z;

    // Compute initial twiddle factor: rou[sBits]^sOff
    Fp curMul(1);
    uint32_t curRou = sBits;
    uint32_t powX = sOff;
    while (curRou > 0) {
        if (powX & 1) {
            curMul = curMul * rou[curRou];
        }
        powX >>= 1;
        curRou--;
    }

    int rouStep = accel_ffs(sSize / sStep);
    Fp stepMul = rou[rouStep];

    for (uint32_t s = sOff; s < sSize; s += sStep) {
        for (uint32_t g = gOff; g < gSize; g += gStep) {
            for (uint32_t c = cOff; c < cSize; c += cStep) {
                Fp a = io[c * nSize + g * 2 * sSize + s];
                Fp b = io[c * nSize + g * 2 * sSize + s + sSize];
                b *= curMul;
                io[c * nSize + g * 2 * sSize + s] = a + b;
                io[c * nSize + g * 2 * sSize + s + sSize] = a - b;
            }
        }
        curMul *= stepMul;
    }
}

/// Inverse NTT step: Gentleman-Sande butterfly.
/// Translated from risc0's multi_ntt_rev_step.
kernel void gs_mixed_radix_narrow(
    device Fp *io [[buffer(0)]],
    const device Fp *rou [[buffer(1)]],
    constant uint32_t &nBits [[buffer(2)]],
    constant uint32_t &sBits [[buffer(3)]],
    constant uint32_t &cSize [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 t_pos [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    uint3 tg_count [[threadgroups_per_grid]]
) {
    uint32_t gSize = 1u << (nBits - sBits);
    uint32_t sSize = 1u << (sBits - 1);
    uint32_t nSize = 1u << nBits;
    uint32_t sOff = t_pos.x + tg_pos.x * tg_size.x;
    uint32_t sStep = tg_size.x * tg_count.x;
    uint32_t gOff = t_pos.y + tg_pos.y * tg_size.y;
    uint32_t gStep = tg_size.y * tg_count.y;
    uint32_t cOff = t_pos.z + tg_pos.z * tg_size.z;
    uint32_t cStep = tg_size.z * tg_count.z;

    Fp curMul(1);
    uint32_t curRou = sBits;
    uint32_t powX = sOff;
    while (curRou > 0) {
        if (powX & 1) {
            curMul = curMul * rou[curRou];
        }
        powX >>= 1;
        curRou--;
    }

    int rouStep = accel_ffs(sSize / sStep);
    Fp stepMul = rou[rouStep];

    for (uint32_t s = sOff; s < sSize; s += sStep) {
        for (uint32_t g = gOff; g < gSize; g += gStep) {
            for (uint32_t c = cOff; c < cSize; c += cStep) {
                Fp a = io[c * nSize + g * 2 * sSize + s];
                Fp b = io[c * nSize + g * 2 * sSize + s + sSize];
                io[c * nSize + g * 2 * sSize + s] = a + b;
                io[c * nSize + g * 2 * sSize + s + sSize] = (a - b) * curMul;
            }
        }
        curMul *= stepMul;
    }
}
