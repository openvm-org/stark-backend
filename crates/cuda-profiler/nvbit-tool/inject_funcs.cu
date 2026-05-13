// SPDX-FileCopyrightText: 2026 Axiom
// SPDX-License-Identifier: MIT
//
// Device-side instrumentation injected into every kernel by the host tool
// in `tool.cu`. Two pieces:
//
//   1. `cta_probe_begin(ctx_ptr, scratch_ptr, scratch_mask, kernel_id)` —
//      fires at the kernel's first SASS instruction. On lane 0 of each CTA
//      only, it reads `%globaltimer` and writes it into a
//      `(smid, block_linear)`-keyed scratch slot so the end probe can
//      recover `t_start`.
//
//   2. `cta_probe_end(ctx_ptr, scratch_ptr, scratch_mask, kernel_id)` —
//      fires before each RET/EXIT (or, if no exit-class opcode is found, on
//      the final instruction). Reads `t_end` and `smid`, atomicExchanges
//      `t_start` out of the scratch slot (so duplicate firings within one
//      CTA emit at most one record), atomically allocates a ring slot,
//      writes the 32-byte `CtaRecord`, fences, then publishes the seq_tag.
//
// Cost: lane-0 only, two `mov` + one atomicExch + one atomicAdd + one
// 32-byte store per CTA. We enforce `--keep-device-functions` and
// `-Xptxas -astoolspatch` in the Makefile so NVBit can splice these calls
// into the target kernel without a linker resolve step.

#include <cstdint>
#include <cuda_runtime.h>

#include "utils/utils.h"

// Mirrors `CtaRecord` in `crates/cuda-profiler/src/record.rs`.
// 32 bytes total, alignof 8.
struct CtaRecord {
    uint32_t kernel_id;
    uint32_t smid;
    uint32_t block_linear;
    uint32_t seq_tag;
    uint64_t t_start;
    uint64_t t_end;
};

// Mirrors `CtaProbeCtx`. ring/head are pinned-mapped pointers from the host.
// `sample_n` (must be a power of two) gates per-CTA emission: a CTA is
// recorded iff `(block_linear & (sample_n - 1)) == 0` OR the kernel's
// total grid size is smaller than sample_n (in which case we'd lose the
// kernel entirely under naive sampling). sample_n == 1 disables sampling.
struct CtaProbeCtx {
    CtaRecord *ring;
    uint32_t mask;
    uint32_t sample_n;
    unsigned long long *head;
};

// Sampling gate. Returns true if this CTA should be recorded. Both probes
// must compute the *same* answer so that BEGIN's scratch slot write and
// END's atomicExch agree — otherwise a non-sampled CTA's END could pick up
// a stale t_start written by a sampled CTA, or a sampled CTA's END could
// see a zeroed slot it never wrote.
static __device__ __forceinline__ bool sample_keep(uint32_t sample_n,
                                                   uint64_t block_linear) {
    if (sample_n <= 1u) return true;
    uint64_t grid_total = static_cast<uint64_t>(gridDim.x) *
                          static_cast<uint64_t>(gridDim.y) *
                          static_cast<uint64_t>(gridDim.z);
    if (grid_total < static_cast<uint64_t>(sample_n)) return true;
    return (block_linear & static_cast<uint64_t>(sample_n - 1u)) == 0ULL;
}

extern "C" __device__ __noinline__ void cta_probe_begin(uint64_t ctx_ptr,
                                                        uint64_t scratch_ptr,
                                                        uint32_t scratch_mask,
                                                        uint32_t /*kernel_id*/) {
    if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }
    CtaProbeCtx *ctx = reinterpret_cast<CtaProbeCtx *>(ctx_ptr);
    if (ctx->mask == 0u) {
        return; // runtime-disabled
    }
    uint64_t block_linear =
        static_cast<uint64_t>(blockIdx.z) * static_cast<uint64_t>(gridDim.x) *
            static_cast<uint64_t>(gridDim.y) +
        static_cast<uint64_t>(blockIdx.y) * static_cast<uint64_t>(gridDim.x) +
        static_cast<uint64_t>(blockIdx.x);
    if (!sample_keep(ctx->sample_n, block_linear)) {
        return;
    }
    uint64_t t_start;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_start));
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    // Hash (smid, block_linear) into the scratch table. Collisions across
    // concurrent kernels would corrupt `t_start` for the unlucky pair, but
    // both the table size (default 8 M slots) and the spread from the prime
    // multipliers keep that under ~1e-4 in practice.
    uint64_t h = (static_cast<uint64_t>(smid) * 9973ULL) ^
                 (block_linear * 6151ULL);
    uint64_t *scratch = reinterpret_cast<uint64_t *>(scratch_ptr);
    scratch[h & static_cast<uint64_t>(scratch_mask)] = t_start;
}

extern "C" __device__ __noinline__ void cta_probe_end(uint64_t ctx_ptr,
                                                      uint64_t scratch_ptr,
                                                      uint32_t scratch_mask,
                                                      uint32_t kernel_id) {
    if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }
    CtaProbeCtx *ctx = reinterpret_cast<CtaProbeCtx *>(ctx_ptr);
    if (ctx->mask == 0u) {
        return;
    }
    uint64_t block_linear =
        static_cast<uint64_t>(blockIdx.z) * static_cast<uint64_t>(gridDim.x) *
            static_cast<uint64_t>(gridDim.y) +
        static_cast<uint64_t>(blockIdx.y) * static_cast<uint64_t>(gridDim.x) +
        static_cast<uint64_t>(blockIdx.x);
    if (!sample_keep(ctx->sample_n, block_linear)) {
        return;
    }
    uint64_t t_end;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_end));
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    uint64_t h = (static_cast<uint64_t>(smid) * 9973ULL) ^
                 (block_linear * 6151ULL);
    uint64_t *scratch = reinterpret_cast<uint64_t *>(scratch_ptr);
    // atomicExch grabs t_start AND zeros the slot atomically, so if the
    // host tool inserted this probe before more than one exit-class
    // instruction (multi-block kernels with several EXITs in divergence
    // patterns, or RET+EXIT pairs on the trailing alignment slots), only
    // the first one to fire publishes a record; subsequent calls for the
    // same CTA see 0 and skip. Cost is one atomicExch on global memory,
    // lane-0 only.
    unsigned long long *slot_ptr =
        reinterpret_cast<unsigned long long *>(
            &scratch[h & static_cast<uint64_t>(scratch_mask)]);
    unsigned long long t_start_ull = atomicExch(slot_ptr, 0ULL);
    if (t_start_ull == 0ULL) {
        return; // already emitted, or begin probe never ran for this CTA
    }
    uint64_t t_start = static_cast<uint64_t>(t_start_ull);
    if (t_start > t_end) {
        // Scratch slot was clobbered by a hash collision with a *later*
        // CTA's begin probe. Clamp to a 1-ns span — the placement is still
        // correct, only the duration is unreliable.
        t_start = t_end > 0ULL ? t_end - 1ULL : 0ULL;
    }
    unsigned long long slot = atomicAdd(ctx->head, 1ULL);
    CtaRecord *r = &ctx->ring[slot & static_cast<unsigned long long>(ctx->mask)];
    r->kernel_id = kernel_id;
    r->smid = smid;
    r->block_linear = static_cast<uint32_t>(block_linear);
    r->t_start = t_start;
    r->t_end = t_end;
    __threadfence_system();
    // Publication: tag store last. Drain reads tag and validates that
    // `tag == ((slot+1) & 0xFFFFFFFF)` before consuming.
    r->seq_tag = static_cast<uint32_t>((slot + 1ULL) & 0xFFFFFFFFULL);
}
