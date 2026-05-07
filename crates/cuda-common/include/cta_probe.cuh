// SPDX-License-Identifier: MIT OR Apache-2.0
// CTA-level GPU probe. Only included when SHADOW_CTA_PROFILE is defined.
//
// Each instrumented kernel records, on lane 0 of every CTA:
//   - a kernel_id (caller-assigned; matched by the host side by name),
//   - the SM the CTA ran on (%smid),
//   - the linear block index,
//   - the per-CTA start/end nanosecond timestamps (%globaltimer).
//
// Records land in a power-of-two ring whose pointer/mask/head come in via a
// CtaProbeCtx struct that is passed as the kernel's last argument. With
// SHADOW_CTA_PROFILE off, the macros and the parameter both compile out.
//
// Cost when enabled and runtime-active: one global-timer read at BEGIN and one
// atomicAdd + 24 B store at END, on lane 0 only. Cost when enabled but
// runtime-inactive (mask == 0): one branch on lane 0; no atomics, no stores.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// 32 bytes per CTA (4 x u32 + 2 x u64 with 8-byte alignment for the
// timestamps). One million CTAs is 32 MB of device-mapped pinned memory —
// still trivial relative to working-set sizes the prover allocates.
//
// The host-side reader assumes this exact layout, both in size and field
// order — see crates/cuda-profiler/src/ffi.rs and src/record.rs.
//
// `seq_tag` is a per-record completion marker the producer writes *last*,
// after a `__threadfence_system()`. It encodes the low 32 bits of (slot_seq +
// 1), so the host drain can distinguish a freshly-written record from a
// stale slot left over from a previous ring wrap: at slot S, the expected
// tag is `((S + 1) & 0xFFFFFFFFu)`.
//
// At process start the host zeroes the entire ring, so a tag of 0 reliably
// means "never written" up until slot index 2^32 - 1. The producer happens
// to write 0 there too (since (2^32 - 1) + 1 ≡ 0 mod 2^32) — and that's
// fine, because the consumer also expects 0 at that slot. The tag wraps
// every 2^32 records; a run that produces more than 4 billion CTA records
// will see false-positive matches across the wrap boundary. At 1 µs/CTA
// that's ~72 minutes — out of scope for the prover.
struct CtaRecord {
    uint32_t kernel_id;
    uint32_t smid;
    uint32_t block_linear;
    uint32_t seq_tag;
    uint64_t t_start;
    uint64_t t_end;
};
static_assert(sizeof(CtaRecord) == 32, "CtaRecord must be 32 bytes");
static_assert(alignof(CtaRecord) == 8, "CtaRecord must be 8-byte aligned");

// Passed to every instrumented kernel as its last parameter when
// SHADOW_CTA_PROFILE is defined. ring is in device-mapped pinned host memory;
// head is a 64-bit monotonically increasing counter; mask = ring_capacity - 1.
//
// mask == 0 is the "runtime-disabled" sentinel: the probe macros short-circuit
// and do nothing. This lets a profiler-enabled binary run with the host-side
// profiler turned off via env var, without an init crash.
struct CtaProbeCtx {
    CtaRecord *ring;
    uint32_t mask;
    uint32_t _pad;
    unsigned long long *head;
};
static_assert(sizeof(CtaProbeCtx) == 24, "CtaProbeCtx layout");

// FNV-1a, 32 bits. Identical to `openvm_cuda_profiler::kernel_id::fnv1a` —
// keep them in lockstep. The host-side `register_kernel(name)` call hashes
// with the same constants so the (id, name) mapping is implicit.
//
// We deliberately bias 0 -> 0xdeadbeef so the assembler can use kernel_id == 0
// as the "uninitialized ring slot" sentinel without ambiguity.
constexpr uint32_t cta_kid_fnv1a(const char *s) {
    uint32_t h = 0x811c9dc5u;
    while (*s) {
        h ^= static_cast<uint32_t>(static_cast<unsigned char>(*s++));
        h *= 0x01000193u;
    }
    return h == 0u ? 0xdeadbeefu : h;
}

// Probe macros take a runtime u32 kernel id (`KERNEL_ID`); the user-facing
// macros in launcher.cuh wrap these and hash the kernel name automatically.
//
// END uses a strict ordering protocol so the host drain never observes a
// torn record:
//
//   (1) atomicAdd reserves a unique slot index `S` (gpu-scope is sufficient;
//       slot uniqueness is the only inter-CTA invariant we need here).
//   (2) The non-tag fields are stored to ring[S & mask].
//   (3) `__threadfence_system()` flushes those stores to host-visible
//       (pinned-mapped) memory and orders them before the tag store.
//   (4) `seq_tag` is stored *last*, with value `(S + 1) & 0xFFFFFFFFu`.
//
// The drain reads the tag at its expected slot S. If `tag == ((S+1)&0xFFFFFFFF)`
// the record is fully published; otherwise the slot is either still being
// written or holds a stale record from a previous wrap. Either way, the drain
// does not consume it on this pass.
#define CTA_PROBE_BEGIN(KERNEL_ID, CTX)                                                            \
    uint64_t __cta_t0 = 0;                                                                         \
    if ((CTX).mask != 0u && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {            \
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(__cta_t0));                               \
    }

#define CTA_PROBE_END(KERNEL_ID, CTX)                                                              \
    if ((CTX).mask != 0u && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {            \
        uint64_t __cta_t1;                                                                         \
        uint32_t __cta_smid;                                                                       \
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(__cta_t1));                               \
        asm volatile("mov.u32 %0, %%smid;" : "=r"(__cta_smid));                                    \
        unsigned long long __cta_slot = atomicAdd((CTX).head, 1ULL);                               \
        CtaRecord *__cta_r = &(CTX).ring[__cta_slot & (CTX).mask];                                 \
        /* 64-bit math for block_linear: a single u32 multiply between two */                      \
        /* dimensions can overflow on a 3D grid even at moderate sizes. */                         \
        uint64_t __cta_bl = static_cast<uint64_t>(blockIdx.z)                                      \
                              * static_cast<uint64_t>(gridDim.x)                                   \
                              * static_cast<uint64_t>(gridDim.y)                                   \
                          + static_cast<uint64_t>(blockIdx.y)                                      \
                              * static_cast<uint64_t>(gridDim.x)                                   \
                          + static_cast<uint64_t>(blockIdx.x);                                     \
        __cta_r->kernel_id = static_cast<uint32_t>(KERNEL_ID);                                     \
        __cta_r->smid = __cta_smid;                                                                \
        __cta_r->block_linear = static_cast<uint32_t>(__cta_bl);                                   \
        __cta_r->t_start = __cta_t0;                                                               \
        __cta_r->t_end = __cta_t1;                                                                 \
        __threadfence_system();                                                                    \
        /* Publication: the tag store must be the last write the host sees. */                    \
        __cta_r->seq_tag = static_cast<uint32_t>((__cta_slot + 1ULL) & 0xFFFFFFFFu);               \
    }
