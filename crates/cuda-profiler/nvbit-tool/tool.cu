// SPDX-FileCopyrightText: 2026 Axiom
// SPDX-License-Identifier: MIT
//
// NVBit-based "instrument every CTA of every kernel" tool that produces a
// SHDWPROF binary log identical to what the source-level probes produce.
// `cuda-profiler-assemble` (the existing Rust binary) parses the log without
// modification.
//
// Lifecycle:
//   nvbit_at_init           -> parse env vars, no GPU work yet
//   nvbit_at_cuda_event     -> on each cuLaunchKernel{,Ex}: lazy init the
//                              ring on first launch, instrument the CUfunction
//                              if not already done, register the kernel name
//   <kernel runs>           -> each CTA's lane-0 thread writes one CtaRecord
//   drain thread (pthread)  -> consumes ring -> framed SHDWPROF records
//   nvbit_at_term           -> stop drain, flush, close log
//
// What we do NOT do (vs the source-level Rust crate):
//   * No CUPTI sidecar. NVBit interposes on cuLaunchKernel and we already
//     get every kernel name + correlation id; if we want timing-only kernel
//     spans, we can record them from the host side at launch time. For now
//     the per-CTA records carry kernel_id and t_start/t_end which is what
//     the timeline cares about.
//   * No NVTX capture. CUPTI is the only good source for that, and skipping
//     CUPTI is the whole point of this tool — it's the "I just installed
//     NVBit, give me a Gantt" path.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <pthread.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <zstd.h>

#include "nvbit.h"
#include "nvbit_tool.h"
#include "utils/utils.h"

// ---- format constants (must match crates/cuda-profiler/src/record.rs) -----

static constexpr uint8_t MAGIC[8] = {'S', 'H', 'D', 'W', 'P', 'R', 'O', 'F'};
static constexpr uint16_t VERSION_MAJOR = 1;
static constexpr uint16_t VERSION_MINOR = 0;

static constexpr uint32_t TAG_PROCESS_START = 0x0001;
static constexpr uint32_t TAG_KERNEL_NAME = 0x0002;
static constexpr uint32_t TAG_CTA_RECORD = 0x0003;
static constexpr uint32_t TAG_DROP = 0x0004;

struct CtaRecord {
    uint32_t kernel_id;
    uint32_t smid;
    uint32_t block_linear;
    uint32_t seq_tag;
    uint64_t t_start;
    uint64_t t_end;
};
static_assert(sizeof(CtaRecord) == 32, "CtaRecord must be 32 bytes");

struct CtaProbeCtx {
    CtaRecord *ring;
    uint32_t mask;
    uint32_t sample_n; // power-of-two; 1 = no sampling
    unsigned long long *head;
};
static_assert(sizeof(CtaProbeCtx) == 24, "CtaProbeCtx layout");

// ---- FNV-1a, identical to `kernel_id::fnv1a` in the Rust crate. The
//     0->0xdeadbeef bias keeps 0 reserved as the "uninitialized slot"
//     sentinel.

static uint32_t fnv1a(const char *s) {
    uint32_t h = 0x811c9dc5u;
    while (*s) {
        h ^= static_cast<uint8_t>(*s++);
        h *= 0x01000193u;
    }
    return h == 0u ? 0xdeadbeefu : h;
}

// ---- log writer + ring -----------------------------------------------------
//
// All record bytes go through a streaming zstd encoder before being
// fwrite'd to the log file. The 16-byte SHDWPROF header is part of the
// compressed stream — the on-disk file is a raw zstd frame whose magic is
// zstd's `0x28 0xb5 0x2f 0xfd`. Consumers detect the format by peeking
// the first 4 bytes; raw and zstd-framed logs are both accepted (raw
// kept for backwards-compat with older logs).
//
// Compression level is fixed at 1 — at the drain rate (~30-50 MB/s on
// this workload) zstd-1 sits comfortably inside the budget of a single
// CPU and gives 3-5x reduction on these record streams. Higher levels
// would burn host CPU without buying meaningfully more reduction.

static FILE *g_log_fp = nullptr;
static ZSTD_CStream *g_zstd = nullptr;
static std::vector<uint8_t> g_zstd_outbuf;
static std::mutex g_log_mu;
static bool g_log_poisoned = false;

// Drain compressed bytes from `inbuf` to disk. Caller holds g_log_mu.
static void zstd_drain_locked(const uint8_t *inbuf, size_t inlen,
                              ZSTD_EndDirective end) {
    ZSTD_inBuffer in = {inbuf, inlen, 0};
    do {
        ZSTD_outBuffer out = {g_zstd_outbuf.data(), g_zstd_outbuf.size(), 0};
        size_t rem = ZSTD_compressStream2(g_zstd, &out, &in, end);
        if (ZSTD_isError(rem)) {
            fprintf(stderr, "nvbit-tool: zstd error: %s\n",
                    ZSTD_getErrorName(rem));
            g_log_poisoned = true;
            return;
        }
        if (out.pos > 0) {
            if (fwrite(g_zstd_outbuf.data(), 1, out.pos, g_log_fp) != out.pos) {
                fprintf(stderr, "nvbit-tool: log fwrite failed\n");
                g_log_poisoned = true;
                return;
            }
        }
        // Continue until input drained AND (for END/FLUSH directive) the
        // encoder reports it's emitted everything.
        if (end == ZSTD_e_continue) {
            if (in.pos == in.size) break;
        } else {
            if (rem == 0 && in.pos == in.size) break;
        }
    } while (true);
}

static void write_raw(const void *data, size_t len) {
    if (g_log_poisoned || g_log_fp == nullptr) return;
    zstd_drain_locked(reinterpret_cast<const uint8_t *>(data), len,
                      ZSTD_e_continue);
}

static void write_frame(uint32_t tag, const void *payload, uint32_t plen) {
    std::lock_guard<std::mutex> lk(g_log_mu);
    if (g_log_poisoned || g_log_fp == nullptr) {
        return;
    }
    write_raw(&tag, 4);
    write_raw(&plen, 4);
    if (plen > 0) write_raw(payload, plen);
}

static void write_header(const char *path) {
    std::lock_guard<std::mutex> lk(g_log_mu);
    g_log_fp = fopen(path, "wb");
    if (!g_log_fp) {
        fprintf(stderr, "nvbit-tool: failed to open %s for write\n", path);
        g_log_poisoned = true;
        return;
    }
    g_zstd = ZSTD_createCStream();
    if (!g_zstd) {
        fprintf(stderr, "nvbit-tool: ZSTD_createCStream failed\n");
        g_log_poisoned = true;
        return;
    }
    size_t init = ZSTD_initCStream(g_zstd, /*compressionLevel=*/1);
    if (ZSTD_isError(init)) {
        fprintf(stderr, "nvbit-tool: ZSTD_initCStream: %s\n",
                ZSTD_getErrorName(init));
        g_log_poisoned = true;
        return;
    }
    // Match zstd's recommended 128 KB output buffer size.
    g_zstd_outbuf.resize(ZSTD_CStreamOutSize());
    setvbuf(g_log_fp, nullptr, _IOFBF, 1 << 20);
    write_raw(MAGIC, 8);
    write_raw(&VERSION_MAJOR, 2);
    write_raw(&VERSION_MINOR, 2);
    uint32_t reserved = 0;
    write_raw(&reserved, 4);
}

static void log_close_locked() {
    if (g_zstd) {
        zstd_drain_locked(nullptr, 0, ZSTD_e_end);
        ZSTD_freeCStream(g_zstd);
        g_zstd = nullptr;
    }
    if (g_log_fp) {
        fflush(g_log_fp);
        fclose(g_log_fp);
        g_log_fp = nullptr;
    }
}

static void write_process_start(uint32_t pid, uint32_t gpu_index,
                                const char *gpu_name) {
    // payload: pid:u32, gpu_index:u32, start_walltime_ns:u64, name_len:u32, name
    uint64_t wt = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
    uint32_t nlen = static_cast<uint32_t>(strlen(gpu_name));
    std::vector<uint8_t> buf(4 + 4 + 8 + 4 + nlen);
    memcpy(&buf[0], &pid, 4);
    memcpy(&buf[4], &gpu_index, 4);
    memcpy(&buf[8], &wt, 8);
    memcpy(&buf[16], &nlen, 4);
    memcpy(&buf[20], gpu_name, nlen);
    write_frame(TAG_PROCESS_START, buf.data(), buf.size());
}

static void write_kernel_name(uint32_t id, const char *name) {
    uint32_t nlen = static_cast<uint32_t>(strlen(name));
    std::vector<uint8_t> buf(4 + 4 + nlen);
    memcpy(&buf[0], &id, 4);
    memcpy(&buf[4], &nlen, 4);
    memcpy(&buf[8], name, nlen);
    write_frame(TAG_KERNEL_NAME, buf.data(), buf.size());
}

static void write_drop(uint64_t count, uint64_t head_at_detection,
                       uint64_t approx_t_ns) {
    uint8_t buf[24];
    memcpy(&buf[0], &count, 8);
    memcpy(&buf[8], &head_at_detection, 8);
    memcpy(&buf[16], &approx_t_ns, 8);
    write_frame(TAG_DROP, buf, sizeof(buf));
}

static void write_cta_record(const CtaRecord *r) {
    write_frame(TAG_CTA_RECORD, r, sizeof(*r));
}

// ---- pinned-mapped ring + drain thread ------------------------------------

static CtaRecord *g_ring_host = nullptr;
static CtaRecord *g_ring_dev = nullptr;
static unsigned long long *g_head_host = nullptr;
static unsigned long long *g_head_dev = nullptr;
static uint32_t g_ring_mask = 0;
static uint64_t *g_scratch_dev = nullptr;
static uint32_t g_scratch_mask = 0;
static CtaProbeCtx *g_ctx_dev = nullptr; // device-resident copy of the probe ctx
static std::atomic<bool> g_drain_stop{false};
static pthread_t g_drain_thr;

static uint64_t round_up_pow2(uint64_t x) {
    if (x <= 1) return 1;
    uint64_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

static void *drain_loop(void *arg) {
    (void)arg;
    uint64_t tail = 0;
    uint64_t cap = static_cast<uint64_t>(g_ring_mask) + 1ULL;
    auto interval = std::chrono::milliseconds(50);
    for (;;) {
        bool stopping = g_drain_stop.load(std::memory_order_acquire);
        // Read head. The probe writes via mapped memory; the host can read
        // it without sync as long as the probe used __threadfence_system()
        // before the seq_tag store, which inject_funcs.cu does.
        uint64_t cur_head = __atomic_load_n(g_head_host, __ATOMIC_ACQUIRE);
        if (cur_head > tail) {
            // overrun?
            if (cur_head - tail > cap) {
                uint64_t dropped = cur_head - tail - cap;
                uint64_t new_tail = cur_head - cap;
                uint64_t probe_slot = new_tail & g_ring_mask;
                uint64_t approx = g_ring_host[probe_slot].t_end;
                if (approx == 0) approx = g_ring_host[probe_slot].t_start;
                write_drop(dropped, cur_head, approx);
                tail = new_tail;
            }
            while (tail < cur_head) {
                CtaRecord rec = g_ring_host[tail & g_ring_mask];
                uint32_t expected =
                    static_cast<uint32_t>((tail + 1ULL) & 0xFFFFFFFFULL);
                if (rec.seq_tag != expected) {
                    // Producer hasn't published yet. Stop the walk; we'll
                    // try again next interval. Don't advance tail past a
                    // torn slot.
                    break;
                }
                write_cta_record(&rec);
                tail++;
            }
            // Flush so on-disk content tracks reality even if the host
            // process aborts (we don't get atexit, NVBit teardown is
            // best-effort). We need a zstd flush (`ZSTD_e_flush`) so the
            // compressor emits a complete sub-frame, not just `fflush` —
            // otherwise the on-disk bytes would be a partial frame and
            // not zstd-decompressible until process exit.
            std::lock_guard<std::mutex> lk(g_log_mu);
            if (g_zstd) zstd_drain_locked(nullptr, 0, ZSTD_e_flush);
            if (g_log_fp) fflush(g_log_fp);
        }
        if (stopping) {
            return nullptr;
        }
        std::this_thread::sleep_for(interval);
    }
}

// ---- one-shot init on first kernel launch ---------------------------------

static std::once_flag g_init_once;
static bool g_active = false;
static std::string g_log_path;

static void parse_env_and_init() {
    const char *enable = std::getenv("SHADOW_PROFILER");
    if (!enable || (strcmp(enable, "1") != 0 && strcasecmp(enable, "true") != 0 &&
                    strcasecmp(enable, "on") != 0)) {
        return; // tool loaded but runtime-disabled
    }

    const char *out = std::getenv("SHADOW_PROFILER_OUT");
    g_log_path = out ? out : "shadow_profile.bin";

    uint64_t ring_bytes = 64ULL * 1024 * 1024;
    if (const char *rb = std::getenv("SHADOW_PROFILER_RING_BYTES")) {
        ring_bytes = strtoull(rb, nullptr, 10);
        if (ring_bytes == 0) ring_bytes = 64ULL * 1024 * 1024;
    }
    uint64_t slots = round_up_pow2(ring_bytes / sizeof(CtaRecord));
    g_ring_mask = static_cast<uint32_t>(slots - 1);

    // Allocate pinned-mapped host memory; the probe writes via the device
    // pointer and the drain reads via the host pointer.
    cudaError_t e = cudaHostAlloc(reinterpret_cast<void **>(&g_ring_host),
                                  slots * sizeof(CtaRecord),
                                  cudaHostAllocMapped);
    if (e != cudaSuccess) {
        fprintf(stderr, "nvbit-tool: cudaHostAlloc(ring) failed: %s\n",
                cudaGetErrorString(e));
        return;
    }
    memset(g_ring_host, 0, slots * sizeof(CtaRecord));
    cudaHostGetDevicePointer(reinterpret_cast<void **>(&g_ring_dev),
                             g_ring_host, 0);

    e = cudaHostAlloc(reinterpret_cast<void **>(&g_head_host),
                      sizeof(unsigned long long), cudaHostAllocMapped);
    if (e != cudaSuccess) {
        fprintf(stderr, "nvbit-tool: cudaHostAlloc(head) failed: %s\n",
                cudaGetErrorString(e));
        return;
    }
    *g_head_host = 0;
    cudaHostGetDevicePointer(reinterpret_cast<void **>(&g_head_dev),
                             g_head_host, 0);

    // Scratch table for begin-probe -> end-probe `t_start` handoff.
    // 8 M slots (64 MB) by default; collisions across concurrent kernels
    // with the same hash give a tiny tail of records with t_start ~= t_end
    // which inject_funcs.cu detects and clamps to a 1-ns span.
    uint64_t scratch_slots = 8ULL * 1024 * 1024;
    if (const char *s = std::getenv("SHADOW_PROFILER_SCRATCH_SLOTS")) {
        scratch_slots = round_up_pow2(strtoull(s, nullptr, 10));
        if (scratch_slots == 0) scratch_slots = 8ULL * 1024 * 1024;
    }
    g_scratch_mask = static_cast<uint32_t>(scratch_slots - 1);
    e = cudaMalloc(reinterpret_cast<void **>(&g_scratch_dev),
                   scratch_slots * sizeof(uint64_t));
    if (e != cudaSuccess) {
        fprintf(stderr, "nvbit-tool: cudaMalloc(scratch) failed: %s\n",
                cudaGetErrorString(e));
        return;
    }
    cudaMemset(g_scratch_dev, 0, scratch_slots * sizeof(uint64_t));

    // Decimation: SHADOW_PROFILER_SAMPLE_N records 1 of every N CTAs from
    // any kernel whose grid >= N, and keeps every CTA from kernels with
    // grid < N (so tiny kernels aren't dropped). Must be a power of two —
    // we round up to the next pow2 if not.
    uint32_t sample_n = 1;
    if (const char *s = std::getenv("SHADOW_PROFILER_SAMPLE_N")) {
        uint64_t v = strtoull(s, nullptr, 10);
        if (v >= 1) {
            uint64_t p = 1;
            while (p < v) p <<= 1;
            sample_n = static_cast<uint32_t>(p);
        }
    }

    // CtaProbeCtx that the injected code reads from constant address space.
    // We pass its device pointer as the first arg to every probe call.
    CtaProbeCtx ctx{};
    ctx.ring = g_ring_dev;
    ctx.mask = g_ring_mask;
    ctx.sample_n = sample_n;
    ctx.head = g_head_dev;
    e = cudaMalloc(reinterpret_cast<void **>(&g_ctx_dev), sizeof(CtaProbeCtx));
    if (e != cudaSuccess) {
        fprintf(stderr, "nvbit-tool: cudaMalloc(ctx) failed: %s\n",
                cudaGetErrorString(e));
        return;
    }
    cudaMemcpy(g_ctx_dev, &ctx, sizeof(CtaProbeCtx), cudaMemcpyHostToDevice);

    write_header(g_log_path.c_str());
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    write_process_start(static_cast<uint32_t>(getpid()),
                        static_cast<uint32_t>(dev), prop.name);

    if (pthread_create(&g_drain_thr, nullptr, drain_loop, nullptr) != 0) {
        fprintf(stderr, "nvbit-tool: pthread_create(drain) failed\n");
        return;
    }
    g_active = true;

    fprintf(stderr,
            "nvbit-tool: profiler activated\n"
            "  log:           %s (zstd-framed)\n"
            "  ring slots:    %llu (%.0f MB)\n"
            "  scratch slots: %llu (%.0f MB)\n"
            "  sample 1/N:    %u\n"
            "  GPU:           %s (sm_%d%d)\n",
            g_log_path.c_str(), (unsigned long long)slots,
            slots * sizeof(CtaRecord) / 1e6,
            (unsigned long long)scratch_slots,
            scratch_slots * sizeof(uint64_t) / 1e6, sample_n, prop.name,
            prop.major, prop.minor);
}

// ---- per-kernel instrumentation -------------------------------------------

static std::unordered_set<CUfunction> g_already;
static std::unordered_set<uint32_t> g_registered_ids;
static std::mutex g_instr_mu;

// Instrument *only* the launched kernel (the `__global__` entry point).
//
// We deliberately do NOT also instrument the device-side helpers reported by
// `nvbit_get_related_functions(ctx, func)`. A previous version of this tool
// did, and it inflated the record count by 10-100x: every `operator*` or
// inlined-but-not-inlined `__device__` arithmetic helper called from inside
// a kernel got its own BEGIN/END probe, so each parent CTA emitted dozens
// of records (one per helper invocation), with the same `smid` /
// `block_linear` but credited to the helper's `kernel_id`. Those records
// are not "CTAs" — they're function calls within a CTA — so attribution
// (top-12 by GPU time, etc.) was meaningless and the on-disk log size
// blew up to 76 GB / 1.9 B records on the prove-app run.
//
// Skipping `nvbit_get_related_functions` keeps everything correct: each CTA
// still emits exactly one record (BEGIN at instrs[0], END before the
// kernel's EXIT/RET, atomicExch dedup'd if there are multiple exit sites).
static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::lock_guard<std::mutex> lk(g_instr_mu);
    if (!g_already.insert(func).second) return;

    const char *name = nvbit_get_func_name(ctx, func, /*mangled=*/false);
    uint32_t kid = fnv1a(name);
    if (g_registered_ids.insert(kid).second) {
        write_kernel_name(kid, name);
    }

    const auto &instrs = nvbit_get_instrs(ctx, func);
    if (instrs.empty()) return;

    // BEGIN at the first instruction.
    nvbit_insert_call(instrs.front(), "cta_probe_begin", IPOINT_BEFORE);
    nvbit_add_call_arg_const_val64(instrs.front(),
                                   reinterpret_cast<uint64_t>(g_ctx_dev));
    nvbit_add_call_arg_const_val64(instrs.front(),
                                   reinterpret_cast<uint64_t>(g_scratch_dev));
    nvbit_add_call_arg_const_val32(instrs.front(), g_scratch_mask);
    nvbit_add_call_arg_const_val32(instrs.front(), kid);

    // END before every exit-class instruction. Hopper uses `EXIT` (and
    // sometimes `EXIT.NO_ATEXIT`); pre-Hopper uses `RET`. Duplicate firings
    // on a single CTA are harmless: `cta_probe_end` uses `atomicExch` on
    // the scratch slot so only the first call publishes a record (see
    // inject_funcs.cu).
    size_t placed = 0;
    for (auto *i : instrs) {
        const char *op = i->getOpcode();
        if (!op) continue;
        bool is_exit = (strncmp(op, "RET", 3) == 0) ||
                       (strncmp(op, "EXIT", 4) == 0);
        if (!is_exit) continue;
        nvbit_insert_call(i, "cta_probe_end", IPOINT_BEFORE);
        nvbit_add_call_arg_const_val64(i, reinterpret_cast<uint64_t>(g_ctx_dev));
        nvbit_add_call_arg_const_val64(i, reinterpret_cast<uint64_t>(g_scratch_dev));
        nvbit_add_call_arg_const_val32(i, g_scratch_mask);
        nvbit_add_call_arg_const_val32(i, kid);
        placed++;
    }
    if (placed == 0) {
        // Last-resort fallback for kernels with no recognizable EXIT/RET in
        // the disassembly (rare, e.g. noreturn variants).
        nvbit_insert_call(instrs.back(), "cta_probe_end", IPOINT_BEFORE);
        nvbit_add_call_arg_const_val64(instrs.back(),
                                       reinterpret_cast<uint64_t>(g_ctx_dev));
        nvbit_add_call_arg_const_val64(instrs.back(),
                                       reinterpret_cast<uint64_t>(g_scratch_dev));
        nvbit_add_call_arg_const_val32(instrs.back(), g_scratch_mask);
        nvbit_add_call_arg_const_val32(instrs.back(), kid);
    }

    if (const char *v = std::getenv("SHADOW_PROFILER_VERBOSE")) {
        if (v[0] == '1') {
            fprintf(stderr,
                    "nvbit-tool: instrumented %s (kid=%#010x, %zu instrs, "
                    "%zu exit sites)\n",
                    name, kid, instrs.size(), placed);
        }
    }
}

// ---- nvbit hooks ----------------------------------------------------------

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    // We can't allocate CUDA buffers here yet (no CUcontext exists). Defer
    // to first launch.
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    (void)name;
    (void)pStatus;
    if (is_exit) return;
    if (cbid != API_CUDA_cuLaunch && cbid != API_CUDA_cuLaunchKernel_ptsz &&
        cbid != API_CUDA_cuLaunchGrid && cbid != API_CUDA_cuLaunchGridAsync &&
        cbid != API_CUDA_cuLaunchKernel && cbid != API_CUDA_cuLaunchKernelEx &&
        cbid != API_CUDA_cuLaunchKernelEx_ptsz) {
        return;
    }
    std::call_once(g_init_once, parse_env_and_init);
    if (!g_active) return;

    CUfunction func;
    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
        cbid == API_CUDA_cuLaunchKernelEx) {
        func = ((cuLaunchKernelEx_params *)params)->f;
    } else {
        func = ((cuLaunchKernel_params *)params)->f;
    }
    instrument_function_if_needed(ctx, func);
    nvbit_enable_instrumented(ctx, func, true);
}

void nvbit_at_term() {
    if (!g_active) return;
    g_drain_stop.store(true, std::memory_order_release);
    pthread_join(g_drain_thr, nullptr);
    {
        std::lock_guard<std::mutex> lk(g_log_mu);
        log_close_locked();
    }
    fprintf(stderr, "nvbit-tool: profiler shut down (log %s)\n", g_log_path.c_str());
}
