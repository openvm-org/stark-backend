// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Host glue for the CTA probe ring.
//
// Every instrumented kernel launch in openvm-cuda-backend reads its ring
// pointers from `shadow_cta_ctx()`, which returns a process-global
// CtaProbeCtx. Until the Rust profiler init publishes a real ctx, we return a
// zeroed struct (mask == 0). The probe macros in cta_probe.cuh treat mask == 0
// as a runtime no-op: no atomic, no globaltimer read, just a single branch on
// CTA lane 0.
//
// Updates are guarded by a single mutex because publication is rare (once at
// init, optionally once at shutdown) and we want a clean acquire/release pair.
// Reads happen on every kernel launch; we copy the struct out under the
// mutex. The struct is 24 bytes — copy cost is negligible relative to a
// kernel launch (~10 us for a small kernel, ns for a struct copy).

#include "cta_probe.cuh"
#include <pthread.h>
#include <string.h>

static CtaProbeCtx g_shadow_ctx = {nullptr, 0u, 0u, nullptr};
static pthread_mutex_t g_shadow_ctx_lock = PTHREAD_MUTEX_INITIALIZER;

extern "C" CtaProbeCtx shadow_cta_ctx() {
    pthread_mutex_lock(&g_shadow_ctx_lock);
    CtaProbeCtx out = g_shadow_ctx;
    pthread_mutex_unlock(&g_shadow_ctx_lock);
    return out;
}

extern "C" void shadow_set_cta_ctx(CtaProbeCtx ctx) {
    pthread_mutex_lock(&g_shadow_ctx_lock);
    g_shadow_ctx = ctx;
    pthread_mutex_unlock(&g_shadow_ctx_lock);
}

extern "C" void shadow_clear_cta_ctx() {
    pthread_mutex_lock(&g_shadow_ctx_lock);
    memset(&g_shadow_ctx, 0, sizeof(g_shadow_ctx));
    pthread_mutex_unlock(&g_shadow_ctx_lock);
}
