#include "base_off.cuh"
#include "fp.h"
#include "fpext.h"
#include "frac_ext.cuh"
#include "launcher.cuh"

#include <cstddef>
#include <cstdint>

namespace logup_zerocheck_ring {

// ============================================================================
// The batched-sumcheck "ring": the device half of `compute_batch_s_poly`.
// ============================================================================
//
// The eager prover closes every steady MLE round on the host
// (`logup_zerocheck/mod.rs:393-422`): it D2Hs the round evaluators' outputs,
// runs ~70 lines of `EF` algebra in `compute_batch_s_poly`
// (`mod.rs:1448-1520`), observes `s(1)..s(s_deg)`, samples `r_round`, and
// evaluates the polynomial again at the sample. That host seam is what stops
// the phase from being one static graph.
//
// The two entry points below are that seam, moved onto the device and split
// at the transcript:
//
//   * `_batch_s_ring_pre`  — everything BEFORE `observe_ext`: raw evaluator loads, numerator
//     normalization, early head accumulation, late equality correction, exhausted tilde scaling,
//     tail accumulation, the missing `s'(0)` reconstruction, Lagrange interpolation, the
//     `eq(xi, X)` multiply, `tail * X`, and Horner evaluation at `1..=s_deg`.
//   * `_batch_s_ring_post` — everything AFTER `sample_ext`: Horner evaluation at `r_round` and the
//     two running equality-product updates.
//
// Both are ONE thread. The work is `O(3 * num_traces * d)` field operations
// on a handful of scalars; a parallel version would need a reduction and a
// second launch for strictly less arithmetic than the launch itself costs.
// The point of moving it here is not throughput, it is that the round's
// challenge never has to exist on the host.

/// One trace's compact evaluator outputs, as offsets into the graph pool.
///
/// The graph's round evaluators emit compact per-batch `[air][num_x]` buffers
/// (`zerocheck_ir.rs:4264-4428`), typed `FpExt` for the constraint family and
/// `FracExt` for the interaction family. Rather than repack them into one
/// canonical array before the ring, each trace carries the address of its own
/// slice; `BASE_OFF_NULL` marks an absent or exhausted family.
struct BatchSRingTraceDesc {
    BaseOff zc_evals;
    BaseOff logup_evals;
    uint32_t n_lift;
    uint32_t flags;
};

static constexpr uint32_t BATCH_S_RING_HAS_CONSTRAINTS = 1u << 0;
static constexpr uint32_t BATCH_S_RING_HAS_INTERACTIONS = 1u << 1;

/// The production SDK's maximum `max_constraint_degree` (4, for recursion;
/// app configs use 3) — `stark-sdk/src/config/mod.rs:41-44`. Every local
/// array below is sized from it and both launchers reject a larger `d`
/// rather than overrunning.
static constexpr uint32_t BATCH_S_RING_MAX_DEGREE = 4;

// The Rust mirror lives in `src/cuda/logup_zerocheck.rs`. It is hand-written,
// so pin the layout here and probe it from Rust (`_batch_s_ring_trace_desc_*`
// below) rather than trusting two independent `#[repr(C)]` / `struct`
// declarations to agree by coincidence.
static_assert(sizeof(BaseOff) == 8, "BaseOff must be a bare uint64_t");
static_assert(sizeof(BatchSRingTraceDesc) == 24, "BatchSRingTraceDesc must be 24 bytes");
static_assert(alignof(BatchSRingTraceDesc) == 8, "BatchSRingTraceDesc must be 8-byte aligned");
static_assert(offsetof(BatchSRingTraceDesc, zc_evals) == 0, "zc_evals must be first");
static_assert(offsetof(BatchSRingTraceDesc, logup_evals) == 8, "logup_evals must follow zc_evals");
static_assert(offsetof(BatchSRingTraceDesc, n_lift) == 16, "n_lift must follow logup_evals");
static_assert(offsetof(BatchSRingTraceDesc, flags) == 20, "flags must be last");

namespace {

__device__ __forceinline__ FpExt ext_zero() { return FpExt(Fp::zero()); }
__device__ __forceinline__ FpExt ext_one() { return FpExt(Fp::one()); }

/// `horner_eval(coeffs[0..len], x)` — `poly_common.rs:231-239`, which folds
/// from the highest coefficient down.
__device__ __forceinline__ FpExt horner_eval(const FpExt *coeffs, uint32_t len, FpExt x) {
    FpExt acc = ext_zero();
    for (int32_t i = static_cast<int32_t>(len) - 1; i >= 0; --i) {
        acc = acc * x + coeffs[i];
    }
    return acc;
}

/// `UnivariatePoly::lagrange_interpolate(&[F::from_usize(0..len)], evals)`
/// (`prover/poly.rs:382-464`), ported literally.
///
/// The host version batch-inverts every base-field denominator
/// `points[i] - points[j]`; here the points are the integers `0..len`, so
/// each denominator is inverted on its own. The inversion is taken in `FpExt`
/// rather than `Fp`: the extension inverse of a subfield element IS the
/// subfield inverse (inverses are unique), so the value is identical, and it
/// avoids relying on which type `Fp`'s inherited `operator-` returns.
///
/// The host's "skip a zero eval" shortcut is a pure optimization and is
/// omitted; `len <= 1` is still special-cased because the general loop would
/// otherwise index `lag[1]`.
__device__ void lagrange_interpolate_small(const FpExt *evals, uint32_t len, FpExt *coeffs) {
    for (uint32_t k = 0; k < len; ++k) {
        coeffs[k] = ext_zero();
    }
    if (len == 0) {
        return;
    }
    if (len == 1) {
        coeffs[0] = evals[0];
        return;
    }
    FpExt lag[BATCH_S_RING_MAX_DEGREE + 1];
    for (uint32_t i = 0; i < len; ++i) {
        uint32_t lag_len = 1;
        lag[0] = ext_one();
        for (uint32_t j = 0; j < len; ++j) {
            if (j == i) {
                continue;
            }
            const FpExt point_i = FpExt(Fp(i));
            const FpExt point_j = FpExt(Fp(j));
            const FpExt scale = inv(point_i - point_j);
            lag[lag_len] = ext_zero();
            lag_len += 1;
            for (uint32_t k = lag_len - 1; k >= 1; --k) {
                const FpExt prev_coeff = lag[k - 1] * scale;
                lag[k] += prev_coeff;
                lag[k - 1] = -(prev_coeff * point_j);
            }
        }
        for (uint32_t k = 0; k < lag_len; ++k) {
            coeffs[k] += evals[i] * lag[k];
        }
    }
}

__global__ void batch_s_ring_pre_kernel(
    const BatchSRingTraceDesc *__restrict__ trace_descs,
    const uint8_t *__restrict__ pool_base,
    const FpExt *__restrict__ tilde_in,
    const FpExt *__restrict__ mu_pows,
    const Fp *__restrict__ norm_factors,
    const FpExt *__restrict__ scalar_state_in,
    const FpExt *__restrict__ xi_j,
    const FpExt *__restrict__ r_prev,
    FpExt *__restrict__ tilde_out,
    FpExt *__restrict__ poly_coeffs_out,
    FpExt *__restrict__ s_evals_out,
    uint32_t num_traces,
    uint32_t constraint_degree,
    uint32_t round
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const uint32_t T = num_traces;
    const uint32_t d = constraint_degree;
    const uint32_t s_deg = d + 1;

    // The whole tilde vector is carried forward; the per-trace branches below
    // overwrite only the slots the eager path writes.
    for (uint32_t i = 0; i < 3u * T; ++i) {
        tilde_out[i] = tilde_in[i];
    }

    FpExt head_zc[BATCH_S_RING_MAX_DEGREE];
    FpExt head_logup[BATCH_S_RING_MAX_DEGREE];
    for (uint32_t i = 0; i < d; ++i) {
        head_zc[i] = ext_zero();
        head_logup[i] = ext_zero();
    }
    FpExt tail = ext_zero();

    const FpExt prev_s_eval = scalar_state_in[0];
    const FpExt eq_n = scalar_state_in[1];
    const FpExt eq_sharp_n = scalar_state_in[2];
    const FpExt r_prev_v = *r_prev;
    const FpExt xi_cur = *xi_j;

    for (uint32_t t = 0; t < T; ++t) {
        const BatchSRingTraceDesc desc = trace_descs[t];
        const uint32_t n_lift = desc.n_lift;
        const bool has_constraints = (desc.flags & BATCH_S_RING_HAS_CONSTRAINTS) != 0u;
        const bool has_interactions = (desc.flags & BATCH_S_RING_HAS_INTERACTIONS) != 0u;
        // The logical slot layout `compute_batch_s_poly` and `mu_pows` share
        // (`mod.rs:1465-1468`): `p_t = 2t`, `q_t = 2t + 1`, `zc_t = 2T + t`.
        const uint32_t numer_idx = 2u * t;
        const uint32_t denom_idx = numer_idx + 1u;
        const uint32_t zc_idx = 2u * T + t;

        const FpExt *zc_ev = base_off_ptr<const FpExt>(pool_base, desc.zc_evals);
        const FracExt *lg_ev = base_off_ptr<const FracExt>(pool_base, desc.logup_evals);
        // `norm_factor = F::from_usize(1 << max(-n, 0)).inverse()` — applied to
        // the interaction NUMERATOR only, never the denominator and never to
        // the constraint family (`mod.rs:1144-1146`, `batch_mle.rs:583, 671`).
        const Fp norm = norm_factors[t];

        if (round <= n_lift) {
            // EARLY: the evaluator produced `d` head values for this trace.
            if (has_constraints && zc_ev != nullptr) {
                for (uint32_t i = 0; i < d; ++i) {
                    head_zc[i] += mu_pows[zc_idx] * zc_ev[i];
                }
            }
            if (has_interactions && lg_ev != nullptr) {
                for (uint32_t i = 0; i < d; ++i) {
                    head_logup[i] += mu_pows[numer_idx] * (lg_ev[i].p * norm) +
                                     mu_pows[denom_idx] * lg_ev[i].q;
                }
            }
            continue;
        }

        if (round == n_lift + 1u) {
            // LATE: the evaluator produced ONE value per family, which is
            // installed into tilde and then corrected by the running equality
            // products (`mod.rs:1314-1316, 1339`, `mod.rs:1467-1472`).
            if (has_constraints && zc_ev != nullptr) {
                tilde_out[zc_idx] = zc_ev[0];
            }
            if (has_interactions && lg_ev != nullptr) {
                tilde_out[numer_idx] = lg_ev[0].p * norm;
                tilde_out[denom_idx] = lg_ev[0].q;
            }
            if (has_constraints) {
                tilde_out[zc_idx] = tilde_out[zc_idx] * eq_n;
            }
            if (has_interactions) {
                tilde_out[numer_idx] = tilde_out[numer_idx] * eq_sharp_n;
                tilde_out[denom_idx] = tilde_out[denom_idx] * eq_sharp_n;
            }
        } else {
            // EXHAUSTED: no launch at all; the eager path only scales the
            // already-held tilde values by the PREVIOUS round's challenge, and
            // it does so per family (`mod.rs:1188-1202`).
            if (has_constraints) {
                tilde_out[zc_idx] = tilde_out[zc_idx] * r_prev_v;
            }
            if (has_interactions) {
                tilde_out[numer_idx] = tilde_out[numer_idx] * r_prev_v;
                tilde_out[denom_idx] = tilde_out[denom_idx] * r_prev_v;
            }
        }
        tail += mu_pows[zc_idx] * tilde_out[zc_idx] +
                mu_pows[numer_idx] * tilde_out[numer_idx] +
                mu_pows[denom_idx] * tilde_out[denom_idx];
    }

    // `sp_head_evals[1..=d]` carry the equality products; `sp_head_evals[0]`
    // is reconstructed from `s_j(0) + s_j(1) = s_{j-1}(r_{j-1})`.
    FpExt sp_head_evals[BATCH_S_RING_MAX_DEGREE + 1];
    sp_head_evals[0] = ext_zero();
    for (uint32_t i = 0; i < d; ++i) {
        sp_head_evals[i + 1] = eq_n * head_zc[i] + eq_sharp_n * head_logup[i];
    }
    {
        const FpExt eq_xi_0 = ext_one() - xi_cur;
        const FpExt eq_xi_1 = xi_cur;
        sp_head_evals[0] =
            (prev_s_eval - eq_xi_1 * sp_head_evals[1] - tail) * inv(eq_xi_0);
    }

    // `s'` has degree `s_deg - 1`; `s(X) = eq(xi, X) * s'(X) + tail * X` has
    // degree `s_deg`, so `s_deg + 1 == d + 2` coefficients.
    FpExt coeffs[BATCH_S_RING_MAX_DEGREE + 2];
    lagrange_interpolate_small(sp_head_evals, s_deg, coeffs);
    coeffs[s_deg] = ext_zero();
    const FpExt b = ext_one() - xi_cur;
    const FpExt a = xi_cur - b;
    for (int32_t i = static_cast<int32_t>(s_deg) - 1; i >= 0; --i) {
        coeffs[i + 1] = a * coeffs[i] + b * coeffs[i + 1];
    }
    coeffs[0] = coeffs[0] * b;
    coeffs[1] += tail;

    for (uint32_t i = 0; i < s_deg + 1u; ++i) {
        poly_coeffs_out[i] = coeffs[i];
    }
    // The prover skips `s(0)`: the verifier infers it (`mod.rs:389-395`).
    for (uint32_t k = 1; k <= s_deg; ++k) {
        s_evals_out[k - 1] = horner_eval(coeffs, s_deg + 1u, FpExt(Fp(k)));
    }
}

__global__ void batch_s_ring_post_kernel(
    const FpExt *__restrict__ poly_coeffs,
    const FpExt *__restrict__ scalar_state_in,
    const FpExt *__restrict__ xi_j,
    const FpExt *__restrict__ r_round,
    FpExt *__restrict__ scalar_state_out,
    uint32_t constraint_degree
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const uint32_t s_deg = constraint_degree + 1;
    const FpExt r = *r_round;
    const FpExt xi = *xi_j;
    // `prev_s_eval' = batch_s.eval_at_point(r_round)` (`mod.rs:401`).
    const FpExt prev_next = horner_eval(poly_coeffs, s_deg + 1u, r);
    // `eq_r = eval_eq_mle(&[xi], &[r])` (`poly_common.rs:7-20`,
    // `mod.rs:1612-1614`).
    const FpExt eq_r = ext_one() - r - xi + (xi * r) * Fp(2u);
    scalar_state_out[0] = prev_next;
    scalar_state_out[1] = scalar_state_in[1] * eq_r;
    scalar_state_out[2] = scalar_state_in[2] * eq_r;
}

} // namespace

// ============================================================================
// Launchers.
// ============================================================================

extern "C" int _batch_s_ring_pre(
    const BatchSRingTraceDesc *trace_descs,
    const uint8_t *pool_base,
    const FpExt *tilde_in,
    const FpExt *mu_pows,
    const Fp *norm_factors,
    const FpExt *scalar_state_in,
    const FpExt *xi_j,
    const FpExt *r_prev,
    FpExt *tilde_out,
    FpExt *poly_coeffs_out,
    FpExt *s_evals_out,
    uint32_t num_traces,
    uint32_t constraint_degree,
    uint32_t round,
    cudaStream_t stream
) {
    // The local arrays are sized for the production maximum; a caller outside
    // that range must fail loudly rather than overrun them. The lower bound is
    // real too: the `s'(0)` reconstruction indexes head value 1.
    if (constraint_degree < 1 || constraint_degree > BATCH_S_RING_MAX_DEGREE) {
        return cudaErrorInvalidValue;
    }
    if (num_traces == 0) {
        return cudaErrorInvalidValue;
    }
    batch_s_ring_pre_kernel<<<1, 1, 0, stream>>>(
        trace_descs,
        pool_base,
        tilde_in,
        mu_pows,
        norm_factors,
        scalar_state_in,
        xi_j,
        r_prev,
        tilde_out,
        poly_coeffs_out,
        s_evals_out,
        num_traces,
        constraint_degree,
        round
    );
    return CHECK_KERNEL();
}

extern "C" int _batch_s_ring_post(
    const FpExt *poly_coeffs,
    const FpExt *scalar_state_in,
    const FpExt *xi_j,
    const FpExt *r_round,
    FpExt *scalar_state_out,
    uint32_t constraint_degree,
    cudaStream_t stream
) {
    if (constraint_degree < 1 || constraint_degree > BATCH_S_RING_MAX_DEGREE) {
        return cudaErrorInvalidValue;
    }
    batch_s_ring_post_kernel<<<1, 1, 0, stream>>>(
        poly_coeffs, scalar_state_in, xi_j, r_round, scalar_state_out, constraint_degree
    );
    return CHECK_KERNEL();
}

// Layout probes for the Rust mirror of `BatchSRingTraceDesc`. The C++
// `static_assert`s above pin this side; these let the Rust side assert the
// same numbers instead of assuming two hand-written declarations agree.
extern "C" size_t _batch_s_ring_trace_desc_size() { return sizeof(BatchSRingTraceDesc); }
extern "C" size_t _batch_s_ring_trace_desc_align() { return alignof(BatchSRingTraceDesc); }
extern "C" size_t _batch_s_ring_trace_desc_zc_offset() {
    return offsetof(BatchSRingTraceDesc, zc_evals);
}
extern "C" size_t _batch_s_ring_trace_desc_logup_offset() {
    return offsetof(BatchSRingTraceDesc, logup_evals);
}
extern "C" size_t _batch_s_ring_trace_desc_n_lift_offset() {
    return offsetof(BatchSRingTraceDesc, n_lift);
}
extern "C" size_t _batch_s_ring_trace_desc_flags_offset() {
    return offsetof(BatchSRingTraceDesc, flags);
}
extern "C" uint32_t _batch_s_ring_has_constraints_flag() { return BATCH_S_RING_HAS_CONSTRAINTS; }
extern "C" uint32_t _batch_s_ring_has_interactions_flag() { return BATCH_S_RING_HAS_INTERACTIONS; }
extern "C" uint32_t _batch_s_ring_max_degree() { return BATCH_S_RING_MAX_DEGREE; }

} // namespace logup_zerocheck_ring
