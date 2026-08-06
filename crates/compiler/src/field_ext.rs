//! DSL-side field-inverse primitives for BabyBear and its degree-4 binomial
//! extension `𝔽ₚ[t]/(t⁴ − 11)`.
//!
//! Both helpers are pure `kernel!`-macro emissions — they append hash-consed
//! [`Node`](crate::ir::Node)s to the caller's [`IRBuilder`] and return a
//! [`NodeId`] for the result. They do not emit host closures or blackbox
//! nodes; the compiled kernel evaluates everything on-device.
//!
//! # Shape convention for `ef_inverse`
//!
//! An `FpExt` element `a = a0 + a1·t + a2·t² + a3·t³` is represented as a
//! rank-1 [`ScalarType::BabyBear`](crate::ir::ScalarType::BabyBear) tensor of
//! shape `[D_EF]` with `[a0, a1, a2, a3]`. This is byte-identical to the
//! `ScalarType::FpExt` representation (16 bytes = 4 canonical `u32`s in
//! coefficient order) and composes cleanly with transcript samplers that
//! produce `[1, D_EF]` BabyBear buffers — callers hand us the inner
//! `[D_EF]` slice.
//!
//! # Preconditions
//!
//! Both helpers assume `x != 0` (matching `bb31_t::reciprocal`'s undefined
//! behavior on zero). The caller is responsible for excluding zero inputs.

use crate::{
    ir::{IRBuilder, NodeId},
    kernel,
};

/// The degree of the BabyBear extension `𝔽ₚ[t]/(t⁴ − 11)`.
const D_EF: usize = 4;

/// The binomial constant `β = 11` for the extension `𝔽ₚ[t]/(t⁴ − β)`.
///
/// Note this is the *canonical* representation of `β`, unlike the CUDA
/// `bb31_4_t::BETA` which stores `(11 << 32) % MOD` because the CUDA code
/// works in Montgomery form. The DSL uses canonical `u32`s for BabyBear,
/// so we use `11` directly.
const BETA: u32 = 11;

/// `x^(2^n) · y`, emitted as `n` squarings followed by one multiplication —
/// the primitive of the `bb31_t::reciprocal` addition chain.
fn sqr_n_mul(b: &mut IRBuilder, x: NodeId, n: usize, y: NodeId) -> NodeId {
    let mut acc = x;
    for _ in 0..n {
        acc = kernel!(b, acc * acc);
    }
    kernel!(b, acc * y)
}

/// Compute `x^{-1}` for `x: BabyBear` (a scalar `NodeId` of type
/// `Scalar(BabyBear)`).
///
/// Uses the addition chain of `bb31_t::reciprocal`
/// (`crates/cuda-common/include/ff/baby_bear.hpp`): 31 squarings and 7
/// multiplications, reaching the exponent `p - 2` (Fermat's little theorem)
/// so the result equals `x^{p-2} = x^{-1}` for every non-zero `x`.
///
/// Hash-consing merges repeated squares — every `sqr_n_mul` step reuses the
/// running `acc`, so the emitted DAG has exactly 31 `Mul(a, a)` squarings
/// and 7 `Mul(a, b)` multiplications.
///
/// **Precondition**: `x != 0`. The result on zero is undefined (matches
/// `bb31_t::reciprocal`'s behavior).
pub fn bb_inverse(b: &mut IRBuilder, x: NodeId) -> NodeId {
    // x11 = x^17            (0b10001)
    let x11 = sqr_n_mul(b, x, 4, x);
    // r = x11^3 = x^51      (0b110011)
    let r = sqr_n_mul(b, x11, 1, x11);
    // r = r^2 * x11 = x^119 (0b1110111)
    let r = sqr_n_mul(b, r, 1, x11);
    // xff = r^2 * x11 = x^255 (0b11111111)
    let xff = sqr_n_mul(b, r, 1, x11);
    // r = r^(2^8) * xff     (0b111011111111111)
    let r = sqr_n_mul(b, r, 8, xff);
    // r = r^(2^8) * xff     (0b11101111111111111111111)
    let r = sqr_n_mul(b, r, 8, xff);
    // r = r^(2^8) * xff     (0b1110111111111111111111111111111) = x^(p-2)
    sqr_n_mul(b, r, 8, xff)
}

/// Norm-based scalar inversion of an `FpExt` element given its four
/// coefficient `NodeId`s.
///
/// Given `a = a0 + a1·t + a2·t² + a3·t³ ∈ 𝔽ₚ[t]/(t⁴ − β)` (with `β = 11`),
/// returns the four coefficients of `a⁻¹` in the same basis.
///
/// The algorithm computes the norm `N = b0² − β·b2²` (a base-field element)
/// where
/// ```text
/// b0 = a0² + β·(a2² − 2·a1·a3)
/// b2 = 2·a0·a2 − a1² − β·a3²
/// ```
/// then delegates to [`bb_inverse`] for `1/N` and reconstructs the four
/// output coefficients from `b0`, `b2`, and the original `a`.
///
/// Every operation here is in the *base* field (BabyBear); the extension
/// arithmetic is unrolled into scalar ops, which lets the compiler emit
/// plain `BabyBear` device code without going through the 16-byte `FpExt`
/// path.
pub fn ef_inverse_coeffs(b: &mut IRBuilder, a: [NodeId; D_EF]) -> [NodeId; D_EF] {
    let [a0, a1, a2, a3] = a;
    let beta = b.const_field(BETA);

    // b0 = a0² + β·(a2² − 2·a1·a3)
    let a2_sq = kernel!(b, a2 * a2);
    let a1a3 = kernel!(b, a1 * a3);
    let two_a1a3 = kernel!(b, a1a3 + a1a3);
    let a2_sq_m_2a1a3 = kernel!(b, a2_sq - two_a1a3);
    let beta_term_b0 = kernel!(b, beta * a2_sq_m_2a1a3);
    let a0_sq = kernel!(b, a0 * a0);
    let b0 = kernel!(b, a0_sq + beta_term_b0);

    // b2 = 2·a0·a2 − a1² − β·a3²
    let a0a2 = kernel!(b, a0 * a2);
    let two_a0a2 = kernel!(b, a0a2 + a0a2);
    let a1_sq = kernel!(b, a1 * a1);
    let a3_sq = kernel!(b, a3 * a3);
    let beta_a3_sq = kernel!(b, beta * a3_sq);
    let b2 = kernel!(b, two_a0a2 - a1_sq - beta_a3_sq);

    // N = b0² − β·b2²   (base field)
    let b0_sq = kernel!(b, b0 * b0);
    let b2_sq = kernel!(b, b2 * b2);
    let beta_b2_sq = kernel!(b, beta * b2_sq);
    let norm = kernel!(b, b0_sq - beta_b2_sq);

    // 1/N via bb_inverse (Fermat's little theorem chain).
    let inv_n = bb_inverse(b, norm);
    let b0 = kernel!(b, b0 * inv_n);
    let b2 = kernel!(b, b2 * inv_n);
    let beta_b2 = kernel!(b, beta * b2);

    // ret[0] = a0·b0 − a2·(β·b2)
    let t0a = kernel!(b, a0 * b0);
    let t0b = kernel!(b, a2 * beta_b2);
    let r0 = kernel!(b, t0a - t0b);
    // ret[1] = a3·(β·b2) − a1·b0
    let t1a = kernel!(b, a3 * beta_b2);
    let t1b = kernel!(b, a1 * b0);
    let r1 = kernel!(b, t1a - t1b);
    // ret[2] = a2·b0 − a0·b2
    let t2a = kernel!(b, a2 * b0);
    let t2b = kernel!(b, a0 * b2);
    let r2 = kernel!(b, t2a - t2b);
    // ret[3] = a1·b2 − a3·b0
    let t3a = kernel!(b, a1 * b2);
    let t3b = kernel!(b, a3 * b0);
    let r3 = kernel!(b, t3a - t3b);

    [r0, r1, r2, r3]
}

/// Compute `x^{-1}` for `x` a `BabyBear` rank-1 tensor of shape `[D_EF]`
/// (four canonical BabyBear coefficients `(a0, a1, a2, a3)` for the FpExt
/// element `x = a0 + a1·t + a2·t² + a3·t³`).
///
/// Returns a tensor of the same shape `[D_EF]` holding the coefficients of
/// `x⁻¹`. Emits the norm-based inversion of `bb31_4_t::reciprocal`
/// (`crates/cuda-common/include/ff/baby_bear.hpp`) inline; the base-field
/// reciprocal step delegates to [`bb_inverse`].
///
/// **Precondition**: `x != 0`. The result on zero is undefined.
pub fn ef_inverse(b: &mut IRBuilder, x: NodeId) -> NodeId {
    let (k0, k1, k2, k3) = (0usize, 1usize, 2usize, 3usize);
    let a0 = kernel!(b, x[#k0]);
    let a1 = kernel!(b, x[#k1]);
    let a2 = kernel!(b, x[#k2]);
    let a3 = kernel!(b, x[#k3]);
    let [r0, r1, r2, r3] = ef_inverse_coeffs(b, [a0, a1, a2, a3]);
    b.pack(&[r0, r1, r2, r3])
}

#[cfg(all(test, feature = "planner"))]
mod tests {
    use openvm_cuda_common::{
        copy::{MemCopyD2H, MemCopyH2D},
        d_buffer::DeviceBuffer,
        stream::GpuDeviceCtx,
    };
    use p3_baby_bear::BabyBear;
    use p3_field::{extension::BinomialExtensionField, BasedVectorSpace, Field, PrimeField32};

    use super::*;
    use crate::{
        graph_exe::GraphCompiler,
        graph_ir::GraphModule,
        ir::{IRBuilder, ScalarType},
        kernel,
        test_utils::{from_monty, to_monty},
    };

    type EF = BinomialExtensionField<BabyBear, 4>;
    const P: u64 = 2_013_265_921;

    /// Deterministic pseudo-random canonical BabyBear elements, avoiding zero
    /// (undefined for the inverse algorithm).
    fn splitmix_nonzero(n: usize, seed: u64) -> Vec<u32> {
        let mut x = seed;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let v = (z % P) as u32;
            if v != 0 {
                out.push(v);
            }
        }
        out
    }

    #[test]
    fn bb_inverse_matches_p3() {
        const N: usize = 8;
        let inputs: Vec<u32> = {
            // Include 1 and a few "structured" values plus splitmix randoms.
            let mut v = vec![1u32, 2, 3, (P - 1) as u32];
            v.extend(splitmix_nonzero(N - v.len(), 0xC0FFEE));
            v
        };
        assert_eq!(inputs.len(), N);

        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![N]);
        let body = kernel!(b, compute[N] | i | { bb_inverse(x[i]) });
        let module = b.finish("bb_inverse_test", body);

        // The emitted CUDA operates on Montgomery-encoded BabyBear; this
        // test drives the raw `KernelProgram` so it encodes/decodes itself.
        let mont_inputs: Vec<u32> = inputs.iter().map(|&v| to_monty(v)).collect();
        let ctx = GpuDeviceCtx::for_current_device().unwrap();
        let gm = GraphModule::from_ir(module, &[]).unwrap();
        let mut exe = GraphCompiler::new().compile(gm.into_builder()).unwrap();
        let km = exe.kernel_program(0);
        let d_in: DeviceBuffer<u32> = mont_inputs.as_slice().to_device_on(&ctx).unwrap();
        let d_out = DeviceBuffer::<u32>::with_capacity_on(N, &ctx);
        km.set_input(0, &d_in).unwrap();
        km.set_output(0, &d_out).unwrap();
        km.run(&ctx.stream).unwrap();
        let got: Vec<u32> = d_out
            .to_host_on(&ctx)
            .unwrap()
            .into_iter()
            .map(from_monty)
            .collect();

        let want: Vec<u32> = inputs
            .iter()
            .map(|&v| BabyBear::new(v).inverse().as_canonical_u32())
            .collect();
        assert_eq!(got, want, "bb_inverse mismatch");
    }

    /// Rebuild an `EF` value from the DSL's canonical 4×`u32` coefficient
    /// layout.
    fn ef_from_u32s(raw: &[u32]) -> EF {
        let coeffs: [BabyBear; 4] = [
            BabyBear::new(raw[0]),
            BabyBear::new(raw[1]),
            BabyBear::new(raw[2]),
            BabyBear::new(raw[3]),
        ];
        EF::from_basis_coefficients_slice(&coeffs).unwrap()
    }

    fn ef_to_u32s(v: EF) -> [u32; 4] {
        let s: &[BabyBear] = v.as_basis_coefficients_slice();
        [
            s[0].as_canonical_u32(),
            s[1].as_canonical_u32(),
            s[2].as_canonical_u32(),
            s[3].as_canonical_u32(),
        ]
    }

    #[test]
    fn ef_inverse_matches_p3() {
        const N: usize = 8;
        // Flat `[N * D_EF]` buffer, coefficient-major (`a0, a1, a2, a3` per
        // element). We include the special case `x = (1, 0, 0, 0)` (i.e.
        // `EF::ONE`) and fill the rest with random non-zero-norm elements —
        // splitmix rarely picks a zero-norm EF, but we still verify each
        // element is invertible with `.try_inverse()`.
        let mut flat: Vec<u32> = Vec::with_capacity(N * D_EF);
        flat.extend_from_slice(&[1, 0, 0, 0]); // EF::ONE
        let mut seed = 0xBADC_0FFE_E0DD_F00D_u64;
        while flat.len() < N * D_EF {
            let coeffs = splitmix_nonzero(D_EF, seed);
            seed = seed.wrapping_add(1);
            let ef = ef_from_u32s(&coeffs);
            if ef.try_inverse().is_some() {
                flat.extend_from_slice(&coeffs);
            }
        }
        assert_eq!(flat.len(), N * D_EF);

        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![N, D_EF]);
        // For each row `i`, read the four coefficients as scalars and call
        // the scalar-tuple form directly — that lets us feed a batched
        // `[N, D_EF]` input without materializing a `[D_EF]` slice
        // (`ef_inverse` itself operates on the rank-1 case). The public
        // `ef_inverse` is exercised end-to-end by the single-EF path below.
        let body = b.compute(N, |b, i| {
            let (k0, k1, k2, k3) = (0usize, 1usize, 2usize, 3usize);
            let a0 = kernel!(b, x[i, #k0]);
            let a1 = kernel!(b, x[i, #k1]);
            let a2 = kernel!(b, x[i, #k2]);
            let a3 = kernel!(b, x[i, #k3]);
            let [r0, r1, r2, r3] = ef_inverse_coeffs(b, [a0, a1, a2, a3]);
            b.pack(&[r0, r1, r2, r3])
        });
        let module = b.finish("ef_inverse_test", body);

        // Montgomery-encode at the raw `KernelProgram` boundary (see
        // `bb_inverse_matches_p3`).
        let mont_flat: Vec<u32> = flat.iter().map(|&v| to_monty(v)).collect();
        let ctx = GpuDeviceCtx::for_current_device().unwrap();
        let gm = GraphModule::from_ir(module, &[]).unwrap();
        let mut exe = GraphCompiler::new().compile(gm.into_builder()).unwrap();
        let km = exe.kernel_program(0);
        let d_in: DeviceBuffer<u32> = mont_flat.as_slice().to_device_on(&ctx).unwrap();
        let d_out = DeviceBuffer::<u32>::with_capacity_on(N * D_EF, &ctx);
        km.set_input(0, &d_in).unwrap();
        km.set_output(0, &d_out).unwrap();
        km.run(&ctx.stream).unwrap();
        let got: Vec<u32> = d_out
            .to_host_on(&ctx)
            .unwrap()
            .into_iter()
            .map(from_monty)
            .collect();

        let mut want: Vec<u32> = Vec::with_capacity(N * D_EF);
        for i in 0..N {
            let coeffs = &flat[i * D_EF..(i + 1) * D_EF];
            let ef = ef_from_u32s(coeffs);
            let inv = ef.inverse();
            want.extend_from_slice(&ef_to_u32s(inv));
        }
        assert_eq!(got, want, "ef_inverse coeff-tuple mismatch");

        // Also exercise the public `ef_inverse(b, x: [D_EF]) -> [D_EF]` API
        // end-to-end on a single element (EF::ONE). We wrap the call in a
        // trivial `compute[1]` — a bare top-level `Pack` isn't a valid
        // module output on its own (canonicalize requires the pack to be
        // the result of a compute). The compute body invokes `ef_inverse`
        // directly, so the `x[#k]` index-and-pack path is exercised on the
        // GPU.
        let single: [u32; D_EF] = [to_monty(1), 0, 0, 0];
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![D_EF]);
        let body = b.compute(1, move |b, _i| ef_inverse(b, x));
        let module = b.finish("ef_inverse_single", body);
        let gm = GraphModule::from_ir(module, &[]).unwrap();
        let mut exe = GraphCompiler::new().compile(gm.into_builder()).unwrap();
        let km = exe.kernel_program(0);
        let d_in: DeviceBuffer<u32> = single.as_slice().to_device_on(&ctx).unwrap();
        let d_out = DeviceBuffer::<u32>::with_capacity_on(D_EF, &ctx);
        km.set_input(0, &d_in).unwrap();
        km.set_output(0, &d_out).unwrap();
        km.run(&ctx.stream).unwrap();
        let got: Vec<u32> = d_out
            .to_host_on(&ctx)
            .unwrap()
            .into_iter()
            .map(from_monty)
            .collect();
        assert_eq!(got, vec![1u32, 0, 0, 0], "ef_inverse(EF::ONE) != EF::ONE");
    }
}
