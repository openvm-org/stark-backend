//! Prebuilt DSL modules for the MVP: radix-2 DIT NTT and a Poseidon2-16
//! Merkle compression tree, both over BabyBear.

use p3_baby_bear::{
    BabyBear, BABYBEAR_RC16_EXTERNAL_FINAL, BABYBEAR_RC16_EXTERNAL_INITIAL, BABYBEAR_RC16_INTERNAL,
};
use p3_field::{PrimeCharacteristicRing, PrimeField32, TwoAdicField};

use crate::ir::{IRBuilder, Module, NodeId, ScalarType};

/// Poseidon2-16 BabyBear round constants and internal diagonal, canonical
/// `u32` representation.
#[derive(Clone, Debug)]
pub struct Poseidon2Constants {
    pub external_initial: [[u32; 16]; 4],
    pub external_final: [[u32; 16]; 4],
    pub internal: [u32; 13],
    /// Diagonal `V` of the internal linear layer `s[i] = sum(s) + V[i]*s[i]`.
    pub diag: [u32; 16],
}

impl Poseidon2Constants {
    /// The constants used by `p3_baby_bear::default_babybear_poseidon2_16`.
    pub fn p3_default() -> Self {
        fn inv_2exp(k: u64) -> BabyBear {
            BabyBear::ONE.div_2exp_u64(k)
        }
        let c = |x: BabyBear| x.as_canonical_u32();
        // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4,
        //      1/2^8, 1/4, 1/8, 1/2^27, -1/2^8, -1/16, -1/2^27]
        let diag: [BabyBear; 16] = [
            -BabyBear::TWO,
            BabyBear::ONE,
            BabyBear::TWO,
            BabyBear::ONE.halve(),
            BabyBear::new(3),
            BabyBear::new(4),
            -BabyBear::ONE.halve(),
            -BabyBear::new(3),
            -BabyBear::new(4),
            inv_2exp(8),
            inv_2exp(2),
            inv_2exp(3),
            inv_2exp(27),
            -inv_2exp(8),
            -inv_2exp(4),
            -inv_2exp(27),
        ];
        Self {
            external_initial: BABYBEAR_RC16_EXTERNAL_INITIAL.map(|r| r.map(c)),
            external_final: BABYBEAR_RC16_EXTERNAL_FINAL.map(|r| r.map(c)),
            internal: BABYBEAR_RC16_INTERNAL.map(c),
            diag: diag.map(c),
        }
    }
}

/// `x^7` via 4 multiplications.
fn sbox7(b: &mut IRBuilder, x: NodeId) -> NodeId {
    let x2 = b.mul(x, x);
    let x3 = b.mul(x2, x);
    let x4 = b.mul(x2, x2);
    b.mul(x3, x4)
}

/// The 4x4 MDS matrix [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]], applied via
/// the same addition chain as p3's `apply_mat4`.
fn apply_mat4(b: &mut IRBuilder, x: &mut [NodeId]) {
    let t01 = b.add(x[0], x[1]);
    let t23 = b.add(x[2], x[3]);
    let t0123 = b.add(t01, t23);
    let t01123 = b.add(t0123, x[1]);
    let t01233 = b.add(t0123, x[3]);
    let x0_dbl = b.add(x[0], x[0]);
    let x2_dbl = b.add(x[2], x[2]);
    x[3] = b.add(t01233, x0_dbl);
    x[1] = b.add(t01123, x2_dbl);
    x[0] = b.add(t01123, t01);
    x[2] = b.add(t01233, t23);
}

/// External linear layer: `apply_mat4` per 4-chunk, then `s[i] += sums[i%4]`
/// with `sums[k] = sum_j s[4j+k]`.
fn mds_light(b: &mut IRBuilder, state: &mut [NodeId; 16]) {
    for chunk in state.chunks_exact_mut(4) {
        apply_mat4(b, chunk);
    }
    let mut sums = [state[0]; 4];
    for (k, sum) in sums.iter_mut().enumerate() {
        let mut s = state[k];
        for j in (4..16).step_by(4) {
            s = b.add(s, state[j + k]);
        }
        *sum = s;
    }
    for (i, s) in state.iter_mut().enumerate() {
        *s = b.add(*s, sums[i % 4]);
    }
}

fn external_round(b: &mut IRBuilder, state: &mut [NodeId; 16], rc: &[u32; 16]) {
    for (s, &k) in state.iter_mut().zip(rc) {
        let kc = b.const_field(k);
        let x = b.add(*s, kc);
        *s = sbox7(b, x);
    }
    mds_light(b, state);
}

fn internal_round(b: &mut IRBuilder, state: &mut [NodeId; 16], rc: u32, diag: &[u32; 16]) {
    let kc = b.const_field(rc);
    let x = b.add(state[0], kc);
    state[0] = sbox7(b, x);
    let mut sum = state[0];
    for &s in state[1..].iter() {
        sum = b.add(sum, s);
    }
    for (s, &v) in state.iter_mut().zip(diag) {
        let vc = b.const_field(v);
        let prod = b.mul(vc, *s);
        *s = b.add(sum, prod);
    }
}

/// The full Poseidon2-16 permutation, unrolled into the expression DAG.
pub fn poseidon2_permutation(b: &mut IRBuilder, state: &mut [NodeId; 16], c: &Poseidon2Constants) {
    mds_light(b, state);
    for rc in &c.external_initial {
        external_round(b, state, rc);
    }
    for &rc in &c.internal {
        internal_round(b, state, rc, &c.diag);
    }
    for rc in &c.external_final {
        external_round(b, state, rc);
    }
}

/// Radix-2 DIT NTT of size `2^log_n` over BabyBear.
///
/// Inputs: `a: [n]` (coefficients), `w: [n/2]` (twiddles, `w[j] = omega^j`
/// for the `n`-th root of unity `omega`; see [`ntt_twiddles`]).
/// Output: `y[k] = sum_j a[j] * omega^(j*k)`, matching p3's `Radix2Dit`.
pub fn ntt_module(log_n: usize) -> Module {
    assert!(log_n >= 1, "NTT size must be at least 2");
    let n = 1usize << log_n;
    let mut b = IRBuilder::new();
    let a = b.input("a", ScalarType::BabyBear, vec![n]);
    let w = b.input("w", ScalarType::BabyBear, vec![n / 2]);

    // Bit-reversal permutation: out[i] = a[bitrev(i)].
    let bitrev = b.compute(n, |b, i| {
        let mut rev = b.const_u32(0);
        for bit in 0..log_n {
            let q = if bit == 0 {
                i
            } else {
                let shift = b.const_u32(1 << bit);
                b.div(i, shift)
            };
            let two = b.const_u32(2);
            let bit_val = b.rem(q, two);
            let scale = b.const_u32(1 << (log_n - 1 - bit));
            let term = b.mul(bit_val, scale);
            rev = b.add(rev, term);
        }
        b.index(a, &[rev])
    });
    let mut prev = b.let_bound(bitrev);

    // Butterfly stages: stage s merges blocks of size 2^(s-1) into 2^s.
    for s in 1..=log_n {
        let m = 1usize << s;
        let half = m / 2;
        let step = n / m;
        let stage = b.compute(n, |b, i| {
            let m_c = b.const_u32(m as u32);
            let half_c = b.const_u32(half as u32);
            let j = b.rem(i, m_c);
            let lo = b.lt(j, half_c);
            // Index of the low element of this butterfly. The `i - half`
            // branch may wrap when discarded, but it is never dereferenced:
            // only `base` and `base + half` are, and both are in bounds.
            let i_minus_half = b.sub(i, half_c);
            let base = b.select(lo, i, i_minus_half);
            let hi = b.add(base, half_c);
            let u = b.index(prev, &[base]);
            let v = b.index(prev, &[hi]);
            let jj = b.rem(j, half_c);
            let step_c = b.const_u32(step as u32);
            let tw_idx = b.mul(jj, step_c);
            let tw = b.index(w, &[tw_idx]);
            let t = b.mul(v, tw);
            let add = b.add(u, t);
            let sub = b.sub(u, t);
            b.select(lo, add, sub)
        });
        prev = b.let_bound(stage);
    }

    b.finish(format!("ntt_{n}"), prev)
}

/// Twiddle factors for [`ntt_module`]: `w[j] = omega^j` for `j < n/2`, where
/// `omega = BabyBear::two_adic_generator(log_n)`. Canonical representation.
pub fn ntt_twiddles(log_n: usize) -> Vec<u32> {
    assert!(log_n >= 1);
    BabyBear::two_adic_generator(log_n)
        .powers()
        .take(1 << (log_n - 1))
        .map(|x| x.as_canonical_u32())
        .collect()
}

/// Poseidon2-16 Merkle compression tree over `2^log_leaves` digests.
///
/// Input: `leaves: [2^log_leaves, 8]` (the bottom digest layer). Each layer
/// halves the previous one via `compress(l, r) = perm(l || r)[0..8]`
/// (truncated permutation, no feed-forward), matching p3's
/// `TruncatedPermutation<Poseidon2BabyBear<16>, 2, 8, 16>`.
/// Output: a tuple of all `log_leaves` layers, root last (shape `[1, 8]`).
pub fn merkle_tree_module(log_leaves: usize, c: &Poseidon2Constants) -> Module {
    assert!(log_leaves >= 1, "need at least two leaves");
    let n_leaves = 1usize << log_leaves;
    let mut b = IRBuilder::new();
    let leaves = b.input("leaves", ScalarType::BabyBear, vec![n_leaves, 8]);

    let mut layers: Vec<NodeId> = Vec::new();
    let mut prev = leaves;
    for lvl in 0..log_leaves {
        let n = n_leaves >> (lvl + 1);
        let layer = b.compute(n, |b, i| {
            let two = b.const_u32(2);
            let one = b.const_u32(1);
            let li = b.mul(i, two);
            let ri = b.add(li, one);
            let zero = b.const_u32(0);
            let mut state = [zero; 16];
            for j in 0..8 {
                let jc = b.const_u32(j as u32);
                state[j] = b.index(prev, &[li, jc]);
                state[8 + j] = b.index(prev, &[ri, jc]);
            }
            poseidon2_permutation(b, &mut state, c);
            b.pack(&state[..8])
        });
        prev = b.let_bound(layer);
        layers.push(prev);
    }

    let out = if layers.len() == 1 {
        layers[0]
    } else {
        b.tuple(&layers)
    };
    b.finish(format!("merkle_{n_leaves}"), out)
}
