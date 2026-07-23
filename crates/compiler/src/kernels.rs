//! Prebuilt DSL modules for the MVP: radix-2 DIT NTT and a Poseidon2-16
//! Merkle compression tree, both over BabyBear.

use p3_baby_bear::{
    BabyBear, BABYBEAR_RC16_EXTERNAL_FINAL, BABYBEAR_RC16_EXTERNAL_INITIAL, BABYBEAR_RC16_INTERNAL,
};
use p3_field::{PrimeCharacteristicRing, PrimeField32, TwoAdicField};

use crate::{
    ir::{IRBuilder, Module, NodeId, ScalarType},
    kernel,
};

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

/// `IRBuilder::const_field` as a free function, for `kernel!` call syntax
/// (the `#x` splice only produces `const_u32`).
fn cf(b: &mut IRBuilder, v: u32) -> NodeId {
    b.const_field(v)
}

/// `x^7` via 4 multiplications (hash-consing dedups the repeated squares).
fn sbox7(b: &mut IRBuilder, x: NodeId) -> NodeId {
    kernel!(b, ((x * x) * x) * ((x * x) * (x * x)))
}

/// The 4x4 MDS matrix [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]], applied via
/// the same addition chain as p3's `apply_mat4`.
fn apply_mat4(b: &mut IRBuilder, x: &mut [NodeId]) {
    let (x0, x1, x2, x3) = (x[0], x[1], x[2], x[3]);
    let t01 = kernel!(b, x0 + x1);
    let t23 = kernel!(b, x2 + x3);
    let t0123 = kernel!(b, t01 + t23);
    let t01123 = kernel!(b, t0123 + x1);
    let t01233 = kernel!(b, t0123 + x3);
    x[3] = kernel!(b, t01233 + (x0 + x0));
    x[1] = kernel!(b, t01123 + (x2 + x2));
    x[0] = kernel!(b, t01123 + t01);
    x[2] = kernel!(b, t01233 + t23);
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
            let x = state[j + k];
            s = kernel!(b, s + x);
        }
        *sum = s;
    }
    for (i, s) in state.iter_mut().enumerate() {
        let (x, m) = (*s, sums[i % 4]);
        *s = kernel!(b, x + m);
    }
}

fn external_round(b: &mut IRBuilder, state: &mut [NodeId; 16], rc: &[u32; 16]) {
    for (s, &k) in state.iter_mut().zip(rc) {
        let x = *s;
        *s = kernel!(b, sbox7(x + cf(k)));
    }
    mds_light(b, state);
}

fn internal_round(b: &mut IRBuilder, state: &mut [NodeId; 16], rc: u32, diag: &[u32; 16]) {
    let s0 = state[0];
    state[0] = kernel!(b, sbox7(s0 + cf(rc)));
    let mut sum = state[0];
    for &s in state[1..].iter() {
        sum = kernel!(b, sum + s);
    }
    for (s, &v) in state.iter_mut().zip(diag) {
        let x = *s;
        *s = kernel!(b, sum + cf(v) * x);
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

/// Bit-reversal of `x` over the low `bits` bits, unrolled into div/rem ops.
fn bitrev_expr(b: &mut IRBuilder, x: NodeId, bits: usize) -> NodeId {
    let mut rev = b.const_u32(0);
    for bit in 0..bits {
        let q = if bit == 0 {
            x
        } else {
            let shift = 1u32 << bit;
            kernel!(b, x / #shift)
        };
        let scale = 1u32 << (bits - 1 - bit);
        rev = kernel!(b, rev + q % 2 * #scale);
    }
    rev
}

/// `(x / 2^pos) % 2^width`: the `width`-bit field of `x` at bit `pos`.
fn extract_bits(b: &mut IRBuilder, x: NodeId, pos: usize, width: usize) -> NodeId {
    let q = if pos == 0 {
        x
    } else {
        let shift = 1u32 << pos;
        kernel!(b, x / #shift)
    };
    let m = 1u32 << width;
    kernel!(b, q % #m)
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
    let bitrev = kernel!(b, compute[n] | i | { a[bitrev_expr(i, log_n)] });
    let mut prev = b.let_bound(bitrev);

    // Butterfly stages: stage s merges blocks of size 2^(s-1) into 2^s.
    // `base` is the butterfly's low element (`i` with the half bit cleared),
    // written quasi-affinely so it can be used as an index.
    for s in 1..=log_n {
        let m = 1usize << s;
        let half = m / 2;
        let step = n / m;
        let stage = kernel!(b,
            compute [n] |i| {
                let j = i % #m;
                let lo = j < #half;
                let base = i - j / #half * #half;
                let u = prev[base];
                let v = prev[base + #half];
                let t = v * w[(j % #half) * #step];
                if lo then u + t else u - t
            }
        );
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

/// Largest bit-field a single shared-memory NTT group handles. A group of
/// `b` bits uses a `2^b`-element tile per stage, so `b = 8` means 256-thread
/// blocks and `8 * 256 * 4B = 8KB` of shared memory per block.
const NTT_GROUP_MAX_BITS: usize = 8;

/// Indexes a row-major `[rows, 2^cols_log2]` tensor by flat address.
fn index_flat2(b: &mut IRBuilder, t: NodeId, addr: NodeId, cols_log2: usize) -> NodeId {
    let cols = 1u32 << cols_log2;
    kernel!(b, t[addr / #cols, addr % #cols])
}

/// Splits `log_n` into `ceil(log_n / NTT_GROUP_MAX_BITS)` balanced fields.
fn split_bits(log_n: usize) -> Vec<usize> {
    let g = log_n.div_ceil(NTT_GROUP_MAX_BITS);
    let base = log_n / g;
    let rem = log_n % g;
    (0..g).map(|k| base + usize::from(k < rem)).collect()
}

/// Chains the butterfly stages of one NTT group as let-bound inner computes
/// (shared-memory tiles), the last stage being the block's result.
///
/// `prev` is the tile holding the group's input, `t` the local stage in
/// `1..=bg`, `start` the group's first logical bit, and `p_low` the low
/// `start` logical bits of the element (constant across the tile), which
/// determine the twiddle offset.
#[allow(clippy::too_many_arguments)]
fn ntt_group_stages(
    b: &mut IRBuilder,
    prev: NodeId,
    t: usize,
    bg: usize,
    start: usize,
    p_low: NodeId,
    w: NodeId,
    log_n: usize,
) -> NodeId {
    let tile = 1usize << bg;
    let m = 1usize << t;
    let half = 1usize << (t - 1);
    let fstride = 1usize << start;
    let step = 1usize << (log_n - start - t);
    // As in `ntt_module`, `base` is the butterfly's low element in
    // quasi-affine form. The twiddle index is the global butterfly position
    // modulo half, `p_low + (j % half) * 2^start`, times the stage step.
    let stage = kernel!(b,
        compute [tile] |j| {
            let jm = j % #m;
            let lo = jm < #half;
            let base = j - jm / #half * #half;
            let u = prev[base];
            let v = prev[base + #half];
            let t = v * w[(p_low + (j % #half) * #fstride) * #step];
            if lo then u + t else u - t
        }
    );
    if t == bg {
        stage
    } else {
        b.bind(stage, |b, v| {
            ntt_group_stages(b, v, t + 1, bg, start, p_low, w, log_n)
        })
    }
}

/// Shared-memory radix-2 DIT NTT of size `2^log_n` over BabyBear.
///
/// Same interface and result as [`ntt_module`] (inputs `a: [n]`, `w: [n/2]`,
/// natural order in and out), but the `log_n` butterfly stages are grouped
/// into bit-fields of at most [`NTT_GROUP_MAX_BITS`] bits. Each group is one
/// kernel whose block gathers a `2^b_g`-element tile into shared memory (a
/// let-bound inner compute) and runs all of the group's stages locally.
///
/// Group 0 reads contiguous tiles with a fused bit-reversal gather. Tiles
/// are always written back contiguously (`out[i * tile + j]`), which moves
/// the group's bit-field to the bottom of the address — tracked by `pi[k]`,
/// the physical position of logical field `k`. Later groups gather their
/// field with a strided read, and a final flat kernel restores natural order
/// when there is more than one group (a generalized four-step NTT).
pub fn ntt_shared_module(log_n: usize) -> Module {
    assert!(log_n >= 1, "NTT size must be at least 2");
    let n = 1usize << log_n;
    let bits = split_bits(log_n);
    let groups = bits.len();
    let starts: Vec<usize> = bits
        .iter()
        .scan(0, |acc, &width| {
            let s = *acc;
            *acc += width;
            Some(s)
        })
        .collect();

    let mut b = IRBuilder::new();
    let a = b.input("a", ScalarType::BabyBear, vec![n]);
    let w = b.input("w", ScalarType::BabyBear, vec![n / 2]);

    let mut pi = starts.clone();
    let mut prev = a;
    for g in 0..groups {
        let (bg, start) = (bits[g], starts[g]);
        let tile = 1usize << bg;
        let group = b.compute(n >> bg, |b, i| {
            let gather = if g == 0 {
                // Element (i, j) of the natural-order tile grid is
                // a[bitrev(i * tile + j)] = a[rev(j) * n/tile + rev(i)].
                let hi_bits = log_n - bg;
                let sc = 1usize << hi_bits;
                kernel!(b, compute [tile] |j| {
                    a[bitrev_expr(j, bg) * #sc + bitrev_expr(i, hi_bits)]
                })
            } else {
                // Insert `j` at bit `pi[g]` of `i`: a strided gather of
                // field g from the previous kernel's layout, whose output is
                // [n >> b_prev, 2^b_prev].
                let pg = 1usize << pi[g];
                let pgb = 1usize << (pi[g] + bg);
                let cols = bits[g - 1];
                kernel!(b, compute [tile] |j| {
                    index_flat2(prev, i / #pg * #pgb + i % #pg + j * #pg, cols)
                })
            };
            // Processed fields k < g occupy the physical range [0, start_g),
            // below field g, so their positions within `i` equal pi[k].
            let mut p_low = b.const_u32(0);
            for k in 0..g {
                let (pos, width) = (pi[k], bits[k]);
                let scale = 1usize << starts[k];
                p_low = kernel!(b, p_low + extract_bits(i, pos, width) * #scale);
            }
            b.bind(gather, |b, tile_var| {
                ntt_group_stages(b, tile_var, 1, bg, start, p_low, w, log_n)
            })
        });
        prev = b.let_bound(group);
        for k in 0..groups {
            if k != g && pi[k] < pi[g] {
                pi[k] += bg;
            }
        }
        pi[g] = 0;
    }

    if groups > 1 {
        // Restore natural order: out[p] = prev[sum_k field_k(p) * 2^pi[k]].
        let restored = b.compute(n, |b, p| {
            let mut addr = b.const_u32(0);
            for k in 0..groups {
                let (pos, width) = (starts[k], bits[k]);
                let scale = 1usize << pi[k];
                addr = kernel!(b, addr + extract_bits(p, pos, width) * #scale);
            }
            index_flat2(b, prev, addr, bits[groups - 1])
        });
        prev = b.let_bound(restored);
    }

    b.finish(format!("ntt_shared_{n}"), prev)
}

/// Bits per register-tiled NTT group: `2^13 = 8192` element tiles under
/// `#[grid(threads = 512)]`, so each of the 512 threads owns 16 contiguous
/// logical elements. Under the `(th, s) -> th * 16 + s` par layout, butterfly
/// partners with `half < 16` stay in register slots (Slot), `half ∈ [16, 256]`
/// stay within a warp (Shuffle), and `half >= 512` cross warps and go through
/// a shared-memory mirror (Bounce) — so the group naturally runs its first
/// nine stages in registers and its last four in shared memory.
const NTT_REG_BITS: usize = 13;
const NTT_REG_THREADS: usize = 512;
const NTT_REG_PER_THREAD: i64 = 1 << (NTT_REG_BITS - 9);

/// Chains the butterfly stages of one register-tiled NTT group as let-bound
/// inner computes, each carrying `#[par((th, s) -> th * 16 + s)]`.
#[allow(clippy::too_many_arguments)]
fn ntt_group_stages_reg(
    b: &mut IRBuilder,
    prev: NodeId,
    t: usize,
    bg: usize,
    start: usize,
    p_low: NodeId,
    w: NodeId,
    log_n: usize,
) -> NodeId {
    let tile = 1usize << bg;
    let m = 1usize << t;
    let half = 1usize << (t - 1);
    let fstride = 1usize << start;
    let step = 1usize << (log_n - start - t);
    let par = b.par_map(|th, s, _| th.mul_c(NTT_REG_PER_THREAD).add(s));
    // Select-form butterfly: each thread reads its own slot `prev[j]` and
    // its partner `prev[j ^ half]`, then blends based on the half bit. The
    // partner-read access map is XOR by `half`, so classify_convert sees
    // an identity linear part with an offset — a slot permutation for
    // `half < 16` and an invertible-lane-block warp shuffle for `half >= 16`
    // under the (th, s) -> th*16 + s layout.
    let stage = b.compute_with(tile, None, Some(par), None, |b, j| {
        kernel!(b,
            let own = prev[j];
            let partner = prev[j + #half - j % #m / #half * #m];
            let w_val = w[(p_low + (j % #half) * #fstride) * #step];
            let lo = j % #m < #half;
            if lo then own + w_val * partner else partner - w_val * own
        )
    });
    if t == bg {
        stage
    } else {
        b.bind(stage, |b, v| {
            ntt_group_stages_reg(b, v, t + 1, bg, start, p_low, w, log_n)
        })
    }
}

/// Register-tiled radix-2 DIT NTT of size `2^log_n` over BabyBear.
///
/// Same interface and result as [`ntt_module`], but the `log_n` butterfly
/// stages are grouped into 13-bit register tiles under `#[grid(threads =
/// 512)]` with `#[par((th, s) -> th * 16 + s)]`, so each of the 512 threads
/// owns 16 contiguous logical elements. Inside one group the compiler
/// resolves partners `j ^ half` at register level for `half < 512` (Slot
/// for `half < 16`, warp Shuffle for `16 <= half < 512`) and through a
/// shared-memory mirror for `half >= 512`; those mirrors have disjoint live
/// ranges, so [`plan_shared_mem`] packs them into one reusable region. Any
/// leftover `< 13`-bit tail is split further via the shared-style scheme of
/// [`ntt_shared_module`].
///
/// [`plan_shared_mem`]: crate::passes::plan_shared_mem
pub fn ntt_reg_module(log_n: usize) -> Module {
    assert!(log_n >= 1, "NTT size must be at least 2");
    let n = 1usize << log_n;
    let n_reg = log_n / NTT_REG_BITS;
    let leftover = log_n - n_reg * NTT_REG_BITS;
    let mut bits = vec![NTT_REG_BITS; n_reg];
    if leftover > 0 {
        // The leftover runs shared-style; splitting it further keeps each
        // tile within the shared-memory-per-block budget.
        bits.extend(split_bits(leftover));
    }
    let groups = bits.len();
    let starts: Vec<usize> = bits
        .iter()
        .scan(0, |acc, &width| {
            let s = *acc;
            *acc += width;
            Some(s)
        })
        .collect();

    let mut b = IRBuilder::new();
    let a = b.input("a", ScalarType::BabyBear, vec![n]);
    let w = b.input("w", ScalarType::BabyBear, vec![n / 2]);

    let mut pi = starts.clone();
    let mut prev = a;
    for g in 0..groups {
        let (bg, start) = (bits[g], starts[g]);
        let tile = 1usize << bg;
        let is_reg = bg == NTT_REG_BITS;
        let n_blocks = n >> bg;
        let threads = if is_reg { Some(NTT_REG_THREADS) } else { None };
        let group = b.compute_with(n_blocks, None, None, threads, |b, i| {
            // The gather uses the default (identity) par regardless of
            // the group being register-tiled — that gives contiguous
            // per-thread writes into the tile, and layout_infer inserts
            // the shared-memory bounce that any subsequent
            // `#[par((th, s) -> ...)]` reader needs. The register-first
            // default is what makes the bounce a *single* mirror instead
            // of forcing the whole tile back to shared.
            let gather = if g == 0 {
                let hi_bits = log_n - bg;
                let sc = 1usize << hi_bits;
                b.compute(
                    tile,
                    |b, j| kernel!(b, a[bitrev_expr(j, bg) * #sc + bitrev_expr(i, hi_bits)]),
                )
            } else {
                let pg = 1usize << pi[g];
                let pgb = 1usize << (pi[g] + bg);
                let cols = bits[g - 1];
                b.compute(tile, |b, j| {
                    let addr = kernel!(b, i / #pg * #pgb + i % #pg + j * #pg);
                    index_flat2(b, prev, addr, cols)
                })
            };
            let mut p_low = b.const_u32(0);
            for k in 0..g {
                let (pos, width) = (pi[k], bits[k]);
                let scale = 1usize << starts[k];
                p_low = kernel!(b, p_low + extract_bits(i, pos, width) * #scale);
            }
            b.bind(gather, |b, tile_var| {
                if is_reg {
                    ntt_group_stages_reg(b, tile_var, 1, bg, start, p_low, w, log_n)
                } else {
                    ntt_group_stages(b, tile_var, 1, bg, start, p_low, w, log_n)
                }
            })
        });
        prev = b.let_bound(group);
        for k in 0..groups {
            if k != g && pi[k] < pi[g] {
                pi[k] += bg;
            }
        }
        pi[g] = 0;
    }

    if groups > 1 {
        let restored = b.compute(n, |b, p| {
            let mut addr = b.const_u32(0);
            for k in 0..groups {
                let (pos, width) = (starts[k], bits[k]);
                let scale = 1usize << pi[k];
                addr = kernel!(b, addr + extract_bits(p, pos, width) * #scale);
            }
            index_flat2(b, prev, addr, bits[groups - 1])
        });
        prev = b.let_bound(restored);
    }

    b.finish(format!("ntt_reg_{n}"), prev)
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
            let zero = b.const_u32(0);
            let mut state = [zero; 16];
            for j in 0..8 {
                state[j] = kernel!(b, prev[i * 2, #j]);
                state[8 + j] = kernel!(b, prev[i * 2 + 1, #j]);
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
