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

/// Base-`W` digit width for windowed twiddle multiplication. `W = 2^5 = 32`
/// matches supra's `WINDOW_SIZE`; changing this changes the trade-off
/// between the number of per-lookup multiplies (`n_windows`) and the size
/// of the `partial_twiddles` table (`n_windows * W`).
const NTT_LG_WINDOW: usize = 5;
const NTT_WINDOW: usize = 1 << NTT_LG_WINDOW;

/// Number of base-`W` digits needed to encode any twiddle index in
/// `[0, n/2)`. `log_n <= 1` has a single trivial twiddle `omega^0 = 1`, so
/// one window covers it.
fn ntt_n_windows(log_n: usize) -> usize {
    if log_n <= 1 {
        1
    } else {
        (log_n - 1).div_ceil(NTT_LG_WINDOW)
    }
}

/// Windowed twiddle table for [`ntt_supra_module`]. Layout is
/// `partial_twiddles[wi, wv]` in row-major order (shape `[n_windows, W]`),
/// with `partial_twiddles[wi][wv] = omega^(wv * W^wi)` — one row per
/// base-`W` digit, so `omega^k` is the product of the `n_windows` entries
/// keyed by `k`'s base-`W` digits. Canonical representation.
pub fn ntt_partial_twiddles(log_n: usize) -> Vec<u32> {
    assert!(log_n >= 1);
    let n_windows = ntt_n_windows(log_n);
    let mut base = BabyBear::two_adic_generator(log_n);
    let mut out = Vec::with_capacity(n_windows * NTT_WINDOW);
    for _ in 0..n_windows {
        let mut acc = BabyBear::ONE;
        for _ in 0..NTT_WINDOW {
            out.push(acc.as_canonical_u32());
            acc *= base;
        }
        // base_{wi+1} = base_wi ^ W = (base_wi)^(2^LG_WINDOW), which we
        // reach with `LG_WINDOW` squarings — cheaper than looking `acc`
        // up as `base^W` since we already discarded that intermediate.
        for _ in 0..NTT_LG_WINDOW {
            base = base * base;
        }
    }
    out
}

/// Source of bits contributing to the NTT stage's *pre-step* twiddle
/// index `pre_idx = p_low + k * 2^start` (with `k = j & (2^h - 1)`).
/// `node` is a scalar in which `n_bits` bits starting at bit
/// `src_bit_lo` map to `pre_idx` bits `[pre_bit_lo, pre_bit_lo +
/// n_bits)`. The actual twiddle index is `idx = pre_idx * 2^log_step`,
/// so the mapping to `idx` bits is a per-stage shift by `log_step` that
/// [`window_digit`] applies. Ranges are pairwise disjoint in `pre_idx`
/// by construction (processed fields have non-overlapping `starts[k]`
/// bit slots and the running `j` sits at `[start, start + h)` above
/// them), so window digits are just concatenations of aligned slices.
///
/// Passing raw bit-fields of `i` and `j` instead of composing them into
/// a runtime `p_low = sum(extract_bits(i, pi[k], bits[k]) * 2^starts[k])`
/// integer keeps every operand in Quast's normalized form a `x % c` /
/// `x / c` of a fresh sym — which fold_rems then recognizes cleanly.
/// The composed form produces cross-terms like `-65535 * (i / 256)`
/// that Quast can't fold against the paired `256 * i`, so `layout_infer`
/// refuses the read as "operand may be negative".
#[derive(Clone, Copy)]
struct TwiddleSource {
    node: NodeId,
    src_bit_lo: usize,
    n_bits: usize,
    pre_bit_lo: usize,
}

/// The `wi`-th base-`W` digit of `idx = pre_idx * 2^log_step`,
/// with `pre_idx` decomposed as `sources`. See [`TwiddleSource`].
/// Returns `None` when the window is guaranteed to be all zero — either
/// entirely below `log_step`'s zero-fill or entirely above every source's
/// contribution — so callers can drop the lookup + multiply-by-1 pair
/// instead of loading `partial_twiddles[wi, 0]` for a wasted `* 1`.
fn window_digit(
    b: &mut IRBuilder,
    sources: &[TwiddleSource],
    log_step: usize,
    wi: usize,
) -> Option<NodeId> {
    let win_lo_abs = wi * NTT_LG_WINDOW;
    let win_hi_abs = win_lo_abs + NTT_LG_WINDOW;
    // Window range in `pre_idx` coordinates. Bits below `log_step` in
    // `idx` are always zero (they're the `step` shift's low zero-fill),
    // so a window that starts below `log_step` gets its low
    // `log_step - win_lo_abs` bits from nothing.
    if win_hi_abs <= log_step {
        return None;
    }
    let win_lo = win_lo_abs.saturating_sub(log_step);
    let win_hi = win_hi_abs - log_step;
    let mut acc: Option<NodeId> = None;
    for src in sources {
        let src_pre_hi = src.pre_bit_lo + src.n_bits;
        let lo = win_lo.max(src.pre_bit_lo);
        let hi = win_hi.min(src_pre_hi);
        if lo >= hi {
            continue;
        }
        let src_off = src.src_bit_lo + (lo - src.pre_bit_lo);
        let field = extract_bits(b, src.node, src_off, hi - lo);
        // Position of these bits within `idx`: they start at
        // `lo + log_step` and the window starts at `win_lo_abs`.
        let shift = (lo + log_step) - win_lo_abs;
        let contribution = if shift == 0 {
            field
        } else {
            let mul = 1u32 << shift;
            kernel!(b, field * #mul)
        };
        acc = Some(match acc {
            None => contribution,
            Some(prev) => kernel!(b, prev + contribution),
        });
    }
    acc
}

/// DSL helper: emit the product-of-`n_windows`-lookups form of
/// `omega^idx` against a `[n_windows, W]` windowed twiddle table.
/// `sources` describes how the stage's *pre-step* index decomposes
/// into aligned bit slices of runtime symbols (see [`TwiddleSource`]);
/// `log_step` is the per-stage shift into full `idx`. Windows that
/// [`window_digit`] proves are always zero (below `log_step`'s zero-fill
/// or above every source's contribution) are elided, so early stages
/// with a small `start + t - 1` bit exponent avoid the `* 1` lookups
/// that a fixed unroll count would emit.
fn windowed_twiddle_lookup(
    b: &mut IRBuilder,
    partial_twiddles: NodeId,
    sources: &[TwiddleSource],
    log_step: usize,
    n_windows: usize,
) -> NodeId {
    let mut acc: Option<NodeId> = None;
    for wi in 0..n_windows {
        let Some(digit) = window_digit(b, sources, log_step, wi) else {
            continue;
        };
        let pt = kernel!(b, partial_twiddles[#wi, digit]);
        acc = Some(match acc {
            None => pt,
            Some(prev) => kernel!(b, prev * pt),
        });
    }
    // No source contributed to any window: `omega^0 = 1`. Callers
    // (e.g. stage 1 of group 0) still need a valid factor for their
    // multiply chain.
    acc.unwrap_or_else(|| kernel!(b, partial_twiddles[0, 0]))
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

/// Twiddle sources for the current NTT group's already-processed
/// fields plus the running-stage's `j` (in *pre-step* coordinates —
/// `windowed_twiddle_lookup` shifts by `log_step` per-stage). `p_fields`
/// is one entry per processed field `k`: `TwiddleSource { node: i,
/// src_bit_lo: pi[k], n_bits: bits[k], pre_bit_lo: starts[k] }`. The
/// stage appends its own `(j, 0, t - 1, start)` slot before building
/// each window digit.
fn stage_sources(
    p_fields: &[TwiddleSource],
    j: NodeId,
    start: usize,
    h: usize,
) -> Vec<TwiddleSource> {
    let mut v: Vec<TwiddleSource> = p_fields.to_vec();
    if h > 0 {
        v.push(TwiddleSource {
            node: j,
            src_bit_lo: 0,
            n_bits: h,
            pre_bit_lo: start,
        });
    }
    v
}

/// Chains the butterfly stages of one register-tiled NTT group under
/// `#[par((th, s) -> th * per_thread + s)]`, matching the supra
/// `_CT_NTT` layout: two elements per polynomial per thread (or
/// `2 * z_count` when batching). `per_thread` is passed as an `i64` so
/// the par-map's `mul_c` sees the constant directly. `w_partial` is
/// the `[n_windows, W]` windowed twiddle table; `n_windows` fixes the
/// per-butterfly lookup unroll count. Windowed lookup trades the
/// single-load twiddle for `n_windows` loads and `n_windows - 1`
/// multiplies, saving the memory bandwidth of the flat `[n/2]` table
/// (32 MB at `log_n = 24`) and letting the tiny `n_windows * W` table
/// stay warm in cache — matching supra's twiddle strategy.
#[allow(clippy::too_many_arguments)]
fn ntt_group_stages_supra(
    b: &mut IRBuilder,
    prev: NodeId,
    t: usize,
    bg: usize,
    start: usize,
    p_fields: &[TwiddleSource],
    w_partial: NodeId,
    log_n: usize,
    per_thread: i64,
    n_windows: usize,
) -> NodeId {
    let tile = 1usize << bg;
    let m = 1usize << t;
    let half = 1usize << (t - 1);
    let log_step = log_n - start - t;
    let par = b.par_map(|th, s, _| th.mul_c(per_thread).add(s));
    // Stage 1 of a group with no already-processed fields has a trivial
    // (omega^0 = 1) twiddle — skip the lookup and its multiply.
    let trivial_twiddle = p_fields.is_empty() && t == 1;
    let stage = b.compute_with(tile, None, Some(par), None, |b, j| {
        if trivial_twiddle {
            kernel!(b,
                let own = prev[j];
                let partner = prev[j + #half - j % #m / #half * #m];
                let lo = j % #m < #half;
                if lo then own + partner else partner - own
            )
        } else {
            let sources = stage_sources(p_fields, j, start, t - 1);
            let w_val = windowed_twiddle_lookup(b, w_partial, &sources, log_step, n_windows);
            kernel!(b,
                let own = prev[j];
                let partner = prev[j + #half - j % #m / #half * #m];
                let lo = j % #m < #half;
                if lo then own + w_val * partner else partner - w_val * own
            )
        }
    });
    if t == bg {
        stage
    } else {
        b.bind(stage, |b, v| {
            ntt_group_stages_supra(
                b,
                v,
                t + 1,
                bg,
                start,
                p_fields,
                w_partial,
                log_n,
                per_thread,
                n_windows,
            )
        })
    }
}

/// Shared-memory leftover stages for [`ntt_supra_module`] — same
/// structure as [`ntt_group_stages`] but reads its twiddle through the
/// windowed table so supra's module only needs one twiddle input.
#[allow(clippy::too_many_arguments)]
fn ntt_group_stages_supra_shared(
    b: &mut IRBuilder,
    prev: NodeId,
    t: usize,
    bg: usize,
    start: usize,
    p_fields: &[TwiddleSource],
    w_partial: NodeId,
    log_n: usize,
    n_windows: usize,
) -> NodeId {
    let tile = 1usize << bg;
    let m = 1usize << t;
    let half = 1usize << (t - 1);
    let log_step = log_n - start - t;
    // Same trivial-twiddle short-circuit as the register-tiled helper.
    let trivial_twiddle = p_fields.is_empty() && t == 1;
    // Shared-tile stages don't run under a par-map, so the outer `j`
    // becomes the identity thread index. We still build the twiddle
    // through the same windowed helper as the register stages: the
    // computed digit expressions read only bit-fields of `i` and `j`,
    // each of which has a crisp bound for `layout_infer`.
    let stage = b.compute(tile, |b, j| {
        if trivial_twiddle {
            kernel!(b,
                let jm = j % #m;
                let lo = jm < #half;
                let base = j - jm / #half * #half;
                let u = prev[base];
                let v = prev[base + #half];
                if lo then u + v else u - v
            )
        } else {
            let sources = stage_sources(p_fields, j, start, t - 1);
            let w_val = windowed_twiddle_lookup(b, w_partial, &sources, log_step, n_windows);
            kernel!(b,
                let jm = j % #m;
                let lo = jm < #half;
                let base = j - jm / #half * #half;
                let u = prev[base];
                let v = prev[base + #half];
                let tw = v * w_val;
                if lo then u + tw else u - tw
            )
        }
    });
    if t == bg {
        stage
    } else {
        b.bind(stage, |b, v| {
            ntt_group_stages_supra_shared(
                b,
                v,
                t + 1,
                bg,
                start,
                p_fields,
                w_partial,
                log_n,
                n_windows,
            )
        })
    }
}

/// Radix-2 DIT NTT modeled directly on supra's `_CT_NTT<z_count,
/// coalesced>` (`crates/cuda-backend/cuda/supra/ntt.cu`) but generated
/// end-to-end by the DSL pipeline. Each block covers
/// `radix = 1 + log2(nthreads)` bits of the transform with
/// `per_thread = 2 * z_count` elements per lane — mirroring supra's
/// `fr_t r[2][z_count]` register slab. Under the
/// `(th, s) -> th * per_thread + s` par layout:
///
/// - Butterfly partners with `half < per_thread` sit in the same thread's register slots (Slot
///   conversion, no data movement).
/// - `per_thread <= half < per_thread * 32` cross lanes within a warp;
///   [`layout_infer`](crate::passes::layout_infer) resolves them as `__shfl_sync` shuffles — the
///   exact equivalent of supra's `shfl_bfly` (`__shfl_xor_sync`).
/// - Larger `half` crosses warps and falls back through a shared-memory mirror, matching supra's
///   `xchg[threadIdx.x ^ laneMask]` path.
///
/// A leftover `< radix`-bit tail below the last full register group is
/// processed with the shared-memory scheme of [`ntt_shared_module`]
/// (same helper as [`ntt_reg_module`]), and a final flat kernel
/// restores natural order when there is more than one group.
///
/// Parameters:
///
/// - `log_n`: log₂ of the transform size.
/// - `nthreads`: threads per block. Must be a power of two ≤ 1024. Together with `z_count` this
///   fixes the group width to `radix + log2(z_count) = log2(nthreads) + 1 + log2(z_count)` bits.
/// - `z_count`: butterfly pairs per thread, matching supra's template `Z_COUNT`. Must be a power of
///   two ≥ 1. `per_thread = 2 * z_count` and each register-tiled group covers `1 + log2(nthreads) +
///   log2(z_count)` bits of the transform — the same block-level width supra gets by dividing
///   `num_blocks` by `Z_COUNT`. A single flat 1D input carries the elements; there is no separate
///   batch axis to model since the DSL indexes `a[i]` uniformly.
/// - `coalesced`: accepted but currently a no-op. Our gather compute uses the default identity par
///   so consecutive lanes already read consecutive addresses — supra's `coalesced_load` +
///   `transpose<z_count>` is the same net effect that layout-conversion (Slot/Shuffle
///   classification) achieves automatically here.
pub fn ntt_supra_module(log_n: usize, nthreads: usize, z_count: usize, coalesced: bool) -> Module {
    assert!(log_n >= 1, "NTT size must be at least 2");
    assert!(
        nthreads.is_power_of_two() && (1..=1024).contains(&nthreads),
        "nthreads must be a power of two in 1..=1024"
    );
    assert!(
        z_count.is_power_of_two() && z_count >= 1,
        "z_count must be a power of two >= 1"
    );
    // `coalesced` is degenerate for `z_count == 1` (supra's
    // `coalesced_load<1>` reduces to a plain load); for `z_count > 1`
    // our gather compute already uses the default identity par, so
    // consecutive lanes read consecutive addresses — the same
    // coalescing supra achieves through `coalesced_load` +
    // `transpose<z_count>`. The flag is accepted to preserve API
    // parity with supra's template.
    let _ = coalesced;
    let per_thread = 2i64 * z_count as i64;
    let radix_bits = 1 + nthreads.trailing_zeros() as usize + z_count.trailing_zeros() as usize;
    let n = 1usize << log_n;

    // Same group-splitting scheme as `ntt_reg_module`: a run of full
    // `radix_bits`-bit register groups followed by a shared-style
    // remainder, so any `log_n` compiles.
    let n_reg = log_n / radix_bits;
    let leftover = log_n - n_reg * radix_bits;
    let mut bits = vec![radix_bits; n_reg];
    if leftover > 0 {
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

    let n_windows = ntt_n_windows(log_n);
    let mut b = IRBuilder::new();
    let a = b.input("a", ScalarType::BabyBear, vec![n]);
    // Windowed twiddles: shape `[n_windows, W]` with
    // `w[wi, wv] = omega^(wv * W^wi)`. See [`ntt_partial_twiddles`].
    let w = b.input("w", ScalarType::BabyBear, vec![n_windows, NTT_WINDOW]);

    let mut pi = starts.clone();
    let mut prev = a;
    for g in 0..groups {
        let (bg, start) = (bits[g], starts[g]);
        let tile = 1usize << bg;
        let is_reg = bg == radix_bits;
        let n_blocks = n >> bg;
        let threads = if is_reg { Some(nthreads) } else { None };
        // Multi-group NTTs used to end with a full-N `restored` pass that
        // read `prev` at a permuted address and wrote natural order — pure
        // bandwidth. Fold it into the last group's store as a `#[scatter]`
        // that sends logical `(i, j)` to its natural-order slot directly.
        // The map inserts `j` at logical bits `[start, start + bg)` and
        // fills the other fields from `i`'s bit-fields at `pi[k]` (shifted
        // down by `bg` when they sit above the group's slot, since the
        // gather originally "opened" that hole).
        let scatter = if groups > 1 && g == groups - 1 {
            let pi_snap = pi.clone();
            let starts_snap = starts.clone();
            let bits_snap = bits.clone();
            let g_last = g;
            let bg_last = bg;
            let start_last = start;
            Some(b.scatter_map(2, Some(vec![n]), move |p, _cst| {
                let block_i = &p[0];
                let tile_j = &p[1];
                let mut expr = tile_j.mul_c(1i64 << start_last);
                for k in 0..pi_snap.len() {
                    if k == g_last {
                        continue;
                    }
                    let i_pos = if pi_snap[k] < pi_snap[g_last] {
                        pi_snap[k]
                    } else {
                        pi_snap[k] - bg_last
                    };
                    let field = block_i.floordiv(1i64 << i_pos).rem_c(1i64 << bits_snap[k]);
                    expr = expr.add(&field.mul_c(1i64 << starts_snap[k]));
                }
                vec![expr]
            }))
        } else {
            None
        };
        let group = b.compute_with(n_blocks, scatter, None, threads, |b, i| {
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
            // One `TwiddleSource` per already-processed field, mapping
            // its physical position in `i` to its logical position
            // (`starts[k]`) in the *pre-step* twiddle index. The stage
            // helper appends its own `j`-slot and applies the per-stage
            // `log_step` shift; window_digit then scans the aligned
            // bit-fields for each base-`W` digit without ever composing
            // a runtime `p_low` integer.
            let p_fields: Vec<TwiddleSource> = (0..g)
                .map(|k| TwiddleSource {
                    node: i,
                    src_bit_lo: pi[k],
                    n_bits: bits[k],
                    pre_bit_lo: starts[k],
                })
                .collect();
            b.bind(gather, |b, tile_var| {
                if is_reg {
                    ntt_group_stages_supra(
                        b, tile_var, 1, bg, start, &p_fields, w, log_n, per_thread, n_windows,
                    )
                } else {
                    ntt_group_stages_supra_shared(
                        b, tile_var, 1, bg, start, &p_fields, w, log_n, n_windows,
                    )
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

    b.finish(format!("ntt_supra_{n}_t{nthreads}_z{z_count}"), prev)
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
