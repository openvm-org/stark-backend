//! The gpu.rs end-to-end tests, rewritten with the `kernel!` proc macro.
//! Requires a CUDA GPU.

use crypto_compiler::{
    compile_and_load,
    ir::{IRBuilder, NodeId, ScalarType},
    kernel,
    kernels::{ntt_twiddles, poseidon2_permutation, Poseidon2Constants},
    runtime::CompileOptions,
};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::GpuDeviceCtx,
};
use p3_baby_bear::{default_babybear_poseidon2_16, BabyBear};
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{
    extension::BinomialExtensionField, BasedVectorSpace, PrimeCharacteristicRing, PrimeField32,
};
use p3_symmetric::Permutation;

/// Degree-4 binomial extension of BabyBear, matching the DSL's
/// `ScalarType::FpExt`.
type EF = BinomialExtensionField<BabyBear, 4>;

/// Rebuild an `EF` value from the DSL's canonical 4×`u32` layout
/// (`(a0, a1, a2, a3)` for `a0 + a1 x + a2 x^2 + a3 x^3`).
fn ef_from_u32s(raw: &[u32]) -> EF {
    let coeffs: [BabyBear; 4] = [bb(raw[0]), bb(raw[1]), bb(raw[2]), bb(raw[3])];
    EF::from_basis_coefficients_slice(&coeffs).unwrap()
}

/// Serialize an `EF` back to the DSL layout.
fn ef_to_u32s(v: EF) -> [u32; 4] {
    let s: &[BabyBear] = v.as_basis_coefficients_slice();
    [
        s[0].as_canonical_u32(),
        s[1].as_canonical_u32(),
        s[2].as_canonical_u32(),
        s[3].as_canonical_u32(),
    ]
}

/// Deterministic random `FpExt` elements as a flat `4n` u32 buffer.
fn pseudo_ext_elems(n: usize, seed: u64) -> Vec<u32> {
    pseudo_field_elems(4 * n, seed)
}

/// Assemble an FpExt DSL value from its four BabyBear coefficients using
/// `lift_fpext` and multiplications by the extension basis.
fn make_ext(b: &mut IRBuilder, a0: NodeId, a1: NodeId, a2: NodeId, a3: NodeId) -> NodeId {
    let x1 = b.const_fpext([0, 1, 0, 0]);
    let x2 = b.const_fpext([0, 0, 1, 0]);
    let x3 = b.const_fpext([0, 0, 0, 1]);
    let l0 = b.lift_fpext(a0);
    let l1 = b.lift_fpext(a1);
    let l2 = b.lift_fpext(a2);
    let l3 = b.lift_fpext(a3);
    let t1 = b.mul(l1, x1);
    let t2 = b.mul(l2, x2);
    let t3 = b.mul(l3, x3);
    let s01 = b.add(l0, t1);
    let s23 = b.add(t2, t3);
    b.add(s01, s23)
}

const P: u64 = 2013265921;

/// Deterministic pseudo-random canonical BabyBear elements (splitmix64).
fn pseudo_field_elems(n: usize, seed: u64) -> Vec<u32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            (z % P) as u32
        })
        .collect()
}

fn bb(x: u32) -> BabyBear {
    BabyBear::new(x)
}

/// Compiles `module`, binds inputs/outputs, runs it and returns the outputs.
/// Dumps both IR levels and the generated CUDA under
/// `target/ir-dumps/gpu_macro/`.
fn run_module(module: crypto_compiler::ir::Module, inputs: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let ctx = GpuDeviceCtx::for_current_device().unwrap();
    let options = CompileOptions {
        dump_ir: Some(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../target/ir-dumps/gpu_macro"
            )
            .into(),
        ),
        ..Default::default()
    };
    let mut km = compile_and_load(module, &options).unwrap();
    assert_eq!(km.num_inputs(), inputs.len());

    let in_bufs: Vec<DeviceBuffer<u32>> = inputs
        .iter()
        .map(|data| data.as_slice().to_device_on(&ctx).unwrap())
        .collect();
    for (i, buf) in in_bufs.iter().enumerate() {
        assert_eq!(km.input_size(i), buf.len() * 4, "input {i} size");
        km.set_input(i, buf).unwrap();
    }
    let out_bufs: Vec<DeviceBuffer<u32>> = (0..km.num_outputs())
        .map(|i| DeviceBuffer::with_capacity_on(km.output_size(i) / 4, &ctx))
        .collect();
    for (i, buf) in out_bufs.iter().enumerate() {
        km.set_output(i, buf).unwrap();
    }
    km.ensure_scratch(&ctx);
    km.run(&ctx.stream).unwrap();
    out_bufs
        .iter()
        .map(|b| b.to_host_on(&ctx).unwrap())
        .collect()
}

/// Zero `BabyBear` constant, for use inside `kernel!` bodies (`0bb` is a
/// Rust lex error since `0b` is the binary-literal prefix).
fn zbb(b: &mut IRBuilder) -> NodeId {
    b.const_field(0)
}

/// Lift a BabyBear value to `FpExt` as `(x, 0, 0, 0)`. Callable from
/// inside `kernel!` bodies via the macro's `Call` splice (`__cc_b` is
/// prepended automatically).
fn to_fpext(b: &mut IRBuilder, x: NodeId) -> NodeId {
    b.lift_fpext(x)
}

/// The bit-reversal of `i` over `log_n` bits, as a DSL expression.
fn bitrev_index(b: &mut IRBuilder, i: NodeId, log_n: usize) -> NodeId {
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
    rev
}

#[test]
fn macro_ntt_matches_p3_radix2dit() {
    let log_n = 12usize;
    let n = 1usize << log_n;
    let coeffs = pseudo_field_elems(n, 1);
    let twiddles = ntt_twiddles(log_n);

    let mut ib = IRBuilder::new();
    let a = ib.input("a", ScalarType::BabyBear, vec![n]);
    let w = ib.input("w", ScalarType::BabyBear, vec![n / 2]);

    let bitrev = kernel!(ib, compute[n] | i | { a[bitrev_index(i, log_n)] });
    let mut prev = ib.let_bound(bitrev);

    for s in 1..=log_n {
        let m = 1usize << s;
        let half = m / 2;
        let step = n / m;
        let stage = kernel!(ib,
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
        prev = ib.let_bound(stage);
    }
    let module = ib.finish("ntt_macro", prev);

    let outs = run_module(module, &[coeffs.clone(), twiddles]);
    assert_eq!(outs.len(), 1);

    let input_f: Vec<BabyBear> = coeffs.iter().map(|&c| bb(c)).collect();
    let want: Vec<u32> = Radix2Dit::default()
        .dft(input_f)
        .iter()
        .map(|x| x.as_canonical_u32())
        .collect();
    assert_eq!(outs[0], want);
}

/// `compress(l, r) = perm(l || r)[0..8]`, reading the pair `(2i, 2i+1)` of
/// digests from `prev`. Called from `kernel!` with the builder prepended.
fn compress(b: &mut IRBuilder, prev: NodeId, i: NodeId, c: &Poseidon2Constants) -> NodeId {
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
}

#[test]
fn macro_merkle_tree_matches_p3_poseidon2() {
    let log_leaves = 8usize;
    let n_leaves = 1usize << log_leaves;
    let leaf_data = pseudo_field_elems(n_leaves * 8, 2);
    let c = Poseidon2Constants::p3_default();
    let constants = &c;

    let mut ib = IRBuilder::new();
    let leaves = ib.input("leaves", ScalarType::BabyBear, vec![n_leaves, 8]);

    let mut prev = leaves;
    let mut layers = Vec::new();
    for lvl in 0..log_leaves {
        let n = n_leaves >> (lvl + 1);
        let layer = kernel!(ib, compute[n] | i | { compress(prev, i, constants) });
        prev = ib.let_bound(layer);
        layers.push(prev);
    }
    let out = ib.tuple(&layers);
    let module = ib.finish("merkle_macro", out);

    let outs = run_module(module, std::slice::from_ref(&leaf_data));
    assert_eq!(outs.len(), log_leaves);

    let perm = default_babybear_poseidon2_16();
    let mut prev: Vec<[BabyBear; 8]> = leaf_data
        .chunks(8)
        .map(|c| std::array::from_fn(|i| bb(c[i])))
        .collect();
    for (lvl, got) in outs.iter().enumerate() {
        let next: Vec<[BabyBear; 8]> = prev
            .chunks(2)
            .map(|pair| {
                let mut state = [BabyBear::ZERO; 16];
                state[..8].copy_from_slice(&pair[0]);
                state[8..].copy_from_slice(&pair[1]);
                let out = perm.permute(state);
                out[..8].try_into().unwrap()
            })
            .collect();
        let want: Vec<u32> = next
            .iter()
            .flatten()
            .map(|x| x.as_canonical_u32())
            .collect();
        assert_eq!(got, &want, "merkle layer {lvl}");
        prev = next;
    }
    assert_eq!(prev.len(), 1, "last layer is the root");
}

#[test]
fn macro_matmul_grid_block_reduce() {
    let (n, m, k) = (33usize, 47usize, 29usize);
    let a = pseudo_field_elems(n * k, 3);
    let b = pseudo_field_elems(k * m, 4);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n, k]);
    let b_in = ib.input("b", ScalarType::BabyBear, vec![k, m]);
    let body = kernel!(ib,
        compute [n] |i| {
            compute [m] |j| {
                reduce [k] |t| { a_in[i, t] * b_in[t, j] }
            }
        }
    );
    let module = ib.finish("matmul_macro", body);

    let outs = run_module(module, &[a.clone(), b.clone()]);

    let mut want = vec![0u32; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut acc = BabyBear::ZERO;
            for t in 0..k {
                acc += bb(a[i * k + t]) * bb(b[t * m + j]);
            }
            want[i * m + j] = acc.as_canonical_u32();
        }
    }
    assert_eq!(outs[0], want);
}

#[test]
fn macro_nested_compute_flattening() {
    let (bs, n, m) = (5usize, 12usize, 9usize);
    let x = pseudo_field_elems(bs * n * m, 5);
    let y = pseudo_field_elems(bs * m * n, 6);

    let mut ib = IRBuilder::new();
    let x_in = ib.input("x", ScalarType::BabyBear, vec![bs, n, m]);
    let y_in = ib.input("y", ScalarType::BabyBear, vec![bs, m, n]);
    let body = kernel!(ib,
        compute [bs] |bi| {
            compute [n] |i| {
                compute [m] |j| { x_in[bi, i, j] * 2bb + y_in[bi, j, i] }
            }
        }
    );
    let module = ib.finish("batched_add_t_macro", body);

    let outs = run_module(module, &[x.clone(), y.clone()]);

    let mut want = vec![0u32; bs * n * m];
    for b in 0..bs {
        for i in 0..n {
            for j in 0..m {
                let v = bb(x[(b * n + i) * m + j]) * bb(2) + bb(y[(b * m + j) * n + i]);
                want[(b * n + i) * m + j] = v.as_canonical_u32();
            }
        }
    }
    assert_eq!(outs[0], want);
}

#[test]
fn macro_tuple_outputs_and_scalar_lets() {
    let n = 1000usize;
    let a = pseudo_field_elems(n, 7);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let pair = kernel!(
        ib,
        compute[n] | i | {
            let v = a_in[i];
            (v * v + v, v + v)
        }
    );
    let module = ib.finish("square_plus_macro", pair);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 2);

    let want0: Vec<u32> = a
        .iter()
        .map(|&c| (bb(c) * bb(c) + bb(c)).as_canonical_u32())
        .collect();
    let want1: Vec<u32> = a
        .iter()
        .map(|&c| (bb(c) + bb(c)).as_canonical_u32())
        .collect();
    assert_eq!(outs[0], want0);
    assert_eq!(outs[1], want1);
}

#[test]
fn macro_top_level_reduce_dot_product() {
    let k = 1234usize;
    let a = pseudo_field_elems(k, 8);
    let b = pseudo_field_elems(k, 9);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![k]);
    let b_in = ib.input("b", ScalarType::BabyBear, vec![k]);
    let body = kernel!(ib, reduce[k] | t | { a_in[t] * b_in[t] });
    let module = ib.finish("dot_macro", body);

    let outs = run_module(module, &[a.clone(), b.clone()]);

    let mut acc = BabyBear::ZERO;
    for t in 0..k {
        acc += bb(a[t]) * bb(b[t]);
    }
    assert_eq!(outs[0], vec![acc.as_canonical_u32()]);
}

/// A reduce whose body never touches memory stays inline in its par as a
/// sequential SSA loop (no register hoisting): `out[i] = a[i] + sum_j (i+j)^2`
/// in u32 arithmetic.
#[test]
fn macro_load_free_reduce_inline_loop() {
    let (n, k) = (517usize, 37usize);
    let a = pseudo_field_elems(n, 10);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::U32, vec![n]);
    let body = kernel!(
        ib,
        compute[n] | i | {
            let s = reduce[k] | j | { (i + j) * (i + j) };
            a_in[i] + s
        }
    );
    let module = ib.finish("inline_reduce_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));

    let want: Vec<u32> = (0..n)
        .map(|i| {
            let s: u32 = (0..k).map(|j| ((i + j) * (i + j)) as u32).sum();
            a[i].wrapping_add(s)
        })
        .collect();
    assert_eq!(outs[0], want);
}

/// Shared memory via nested compute: an inner `let buf = compute [t] ...`
/// materializes a per-block shared buffer. Threads then read `buf` at
/// permuted indices (reversal) and a reduce sums the whole tile — both
/// require the staged data to be visible across the block (`__syncthreads`).
#[test]
fn macro_shared_memory_tile() {
    let (blocks, t) = (33usize, 64usize);
    let n = blocks * t;
    let a = pseudo_field_elems(n, 11);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(ib,
        compute [blocks] |i| {
            let buf = compute [t] |j| { a_in[i * #t + j] };
            let total = reduce [t] |j| { buf[j] };
            compute [t] |j| { buf[#t - 1 - j] * 2bb + total }
        }
    );
    let module = ib.finish("shared_tile_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; n];
    for i in 0..blocks {
        let tile = &a[i * t..(i + 1) * t];
        let total: BabyBear = tile.iter().map(|&x| bb(x)).sum();
        for j in 0..t {
            want[i * t + j] = (bb(tile[t - 1 - j]) * bb(2) + total).as_canonical_u32();
        }
    }
    assert_eq!(outs[0], want);
}

/// `#[scatter(...)]`: results are stored through a bijective quasi-affine
/// map. `i -> (i % 4, i / 4)` scatters a length-12 vector into a `[4, 3]`
/// tensor (the transpose of the natural `[3, 4]` reshape).
#[test]
fn macro_scatter_reshape_transpose() {
    let n = 12usize;
    let a = pseudo_field_elems(n, 12);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(
        ib,
        #[scatter(i -> (i % 4, i / 4), [4, 3])]
        compute[n]
            | i
            | { a_in[i] * 2bb }
    );
    let module = ib.finish("scatter_transpose_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; n];
    for i in 0..n {
        want[(i % 4) * 3 + i / 4] = (bb(a[i]) * bb(2)).as_canonical_u32();
    }
    assert_eq!(outs[0], want);
}

/// `N == M` scatter with the output bounds omitted: a pure permutation.
/// Reversal needs constants and negation: `i -> #(n - 1) - i`.
#[test]
fn macro_scatter_reverse_permutation() {
    let n = 1000usize;
    let a = pseudo_field_elems(n, 13);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(
        ib,
        #[scatter(i -> #(n - 1) - i)]
        compute[n]
            | i
            | { a_in[i] + 1bb }
    );
    let module = ib.finish("scatter_reverse_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; n];
    for i in 0..n {
        want[n - 1 - i] = (bb(a[i]) + bb(1)).as_canonical_u32();
    }
    assert_eq!(outs[0], want);
}

/// Even/odd deinterleave as a 1-D quasi-affine permutation:
/// `i -> (i % 2) * (n / 2) + i / 2` (evens to the first half, odds to the
/// second).
#[test]
fn macro_scatter_deinterleave() {
    let n = 512usize;
    let a = pseudo_field_elems(n, 14);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(
        ib,
        #[scatter(i -> i % 2 * #(n / 2) + i / 2)]
        compute[n]
            | i
            | { a_in[i] * 3bb }
    );
    let module = ib.finish("scatter_deinterleave_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; n];
    for i in 0..n {
        want[i % 2 * (n / 2) + i / 2] = (bb(a[i]) * bb(3)).as_canonical_u32();
    }
    assert_eq!(outs[0], want);
}

/// Scatter over a nested compute: the map sees the full logical index tuple
/// `(i, j)` of the `[r, c]` output and transposes it into `[c, r]`.
#[test]
fn macro_scatter_nested_transpose() {
    let (r, c) = (10usize, 7usize);
    let x = pseudo_field_elems(r * c, 15);

    let mut ib = IRBuilder::new();
    let x_in = ib.input("x", ScalarType::BabyBear, vec![r, c]);
    let body = kernel!(ib,
        #[scatter((i, j) -> (j, i), [c, r])]
        compute [r] |i| { compute [c] |j| { x_in[i, j] * 2bb } }
    );
    let module = ib.finish("scatter_nested_transpose_macro", body);

    let outs = run_module(module, std::slice::from_ref(&x));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; r * c];
    for i in 0..r {
        for j in 0..c {
            want[j * r + i] = (bb(x[i * c + j]) * bb(2)).as_canonical_u32();
        }
    }
    assert_eq!(outs[0], want);
}

/// `#[scatter(...)]` on the trailing inner compute of a grid kernel: the map
/// permutes where each element of a block's row is stored, here reversing
/// within the row. `out[i*t + (t-1-j)] = a[i*t + j] * 2`.
#[test]
fn macro_scatter_inner_compute() {
    let (blocks, t) = (9usize, 48usize);
    let n = blocks * t;
    let a = pseudo_field_elems(n, 16);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(ib,
        compute [blocks] |i| {
            #[scatter(j -> #(t - 1) - j)]
            compute [t] |j| { a_in[i * #t + j] * 2bb }
        }
    );
    let module = ib.finish("scatter_inner_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; n];
    for i in 0..blocks {
        for j in 0..t {
            want[i * t + (t - 1 - j)] = (bb(a[i * t + j]) * bb(2)).as_canonical_u32();
        }
    }
    assert_eq!(outs[0], want);
}

/// `#[scatter(...)]` on a let-bound tile: the shared-memory writes go through
/// the map (a reversal), and readers index the physical layout. Reading slot
/// `j` crosses threads, so the derived barrier is still required.
#[test]
fn macro_scatter_shared_tile() {
    let (blocks, t) = (11usize, 64usize);
    let n = blocks * t;
    let a = pseudo_field_elems(n, 17);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(ib,
        compute [blocks] |i| {
            let buf = #[scatter(j -> #(t - 1) - j)]
            compute [t] |j| { a_in[i * #t + j] };
            compute [t] |j| { buf[j] + 1bb }
        }
    );
    let module = ib.finish("scatter_tile_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    // Slot j of the tile holds body(t-1-j), so out[i*t+j] = a[i*t+(t-1-j)]+1.
    let mut want = vec![0u32; n];
    for i in 0..blocks {
        for j in 0..t {
            want[i * t + j] = (bb(a[i * t + (t - 1 - j)]) + bb(1)).as_canonical_u32();
        }
    }
    assert_eq!(outs[0], want);
}

/// Register promotion: tiles written and read at the par's own index chain
/// through registers (no shared round-trip, no barrier). `t` is twice the
/// block size, so each thread owns two register slots.
#[test]
fn macro_register_promoted_tile_chain() {
    let (blocks, t) = (17usize, 512usize);
    let n = blocks * t;
    let a = pseudo_field_elems(n, 18);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(ib,
        compute [blocks] |i| {
            let b1 = compute [t] |j| { a_in[i * #t + j] * 2bb };
            let b2 = compute [t] |j| { b1[j] + 1bb };
            compute [t] |j| { b2[j] * b2[j] }
        }
    );
    let module = ib.finish("register_chain_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let want: Vec<u32> = a
        .iter()
        .map(|&c| {
            let v = bb(c) * bb(2) + bb(1);
            (v * v).as_canonical_u32()
        })
        .collect();
    assert_eq!(outs[0], want);
}

/// Warp-shuffle layout conversion: the tile is register-promoted and read
/// through a rotation of the five lane bits, `j -> j/32*32 + (j%16)*2 +
/// (j%32)/16`, which stays within each warp.
#[test]
fn macro_register_shuffle_lane_rotation() {
    let (blocks, t) = (13usize, 512usize);
    let n = blocks * t;
    let a = pseudo_field_elems(n, 19);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(ib,
        compute [blocks] |i| {
            let buf = compute [t] |j| { a_in[i * #t + j] * 2bb };
            compute [t] |j| { buf[j / 32 * 32 + j % 16 * 2 + j % 32 / 16] + 1bb }
        }
    );
    let module = ib.finish("shuffle_rotation_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; n];
    for i in 0..blocks {
        for j in 0..t {
            let src = j / 32 * 32 + (j % 16) * 2 + (j % 32) / 16;
            want[i * t + j] = (bb(a[i * t + src]) * bb(2) + bb(1)).as_canonical_u32();
        }
    }
    assert_eq!(outs[0], want);
}

/// Shuffle conversion whose sent register slot depends on the lane:
/// `j -> j + (j % 2) * 512` flips the sequential-slot bit for odd lanes
/// (lanes stay fixed, so each thread exchanges its own two slots).
#[test]
fn macro_register_shuffle_slot_xor() {
    let (blocks, t) = (7usize, 1024usize);
    let n = blocks * t;
    let a = pseudo_field_elems(n, 20);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(ib,
        compute [blocks] |i| {
            let buf = compute [t] |j| { a_in[i * #t + j] * 2bb };
            compute [t] |j| { buf[j + j % 2 * 512 - (j % 2 + j / 512) / 2 * 1024] + 1bb }
        }
    );
    let module = ib.finish("shuffle_slot_xor_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; n];
    for i in 0..blocks {
        for j in 0..t {
            let src = j ^ ((j % 2) * 512);
            want[i * t + j] = (bb(a[i * t + src]) * bb(2) + bb(1)).as_canonical_u32();
        }
    }
    assert_eq!(outs[0], want);
}

/// Mixed readers force the shared bounce: one reader matches the register
/// layout, the other reads a reversal (not XOR-linear), so the promoted tile
/// is mirrored back through shared memory for it.
#[test]
fn macro_register_tile_shared_bounce() {
    let (blocks, t) = (21usize, 64usize);
    let n = blocks * t;
    let a = pseudo_field_elems(n, 21);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(ib,
        compute [blocks] |i| {
            let buf = compute [t] |j| { a_in[i * #t + j] * 2bb };
            let c1 = compute [t] |j| { buf[j] + 1bb };
            compute [t] |j| { buf[#t - 1 - j] + c1[j] }
        }
    );
    let module = ib.finish("register_bounce_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; n];
    for i in 0..blocks {
        for j in 0..t {
            let rev = bb(a[i * t + (t - 1 - j)]) * bb(2);
            let same = bb(a[i * t + j]) * bb(2) + bb(1);
            want[i * t + j] = (rev + same).as_canonical_u32();
        }
    }
    assert_eq!(outs[0], want);
}

/// `#[grid(threads = 32)]` with a `#[par((th, s) -> th*16 + s)]` compute:
/// each of the 32 threads owns 16 contiguous logical indices, while the
/// plain gather tile keeps the identity (coalesced) schedule.
#[test]
fn macro_grid_threads_par_layout() {
    let (blocks, t) = (5usize, 512usize);
    let n = blocks * t;
    let a = pseudo_field_elems(n, 22);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(ib,
        #[grid(threads = 32)]
        compute [blocks] |i| {
            let buf = compute [t] |j| { a_in[i * #t + j] * 2bb };
            #[par((th, s) -> th * 16 + s)]
            compute [t] |j| { buf[j] + 1bb }
        }
    );
    let module = ib.finish("grid_par_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let want: Vec<u32> = a
        .iter()
        .map(|&c| (bb(c) * bb(2) + bb(1)).as_canonical_u32())
        .collect();
    assert_eq!(outs[0], want);
}

/// Butterfly-partner reads under a `#[par]` layout: with each thread holding
/// 16 contiguous elements, `j ^ 8` stays within a thread's own slots and
/// `j ^ 64` crosses lanes within a warp once the tile chain is
/// register-promoted; either way the result must match the host reference.
#[test]
fn macro_par_tile_partner_xor() {
    let (blocks, t) = (9usize, 512usize);
    let n = blocks * t;
    let a = pseudo_field_elems(n, 23);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(ib,
        #[grid(threads = 32)]
        compute [blocks] |i| {
            let buf = compute [t] |j| { a_in[i * #t + j] * 2bb };
            let c1 =
                #[par((th, s) -> th * 16 + s)]
                compute [t] |j| { buf[j] + buf[j + 8 - j % 16 / 8 * 16] };
            #[par((th, s) -> th * 16 + s)]
            compute [t] |j| { c1[j] * c1[j + 64 - j % 128 / 64 * 128] }
        }
    );
    let module = ib.finish("par_partner_xor_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; n];
    for i in 0..blocks {
        let c1: Vec<BabyBear> = (0..t)
            .map(|j| bb(a[i * t + j]) * bb(2) + bb(a[i * t + (j ^ 8)]) * bb(2))
            .collect();
        for j in 0..t {
            want[i * t + j] = (c1[j] * c1[j ^ 64]).as_canonical_u32();
        }
    }
    assert_eq!(outs[0], want);
}

/// Pack (array literal) syntax: each row of the output is `[v*v, v + 2]`.
#[test]
fn macro_pack_rows() {
    let n = 512usize;
    let a = pseudo_field_elems(n, 10);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(
        ib,
        compute[n] | i | {
            let v = a_in[i];
            [v * v, v + 2bb]
        }
    );
    let module = ib.finish("pack_rows_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = Vec::with_capacity(2 * n);
    for &c in &a {
        want.push((bb(c) * bb(c)).as_canonical_u32());
        want.push((bb(c) + bb(2)).as_canonical_u32());
    }
    assert_eq!(outs[0], want);
}

// ============================================================================
// cuda-backend kernel ports (Fp/BabyBear-only variants).
//
// Each test mirrors a `__global__` kernel from `crates/cuda-backend/cuda/src/`,
// re-expressed as a functional DSL kernel and checked against the same
// operation computed on the host.
// ============================================================================

/// Port of `matrix.cu::batch_expand_pad_kernel`: for a col-major bundle of
/// `poly_count` polynomials of length `in_size`, produce the padded version
/// of length `out_size` (zero-filled beyond `in_size`).
#[test]
fn macro_batch_expand_pad() {
    let (poly_count, in_size, out_size) = (5usize, 300usize, 512usize);
    let a = pseudo_field_elems(poly_count * in_size, 30);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![poly_count, in_size]);
    // `a_in[p, i]` would be OOB for `i >= in_size`; both branches of a select
    // evaluate their loads, so clamp with `i % in_size` (which is affine on the
    // constant `in_size`) to keep the discarded load in bounds.
    let body = kernel!(ib,
        compute [poly_count] |p| {
            compute [out_size] |i| {
                if i < #in_size then a_in[p, i % #in_size] else zbb()
            }
        }
    );
    let module = ib.finish("batch_expand_pad_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; poly_count * out_size];
    for p in 0..poly_count {
        for i in 0..in_size {
            want[p * out_size + i] = a[p * in_size + i];
        }
    }
    assert_eq!(outs[0], want);
}

/// Port of `matrix.cu::batch_expand_pad_wide_kernel`: the same padding op with
/// a `[width, padded_height]` output layout instead of `[polyCount, outSize]`.
/// The DSL body is identical; only the shapes change.
#[test]
fn macro_batch_expand_pad_wide() {
    let (width, height, padded_height) = (17usize, 12usize, 32usize);
    let a = pseudo_field_elems(width * height, 31);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![width, height]);
    let body = kernel!(ib,
        compute [width] |c| {
            compute [padded_height] |r| {
                if r < #height then a_in[c, r % #height] else zbb()
            }
        }
    );
    let module = ib.finish("batch_expand_pad_wide_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; width * padded_height];
    for c in 0..width {
        for r in 0..height {
            want[c * padded_height + r] = a[c * height + r];
        }
    }
    assert_eq!(outs[0], want);
}

/// Port of `matrix.cu::collapse_strided_matrix_kernel`: keep every `stride`-th
/// row of each column (`out[c, r] = in[c, r * stride]`).
#[test]
fn macro_collapse_strided_matrix() {
    let (width, height, stride) = (13usize, 20usize, 3usize);
    let lifted = height * stride;
    let a = pseudo_field_elems(width * lifted, 32);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![width, lifted]);
    let body = kernel!(ib,
        compute [width] |c| {
            compute [height] |r| { a_in[c, r * #stride] }
        }
    );
    let module = ib.finish("collapse_strided_matrix_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; width * height];
    for c in 0..width {
        for r in 0..height {
            want[c * height + r] = a[c * lifted + r * stride];
        }
    }
    assert_eq!(outs[0], want);
}

/// Port of `matrix.cu::lift_padded_matrix_evals_kernel`: the input matrix
/// already has `padded_height` rows; the first `height` rows carry data and
/// the rows `[height, lifted_height)` are overwritten with the periodic
/// extension `matrix[c, r % height]`. Rows `[lifted_height, padded_height)`
/// pass through unchanged. `lifted_height` must equal `height * stride`.
#[test]
fn macro_lift_padded_matrix_evals() {
    let (width, height, stride, padded_height) = (7usize, 5usize, 3usize, 20usize);
    let lifted_height = height * stride;
    assert!(lifted_height <= padded_height);
    let a = pseudo_field_elems(width * padded_height, 33);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![width, padded_height]);
    let body = kernel!(ib,
        compute [width] |c| {
            compute [padded_height] |r| {
                if r < #lifted_height then a_in[c, r % #height] else a_in[c, r]
            }
        }
    );
    let module = ib.finish("lift_padded_matrix_evals_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; width * padded_height];
    for c in 0..width {
        for r in 0..padded_height {
            let src = if r < lifted_height { r % height } else { r };
            want[c * padded_height + r] = a[c * padded_height + src];
        }
    }
    assert_eq!(outs[0], want);
}

/// Port of `matrix.cu::batch_rotate_pad_kernel`. `in` has layout
/// `[width * num_x, domain]` (col-major within each `num_x` batch), `out` has
/// layout `[width * num_x, padded]`. The rotation is by one over
/// `domain * num_x`: within each batch of `num_x` adjacent columns, the flat
/// index `f = (p mod num_x) * domain + t` moves to `(f + 1) mod (num_x *
/// domain)`. Positions past `domain` in the row are zero-padded.
#[test]
fn macro_batch_rotate_pad() {
    let (width, num_x, domain, padded) = (3usize, 4usize, 8usize, 12usize);
    let n_dom = num_x * domain;
    let cols = width * num_x;
    let a = pseudo_field_elems(cols * domain, 34);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![cols, domain]);
    let body = kernel!(ib,
        compute [cols] |p| {
            compute [padded] |t| {
                if t < #domain then
                    let f = p % #num_x * #domain + t + 1;
                    let new_f = f % #n_dom;
                    let new_p = p / #num_x * #num_x + new_f / #domain;
                    let new_t = new_f % #domain;
                    a_in[new_p, new_t]
                else zbb()
            }
        }
    );
    let module = ib.finish("batch_rotate_pad_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; cols * padded];
    for p in 0..cols {
        for t in 0..padded {
            let v = if t < domain {
                let f = (p % num_x) * domain + t + 1;
                let new_f = f % n_dom;
                let new_p = (p / num_x) * num_x + new_f / domain;
                let new_t = new_f % domain;
                a[new_p * domain + new_t]
            } else {
                0
            };
            want[p * padded + t] = v;
        }
    }
    assert_eq!(outs[0], want);
}

/// Port of `mle_interpolate.cu::mle_interpolate_stage_kernel` (coeff_to_eval
/// direction, `Fp`): the "in-place" butterfly `buf[base + step] += buf[base]`
/// for every pair `base = chunk * span + offset`, `offset < step`, `span =
/// 2 * step`. Base positions are unchanged; second positions absorb the base.
///
/// The functional rewrite avoids the OOB `a[i - step]` load that a flat
/// `compute [n] |i|` version would produce for base-half indices `i < step`:
/// iterate `(chunk, off)` and derive an always-in-bounds `base_r = chunk *
/// span + (off % step)` so both loads land inside `[chunk * span, (chunk +
/// 1) * span)`.
#[test]
fn macro_mle_interpolate_stage_coeff_to_eval() {
    let (log_n, log_step) = (10usize, 3usize);
    let n = 1usize << log_n;
    let step = 1usize << log_step;
    let span = 2 * step;
    let chunks = n / span;
    let a = pseudo_field_elems(n, 35);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(ib,
        compute [chunks] |c| {
            compute [span] |off| {
                let base_r = c * #span + off % #step;
                if off < #step then a_in[base_r] else a_in[base_r] + a_in[base_r + #step]
            }
        }
    );
    let module = ib.finish("mle_interpolate_stage_c2e_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let want: Vec<u32> = (0..n)
        .map(|i| {
            let off = i % span;
            let v = if off < step {
                bb(a[i])
            } else {
                bb(a[i]) + bb(a[i - step])
            };
            v.as_canonical_u32()
        })
        .collect();
    assert_eq!(outs[0], want);
}

/// Port of `mle_interpolate.cu::mle_interpolate_stage_kernel` (eval_to_coeff
/// direction): same reformulation as above, subtracting instead of adding.
#[test]
fn macro_mle_interpolate_stage_eval_to_coeff() {
    let (log_n, log_step) = (10usize, 5usize);
    let n = 1usize << log_n;
    let step = 1usize << log_step;
    let span = 2 * step;
    let chunks = n / span;
    let a = pseudo_field_elems(n, 36);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(ib,
        compute [chunks] |c| {
            compute [span] |off| {
                let base_r = c * #span + off % #step;
                if off < #step then a_in[base_r] else a_in[base_r + #step] - a_in[base_r]
            }
        }
    );
    let module = ib.finish("mle_interpolate_stage_e2c_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let want: Vec<u32> = (0..n)
        .map(|i| {
            let off = i % span;
            let v = if off < step {
                bb(a[i])
            } else {
                bb(a[i]) - bb(a[i - step])
            };
            v.as_canonical_u32()
        })
        .collect();
    assert_eq!(outs[0], want);
}

/// Port of `mle_interpolate.cu::mle_interpolate_stage_2d_kernel` (coeff to
/// eval): the same butterfly per column of a `[width, padded_height]`
/// col-major matrix.
#[test]
fn macro_mle_interpolate_stage_2d() {
    let (width, padded_height, log_step) = (7usize, 512usize, 4usize);
    let step = 1usize << log_step;
    let span = 2 * step;
    let chunks = padded_height / span;
    let a = pseudo_field_elems(width * padded_height, 37);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![width, padded_height]);
    let body = kernel!(ib,
        compute [width] |col| {
            compute [chunks] |c| {
                compute [span] |off| {
                    let base_r = c * #span + off % #step;
                    if off < #step then a_in[col, base_r]
                    else a_in[col, base_r] + a_in[col, base_r + #step]
                }
            }
        }
    );
    let module = ib.finish("mle_interpolate_stage_2d_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; width * padded_height];
    for c in 0..width {
        for r in 0..padded_height {
            let off = r % span;
            let idx = c * padded_height + r;
            let v = if off < step {
                bb(a[idx])
            } else {
                bb(a[idx]) + bb(a[idx - step])
            };
            want[idx] = v.as_canonical_u32();
        }
    }
    assert_eq!(outs[0], want);
}

/// The strided merkle-compression step: `compress(prev[2*x*stride + y],
/// prev[(2*x + 1) * stride + y])` with `x = gid / stride, y = gid % stride`,
/// invoking the shared `poseidon2_permutation`.
fn compress_strided(
    b: &mut IRBuilder,
    prev: NodeId,
    gid: NodeId,
    stride: usize,
    c: &Poseidon2Constants,
) -> NodeId {
    let stride_c = b.const_u32(stride as u32);
    let two_stride = b.const_u32((2 * stride) as u32);
    // idx0 = (gid / stride) * (2 * stride) + gid % stride
    let x = b.div(gid, stride_c);
    let two_x_stride = b.mul(x, two_stride);
    let y = b.rem(gid, stride_c);
    let idx0 = b.add(two_x_stride, y);
    let idx1 = b.add(idx0, stride_c);

    let zero = b.const_u32(0);
    let mut state = [zero; 16];
    for j in 0..8 {
        let jc = b.const_u32(j as u32);
        state[j] = b.index(prev, &[idx0, jc]);
        state[8 + j] = b.index(prev, &[idx1, jc]);
    }
    poseidon2_permutation(b, &mut state, c);
    b.pack(&state[..8])
}

/// Port of `merkle_tree.cu::poseidon2_strided_compress_layer_kernel` (adjacent
/// case with `stride > 1`): compress pairs of adjacent-in-stride digests.
#[test]
fn macro_poseidon2_strided_compress_layer() {
    let (output_size, stride) = (16usize, 4usize);
    let prev_size = 2 * output_size;
    let prev_data = pseudo_field_elems(prev_size * 8, 38);
    let c = Poseidon2Constants::p3_default();
    let constants = &c;

    let mut ib = IRBuilder::new();
    let prev = ib.input("prev", ScalarType::BabyBear, vec![prev_size, 8]);
    let body = kernel!(
        ib,
        compute[output_size] | gid | { compress_strided(prev, gid, stride, constants) }
    );
    let module = ib.finish("poseidon2_strided_compress_macro", body);

    let outs = run_module(module, std::slice::from_ref(&prev_data));
    assert_eq!(outs.len(), 1);

    let perm = default_babybear_poseidon2_16();
    let prev_digests: Vec<[BabyBear; 8]> = prev_data
        .chunks(8)
        .map(|c| std::array::from_fn(|i| bb(c[i])))
        .collect();
    let mut want = vec![0u32; output_size * 8];
    for gid in 0..output_size {
        let x = gid / stride;
        let y = gid % stride;
        let idx0 = 2 * x * stride + y;
        let idx1 = idx0 + stride;
        let mut state = [BabyBear::ZERO; 16];
        state[..8].copy_from_slice(&prev_digests[idx0]);
        state[8..].copy_from_slice(&prev_digests[idx1]);
        let out = perm.permute(state);
        for j in 0..8 {
            want[gid * 8 + j] = out[j].as_canonical_u32();
        }
    }
    assert_eq!(outs[0], want);
}

/// Port of `merkle_tree.cu::poseidon2_strided_compress_layer_kernel` with
/// `stride = 1` (the "adjacent" launcher path): the same as the merkle-tree
/// test's per-layer compression but exercised as a standalone kernel.
#[test]
fn macro_poseidon2_adjacent_compress_layer() {
    let output_size = 128usize;
    let prev_size = 2 * output_size;
    let prev_data = pseudo_field_elems(prev_size * 8, 39);
    let c = Poseidon2Constants::p3_default();
    let constants = &c;

    let stride = 1usize;
    let mut ib = IRBuilder::new();
    let prev = ib.input("prev", ScalarType::BabyBear, vec![prev_size, 8]);
    let body = kernel!(
        ib,
        compute[output_size] | gid | { compress_strided(prev, gid, stride, constants) }
    );
    let module = ib.finish("poseidon2_adjacent_compress_macro", body);

    let outs = run_module(module, std::slice::from_ref(&prev_data));
    assert_eq!(outs.len(), 1);

    let perm = default_babybear_poseidon2_16();
    let prev_digests: Vec<[BabyBear; 8]> = prev_data
        .chunks(8)
        .map(|c| std::array::from_fn(|i| bb(c[i])))
        .collect();
    let mut want = vec![0u32; output_size * 8];
    for gid in 0..output_size {
        let idx0 = 2 * gid;
        let idx1 = idx0 + 1;
        let mut state = [BabyBear::ZERO; 16];
        state[..8].copy_from_slice(&prev_digests[idx0]);
        state[8..].copy_from_slice(&prev_digests[idx1]);
        let out = perm.permute(state);
        for j in 0..8 {
            want[gid * 8 + j] = out[j].as_canonical_u32();
        }
    }
    assert_eq!(outs[0], want);
}

/// Port of `merkle_tree.cu::poseidon2_compressing_row_hashes_kernel` for a
/// width that is a multiple of `CELLS_RATE = 8`, with `log_rows_per_query =
/// 0` (a plain row-hash kernel with no tree layer on top): for each row
/// of a `[width, height]` col-major matrix, absorb the row's values in
/// chunks of 8 and return the top 8 permutation cells. Matches the CUDA
/// launcher's `_poseidon2_compressing_row_hashes` for `log_rows_per_query =
/// 0` (i.e. `query_stride = height`).
#[test]
fn macro_poseidon2_row_hashes_plain() {
    let (width, height) = (24usize, 64usize);
    assert!(width % 8 == 0, "row-hash DSL port requires width % 8 == 0");
    let matrix = pseudo_field_elems(width * height, 40);
    let c = Poseidon2Constants::p3_default();
    let constants = &c;

    fn row_hash(
        b: &mut IRBuilder,
        matrix: NodeId,
        row: NodeId,
        width: usize,
        c: &Poseidon2Constants,
    ) -> NodeId {
        let zero = b.const_field(0);
        let mut state = [zero; 16];
        // Absorb `width` values in chunks of 8 = CELLS_RATE.
        for col in 0..width {
            let col_c = b.const_u32(col as u32);
            let v = b.index(matrix, &[col_c, row]);
            state[col % 8] = v;
            if col % 8 == 7 {
                poseidon2_permutation(b, &mut state, c);
                // Reset capacity half? p3's Poseidon2 sponge keeps the full
                // permuted state between absorbs; CUDA reuses `cells` in
                // place, so nothing to zero here.
            }
        }
        b.pack(&state[..8])
    }

    let mut ib = IRBuilder::new();
    let mat_in = ib.input("m", ScalarType::BabyBear, vec![width, height]);
    let body = kernel!(
        ib,
        compute[height] | r | { row_hash(mat_in, r, width, constants) }
    );
    let module = ib.finish("poseidon2_row_hashes_macro", body);

    let outs = run_module(module, std::slice::from_ref(&matrix));
    assert_eq!(outs.len(), 1);

    let perm = default_babybear_poseidon2_16();
    let mut want = vec![0u32; height * 8];
    for r in 0..height {
        let mut state = [BabyBear::ZERO; 16];
        for col in 0..width {
            state[col % 8] = bb(matrix[col * height + r]);
            if col % 8 == 7 {
                state = perm.permute(state);
            }
        }
        for j in 0..8 {
            want[r * 8 + j] = state[j].as_canonical_u32();
        }
    }
    assert_eq!(outs[0], want);
}

// ---------------------------------------------------------------------------
// Layout-conversion torture tests
// ---------------------------------------------------------------------------
//
// These four tests stress the register-first `layout_infer` pass along the
// three axes called out in the design notes:
//
//   1. **Chained conversions** — a multi-stage pipeline where every stage promotes its predecessor
//      and adds a *fresh* view or mirror on top, exercising the register-view chain plus
//      insert_sync's aliased-mirror handling.
//   2. **Diverse reader layouts** — a single tile with several readers that each demand a different
//      `ConvertKind` (Direct / Slot / Shuffle / Bounce), verifying that view- and mirror-emission
//      handles the full mix and that the shuffle fast path stays intact for the const-slot cases.
//   3. **Par blocks with different shapes** — tiles whose `par` bounds differ from the outer
//      reader's, forcing the reader to bounce off a shared mirror even for register-friendly
//      writers.

/// Chain of four register-tiled stages, each partner-XOR at a different
/// bit under `#[grid(threads = 32)]` with `#[par((th, s) -> th*16 + s)]`.
/// `half=1` and `half=8` flip seq bits (Slot); `half=32` and `half=128`
/// flip lane bits (Shuffle). Every stage keeps its data in registers and
/// gets a fresh view for its partner read; the whole pipeline runs
/// entirely through the register/view chain with no shared memory bounce.
#[test]
fn torture_chained_layout_conversions() {
    let (blocks, t) = (7usize, 512usize);
    let n = blocks * t;
    let a = pseudo_field_elems(n, 30);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(ib,
        #[grid(threads = 32)]
        compute [blocks] |i| {
            let s0 =
                #[par((th, s) -> th * 16 + s)]
                compute [t] |j| { a_in[i * #t + j] };
            let s1 =
                #[par((th, s) -> th * 16 + s)]
                compute [t] |j| { s0[j] + s0[j + 1 - j % 2 * 2] };
            let s2 =
                #[par((th, s) -> th * 16 + s)]
                compute [t] |j| { s1[j] + s1[j + 32 - j % 64 / 32 * 64] };
            let s3 =
                #[par((th, s) -> th * 16 + s)]
                compute [t] |j| { s2[j] + s2[j + 8 - j % 16 / 8 * 16] };
            #[par((th, s) -> th * 16 + s)]
            compute [t] |j| { s3[j] + s3[j + 128 - j % 256 / 128 * 256] }
        }
    );
    let module = ib.finish("torture_chain", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; n];
    for bi in 0..blocks {
        let s0: Vec<BabyBear> = (0..t).map(|j| bb(a[bi * t + j])).collect();
        let s1: Vec<BabyBear> = (0..t).map(|j| s0[j] + s0[j ^ 1]).collect();
        let s2: Vec<BabyBear> = (0..t).map(|j| s1[j] + s1[j ^ 32]).collect();
        let s3: Vec<BabyBear> = (0..t).map(|j| s2[j] + s2[j ^ 8]).collect();
        for j in 0..t {
            want[bi * t + j] = (s3[j] + s3[j ^ 128]).as_canonical_u32();
        }
    }
    assert_eq!(outs[0], want);
}

/// One register tile read four times in a single par at four distinct
/// offsets. Under `(th, s) -> th*16 + s` on a 32-thread block, `j`
/// resolves to a Direct own-slot read, `j^4` to a Slot conversion,
/// `j^64` and `j^128` to Shuffle conversions — three unique effective
/// maps, three views, no shared memory. The multipliers `2, 3, 5, 7`
/// distinguish the readers so a shuffled-value mixup would break the
/// per-element comparison.
#[test]
fn torture_multiple_view_layouts() {
    let (blocks, t) = (9usize, 512usize);
    let n = blocks * t;
    let a = pseudo_field_elems(n, 31);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(ib,
        #[grid(threads = 32)]
        compute [blocks] |i| {
            let tile =
                #[par((th, s) -> th * 16 + s)]
                compute [t] |j| { a_in[i * #t + j] };
            #[par((th, s) -> th * 16 + s)]
            compute [t] |j| {
                tile[j] * 2bb
                    + tile[j + 4 - j % 8 / 4 * 8] * 3bb
                    + tile[j + 64 - j % 128 / 64 * 128] * 5bb
                    + tile[j + 128 - j % 256 / 128 * 256] * 7bb
            }
        }
    );
    let module = ib.finish("torture_views", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; n];
    for bi in 0..blocks {
        let tile: Vec<BabyBear> = (0..t).map(|j| bb(a[bi * t + j])).collect();
        for j in 0..t {
            let v = tile[j] * bb(2)
                + tile[j ^ 4] * bb(3)
                + tile[j ^ 64] * bb(5)
                + tile[j ^ 128] * bb(7);
            want[bi * t + j] = v.as_canonical_u32();
        }
    }
    assert_eq!(outs[0], want);
}

/// One register tile with a mix of `ConvertKind`s in a single reader
/// par: Direct (own slot), Slot (`half=4`, seq bit), Shuffle (`half=32`,
/// lane bit), and Bounce (`half=1024`, warp bit). Under
/// `#[grid(threads = 512)]` with `#[par((th, s) -> th*16 + s)]` the
/// `1024` XOR crosses warps so `layout_infer` promotes the tile to
/// registers and emits *one* shared mirror covering just the bouncing
/// read; the other three still resolve to registers.
#[test]
fn torture_mixed_kinds_in_one_par() {
    let (blocks, t) = (3usize, 8192usize);
    let n = blocks * t;
    let a = pseudo_field_elems(n, 32);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(ib,
        #[grid(threads = 512)]
        compute [blocks] |i| {
            let tile =
                #[par((th, s) -> th * 16 + s)]
                compute [t] |j| { a_in[i * #t + j] };
            #[par((th, s) -> th * 16 + s)]
            compute [t] |j| {
                tile[j] * 2bb
                    + tile[j + 4 - j % 8 / 4 * 8] * 3bb
                    + tile[j + 32 - j % 64 / 32 * 64] * 5bb
                    + tile[j + 1024 - j % 2048 / 1024 * 2048] * 7bb
            }
        }
    );
    let module = ib.finish("torture_mixed_kinds", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; n];
    for bi in 0..blocks {
        let tile: Vec<BabyBear> = (0..t).map(|j| bb(a[bi * t + j])).collect();
        for j in 0..t {
            let v = tile[j] * bb(2)
                + tile[j ^ 4] * bb(3)
                + tile[j ^ 32] * bb(5)
                + tile[j ^ 1024] * bb(7);
            want[bi * t + j] = v.as_canonical_u32();
        }
    }
    assert_eq!(outs[0], want);
}

/// Two register tiles of different sizes in one block. The 64-element
/// `small` tile is written by a bound-64 par and read by a bound-128
/// consumer at `j % 64`; the bound mismatch forces layout_infer down
/// the Mirror plan even though the writer is trivially XOR-linear, so
/// `small` becomes a register buffer *plus* a shared mirror while the
/// 128-element `big` tile stays purely in registers with a Direct
/// reader. Same-block par shapes differ; both readers combine into the
/// final compute.
#[test]
fn torture_different_par_bounds() {
    let (blocks, small_t, big_t) = (5usize, 64usize, 128usize);
    let n = blocks * big_t;
    let a = pseudo_field_elems(n, 33);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(ib,
        compute [blocks] |i| {
            let small = compute [small_t] |j| { a_in[i * #big_t + j] };
            let big = compute [big_t] |j| { a_in[i * #big_t + j] * 2bb };
            compute [big_t] |j| { small[j % #small_t] + big[j] }
        }
    );
    let module = ib.finish("torture_bounds", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; n];
    for bi in 0..blocks {
        for j in 0..big_t {
            let small_val = bb(a[bi * big_t + (j % small_t)]);
            let big_val = bb(a[bi * big_t + j]) * bb(2);
            want[bi * big_t + j] = (small_val + big_val).as_canonical_u32();
        }
    }
    assert_eq!(outs[0], want);
}

/// Reference add of two FpExt tensors: `FpExt` is `[BabyBear; 4]`
/// interpreted as `a0 + a1 x + a2 x^2 + a3 x^3`; add is elementwise.
fn fpext_add_ref(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (bb(x) + bb(y)).as_canonical_u32())
        .collect()
}

/// Reference mul of two FpExt tensors, iterating four coefficients at
/// a time and applying the `x^4 - 11` reduction.
fn fpext_mul_ref(a: &[u32], b: &[u32]) -> Vec<u32> {
    assert_eq!(a.len() % 4, 0);
    let mut out = vec![0u32; a.len()];
    let b_c = BabyBear::new(11);
    for i in (0..a.len()).step_by(4) {
        let (a0, a1, a2, a3) = (bb(a[i]), bb(a[i + 1]), bb(a[i + 2]), bb(a[i + 3]));
        let (b0, b1, b2, b3) = (bb(b[i]), bb(b[i + 1]), bb(b[i + 2]), bb(b[i + 3]));
        let c0 = a0 * b0 + b_c * (a1 * b3 + a2 * b2 + a3 * b1);
        let c1 = a0 * b1 + a1 * b0 + b_c * (a2 * b3 + a3 * b2);
        let c2 = a0 * b2 + a1 * b1 + a2 * b0 + b_c * (a3 * b3);
        let c3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;
        out[i] = c0.as_canonical_u32();
        out[i + 1] = c1.as_canonical_u32();
        out[i + 2] = c2.as_canonical_u32();
        out[i + 3] = c3.as_canonical_u32();
    }
    out
}

/// Elementwise FpExt addition. The generated kernel loads each input
/// through a `FpExt*` pointer (giving `LDG.128`) and stores the sum
/// through a `FpExt*` pointer (`STG.128`).
#[test]
fn fpext_add_matches_reference() {
    let n = 1024usize;
    let a = pseudo_field_elems(n * 4, 60);
    let b = pseudo_field_elems(n * 4, 61);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::FpExt, vec![n]);
    let b_in = ib.input("b", ScalarType::FpExt, vec![n]);
    let body = kernel!(ib, compute[n] | i | { a_in[i] + b_in[i] });
    let module = ib.finish("fpext_add", body);

    let outs = run_module(module, &[a.clone(), b.clone()]);
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0], fpext_add_ref(&a, &b));
}

/// Elementwise FpExt multiplication over `x^4 - 11`. Each result is
/// computed by the inline `fpext_mul` from the codegen prelude; the
/// buffer accesses use `LDG.128`/`STG.128`.
#[test]
fn fpext_mul_matches_reference() {
    let n = 1024usize;
    let a = pseudo_field_elems(n * 4, 62);
    let b = pseudo_field_elems(n * 4, 63);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::FpExt, vec![n]);
    let b_in = ib.input("b", ScalarType::FpExt, vec![n]);
    let body = kernel!(ib, compute[n] | i | { a_in[i] * b_in[i] });
    let module = ib.finish("fpext_mul", body);

    let outs = run_module(module, &[a.clone(), b.clone()]);
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0], fpext_mul_ref(&a, &b));
}

/// FpExt constant construction plus `BabyBear -> FpExt` lift. For each
/// input `x` (BabyBear), the kernel emits `lift_fpext(x) * FpExt(1, 2,
/// 3, 4)`. By the Karatsuba-mod-`x^4 - 11` product with `a = (x, 0, 0,
/// 0)` and `b = (1, 2, 3, 4)`, all `a1..a3` terms vanish and the result
/// is `(x, 2x, 3x, 4x)`. The test lifts, multiplies by a FpExt const,
/// then adds a second FpExt const to check that the emitted code
/// initializes and consumes both `LiftFpExt` and `ConstFpExt`.
#[test]
fn fpext_const_and_lift() {
    let n = 512usize;
    let a = pseudo_field_elems(n, 70);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let k1 = ib.const_fpext([1, 2, 3, 4]);
    let k2 = ib.const_fpext([11, 22, 33, 44]);
    let body = kernel!(ib, compute[n] | i | { to_fpext(a_in[i]) * k1 + k2 });
    let module = ib.finish("fpext_const_lift", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; n * 4];
    for i in 0..n {
        let x = bb(a[i]);
        // (x, 0, 0, 0) * (1, 2, 3, 4) = (x, 2x, 3x, 4x)
        // + (11, 22, 33, 44)
        let coeffs = [
            (x + bb(11)).as_canonical_u32(),
            (x * bb(2) + bb(22)).as_canonical_u32(),
            (x * bb(3) + bb(33)).as_canonical_u32(),
            (x * bb(4) + bb(44)).as_canonical_u32(),
        ];
        want[i * 4..i * 4 + 4].copy_from_slice(&coeffs);
    }
    assert_eq!(outs[0], want);
}

/// FpExt through a register-tile layout conversion. Under
/// `#[grid(threads = 32)]` with the `(th, s) -> th*16 + s` par layout,
/// a bound-512 register tile of `FpExt` gets an own-slot Direct read
/// on `tile[j]` and a Slot-classified conversion for `tile[j ^ 4]`
/// (the XOR flips seq bit 2, which under this layout lands on physical
/// slot bit 7 — a within-thread slot permutation, no `__shfl_sync`
/// needed). The generated view assignments become 16-byte-per-element
/// `FpExt` struct copies between register arrays. Result checked
/// elementwise on the CPU.
#[test]
fn fpext_slot_layout_conversion() {
    let (blocks, t) = (5usize, 512usize);
    let n = blocks * t;
    let a = pseudo_field_elems(n * 4, 71);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::FpExt, vec![n]);
    let body = kernel!(ib,
        #[grid(threads = 32)]
        compute [blocks] |i| {
            let tile =
                #[par((th, s) -> th * 16 + s)]
                compute [t] |j| { a_in[i * #t + j] };
            #[par((th, s) -> th * 16 + s)]
            compute [t] |j| { tile[j] + tile[j + 4 - j % 8 / 4 * 8] }
        }
    );
    let module = ib.finish("fpext_slot_conv", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    // Reference: elementwise add of tile[j] and tile[j ^ 4], where each
    // FpExt lives in four consecutive u32s of `a` and the tile itself
    // matches the input row unchanged.
    let mut want = vec![0u32; n * 4];
    for bi in 0..blocks {
        let base = bi * t * 4;
        for j in 0..t {
            let jp = j ^ 4;
            for lane in 0..4 {
                let x = bb(a[base + j * 4 + lane]);
                let y = bb(a[base + jp * 4 + lane]);
                want[base + j * 4 + lane] = (x + y).as_canonical_u32();
            }
        }
    }
    assert_eq!(outs[0], want);
}

// ============================================================================
// cuda-backend kernel ports (FpExt variants).
//
// Each test mirrors an FpExt-parametrized `__global__` kernel from
// `crates/cuda-backend/cuda/src/`, re-expressed against `ScalarType::FpExt`
// and validated against an `EF = BinomialExtensionField<BabyBear, 4>` host
// reference.
// ============================================================================

/// Port of `mle_interpolate.cu::mle_interpolate_stage_kernel<FpExt, /*
/// EvalToCoeff=*/false>` (coeff to eval): identical control flow to the
/// `Fp` port, but with `ScalarType::FpExt` buffers so codegen picks up
/// `fpext_add` and `LDG.128`/`STG.128`.
#[test]
fn macro_mle_interpolate_stage_ext_c2e() {
    let (log_n, log_step) = (10usize, 3usize);
    let n = 1usize << log_n;
    let step = 1usize << log_step;
    let span = 2 * step;
    let chunks = n / span;
    let a = pseudo_ext_elems(n, 41);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::FpExt, vec![n]);
    let body = kernel!(ib,
        compute [chunks] |c| {
            compute [span] |off| {
                let base_r = c * #span + off % #step;
                if off < #step then a_in[base_r] else a_in[base_r] + a_in[base_r + #step]
            }
        }
    );
    let module = ib.finish("mle_interpolate_stage_ext_c2e_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let a_ef: Vec<EF> = a.chunks(4).map(ef_from_u32s).collect();
    let mut want: Vec<u32> = Vec::with_capacity(4 * n);
    for i in 0..n {
        let off = i % span;
        let v = if off < step {
            a_ef[i]
        } else {
            a_ef[i] + a_ef[i - step]
        };
        want.extend_from_slice(&ef_to_u32s(v));
    }
    assert_eq!(outs[0], want);
}

/// Port of `mle_interpolate.cu::mle_interpolate_stage_kernel<FpExt, /*
/// EvalToCoeff=*/true>` (eval to coeff).
#[test]
fn macro_mle_interpolate_stage_ext_e2c() {
    let (log_n, log_step) = (10usize, 5usize);
    let n = 1usize << log_n;
    let step = 1usize << log_step;
    let span = 2 * step;
    let chunks = n / span;
    let a = pseudo_ext_elems(n, 42);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::FpExt, vec![n]);
    let body = kernel!(ib,
        compute [chunks] |c| {
            compute [span] |off| {
                let base_r = c * #span + off % #step;
                if off < #step then a_in[base_r] else a_in[base_r + #step] - a_in[base_r]
            }
        }
    );
    let module = ib.finish("mle_interpolate_stage_ext_e2c_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let a_ef: Vec<EF> = a.chunks(4).map(ef_from_u32s).collect();
    let mut want: Vec<u32> = Vec::with_capacity(4 * n);
    for i in 0..n {
        let off = i % span;
        let v = if off < step {
            a_ef[i]
        } else {
            a_ef[i] - a_ef[i - step]
        };
        want.extend_from_slice(&ef_to_u32s(v));
    }
    assert_eq!(outs[0], want);
}

/// Port of `mle_interpolate.cu::mle_interpolate_stage_2d_kernel<FpExt,
/// /*EvalToCoeff=*/false>`: per-column butterfly on a `[width,
/// padded_height]` FpExt matrix.
#[test]
fn macro_mle_interpolate_stage_2d_ext() {
    let (width, padded_height, log_step) = (5usize, 512usize, 4usize);
    let step = 1usize << log_step;
    let span = 2 * step;
    let chunks = padded_height / span;
    let a = pseudo_ext_elems(width * padded_height, 43);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::FpExt, vec![width, padded_height]);
    let body = kernel!(ib,
        compute [width] |col| {
            compute [chunks] |c| {
                compute [span] |off| {
                    let base_r = c * #span + off % #step;
                    if off < #step then a_in[col, base_r]
                    else a_in[col, base_r] + a_in[col, base_r + #step]
                }
            }
        }
    );
    let module = ib.finish("mle_interpolate_stage_2d_ext_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let a_ef: Vec<EF> = a.chunks(4).map(ef_from_u32s).collect();
    let mut want: Vec<u32> = vec![0u32; 4 * width * padded_height];
    for col in 0..width {
        for r in 0..padded_height {
            let off = r % span;
            let idx = col * padded_height + r;
            let v = if off < step {
                a_ef[idx]
            } else {
                a_ef[idx] + a_ef[idx - step]
            };
            let coeffs = ef_to_u32s(v);
            want[idx * 4..idx * 4 + 4].copy_from_slice(&coeffs);
        }
    }
    assert_eq!(outs[0], want);
}

/// Port of `matrix.cu::matrix_transpose_kernel<FpExt>`: transpose a `[r,
/// c]` FpExt matrix to `[c, r]` via `#[scatter((i, j) -> (j, i))]`.
#[test]
fn macro_matrix_transpose_ext() {
    let (r, c) = (10usize, 7usize);
    let x = pseudo_ext_elems(r * c, 44);

    let mut ib = IRBuilder::new();
    let x_in = ib.input("x", ScalarType::FpExt, vec![r, c]);
    let body = kernel!(ib,
        #[scatter((i, j) -> (j, i), [c, r])]
        compute [r] |i| { compute [c] |j| { x_in[i, j] } }
    );
    let module = ib.finish("matrix_transpose_ext_macro", body);

    let outs = run_module(module, std::slice::from_ref(&x));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; 4 * r * c];
    for i in 0..r {
        for j in 0..c {
            let src = (i * c + j) * 4;
            let dst = (j * r + i) * 4;
            want[dst..dst + 4].copy_from_slice(&x[src..src + 4]);
        }
    }
    assert_eq!(outs[0], want);
}

/// Port of `poly.cu::vector_scalar_multiply_kernel<FpExt>`: multiply an
/// `FpExt` vector by a compile-time-baked `FpExt` scalar.
#[test]
fn macro_vector_scalar_multiply_ext() {
    let n = 1024usize;
    let a = pseudo_ext_elems(n, 45);
    let s_raw = pseudo_ext_elems(1, 46);
    let s_arr: [u32; 4] = [s_raw[0], s_raw[1], s_raw[2], s_raw[3]];

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::FpExt, vec![n]);
    let s_const = ib.const_fpext(s_arr);
    let body = kernel!(ib, compute[n] | i | { a_in[i] * s_const });
    let module = ib.finish("vector_scalar_multiply_ext_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let s_ef = ef_from_u32s(&s_arr);
    let mut want = vec![0u32; 4 * n];
    for (i, chunk) in a.chunks(4).enumerate() {
        let v = ef_from_u32s(chunk) * s_ef;
        let c = ef_to_u32s(v);
        want[i * 4..i * 4 + 4].copy_from_slice(&c);
    }
    assert_eq!(outs[0], want);
}

/// Port of `poly.cu::transpose_fp_to_fpext_vec_kernel`: read a
/// `[4, height]` col-major BabyBear matrix and pack each column into an
/// FpExt result vector. The DSL rebuilds each FpExt through
/// `lift_fpext(a[i, idx]) * X^i` sums.
#[test]
fn macro_transpose_fp_to_fpext_vec() {
    let height = 256usize;
    let input = pseudo_field_elems(4 * height, 47);

    let mut ib = IRBuilder::new();
    let in_in = ib.input("in", ScalarType::BabyBear, vec![4, height]);
    let body = kernel!(ib,
        compute [height] |idx| {
            make_ext(in_in[0, idx], in_in[1, idx], in_in[2, idx], in_in[3, idx])
        }
    );
    let module = ib.finish("transpose_fp_to_fpext_vec_macro", body);

    let outs = run_module(module, std::slice::from_ref(&input));
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; 4 * height];
    for idx in 0..height {
        for i in 0..4 {
            want[idx * 4 + i] = input[i * height + idx];
        }
    }
    assert_eq!(outs[0], want);
}

/// Port of `poly.cu::eq_hypercube_stage_ext_kernel` (in-place butterfly
/// over an FpExt hypercube stage, inserting `x_i` at the back):
///   `hi = out[y] * x_i`
///   `out[y | step] = hi`
///   `out[y] -= hi`  (i.e. `out[y] *= 1 - x_i`)
/// Rewritten as a functional pass over `[0, 2*step)`. `x_i` is baked in
/// as a `ConstFpExt`.
#[test]
fn macro_eq_hypercube_stage_ext() {
    let step = 1024usize;
    let n = 2 * step;
    let a = pseudo_ext_elems(n, 48);
    let x_raw = pseudo_ext_elems(1, 49);
    let x_arr: [u32; 4] = [x_raw[0], x_raw[1], x_raw[2], x_raw[3]];

    let mut ib = IRBuilder::new();
    let out_in = ib.input("out", ScalarType::FpExt, vec![n]);
    let x_i = ib.const_fpext(x_arr);
    let body = kernel!(ib,
        compute [n] |i| {
            let y = i % #step;
            let hi = out_in[y] * x_i;
            if i < #step then out_in[y] - hi else hi
        }
    );
    let module = ib.finish("eq_hypercube_stage_ext_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let x_ef = ef_from_u32s(&x_arr);
    let mut want = vec![0u32; 4 * n];
    for i in 0..n {
        let y = i % step;
        let prev = ef_from_u32s(&a[y * 4..y * 4 + 4]);
        let hi = prev * x_ef;
        let v = if i < step { prev - hi } else { hi };
        want[i * 4..i * 4 + 4].copy_from_slice(&ef_to_u32s(v));
    }
    assert_eq!(outs[0], want);
}

/// Port of `poly.cu::eq_hypercube_nonoverlapping_stage_ext_kernel`: the
/// same butterfly as `eq_hypercube_stage_ext`, but the reads come from a
/// separate input tensor `in` (not the same as `out`).
#[test]
fn macro_eq_hypercube_nonoverlapping_stage_ext() {
    let step = 1024usize;
    let n = 2 * step;
    let a = pseudo_ext_elems(n, 50);
    let x_raw = pseudo_ext_elems(1, 51);
    let x_arr: [u32; 4] = [x_raw[0], x_raw[1], x_raw[2], x_raw[3]];

    let mut ib = IRBuilder::new();
    let in_in = ib.input("in", ScalarType::FpExt, vec![step]);
    let x_i = ib.const_fpext(x_arr);
    let body = kernel!(ib,
        compute [n] |i| {
            let y = i % #step;
            let prev = in_in[y];
            let hi = prev * x_i;
            if i < #step then prev - hi else hi
        }
    );
    let module = ib.finish("eq_hypercube_nonoverlap_ext_macro", body);

    // `in` is length `step`; the low half of `a` supplies those values.
    let outs = run_module(module, &[a[..step * 4].to_vec()]);
    assert_eq!(outs.len(), 1);

    let x_ef = ef_from_u32s(&x_arr);
    let mut want = vec![0u32; 4 * n];
    for i in 0..n {
        let y = i % step;
        let prev = ef_from_u32s(&a[y * 4..y * 4 + 4]);
        let hi = prev * x_ef;
        let v = if i < step { prev - hi } else { hi };
        want[i * 4..i * 4 + 4].copy_from_slice(&ef_to_u32s(v));
    }
    assert_eq!(outs[0], want);
}

/// Port of `poly.cu::mobius_eq_hypercube_stage_ext_kernel`: the Möbius
/// variant. Low half becomes `prev * (1 - 2*omega_i) = prev - hi - hi`
/// (computed with two subtractions to avoid materializing `1 - 2*omega`).
#[test]
fn macro_mobius_eq_hypercube_stage_ext() {
    let step = 1024usize;
    let n = 2 * step;
    let a = pseudo_ext_elems(n, 52);
    let w_raw = pseudo_ext_elems(1, 53);
    let w_arr: [u32; 4] = [w_raw[0], w_raw[1], w_raw[2], w_raw[3]];

    let mut ib = IRBuilder::new();
    let out_in = ib.input("out", ScalarType::FpExt, vec![n]);
    let omega = ib.const_fpext(w_arr);
    let body = kernel!(ib,
        compute [n] |i| {
            let y = i % #step;
            let prev = out_in[y];
            let hi = prev * omega;
            if i < #step then prev - hi - hi else hi
        }
    );
    let module = ib.finish("mobius_eq_hypercube_stage_ext_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let w_ef = ef_from_u32s(&w_arr);
    let mut want = vec![0u32; 4 * n];
    for i in 0..n {
        let y = i % step;
        let prev = ef_from_u32s(&a[y * 4..y * 4 + 4]);
        let hi = prev * w_ef;
        let v = if i < step { prev - hi - hi } else { hi };
        want[i * 4..i * 4 + 4].copy_from_slice(&ef_to_u32s(v));
    }
    assert_eq!(outs[0], want);
}

/// Port of `poly.cu::eq_hypercube_interleaved_stage_ext_kernel`: same
/// stage as `eq_hypercube_nonoverlapping_stage_ext`, but the results are
/// interleaved (`out[2y]` gets the low half, `out[2y | 1]` the high half)
/// rather than concatenated. Expressed as `compute [2*step] |i| ...` with
/// `y = i / 2` and a parity-based select.
#[test]
fn macro_eq_hypercube_interleaved_stage_ext() {
    let step = 512usize;
    let n = 2 * step;
    let a = pseudo_ext_elems(step, 54);
    let x_raw = pseudo_ext_elems(1, 55);
    let x_arr: [u32; 4] = [x_raw[0], x_raw[1], x_raw[2], x_raw[3]];

    let mut ib = IRBuilder::new();
    let in_in = ib.input("in", ScalarType::FpExt, vec![step]);
    let x_i = ib.const_fpext(x_arr);
    let body = kernel!(ib,
        compute [n] |i| {
            let y = i / 2;
            let prev = in_in[y];
            let hi = prev * x_i;
            if i % 2 == 0 then prev - hi else hi
        }
    );
    let module = ib.finish("eq_hypercube_interleaved_ext_macro", body);

    let outs = run_module(module, std::slice::from_ref(&a));
    assert_eq!(outs.len(), 1);

    let x_ef = ef_from_u32s(&x_arr);
    let mut want = vec![0u32; 4 * n];
    for y in 0..step {
        let prev = ef_from_u32s(&a[y * 4..y * 4 + 4]);
        let hi = prev * x_ef;
        let lo_v = prev - hi;
        let hi_v = hi;
        want[(2 * y) * 4..(2 * y) * 4 + 4].copy_from_slice(&ef_to_u32s(lo_v));
        want[(2 * y + 1) * 4..(2 * y + 1) * 4 + 4].copy_from_slice(&ef_to_u32s(hi_v));
    }
    assert_eq!(outs[0], want);
}

/// Port of `poly.cu::batch_eq_hypercube_stage_kernel` (Fp, 2D): per
/// column `x_idx`, apply the `Fp` eq-hypercube butterfly with scalar
/// `x_i = x[x_idx]`. Positions `[0, step)` of each column become
/// `prev * (1 - x_i)`, positions `[step, 2*step)` become
/// `prev * x_i` reading from the low half; positions `>= 2*step`
/// pass through unchanged.
#[test]
fn macro_batch_eq_hypercube_stage() {
    let (width, height, log_step) = (7usize, 128usize, 4usize);
    let step = 1usize << log_step;
    let two_step = 2 * step;
    assert!(two_step <= height);
    let out = pseudo_field_elems(width * height, 56);
    let x = pseudo_field_elems(width, 57);

    let mut ib = IRBuilder::new();
    let out_in = ib.input("out", ScalarType::BabyBear, vec![width, height]);
    let x_in = ib.input("x", ScalarType::BabyBear, vec![width]);
    let body = kernel!(ib,
        compute [width] |x_idx| {
            let x_i = x_in[x_idx];
            compute [height] |y| {
                // For y in [0, 2*step) the base index alternates between
                // `y` and `y - step` (the low half of the current pair);
                // beyond that it just points at `y` (pass-through).
                let base_lo = y - y / #step % 2 * #step;
                let prev = out_in[x_idx, base_lo];
                let hi = prev * x_i;
                if y < #step then prev - hi
                else if y < #two_step then hi
                else out_in[x_idx, y]
            }
        }
    );
    let module = ib.finish("batch_eq_hypercube_stage_macro", body);

    let outs = run_module(module, &[out.clone(), x.clone()]);
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; width * height];
    for x_idx in 0..width {
        let x_i = bb(x[x_idx]);
        for y in 0..height {
            let base = x_idx * height + y;
            let v = if y < step {
                let prev = bb(out[base]);
                (prev - prev * x_i).as_canonical_u32()
            } else if y < two_step {
                let prev_lo = bb(out[x_idx * height + (y - step)]);
                (prev_lo * x_i).as_canonical_u32()
            } else {
                out[base]
            };
            want[base] = v;
        }
    }
    assert_eq!(outs[0], want);
}

/// `Node::Select` must short-circuit: only the taken branch's body
/// runs, so a load at an index that would be out of bounds when the
/// other branch is taken does not execute. This kernel splits a
/// bound-`2n` iteration into `[0, n)` (returns `a[i]`) and `[n, 2n)`
/// (returns `b[i - n]`). Under eager `?:` semantics both `a[i]` (OOB
/// when `i >= n`) and `b[i - n]` (via a huge unsigned wrap when `i <
/// n`) would evaluate on every iteration and the read past the end
/// would either page-fault or silently return garbage; with the
/// short-circuit lowering only the in-bounds side runs and the result
/// matches the CPU reference exactly.
#[test]
fn select_short_circuits_out_of_bounds_loads() {
    let n = 256usize;
    let two_n = 2 * n;
    let a = pseudo_field_elems(n, 90);
    let b = pseudo_field_elems(n, 91);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let b_in = ib.input("b", ScalarType::BabyBear, vec![n]);
    let body = kernel!(ib,
        compute [two_n] |i| {
            if i < #n then a_in[i] else b_in[i - #n]
        }
    );
    let module = ib.finish("select_short_circuit", body);

    let outs = run_module(module, &[a.clone(), b.clone()]);
    assert_eq!(outs.len(), 1);

    let mut want = vec![0u32; two_n];
    want[..n].copy_from_slice(&a);
    want[n..].copy_from_slice(&b);
    assert_eq!(outs[0], want);
}

/// Structural check: the emitted CU uses `if / else` blocks for
/// `Node::Select`, not the eager ternary `?:` we used to emit. The
/// kernel body itself must never contain `?:` on user SSA values —
/// only the arithmetic prelude does, and that lives above `__global__`.
#[test]
fn select_emits_if_else_block_not_ternary() {
    use crypto_compiler::passes::{
        canonicalize, codegen, insert_sync, layout_infer, lower_to_kir, plan_global_scratch,
        type_infer,
    };
    let n = 32usize;

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(ib,
        compute [n] |i| {
            if i < 8 then a_in[i] * 2bb else a_in[i] + 1bb
        }
    );
    let module = ib.finish("select_structure", body);

    let types = type_infer(&module).unwrap();
    let program = canonicalize(module, types).unwrap();
    let scratch = plan_global_scratch(&program).unwrap();
    let mut kprog = lower_to_kir(&program, &scratch).unwrap();
    layout_infer(&mut kprog);
    insert_sync(&mut kprog);
    let source = codegen(&kprog).unwrap();

    let kernel_body_start = source.find("__global__").unwrap();
    let kernel_body = &source[kernel_body_start..];
    assert!(
        kernel_body.contains("if ("),
        "expected an `if (` in the kernel body:\n{kernel_body}"
    );
    assert!(
        kernel_body.contains("} else {"),
        "expected `}} else {{` in the kernel body:\n{kernel_body}"
    );
    assert!(
        !kernel_body.contains(" ? v"),
        "expected no `?:` on SSA values in the kernel body:\n{kernel_body}"
    );
}
