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
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_symmetric::Permutation;

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
fn run_module(module: crypto_compiler::ir::Module, inputs: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let ctx = GpuDeviceCtx::for_current_device().unwrap();
    let mut km = compile_and_load(module, &CompileOptions::default()).unwrap();
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
                let base = if lo then i else i - #half;
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
