//! End-to-end GPU tests: build DSL modules, JIT them through nvcc, run on
//! the GPU and compare against p3 CPU references. Requires a CUDA GPU.

use crypto_compiler::{
    compile_and_load,
    ir::{IRBuilder, ScalarType},
    kernels::{
        merkle_tree_module, ntt_module, ntt_reg_module, ntt_shared_module, ntt_twiddles,
        Poseidon2Constants,
    },
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
/// Dumps both IR levels and the generated CUDA under `target/ir-dumps/gpu/`.
fn run_module(module: crypto_compiler::ir::Module, inputs: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let ctx = GpuDeviceCtx::for_current_device().unwrap();
    let options = CompileOptions {
        dump_ir: Some(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/ir-dumps/gpu").into()),
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

#[test]
fn ntt_matches_p3_radix2dit() {
    let log_n = 12;
    let n = 1usize << log_n;
    let coeffs = pseudo_field_elems(n, 1);
    let twiddles = ntt_twiddles(log_n);

    let outs = run_module(ntt_module(log_n), &[coeffs.clone(), twiddles]);
    assert_eq!(outs.len(), 1);

    let input_f: Vec<BabyBear> = coeffs.iter().map(|&c| bb(c)).collect();
    let want: Vec<u32> = Radix2Dit::default()
        .dft(input_f)
        .iter()
        .map(|x| x.as_canonical_u32())
        .collect();
    assert_eq!(outs[0], want);
}

/// The shared-memory NTT must match p3 for a single group (log_n = 8), two
/// groups plus the restore pass (12), and three groups (17).
#[test]
fn shared_ntt_matches_p3_radix2dit() {
    for log_n in [8usize, 12, 17] {
        let n = 1usize << log_n;
        let coeffs = pseudo_field_elems(n, 3);
        let twiddles = ntt_twiddles(log_n);

        let outs = run_module(ntt_shared_module(log_n), &[coeffs.clone(), twiddles]);
        assert_eq!(outs.len(), 1);

        let input_f: Vec<BabyBear> = coeffs.iter().map(|&c| bb(c)).collect();
        let want: Vec<u32> = Radix2Dit::default()
            .dft(input_f)
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect();
        assert_eq!(outs[0], want, "shared NTT mismatch at log_n={log_n}");
    }
}

/// The register-tiled NTT must match p3 at the single-group boundary
/// (log_n = 13, one 13-bit reg group only), across the 1-bit and 2-bit
/// shared leftovers (14 and 15), and at the target size for 512-thread
/// blocks with an 8-bit shared tail (21). The 2-bit tail case in
/// particular exercises the disjoint-lifetime shared-memory aliasing
/// through [`plan_shared_mem`] and its accompanying barrier insertion.
#[test]
fn reg_ntt_matches_p3_radix2dit() {
    for log_n in [13usize, 14, 15, 21] {
        let n = 1usize << log_n;
        let coeffs = pseudo_field_elems(n, 4);
        let twiddles = ntt_twiddles(log_n);

        let outs = run_module(ntt_reg_module(log_n), &[coeffs.clone(), twiddles]);
        assert_eq!(outs.len(), 1);

        let input_f: Vec<BabyBear> = coeffs.iter().map(|&c| bb(c)).collect();
        let want: Vec<u32> = Radix2Dit::default()
            .dft(input_f)
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect();
        assert_eq!(outs[0], want, "reg NTT mismatch at log_n={log_n}");
    }
}

#[test]
fn merkle_tree_matches_p3_poseidon2() {
    let log_leaves = 8;
    let n_leaves = 1usize << log_leaves;
    let leaves = pseudo_field_elems(n_leaves * 8, 2);
    let constants = Poseidon2Constants::p3_default();

    let outs = run_module(
        merkle_tree_module(log_leaves, &constants),
        std::slice::from_ref(&leaves),
    );
    assert_eq!(outs.len(), log_leaves);

    let perm = default_babybear_poseidon2_16();
    let mut prev: Vec<[BabyBear; 8]> = leaves
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

/// `compute [n] |i| { compute [m] |j| { reduce [k] ... } }` exercises the
/// grid/block launch shape with a sequential reduction loop.
#[test]
fn matmul_grid_block_reduce() {
    let (n, m, k) = (33usize, 47usize, 29usize);
    let a = pseudo_field_elems(n * k, 3);
    let b = pseudo_field_elems(k * m, 4);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n, k]);
    let b_in = ib.input("b", ScalarType::BabyBear, vec![k, m]);
    let body = ib.compute(n, |ib, i| {
        ib.compute(m, |ib, j| {
            ib.reduce_add(k, |ib, t| {
                let x = ib.index(a_in, &[i, t]);
                let y = ib.index(b_in, &[t, j]);
                ib.mul(x, y)
            })
        })
    });
    let module = ib.finish("matmul", body);

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

/// A 3-deep compute nest is flattened into outer grid + one inner compute,
/// with the two inner indices recovered via div/mod.
#[test]
fn nested_compute_flattening() {
    let (bs, n, m) = (5usize, 12usize, 9usize);
    let x = pseudo_field_elems(bs * n * m, 5);
    let y = pseudo_field_elems(bs * m * n, 6);

    let mut ib = IRBuilder::new();
    let x_in = ib.input("x", ScalarType::BabyBear, vec![bs, n, m]);
    let y_in = ib.input("y", ScalarType::BabyBear, vec![bs, m, n]);
    let body = ib.compute(bs, |ib, bi| {
        ib.compute(n, |ib, i| {
            ib.compute(m, |ib, j| {
                let xv = ib.index(x_in, &[bi, i, j]);
                let yv = ib.index(y_in, &[bi, j, i]);
                let two = ib.const_field(2);
                let t = ib.mul(xv, two);
                ib.add(t, yv)
            })
        })
    });
    let module = ib.finish("batched_add_t", body);

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

/// Scalar lets inside the body, a tuple result (two output buffers), and a
/// chained second kernel reading the first one (scratch is not involved
/// since both tuple members are module outputs).
#[test]
fn tuple_outputs_and_scalar_lets() {
    let n = 1000usize;
    let a = pseudo_field_elems(n, 7);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![n]);
    let pair = ib.compute(n, |ib, i| {
        let x = ib.index(a_in, &[i]);
        ib.bind(x, |ib, v| {
            let sq = ib.mul(v, v);
            let sum = ib.add(sq, v);
            let dbl = ib.add(v, v);
            ib.tuple(&[sum, dbl])
        })
    });
    let module = ib.finish("square_plus", pair);

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

/// A top-level reduce is wrapped into `compute [1]`; the result is a
/// one-element output buffer.
#[test]
fn top_level_reduce_dot_product() {
    let k = 1234usize;
    let a = pseudo_field_elems(k, 8);
    let b = pseudo_field_elems(k, 9);

    let mut ib = IRBuilder::new();
    let a_in = ib.input("a", ScalarType::BabyBear, vec![k]);
    let b_in = ib.input("b", ScalarType::BabyBear, vec![k]);
    let body = ib.reduce_add(k, |ib, t| {
        let x = ib.index(a_in, &[t]);
        let y = ib.index(b_in, &[t]);
        ib.mul(x, y)
    });
    let module = ib.finish("dot", body);

    let outs = run_module(module, &[a.clone(), b.clone()]);

    let mut acc = BabyBear::ZERO;
    for t in 0..k {
        acc += bb(a[t]) * bb(b[t]);
    }
    assert_eq!(outs[0], vec![acc.as_canonical_u32()]);
}
