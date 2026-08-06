//! End-to-end GPU test for the graph kernel-fusion pass: the same graph
//! compiled with fusion (the default) and with `.without_fusion()` must
//! produce identical outputs, with the fused exe launching fewer kernels.
//! Requires a CUDA GPU and the `planner` feature.
#![cfg(feature = "planner")]

use crypto_compiler::{
    graph_exe::{GraphCompiler, GraphExe},
    graph_ir::{BufId, BufInfo, ConstBuf, DeviceType, GraphBuilder},
    ir::{IRBuilder, Module, ScalarType},
    kernels,
    quast::Quast,
};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, stream::GpuDeviceCtx};

const N: usize = 1 << 12;

/// Deterministic pseudo-random canonical BabyBear elements (splitmix64).
fn pseudo_field_elems(n: usize, seed: u64) -> Vec<u32> {
    const P: u64 = 2013265921;
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

/// `a = x*2; b[i] = a[i] + a[(i+1) % N]; out = b*3` — `insert_kernel`
/// splits this into three chained kernels; fusion collapses them to one.
///
/// `shift > 0` burns that many `VarId`s first without touching the module
/// body (the unreachable arena nodes are never hashed), producing an
/// α-variant of the `shift = 0` module with a different module hash.
fn chain_module(shift: usize) -> Module {
    let mut b = IRBuilder::new();
    for _ in 0..shift {
        let z = b.const_u32(0);
        b.bind(z, |_, v| v);
    }
    let x = b.input("x", ScalarType::BabyBear, vec![N]);
    let two = b.const_field(2);
    let a = b.compute(N, |b, i| {
        let xi = b.index(x, &[i]);
        b.mul(xi, two)
    });
    let body = b.bind(a, |b, av| {
        let mid = b.compute(N, |b, i| {
            let ai = b.index(av, &[i]);
            let c1 = b.const_u32(1);
            let cn = b.const_u32(N as u32);
            let i1 = b.add(i, c1);
            let iw = b.rem(i1, cn);
            let ai1 = b.index(av, &[iw]);
            b.add(ai, ai1)
        });
        b.bind(mid, |b, mv| {
            let three = b.const_field(3);
            b.compute(N, |b, i| {
                let mi = b.index(mv, &[i]);
                b.mul(mi, three)
            })
        })
    });
    b.finish("gpu_graph_chain", body)
}

fn add_buf(g: &mut GraphBuilder, name: &str, bytes: usize) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.into()),
        device_type: DeviceType::Cuda(0),
        size: Quast::cst(bytes as i64),
        elem_size: 4,
    })
}

fn build_graph() -> GraphBuilder {
    let mut g = GraphBuilder::new();
    let x = add_buf(&mut g, "x", N * 4);
    let out = add_buf(&mut g, "out", N * 4);
    g.register_input(x);
    g.register_output(out);
    g.insert_kernel(chain_module(0), [x], [out], &[]);
    g
}

fn to_dev(ctx: &GpuDeviceCtx, v: &[u32]) -> DeviceBuffer<u8> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    bytes.as_slice().to_device_on(ctx).unwrap()
}

fn read_output(exe: &GraphExe, ctx: &GpuDeviceCtx, i: usize) -> Vec<u32> {
    let bytes: Vec<u8> = exe.get_output(i).to_host_on(ctx).unwrap();
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn run(exe: &mut GraphExe, ctx: &GpuDeviceCtx, input: &[u32]) -> Vec<u32> {
    let input_buf = to_dev(ctx, input);
    exe.set_input(ctx, 0, &input_buf).unwrap();
    exe.run(ctx).unwrap();
    read_output(exe, ctx, 0)
}

/// Host reference for [`chain_module`]:
/// `((x[i]*2) + (x[(i+1)%N]*2)) * 3 mod p`.
fn chain_reference(input: &[u32]) -> Vec<u64> {
    const P: u64 = 2013265921;
    (0..input.len())
        .map(|i| {
            let a0 = (input[i] as u64) * 2 % P;
            let a1 = (input[(i + 1) % input.len()] as u64) * 2 % P;
            ((a0 + a1) % P) * 3 % P
        })
        .collect()
}

#[test]
fn fused_matches_unfused() {
    let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");
    let input = pseudo_field_elems(N, 7);

    let mut fused = GraphCompiler::new().compile(build_graph()).unwrap();
    let mut unfused = GraphCompiler::new()
        .without_fusion()
        .compile(build_graph())
        .unwrap();

    assert!(unfused.fusion_report().is_none());
    let report = fused.fusion_report().unwrap();
    assert_eq!(report.nodes_before, 3);
    assert_eq!(report.nodes_after, 1);
    assert_eq!(report.fused.len(), 2);

    let got_fused = run(&mut fused, &ctx, &input);
    let got_unfused = run(&mut unfused, &ctx, &input);
    assert_eq!(got_fused.len(), N);
    assert_eq!(got_fused, got_unfused);

    let want = chain_reference(&input);
    for i in 0..N {
        assert_eq!(got_fused[i] as u64, want[i], "mismatch at index {i}");
    }

    // Re-binding the input and re-running reuses the same pool (stable
    // node pointers — the cuda-graph-capture contract).
    let input2 = pseudo_field_elems(N, 99);
    let got2 = run(&mut fused, &ctx, &input2);
    let want2 = chain_reference(&input2);
    for i in 0..N {
        assert_eq!(got2[i] as u64, want2[i], "rerun mismatch at index {i}");
    }
}

/// A caller-declared but *unregistered* staging buffer between two
/// separately inserted kernels is internal now: fusion collapses the two
/// kernels into one and the staging buffer disappears from the graph.
#[test]
fn fusion_crosses_unregistered_staging_buf() {
    let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");

    let scale = |name: &str, k: u32| -> Module {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![N]);
        let c = b.const_field(k);
        let body = b.compute(N, |b, i| {
            let xi = b.index(x, &[i]);
            b.mul(xi, c)
        });
        b.finish(name, body)
    };
    let build = || {
        let mut g = GraphBuilder::new();
        let x = add_buf(&mut g, "x", N * 4);
        let staging = add_buf(&mut g, "staging", N * 4);
        let out = add_buf(&mut g, "out", N * 4);
        g.register_input(x);
        g.register_output(out);
        g.insert_kernel(scale("scale2", 2), [x], [staging], &[]);
        g.insert_kernel(scale("scale3", 3), [staging], [out], &[]);
        g
    };

    let mut fused = GraphCompiler::new().compile(build()).unwrap();
    let report = fused.fusion_report().unwrap();
    assert_eq!(report.nodes_before, 2);
    assert_eq!(report.nodes_after, 1);
    assert_eq!(report.fused.len(), 1);

    let mut unfused = GraphCompiler::new()
        .without_fusion()
        .compile(build())
        .unwrap();

    let input = pseudo_field_elems(N, 13);
    let got_fused = run(&mut fused, &ctx, &input);
    let got_unfused = run(&mut unfused, &ctx, &input);
    assert_eq!(got_fused, got_unfused);

    const P: u64 = 2013265921;
    for i in 0..N {
        let want = (input[i] as u64) * 6 % P;
        assert_eq!(got_fused[i] as u64, want, "mismatch at index {i}");
    }
}

/// The NTT chain (bit-reversal + butterfly stages, split into one kernel
/// per stage by `insert_kernel`) must survive fusion unchanged: fused and
/// unfused compiles of the same graph agree element-for-element.
#[test]
fn ntt_fused_matches_unfused() {
    let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");
    for log_n in [1usize, 2, 3, 4, 6, 10] {
        let n = 1usize << log_n;
        let build = || {
            let mut g = GraphBuilder::new();
            let a = add_buf(&mut g, "a", n * 4);
            let w = add_buf(&mut g, "w", (n / 2).max(1) * 4);
            let out = add_buf(&mut g, "out", n * 4);
            g.register_input(a);
            g.register_output(out);
            let twiddles = kernels::ntt_twiddles(log_n);
            let tw_bytes: Vec<u8> = twiddles.iter().flat_map(|v| v.to_le_bytes()).collect();
            g.insert_const(w, ConstBuf::HostBuf(tw_bytes));
            g.insert_kernel(kernels::ntt_module(log_n), [a, w], [out], &[]);
            g
        };
        let input = pseudo_field_elems(n, 3);

        let mut fused = GraphCompiler::new().compile(build()).unwrap();
        let mut unfused = GraphCompiler::new()
            .without_fusion()
            .compile(build())
            .unwrap();
        let report = fused.fusion_report().unwrap().clone();

        let got_fused = run(&mut fused, &ctx, &input);
        let got_unfused = run(&mut unfused, &ctx, &input);
        assert_eq!(
            got_fused, got_unfused,
            "log_n = {log_n}: fused NTT diverges from unfused (report: {report:?})"
        );
    }
}

/// Two α-variant copies of the chain (independent builders with shifted
/// `VarId` counters) over separate buffers: the stage-6 dedup sweep must
/// fold the two fused modules onto one `Arc<Module>` — one JIT build, two
/// launches — without changing numerics.
#[test]
fn alpha_variant_chains_dedup_and_match() {
    let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");
    let mut g = GraphBuilder::new();
    let x0 = add_buf(&mut g, "x0", N * 4);
    let x1 = add_buf(&mut g, "x1", N * 4);
    let out0 = add_buf(&mut g, "out0", N * 4);
    let out1 = add_buf(&mut g, "out1", N * 4);
    g.register_input(x0);
    g.register_input(x1);
    g.register_output(out0);
    g.register_output(out1);
    g.insert_kernel(chain_module(0), [x0], [out0], &[]);
    g.insert_kernel(chain_module(3), [x1], [out1], &[]);

    let mut exe = GraphCompiler::new().compile(g).unwrap();
    let report = exe.fusion_report().unwrap();
    assert_eq!(report.nodes_before, 6);
    assert_eq!(report.nodes_after, 2);
    assert_eq!(report.fused.len(), 4);
    assert_eq!(report.deduped, 1);
    assert_eq!(exe.num_inputs(), 2);
    assert_eq!(exe.num_outputs(), 2);
    assert_eq!(exe.input_buf_id(0), x0);
    assert_eq!(exe.input_buf_id(1), x1);
    assert_eq!(exe.output_buf_id(0), out0);
    assert_eq!(exe.output_buf_id(1), out1);

    let in0 = pseudo_field_elems(N, 7);
    let in1 = pseudo_field_elems(N, 21);
    let (d0, d1) = (to_dev(&ctx, &in0), to_dev(&ctx, &in1));
    exe.set_input(&ctx, 0, &d0).unwrap();
    exe.set_input(&ctx, 1, &d1).unwrap();
    exe.run(&ctx).unwrap();

    for (i, input) in [(0, &in0), (1, &in1)] {
        let got = read_output(&exe, &ctx, i);
        let want = chain_reference(input);
        for j in 0..N {
            assert_eq!(got[j] as u64, want[j], "output {i}: mismatch at index {j}");
        }
    }
}

/// The caller can hand the exe a pre-allocated pool arena; results are
/// unchanged.
#[test]
fn caller_supplied_pool_arena() {
    let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");
    let mut exe = GraphCompiler::new().compile(build_graph()).unwrap();
    let arena: DeviceBuffer<u8> = DeviceBuffer::with_capacity_on(exe.scratch_bytes().max(1), &ctx);
    exe.set_scratch(arena).unwrap();

    let input = pseudo_field_elems(N, 42);
    let got = run(&mut exe, &ctx, &input);
    let want = chain_reference(&input);
    for i in 0..N {
        assert_eq!(got[i] as u64, want[i], "mismatch at index {i}");
    }
}

/// Post-Phase-7: `lower_to_kir` requires single-kernel modules. A
/// two-stage parallel-reduce that survives to graph compilation without
/// being split first triggers a lowering error. The full P7.5 pipeline
/// will canonicalize + split such modules before lowering; until then,
/// this pins the error contract at the lowering boundary.
///
/// Replaced in P7.12 by a test that compiles this same graph
/// successfully by driving canonicalize → split_module.
#[test]
fn module_with_intermediate_buffers_is_rejected() {
    const K: usize = 1 << 12;
    let mut b = IRBuilder::new();
    let x = b.input("x", ScalarType::BabyBear, vec![K]);
    let body = b.reduce_add(K, |b, i| b.index(x, &[i]));
    let m = b.finish("graph_two_stage_reduce", body);

    let mut g = GraphBuilder::new();
    let x = add_buf(&mut g, "x", K * 4);
    let out = add_buf(&mut g, "out", 4);
    g.register_input(x);
    g.register_output(out);
    g.insert_kernel(m, [x], [out], &[]);

    let err = match GraphCompiler::new().compile(g) {
        Ok(_) => panic!("expected graph compilation to reject multi-kernel modules at lowering"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("intermediate tensor") && msg.contains("single-kernel"),
        "unexpected error: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Symbolic-shape refactor north-star test (see refactor-plan.md). Pins the
// graph-side API: param bindings are INFERRED at `insert_kernel` from the
// concrete buffer shapes (no explicit params arg), and
// `GraphExe::num_unique_modules()` exposes how many distinct compiled
// variants back the graph's kernel nodes.
mod symbolic {
    use super::*;

    fn sym_scale(shift: usize) -> Module {
        let mut b = IRBuilder::new();
        for _ in 0..shift {
            let z = b.const_u32(0);
            b.bind(z, |_, v| v);
        }
        let n = b.symbol("n");
        let x = b.input("x", ScalarType::BabyBear, vec![n]);
        let c = b.const_field(3);
        let body = b.compute(n, |b, i| {
            let xi = b.index(x, &[i]);
            b.mul(xi, c)
        });
        b.finish("sym_scale", body)
    }

    /// `a = x*2; out = a*3` over a symbolic bound `n` — `insert_kernel`
    /// splits the bind into a producer and a consumer kernel around a
    /// staging buffer.
    fn sym_chain(shift: usize) -> Module {
        let mut b = IRBuilder::new();
        for _ in 0..shift {
            let z = b.const_u32(0);
            b.bind(z, |_, v| v);
        }
        let n = b.symbol("n");
        let x = b.input("x", ScalarType::BabyBear, vec![n]);
        let two = b.const_field(2);
        let a = b.compute(n, |b, i| {
            let xi = b.index(x, &[i]);
            b.mul(xi, two)
        });
        let body = b.bind(a, |b, av| {
            let three = b.const_field(3);
            b.compute(n, |b, i| {
                let ai = b.index(av, &[i]);
                b.mul(ai, three)
            })
        });
        b.finish("sym_chain", body)
    }

    /// Two α-variant symbolic producer→consumer chains at DIFFERENT sizes:
    /// fusion merges each chain across the staging seam by unifying the
    /// producer's bare param with the consumer's, so the merged modules stay
    /// fully symbolic and collapse onto ONE compiled variant across sizes.
    #[test]
    fn symbolic_chain_fuses_and_dedupes_across_sizes() {
        const P: u64 = 2013265921;
        let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");
        let (n0, n1) = (1usize << 10, 1usize << 12);
        let mut g = GraphBuilder::new();
        let x0 = add_buf(&mut g, "x0", n0 * 4);
        let x1 = add_buf(&mut g, "x1", n1 * 4);
        let out0 = add_buf(&mut g, "out0", n0 * 4);
        let out1 = add_buf(&mut g, "out1", n1 * 4);
        g.register_input(x0);
        g.register_input(x1);
        g.register_output(out0);
        g.register_output(out1);
        g.insert_kernel(sym_chain(0), [x0], [out0], &[]);
        g.insert_kernel(sym_chain(3), [x1], [out1], &[]);

        let mut exe = GraphCompiler::new().compile(g).unwrap();
        let report = exe.fusion_report().unwrap();
        assert_eq!(report.nodes_before, 4);
        assert_eq!(report.nodes_after, 2);
        assert_eq!(report.fused.len(), 2);
        assert_eq!(exe.num_unique_modules(), 1);

        let in0 = pseudo_field_elems(n0, 5);
        let in1 = pseudo_field_elems(n1, 6);
        let (d0, d1) = (to_dev(&ctx, &in0), to_dev(&ctx, &in1));
        exe.set_input(&ctx, 0, &d0).unwrap();
        exe.set_input(&ctx, 1, &d1).unwrap();
        exe.run(&ctx).unwrap();
        for (idx, input) in [(0usize, &in0), (1, &in1)] {
            let got = read_output(&exe, &ctx, idx);
            assert_eq!(got.len(), input.len());
            for j in 0..input.len() {
                let want = (input[j] as u64) * 6 % P;
                assert_eq!(got[j] as u64, want, "output {idx}: mismatch at {j}");
            }
        }
    }

    /// Two α-variant instances of the same symbolic module at DIFFERENT
    /// sizes: bindings come from the buffer shapes, and both nodes collapse
    /// onto one compiled variant — the core dedup goal of the refactor.
    /// (With concrete shapes these would be two distinct modules/compiles.)
    #[test]
    fn symbolic_module_dedupes_across_sizes() {
        const P: u64 = 2013265921;
        let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");
        let (n0, n1) = (1usize << 10, 1usize << 12);
        let mut g = GraphBuilder::new();
        let x0 = add_buf(&mut g, "x0", n0 * 4);
        let x1 = add_buf(&mut g, "x1", n1 * 4);
        let out0 = add_buf(&mut g, "out0", n0 * 4);
        let out1 = add_buf(&mut g, "out1", n1 * 4);
        g.register_input(x0);
        g.register_input(x1);
        g.register_output(out0);
        g.register_output(out1);
        g.insert_kernel(sym_scale(0), [x0], [out0], &[]);
        g.insert_kernel(sym_scale(3), [x1], [out1], &[]);

        let mut exe = GraphCompiler::new().compile(g).unwrap();
        assert_eq!(exe.num_unique_modules(), 1);

        let in0 = pseudo_field_elems(n0, 7);
        let in1 = pseudo_field_elems(n1, 8);
        let (d0, d1) = (to_dev(&ctx, &in0), to_dev(&ctx, &in1));
        exe.set_input(&ctx, 0, &d0).unwrap();
        exe.set_input(&ctx, 1, &d1).unwrap();
        exe.run(&ctx).unwrap();
        for (idx, input) in [(0usize, &in0), (1, &in1)] {
            let got = read_output(&exe, &ctx, idx);
            assert_eq!(got.len(), input.len());
            for j in 0..input.len() {
                let want = (input[j] as u64) * 3 % P;
                assert_eq!(got[j] as u64, want, "output {idx}: mismatch at {j}");
            }
        }
    }

    /// A parameter used ONLY as a `const_sym` index splice: not inferable
    /// from any input shape, so each insert's `shape_hint` argument
    /// supplies it. Two inserts of the same HIR with different hint values
    /// must keep their own bindings (regression: the subgraph cache used to
    /// key on the hint-excluding module hash and replayed the first hint)
    /// while still collapsing onto ONE compiled kernel — the splice
    /// survives monomorphization as a runtime parameter.
    #[test]
    fn symbolic_index_splice_dedupes_across_hints() {
        fn sym_pick() -> Module {
            let mut b = IRBuilder::new();
            let k = b.symbol("k");
            let x = b.input("x", ScalarType::BabyBear, vec![8]);
            let body = b.compute(1usize, move |b, _i| {
                let idx = b.const_sym(k);
                b.index(x, &[idx])
            });
            b.finish("sym_pick", body)
        }

        let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");
        let mut g = GraphBuilder::new();
        let x = add_buf(&mut g, "x", 8 * 4);
        let out0 = add_buf(&mut g, "out0", 4);
        let out1 = add_buf(&mut g, "out1", 4);
        g.register_input(x);
        g.register_output(out0);
        g.register_output(out1);
        g.insert_kernel(sym_pick(), [x], [out0], &[("k", 2)]);
        g.insert_kernel(sym_pick(), [x], [out1], &[("k", 5)]);

        let mut exe = GraphCompiler::new().compile(g).unwrap();
        assert_eq!(exe.num_unique_modules(), 1);

        let input = pseudo_field_elems(8, 11);
        let d = to_dev(&ctx, &input);
        exe.set_input(&ctx, 0, &d).unwrap();
        exe.run(&ctx).unwrap();
        assert_eq!(read_output(&exe, &ctx, 0), vec![input[2]]);
        assert_eq!(read_output(&exe, &ctx, 1), vec![input[5]]);
    }

    /// Fold-style symbolic gather: `out[i] = x[i + (i/q)*q] + x[i + (i/q)*q
    /// + q]` with `q` inferred from the input shape `[q*4]` and a symbolic
    /// compute bound `q*2`. The index expression floor-divides and
    /// multiplies by the surviving parameter, so it lowers to
    /// `IndexMap::SExpr`; two sizes collapse onto one compiled kernel.
    #[test]
    fn symbolic_div_mul_index_dedupes_across_sizes() {
        fn sym_fold_gather() -> Module {
            let mut b = IRBuilder::new();
            let q = b.symbol("q");
            let x = b.input("x", ScalarType::BabyBear, vec![q * 4]);
            let body = b.compute(q * 2, move |b, i| {
                let qc = b.const_sym(q);
                let grp = b.div(i, qc);
                let off = b.mul(grp, qc);
                let a_idx = b.add(i, off);
                let b_idx = b.add(a_idx, qc);
                let av = b.index(x, &[a_idx]);
                let bv = b.index(x, &[b_idx]);
                b.add(av, bv)
            });
            b.finish("sym_fold_gather", body)
        }

        const P: u64 = 2013265921;
        let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");
        let (q0, q1) = (4usize, 1usize << 8);
        let mut g = GraphBuilder::new();
        let x0 = add_buf(&mut g, "x0", q0 * 4 * 4);
        let x1 = add_buf(&mut g, "x1", q1 * 4 * 4);
        let out0 = add_buf(&mut g, "out0", q0 * 2 * 4);
        let out1 = add_buf(&mut g, "out1", q1 * 2 * 4);
        g.register_input(x0);
        g.register_input(x1);
        g.register_output(out0);
        g.register_output(out1);
        g.insert_kernel(sym_fold_gather(), [x0], [out0], &[]);
        g.insert_kernel(sym_fold_gather(), [x1], [out1], &[]);

        let mut exe = GraphCompiler::new().compile(g).unwrap();
        assert_eq!(exe.num_unique_modules(), 1);

        let in0 = pseudo_field_elems(q0 * 4, 12);
        let in1 = pseudo_field_elems(q1 * 4, 13);
        let (d0, d1) = (to_dev(&ctx, &in0), to_dev(&ctx, &in1));
        exe.set_input(&ctx, 0, &d0).unwrap();
        exe.set_input(&ctx, 1, &d1).unwrap();
        exe.run(&ctx).unwrap();
        for (idx, input, q) in [(0usize, &in0, q0), (1, &in1, q1)] {
            let got = read_output(&exe, &ctx, idx);
            assert_eq!(got.len(), q * 2);
            for (i, &got_i) in got.iter().enumerate() {
                let a = i + (i / q) * q;
                let want = ((input[a] as u64) + (input[a + q] as u64)) % P;
                assert_eq!(got_i as u64, want, "output {idx}: mismatch at {i}");
            }
        }
    }

    /// Per-template block selection: two instances whose node-local block
    /// policies differ (32 → block 32, 4096 → block 256) still collapse
    /// onto ONE compiled variant, because the block hint comes from the
    /// template group's max concrete size, not from each node's own.
    #[test]
    fn block_policy_uses_template_group_max() {
        const P: u64 = 2013265921;
        let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");
        let (n0, n1) = (32usize, 1usize << 12);
        let mut g = GraphBuilder::new();
        let x0 = add_buf(&mut g, "x0", n0 * 4);
        let x1 = add_buf(&mut g, "x1", n1 * 4);
        let out0 = add_buf(&mut g, "out0", n0 * 4);
        let out1 = add_buf(&mut g, "out1", n1 * 4);
        g.register_input(x0);
        g.register_input(x1);
        g.register_output(out0);
        g.register_output(out1);
        g.insert_kernel(sym_scale(0), [x0], [out0], &[]);
        g.insert_kernel(sym_scale(3), [x1], [out1], &[]);

        let mut exe = GraphCompiler::new().compile(g).unwrap();
        assert_eq!(exe.num_unique_modules(), 1);

        let in0 = pseudo_field_elems(n0, 11);
        let in1 = pseudo_field_elems(n1, 12);
        let (d0, d1) = (to_dev(&ctx, &in0), to_dev(&ctx, &in1));
        exe.set_input(&ctx, 0, &d0).unwrap();
        exe.set_input(&ctx, 1, &d1).unwrap();
        exe.run(&ctx).unwrap();
        for (idx, input) in [(0usize, &in0), (1, &in1)] {
            let got = read_output(&exe, &ctx, idx);
            assert_eq!(got.len(), input.len());
            for j in 0..input.len() {
                let want = (input[j] as u64) * 3 % P;
                assert_eq!(got[j] as u64, want, "output {idx}: mismatch at {j}");
            }
        }
    }

    /// Partial monomorphization: producer
    /// `y = compute [n] |i| { compute [t] |j| { x[i+a] * x[i+j] } }` feeds
    /// consumer `out = compute [k] |i| { y[i] + 3 }` through an
    /// unregistered staging buffer, with all of `n`, `t`, `a`, `k`
    /// symbolic.
    ///
    /// Unfused, only `t` is baked into the producer (it is an inner
    /// compute bound, so it must be concrete); `n` stays the symbolic
    /// grid bound, `a` a runtime device param, and the input's `n + t + a`
    /// extent is runtime-sized. The consumer bakes nothing (`k` is
    /// inferred from the staging buffer and stays the symbolic grid
    /// bound).
    ///
    /// Fused, the producer's row-major write map is inverted at the seam
    /// with a concrete `DivMod { tile }`, hardcoding `v / t`, `v % t`
    /// into the merged body — so `t` is specialized by fusion itself and
    /// `baked` is empty, while `k`, `n`, `a` all survive symbolically.
    ///
    /// `x`'s single shape dim `n + t + a` has three unknowns, which
    /// `insert_kernel`'s binding inference cannot solve from the buffer
    /// size alone — the producer's shape hint supplies the bindings and
    /// the concrete buffer size is verified against them. The consumer
    /// needs no hint: bare `k` is solved from the staging buffer.
    #[test]
    fn partial_monomorphization_and_fusion() {
        const P: u64 = 2013265921;
        // Distinct values so `baked == [t0]` is unambiguous.
        let (n0, t0, a0) = (64usize, 8usize, 3usize);
        let k0 = n0 * t0;
        let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");

        let producer = || -> Module {
            let mut b = IRBuilder::new();
            let n = b.symbol("n");
            let t = b.symbol("t");
            let a = b.symbol("a");
            let x = b.input("x", ScalarType::BabyBear, [n + t + a]);
            let body = b.compute(n, |b, i| {
                b.compute(t, |b, j| {
                    let ca = b.const_sym(a);
                    let ia = b.add(i, ca);
                    let xia = b.index(x, &[ia]);
                    let ij = b.add(i, j);
                    let xij = b.index(x, &[ij]);
                    b.mul(xia, xij)
                })
            });
            b.finish("partial_mono_producer", body)
        };
        let consumer = || -> Module {
            let mut b = IRBuilder::new();
            let k = b.symbol("k");
            let y = b.input("y", ScalarType::BabyBear, vec![k]);
            let three = b.const_field(3);
            let body = b.compute(k, |b, i| {
                let yi = b.index(y, &[i]);
                b.add(yi, three)
            });
            b.finish("partial_mono_consumer", body)
        };
        let build = || {
            let mut g = GraphBuilder::new();
            let x = add_buf(&mut g, "x", (n0 + t0 + a0) * 4);
            let staging = add_buf(&mut g, "y", k0 * 4);
            let out = add_buf(&mut g, "out", k0 * 4);
            g.register_input(x);
            g.register_output(out);
            g.insert_kernel(
                producer(),
                [x],
                [staging],
                &[("n", n0 as i64), ("t", t0 as i64), ("a", a0 as i64)],
            );
            g.insert_kernel(consumer(), [staging], [out], &[]);
            g
        };

        let mut unfused = GraphCompiler::new()
            .without_fusion()
            .compile(build())
            .unwrap();
        assert_eq!(unfused.num_unique_modules(), 2);

        let mut fused = GraphCompiler::new().compile(build()).unwrap();
        let report = fused.fusion_report().unwrap();
        assert_eq!(report.nodes_before, 2);
        assert_eq!(report.nodes_after, 1);
        assert_eq!(report.fused.len(), 1);
        assert_eq!(fused.num_unique_modules(), 1);

        // `x[i+a] * x[i+j]` is a genuine var*var multiply — a Montgomery
        // mul on raw bits — so (unlike the linear chains above, which are
        // representation-transparent) inputs must be Montgomery-encoded
        // and outputs decoded.
        fn to_mont(x: u32) -> u32 {
            (((x as u64) << 32) % P) as u32
        }
        fn from_mont(x: u32) -> u32 {
            const M0: u32 = 0x77ff_ffff; // -P^{-1} mod 2^32
            let red = x.wrapping_mul(M0);
            let t = ((x as u64 + red as u64 * P) >> 32) as u32;
            if t >= P as u32 {
                t - P as u32
            } else {
                t
            }
        }
        let input = pseudo_field_elems(n0 + t0 + a0, 17);
        let encoded: Vec<u32> = input.iter().map(|&v| to_mont(v)).collect();
        let got_fused: Vec<u32> = run(&mut fused, &ctx, &encoded)
            .into_iter()
            .map(from_mont)
            .collect();
        let got_unfused: Vec<u32> = run(&mut unfused, &ctx, &encoded)
            .into_iter()
            .map(from_mont)
            .collect();
        assert_eq!(got_fused, got_unfused);
        assert_eq!(got_fused.len(), k0);
        for (v, &got) in got_fused.iter().enumerate() {
            let (i, j) = (v / t0, v % t0);
            let want = ((input[i + a0] as u64 * input[i + j] as u64) % P + 3) % P;
            assert_eq!(got as u64, want, "mismatch at {v} (i={i}, j={j})");
        }
    }
}
