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
    g.insert_kernel(chain_module(0), [x], [out]);
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
        g.insert_kernel(scale("scale2", 2), [x], [staging]);
        g.insert_kernel(scale("scale3", 3), [staging], [out]);
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
            g.insert_kernel(kernels::ntt_module(log_n), [a, w], [out]);
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
    g.insert_kernel(chain_module(0), [x0], [out0]);
    g.insert_kernel(chain_module(3), [x1], [out1]);

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
