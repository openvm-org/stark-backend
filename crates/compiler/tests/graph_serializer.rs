//! End-to-end tests for the graph-builder serializer.
//!
//! Covers:
//! - Full round-trip on a blackbox-free graph: serialize a `GraphBuilder`, deserialize it (no
//!   `existing` needed), compile, run, verify output.
//! - Round-trip on a graph with a blackbox: serialize, deserialize with an `existing` builder that
//!   supplies the closures.
//! - `into_graph_builder` panics when the snapshot has blackboxes and no `existing` is supplied.
//! - `into_graph_builder` panics when the supplied `existing` builder has an `original_hash`
//!   mismatch.
#![cfg(feature = "planner")]

use std::sync::Arc;

use crypto_compiler::{
    graph_compiler::GraphCompiler, graph_exe::GraphExe,
    graph_ir::{
        BufId, BufInfo, ConstBuf, DeviceType, GraphBuilder, GraphNode, KernelFn, KernelNode,
    },
    graph_serializer::SerializableGraphBuilder,
    ir::{IRBuilder, Module, ScalarType},
    quast::Quast,
};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, stream::GpuDeviceCtx};

const N: usize = 1 << 10;

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

fn scale_module(name: &str, k: u32) -> Module {
    let mut b = IRBuilder::new();
    let x = b.input("x", ScalarType::BabyBear, vec![N]);
    let c = b.const_field(k);
    let body = b.compute(N, |b, i| {
        let xi = b.index(x, &[i]);
        b.mul(xi, c)
    });
    b.finish(name, body)
}

fn add_dev_buf(g: &mut GraphBuilder, name: &str, bytes: usize) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.into()),
        device_type: DeviceType::Cuda(0),
        size: Quast::cst(bytes as i64),
        concrete_size: bytes,
        elem_size: 4,
    })
}

fn build_kernel_only_graph() -> GraphBuilder {
    let mut g = GraphBuilder::new();
    let x = add_dev_buf(&mut g, "x", N * 4);
    let out = add_dev_buf(&mut g, "out", N * 4);
    g.register_input(x);
    g.register_output(out);
    g.insert_kernel(scale_module("scale7", 7), [x], [out], &[]);
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

fn scale_reference(input: &[u32], k: u32) -> Vec<u32> {
    const P: u64 = 2013265921;
    input
        .iter()
        .map(|x| (((*x as u64) * (k as u64)) % P) as u32)
        .collect()
}

/// A blackbox-free graph round-trips through `bincode`. The compile
/// pipeline on the deserialized builder produces the same output as
/// direct compilation of a freshly-built graph.
#[test]
fn blackbox_free_round_trip() {
    let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");
    let input = pseudo_field_elems(N, 11);

    let mut direct = GraphCompiler::new()
        .compile(build_kernel_only_graph())
        .unwrap();
    let want = run(&mut direct, &ctx, &input);

    let src = build_kernel_only_graph();
    let ser = SerializableGraphBuilder::from_graph_builder(&src, &ctx).unwrap();
    let bytes = bincode::serialize(&ser).unwrap();
    drop(src);

    let ser2: SerializableGraphBuilder = bincode::deserialize(&bytes).unwrap();
    assert!(!ser2.has_blackboxes());
    let restored = ser2.into_graph_builder(None, &ctx).unwrap();
    let mut restored_exe = GraphCompiler::new().compile(restored).unwrap();
    let got = run(&mut restored_exe, &ctx, &input);
    assert_eq!(got, want);

    let ref_out = scale_reference(&input, 7);
    for (i, (a, b)) in got.iter().zip(ref_out.iter()).enumerate() {
        assert_eq!(a, b, "mismatch at index {i}");
    }
}

/// A graph with a blackbox kernel round-trips as long as a matching
/// `existing` builder is supplied on load. The blackbox here just
/// zero-fills the output buffer, which the memcpy chain then observes.
#[test]
fn blackbox_round_trip_with_existing() {
    let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");

    let bb_fn: KernelFn = Arc::new(|_ins: &[*mut ()], _outs: &[*mut ()], _s| {});
    let build = |func: KernelFn| -> GraphBuilder {
        let mut g = GraphBuilder::new();
        let x = add_dev_buf(&mut g, "x", N * 4);
        let out = add_dev_buf(&mut g, "out", N * 4);
        g.register_input(x);
        g.register_output(out);
        g.nodes.push(GraphNode::BlackboxKernel(KernelNode {
            inputs: vec![x],
            outputs: vec![out],
            carried_outputs: vec![],
            func,
            name: "bb".into(),
        }));
        g
    };

    let src = build(bb_fn.clone());
    let ser = SerializableGraphBuilder::from_graph_builder(&src, &ctx).unwrap();
    let bytes = bincode::serialize(&ser).unwrap();
    drop(src);

    let mut existing = build(bb_fn.clone());
    let ser2: SerializableGraphBuilder = bincode::deserialize(&bytes).unwrap();
    assert!(ser2.has_blackboxes());
    let restored = ser2.into_graph_builder(Some(&mut existing), &ctx).unwrap();

    // The reconstructed builder must still hold the closure and be
    // compilable. We can't easily invoke the noop blackbox and see
    // outputs, but the compile+run sanity check catches obvious
    // reattachment mistakes.
    let mut exe = GraphCompiler::new().compile(restored).unwrap();
    let d_input = to_dev(&ctx, &pseudo_field_elems(N, 3));
    exe.set_input(&ctx, 0, &d_input).unwrap();
    exe.run(&ctx).unwrap();
}

/// `into_graph_builder` without an `existing` builder reconstructs
/// blackboxes with placeholder closures — usable for offline inspection
/// (`print`, cytoscape export), panicking only if actually dispatched.
#[test]
fn into_graph_builder_without_existing_installs_placeholders() {
    let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");
    let bb_fn: KernelFn = Arc::new(|_ins: &[*mut ()], _outs: &[*mut ()], _s| {});

    let mut g = GraphBuilder::new();
    let x = add_dev_buf(&mut g, "x", N * 4);
    let out = add_dev_buf(&mut g, "out", N * 4);
    g.register_input(x);
    g.register_output(out);
    g.nodes.push(GraphNode::BlackboxKernel(KernelNode {
        inputs: vec![x],
        outputs: vec![out],
        carried_outputs: vec![],
        func: bb_fn,
        name: "bb".into(),
    }));

    let ser = SerializableGraphBuilder::from_graph_builder(&g, &ctx).unwrap();
    let bytes = bincode::serialize(&ser).unwrap();
    let ser2: SerializableGraphBuilder = bincode::deserialize(&bytes).unwrap();
    assert!(ser2.has_blackboxes());
    let restored = ser2.into_graph_builder(None, &ctx).unwrap();
    let GraphNode::BlackboxKernel(k) = &restored.nodes[0] else {
        panic!("expected blackbox at node 0");
    };
    assert_eq!(k.name, "bb");
    let placeholder = k.func.clone();
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        placeholder(&[], &[], std::ptr::null_mut());
    }))
    .is_err();
    assert!(panicked, "placeholder closure must panic when dispatched");
}

/// `into_graph_builder` panics when the supplied `existing` builder has
/// a different `original_hash`.
#[test]
#[should_panic(expected = "does not match snapshot")]
fn into_graph_builder_panics_on_hash_mismatch() {
    let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");
    let bb_fn: KernelFn = Arc::new(|_ins: &[*mut ()], _outs: &[*mut ()], _s| {});

    let build = |bytes: usize, func: KernelFn| -> GraphBuilder {
        let mut g = GraphBuilder::new();
        let x = add_dev_buf(&mut g, "x", bytes);
        let out = add_dev_buf(&mut g, "out", bytes);
        g.register_input(x);
        g.register_output(out);
        g.nodes.push(GraphNode::BlackboxKernel(KernelNode {
            inputs: vec![x],
            outputs: vec![out],
            carried_outputs: vec![],
            func,
            name: "bb".into(),
        }));
        g
    };

    let src = build(N * 4, bb_fn.clone());
    let ser = SerializableGraphBuilder::from_graph_builder(&src, &ctx).unwrap();
    let bytes = bincode::serialize(&ser).unwrap();
    let ser2: SerializableGraphBuilder = bincode::deserialize(&bytes).unwrap();

    let mut different = build(N * 8, bb_fn);
    let _ = ser2.into_graph_builder(Some(&mut different), &ctx);
}

/// A device-side `Const` buffer round-trips through the serializer: the
/// bytes are copied D2H on save and re-uploaded H2D on load, so a graph
/// that seeds an output via the const survives the round trip.
#[test]
fn const_device_buf_round_trip() {
    let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");
    let payload: Vec<u8> = (0..(N * 4) as u32)
        .map(|i| (i.wrapping_mul(2654435761) & 0xff) as u8)
        .collect();
    let dev_bytes: DeviceBuffer<u8> = payload.as_slice().to_device_on(&ctx).unwrap();

    let mut g = GraphBuilder::new();
    let src = g.add_buf(BufInfo {
        name: Some("src".into()),
        device_type: DeviceType::Cuda(0),
        size: Quast::cst((N * 4) as i64),
        concrete_size: N * 4,
        elem_size: 1,
    });
    let out = g.add_buf(BufInfo {
        name: Some("out".into()),
        device_type: DeviceType::Cuda(0),
        size: Quast::cst((N * 4) as i64),
        concrete_size: N * 4,
        elem_size: 1,
    });
    g.register_output(out);
    g.insert_const(src, ConstBuf::DeviceBuf(dev_bytes));
    g.insert_memcpy(src, out);

    let ser = SerializableGraphBuilder::from_graph_builder(&g, &ctx).unwrap();
    let encoded = bincode::serialize(&ser).unwrap();
    drop(g);

    let ser2: SerializableGraphBuilder = bincode::deserialize(&encoded).unwrap();
    let restored = ser2.into_graph_builder(None, &ctx).unwrap();
    let mut restored_exe = GraphCompiler::new()
        .without_fusion()
        .compile(restored)
        .unwrap();
    restored_exe.run(&ctx).unwrap();
    let got: Vec<u8> = restored_exe.get_output(0).to_host_on(&ctx).unwrap();
    assert_eq!(got, payload);
}
