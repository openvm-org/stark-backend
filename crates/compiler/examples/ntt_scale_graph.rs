//! Example: build a [`graph_ir`] graph that chains this crate's radix-2 NTT
//! kernel with a small elementwise "multiply by 2" kernel built through the
//! `kernel!` macro, then run it end-to-end on the GPU via [`GraphCompiler`]
//! / [`GraphExe`] and compare the result against a plonky3 `Radix2Dit`
//! reference (scaled by 2 in BabyBear).
//!
//! Requires the `planner` feature; needs an OR-Tools install (see the
//! `planner` docs in `Cargo.toml`) and a working CUDA setup.
//!
//! Run with:
//!     cargo run -p crypto-compiler --features planner \
//!         --example ntt_scale_graph -- 10

use crypto_compiler::{
    graph_exe::GraphCompiler,
    graph_ir::{BufId, BufInfo, ConstBuf, DeviceType, GraphBuilder},
    ir::{IRBuilder, ScalarType},
    kernel, kernels,
    quast::Quast,
    runtime::CompileOptions,
};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::GpuDeviceCtx,
};
use p3_baby_bear::BabyBear;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{PrimeCharacteristicRing, PrimeField32};

fn main() {
    let log_n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let n = 1usize << log_n;
    println!("Graph: NTT(log_n = {log_n}, n = {n}) then * 2 in BabyBear");

    // ---- Build graph -------------------------------------------------------
    let mut g = GraphBuilder::new();
    let n_bytes_sym = g.register_symbol("n_bytes");
    let n_bytes = Quast::sym(n_bytes_sym);
    let half_n_bytes = n_bytes.floordiv(2);

    let a_buf = add_dev_buf(&mut g, "a", n_bytes.clone());
    let w_buf = add_dev_buf(&mut g, "twiddles", half_n_bytes);
    let ntt_out = add_dev_buf(&mut g, "ntt_out", n_bytes.clone());
    let scaled_out = add_dev_buf(&mut g, "scaled_out", n_bytes);

    // Twiddle table: baked into a HostBuf Const, memcpy'd to device by the
    // graph runtime.
    let twiddles = kernels::ntt_twiddles(log_n);
    g.insert_const(w_buf, ConstBuf::HostBuf(u32_to_le_bytes(&twiddles)));

    let ntt = kernels::ntt_module(log_n);
    g.insert_kernel(ntt, [a_buf, w_buf], [ntt_out]);

    let scale = build_scale_by_two_module(n);
    g.insert_kernel(scale, [ntt_out], [scaled_out]);

    // ---- Dump builder IR ---------------------------------------------------
    println!("\n----- GraphBuilder.print() -----");
    print!("{}", g.print());

    // ---- Compile ------------------------------------------------------------
    let mut exe = GraphCompiler::new()
        .device(DeviceType::Cuda(0))
        .symbol(n_bytes_sym, (n * 4) as i64)
        .compile_options(CompileOptions::default())
        .compile(g)
        .expect("graph compile");

    println!("\n----- GraphExe.print() -----");
    print!("{}", exe.print());

    assert_eq!(exe.num_inputs(), 1, "only `a` is a graph input");
    assert_eq!(exe.num_outputs(), 1, "only `scaled_out` is a graph output");
    assert_eq!(exe.input_buf_id(0), a_buf);
    assert_eq!(exe.output_buf_id(0), scaled_out);

    // ---- Run ---------------------------------------------------------------
    let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");
    let coeffs = pseudo_field_elems(n, 1);
    let coeffs_bytes = u32_to_le_bytes(&coeffs);

    let input_buf: DeviceBuffer<u8> = coeffs_bytes
        .as_slice()
        .to_device_on(&ctx)
        .expect("input H2D");
    let mut output_buf: DeviceBuffer<u8> = DeviceBuffer::with_capacity_on(exe.output_size(0), &ctx);
    let mut scratch: DeviceBuffer<u8> =
        DeviceBuffer::with_capacity_on(exe.scratch_bytes().max(1), &ctx);

    exe.run(
        &ctx,
        &[input_buf],
        std::slice::from_mut(&mut output_buf),
        &mut scratch,
    )
    .expect("graph run");

    let got_bytes: Vec<u8> = output_buf.to_host_on(&ctx).expect("D2H");
    let got = le_bytes_to_u32(&got_bytes);

    // ---- Reference ---------------------------------------------------------
    let want: Vec<u32> = {
        let input_f: Vec<BabyBear> = coeffs.iter().map(|&c| BabyBear::new(c)).collect();
        Radix2Dit::default()
            .dft(input_f)
            .into_iter()
            .map(|x| (x * BabyBear::TWO).as_canonical_u32())
            .collect()
    };

    assert_eq!(
        got.len(),
        want.len(),
        "output length {} != reference length {}",
        got.len(),
        want.len()
    );
    for i in 0..want.len() {
        assert_eq!(
            got[i], want[i],
            "mismatch at index {i}: got {}, want {}",
            got[i], want[i]
        );
    }
    println!("OK — GraphExe output matches p3 Radix2Dit * 2 across all {n} elements.");
}

fn add_dev_buf(g: &mut GraphBuilder, name: &str, size: Quast) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: DeviceType::Cuda(0),
        size,
        elem_size: 4,
    })
}

/// `scale_by_two(a: BB[n]) -> BB[n]`, expressed through the `kernel!` macro.
fn build_scale_by_two_module(n: usize) -> crypto_compiler::ir::Module {
    let mut b = IRBuilder::new();
    let a = b.input("a", ScalarType::BabyBear, vec![n]);
    let body = kernel!(b, compute[n] | i | { a[i] * 2bb });
    b.finish(format!("scale_by_two_{n}"), body)
}

/// Deterministic pseudo-random canonical BabyBear elements (splitmix64) —
/// mirrors the helper used in `tests/gpu.rs`.
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

fn u32_to_le_bytes(v: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn le_bytes_to_u32(v: &[u8]) -> Vec<u32> {
    v.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
