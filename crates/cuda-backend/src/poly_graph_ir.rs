//! Graph-IR helpers for extension-field batch NTTs over small (`2^l_skip`)
//! domains.
//!
//! This module mirrors, on a [`GraphBuilder`], the batch DFT / iDFT that the
//! host performs over `EF` polynomials in the logup-zerocheck round-0 path
//! (`openvm_stark_backend::prover::poly::UnivariatePoly::from_evals_idft` and
//! `::from_geometric_cosets_evals_idft`, which call
//! [`openvm_stark_backend::dft::Radix2BowersSerial`] on a
//! `RowMajorMatrix<EF>`). The eager CUDA backend has no `EF` small-domain NTT
//! primitive at all — [`crate::cuda::batch_ntt_small::batch_ntt_small`] is
//! base-field only — so this is the missing device primitive rather than a
//! mirror of an existing eager CUDA call.
//!
//! # How it works
//!
//! A DFT over a two-adic subgroup of `F` is `F`-linear, and `EF = F^{D_EF}` as
//! an `F`-vector space with coordinates `elems[0..D_EF]`. So transforming an
//! `EF` polynomial is exactly transforming each of its `D_EF` base-field
//! *lanes* independently. The composition is three existing kernels:
//!
//! 1. [`split_ext_to_base_col_major_matrix`] with `poly_len == matrix_height == m` writes
//!    `lanes[lane * m + idx] = input[idx].elems[lane]`, i.e. four contiguous lane-major arrays.
//! 2. [`batch_ntt_small`] transforms `D_EF * num_polys` unit-stride blocks of `2^l_skip` elements
//!    each. In the layout above, block `lane * num_polys + poly` is exactly lane `lane` of
//!    polynomial `poly`.
//! 3. [`transpose_fp_to_fpext_vec`] performs the exact inverse lane mapping, `out[idx].elems[lane]
//!    = lanes[lane * m + idx]`, reassembling `EF`.
//!
//! # Ordering
//!
//! **Natural order in, natural order out. Do not add a bit-reversal step.**
//! The `batch_ntt_kernel` helper leaves thread `i` holding the DIF
//! bit-reversed-position value, but the kernel's final store scatters it to
//! `rev_len(i, l_skip)` (`cuda/src/batch_ntt_small.cu:110-112`), so the
//! externally visible buffer is in natural order. Both eager wrappers document
//! that contract ("Use natural ordering", [`crate::poly::PleMatrix`]). The
//! internal helper name `ntt_natural_to_bitrev` describes the *register*
//! convention, not the buffer contract; inserting a reconciliation permutation
//! here would silently corrupt coefficients.
//!
//! # Scope
//!
//! This is reusable enablement only. Nothing in the eager prover is routed
//! through it, and the round-0 host seam is *not* closed: the live path still
//! synchronously exports round-0 evaluator outputs
//! (`logup_zerocheck/mod.rs:857`, `:914`) and does the geometric unshift and
//! `width x width` Lagrange assembly on the host
//! (`openvm_stark_backend::prover::poly`, `from_geometric_cosets_evals_idft`).

use std::mem::{forget, size_of};

use crypto_compiler::{
    graph_ir::{BufId, BufInfo, DeviceType, GraphBuilder},
    quast::Quast,
};
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_stark_backend::prover::fractional_sumcheck_gkr::Frac;

use crate::{
    cuda::{
        batch_ntt_small::{
            batch_ntt_small, ensure_device_ntt_twiddles_initialized, validate_gpu_l_skip,
        },
        matrix::{matrix_transpose_fpext, split_ext_to_base_col_major_matrix},
        poly::transpose_fp_to_fpext_vec,
    },
    prelude::{EF, F},
    types::D_EF,
};

// ---------------------------------------------------------------------------
// Buffer allocation helper.

/// Allocate a device buffer of `n` `T` elements on `device`.
///
/// Local twin of `logup_zerocheck::fractional_ir::add_ef_buf` /
/// `logup_zerocheck::zerocheck_ir::add_f_buf`, kept here so this module has no
/// build-order coupling to either mirror.
fn add_typed_buf<T>(g: &mut GraphBuilder, device: DeviceType, name: &str, n: usize) -> BufId {
    let elem_size = size_of::<T>();
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: device,
        size: Quast::cst((n * elem_size) as i64),
        concrete_size: n * elem_size,
        elem_size,
    })
}

// ---------------------------------------------------------------------------
// Blackbox kernel wrappers (one `insert_blackbox_kernel` each).

/// Insert a [`split_ext_to_base_col_major_matrix`] node with
/// `poly_len == matrix_height == m`: `input: EF[m] -> lanes: F[D_EF * m]`,
/// laid out lane-major (`lanes[lane * m + idx] = input[idx].elems[lane]`).
pub fn split_ext_to_base_lanes_ir(g: &mut GraphBuilder, input: BufId, lanes: BufId, m: usize) {
    g.insert_blackbox_kernel(
        "split_ext_to_base_col_major_matrix",
        std::iter::once(input),
        std::iter::once(lanes),
        std::iter::once(false),
        move |inputs, outputs, stream| unsafe {
            let src = DeviceBuffer::<EF>::from_raw_parts(inputs[0] as *mut EF, m);
            let mut dst = DeviceBuffer::<F>::from_raw_parts(outputs[0] as *mut F, D_EF * m);
            split_ext_to_base_col_major_matrix(&mut dst, &src, m as u64, m as u32, stream)
                .expect("split_ext_to_base_col_major_matrix");
            forget(src);
            forget(dst);
        },
    );
}

/// Insert an in-place [`batch_ntt_small`] node over `cnt_blocks` unit-stride
/// blocks of `2^l_skip` base-field elements. `buf_len` is the number of `F`
/// elements owned by the buffer (used only to rebuild a borrowed view).
pub fn batch_ntt_small_ir(
    g: &mut GraphBuilder,
    buf: BufId,
    buf_len: usize,
    l_skip: usize,
    cnt_blocks: usize,
    is_intt: bool,
) {
    g.insert_blackbox_kernel(
        "batch_ntt_small",
        std::iter::once(buf),
        std::iter::empty(),
        std::iter::once(true),
        move |inputs, _outputs, stream| unsafe {
            let mut view = DeviceBuffer::<F>::from_raw_parts(inputs[0] as *mut F, buf_len);
            batch_ntt_small(&mut view, l_skip, cnt_blocks, is_intt, stream)
                .expect("batch_ntt_small");
            forget(view);
        },
    );
}

/// Insert a [`transpose_fp_to_fpext_vec`] node: `lanes: F[D_EF * m] -> out:
/// EF[m]`, the exact inverse of [`split_ext_to_base_lanes_ir`].
pub fn join_base_lanes_to_ext_ir(g: &mut GraphBuilder, lanes: BufId, out: BufId, m: usize) {
    g.insert_blackbox_kernel(
        "transpose_fp_to_fpext_vec",
        std::iter::once(lanes),
        std::iter::once(out),
        std::iter::once(false),
        move |inputs, outputs, stream| unsafe {
            let src = DeviceBuffer::<F>::from_raw_parts(inputs[0] as *mut F, D_EF * m);
            let mut dst = DeviceBuffer::<EF>::from_raw_parts(outputs[0] as *mut EF, m);
            transpose_fp_to_fpext_vec(&mut dst, &src, stream).expect("transpose_fp_to_fpext_vec");
            forget(src);
            forget(dst);
        },
    );
}

/// Insert a [`matrix_transpose_fpext`] node transposing a row-major
/// `height x width` `EF` matrix into a row-major `width x height` one.
pub fn matrix_transpose_ext_ir(
    g: &mut GraphBuilder,
    input: BufId,
    output: BufId,
    width: usize,
    height: usize,
) {
    g.insert_blackbox_kernel(
        "matrix_transpose_fpext",
        std::iter::once(input),
        std::iter::once(output),
        std::iter::once(false),
        move |inputs, outputs, stream| unsafe {
            let src = DeviceBuffer::<EF>::from_raw_parts(inputs[0] as *mut EF, width * height);
            let dst = DeviceBuffer::<EF>::from_raw_parts(outputs[0] as *mut EF, width * height);
            matrix_transpose_fpext(&dst, &src, width, height, stream)
                .expect("matrix_transpose_fpext");
            forget(src);
            forget(dst);
        },
    );
}

// ---------------------------------------------------------------------------
// Compositions.

/// Batch DFT (`is_intt == false`) or iDFT (`is_intt == true`) of `num_polys`
/// `EF` polynomials of length `2^l_skip` each, over the two-adic subgroup of
/// `F` of that size.
///
/// `input` is `EF[num_polys << l_skip]` in **poly-major** order: polynomial
/// `j`'s coefficients (resp. evaluations) occupy
/// `input[j << l_skip .. (j + 1) << l_skip]` in **natural** order. That is the
/// column-major view of the `RowMajorMatrix<EF>` of height `2^l_skip` and
/// width `num_polys` that
/// [`openvm_stark_backend::dft::Radix2BowersSerial::dft_batch`] /
/// `idft_batch` operate on. The returned buffer has the same shape and
/// ordering.
///
/// `input` is only read, so it may be a graph const buffer.
///
/// # Panics
/// If `num_polys == 0` or `l_skip > MAX_SMALL_NTT_LEVEL`.
#[allow(clippy::too_many_arguments)]
pub fn batch_ntt_small_ext_ir(
    g: &mut GraphBuilder,
    input: BufId,
    num_polys: usize,
    l_skip: usize,
    is_intt: bool,
    device: DeviceType,
    name: &str,
) -> BufId {
    assert!(num_polys > 0, "batch_ntt_small_ext_ir: num_polys == 0");
    validate_gpu_l_skip(l_skip).expect("batch_ntt_small_ext_ir: l_skip > MAX_SMALL_NTT_LEVEL");
    let m = num_polys << l_skip;
    assert!(
        m <= u32::MAX as usize,
        "batch_ntt_small_ext_ir: {m} elements exceeds the kernel's u32 matrix_height"
    );

    let lanes = add_typed_buf::<F>(g, device, &format!("{name}_lanes"), D_EF * m);
    let out = add_typed_buf::<EF>(g, device, &format!("{name}_out"), m);

    split_ext_to_base_lanes_ir(g, input, lanes, m);
    if l_skip > 0 {
        // `batch_ntt_small` is a documented no-op at `l_skip == 0`; skip the
        // node entirely rather than emit one that does nothing.
        //
        // The twiddle table lives in CUDA constant memory and its first-use
        // upload synchronizes, which a blackbox closure must never do. Force
        // it here, at graph-build time, so the closure hits the initialized
        // fast path.
        //
        // TODO(cc-ir): this makes graph *building* require a live CUDA
        // context on the same device the graph will later run on.
        // WHY: `ensure_device_ntt_twiddles_initialized` keys off `get_device()`
        // at call time; there is no way to name the target device from a
        // `DeviceType` here without widening the cuda-common API.
        // RISK: building the graph on a thread bound to a different device
        // than the one that executes it would leave the executing device's
        // constant memory uninitialized, and the *first* in-graph launch would
        // then synchronize instead of failing loudly.
        ensure_device_ntt_twiddles_initialized()
            .expect("batch_ntt_small_ext_ir: device NTT twiddles");
        batch_ntt_small_ir(g, lanes, D_EF * m, l_skip, D_EF * num_polys, is_intt);
    }
    join_base_lanes_to_ext_ir(g, lanes, out, m);
    out
}

/// Batch DFT / iDFT of the numerator and denominator halves of
/// `Frac<EF>[num_polys << l_skip]`.
///
/// `input` is in poly-major order (`input[(j << l_skip) + i]` is fraction `i`
/// of polynomial `j`). The output is `EF[2 * num_polys << l_skip]`: all
/// transformed `p` values first (poly-major), then all transformed `q` values,
/// matching the `unzip` the eager round-0 path performs on the host.
///
/// Normalization (e.g. the negative-lift scaling the eager path applies to
/// numerators only) is deliberately *not* part of this primitive — it is not
/// symmetric in `p` and `q`.
#[allow(clippy::too_many_arguments)]
pub fn batch_ntt_small_frac_ext_ir(
    g: &mut GraphBuilder,
    input: BufId,
    num_polys: usize,
    l_skip: usize,
    is_intt: bool,
    device: DeviceType,
    name: &str,
) -> BufId {
    assert!(num_polys > 0, "batch_ntt_small_frac_ext_ir: num_polys == 0");
    let m = num_polys << l_skip;

    // `Frac<EF>` is `#[repr(C)] { p: EF, q: EF }`, so the buffer is a row-major
    // `m x 2` matrix of `EF`. Transposing it yields `[all p][all q]`.
    debug_assert_eq!(size_of::<Frac<EF>>(), 2 * size_of::<EF>());
    let unzipped = add_typed_buf::<EF>(g, device, &format!("{name}_unzipped"), 2 * m);
    matrix_transpose_ext_ir(g, input, unzipped, 2, m);

    batch_ntt_small_ext_ir(g, unzipped, 2 * num_polys, l_skip, is_intt, device, name)
}

// ---------------------------------------------------------------------------
// Tests.

#[cfg(test)]
mod poly_graph_ir_tests {
    use crypto_compiler::{graph_compiler::GraphCompiler, graph_ir::ConstBuf};
    use openvm_cuda_common::{
        common::get_device,
        stream::{CudaStream, GpuDeviceCtx, StreamGuard},
    };
    use openvm_stark_backend::{
        dft::Radix2BowersSerial, p3_matrix::dense::RowMajorMatrix,
        prover::fractional_sumcheck_gkr::Frac,
    };
    use p3_dft::TwoAdicSubgroupDft;
    use p3_field::PrimeCharacteristicRing;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;

    /// Shapes under test: `(num_polys, l_skip)`. `l_skip = 10` is
    /// `MAX_SMALL_NTT_LEVEL`, the shared-memory path of the CUDA kernel.
    const NUM_POLYS: [usize; 4] = [1, 2, 3, 4];
    const L_SKIPS: [usize; 5] = [0, 1, 2, 4, 10];

    fn test_ctx() -> GpuDeviceCtx {
        GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        }
    }

    fn as_bytes<T>(data: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        }
    }

    /// Stage `data` as a read-only graph const buffer.
    fn const_buf<T>(g: &mut GraphBuilder, device: DeviceType, name: &str, data: &[T]) -> BufId {
        let buf = add_typed_buf::<T>(g, device, name, data.len());
        g.insert_const(buf, ConstBuf::HostBuf(as_bytes(data).to_vec()));
        buf
    }

    /// Copy `src` into a freshly registered output buffer of `n` `EF`s and
    /// return the output buffer id.
    fn export_ef(
        g: &mut GraphBuilder,
        device: DeviceType,
        name: &str,
        src: BufId,
        n: usize,
    ) -> BufId {
        let out = add_typed_buf::<EF>(g, device, name, n);
        g.insert_memcpy(src, out);
        g.register_output(out);
        out
    }

    /// Compile a graph with no runtime inputs, run it, and read back the given
    /// registered output buffers as raw device bytes.
    fn run_graph_read_bufs(g: GraphBuilder, bufs: &[BufId], ctx: &GpuDeviceCtx) -> Vec<Vec<u8>> {
        let mut exe = GraphCompiler::new()
            .device(DeviceType::Cuda(0))
            .compile(g)
            .expect("graph compile");
        exe.run(ctx).expect("graph run");
        bufs.iter()
            .map(|&bid| {
                let idx = (0..exe.num_outputs())
                    .find(|&i| exe.output_buf_id(i) == bid)
                    .expect("output buf");
                exe.get_output(idx).to_host_on(ctx).expect("D2H")
            })
            .collect()
    }

    fn seeded_ef(len: usize, seed: u64) -> Vec<EF> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..len).map(|_| rng.random::<EF>()).collect()
    }

    /// Host oracle: [`Radix2BowersSerial`] applied to the `2^l_skip x
    /// num_polys` `RowMajorMatrix<EF>` that `input` (poly-major) represents,
    /// returned back in poly-major order.
    ///
    /// Forward and inverse go through `dft_batch` / `idft_batch` separately —
    /// deliberately *not* a roundtrip, which would cancel a matching ordering
    /// error on both sides.
    fn host_batch_ntt_ext(input: &[EF], num_polys: usize, l_skip: usize, is_intt: bool) -> Vec<EF> {
        let n = 1usize << l_skip;
        assert_eq!(input.len(), num_polys * n);
        // poly-major (column-major) -> row-major
        let mut values = Vec::with_capacity(input.len());
        for i in 0..n {
            for j in 0..num_polys {
                values.push(input[j * n + i]);
            }
        }
        let mat = RowMajorMatrix::new(values, num_polys);
        let dft = Radix2BowersSerial;
        let out = if is_intt {
            dft.idft_batch(mat)
        } else {
            dft.dft_batch(mat)
        };
        // row-major -> poly-major
        let mut res = vec![EF::ZERO; input.len()];
        for i in 0..n {
            for j in 0..num_polys {
                res[j * n + i] = out.values[i * num_polys + j];
            }
        }
        res
    }

    #[test]
    fn ef_batch_ntt_ir_matches_bowers_serial() {
        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);
        let mut g = GraphBuilder::new();

        struct Case {
            num_polys: usize,
            l_skip: usize,
            is_intt: bool,
            out: BufId,
            expected: Vec<EF>,
        }
        let mut cases = Vec::new();
        let mut seed = 0u64;
        for &num_polys in NUM_POLYS.iter() {
            for &l_skip in L_SKIPS.iter() {
                for is_intt in [false, true] {
                    seed = seed.wrapping_add(1);
                    let m = num_polys << l_skip;
                    let input = seeded_ef(m, 0xE7_0000 + seed);
                    let expected = host_batch_ntt_ext(&input, num_polys, l_skip, is_intt);
                    let name = format!("ef_p{num_polys}_l{l_skip}_i{}", u8::from(is_intt));
                    let inbuf = const_buf(&mut g, device, &format!("{name}_in"), &input);
                    let got = batch_ntt_small_ext_ir(
                        &mut g, inbuf, num_polys, l_skip, is_intt, device, &name,
                    );
                    let out = export_ef(&mut g, device, &format!("{name}_export"), got, m);
                    cases.push(Case {
                        num_polys,
                        l_skip,
                        is_intt,
                        out,
                        expected,
                    });
                }
            }
        }

        let ids: Vec<BufId> = cases.iter().map(|c| c.out).collect();
        let got = run_graph_read_bufs(g, &ids, &ctx);
        for (case, bytes) in cases.iter().zip(got) {
            assert_eq!(
                bytes,
                as_bytes(&case.expected),
                "mismatch for (num_polys={}, l_skip={}, is_intt={})",
                case.num_polys,
                case.l_skip,
                case.is_intt,
            );
        }
    }

    #[test]
    fn frac_ext_batch_ntt_ir_matches_bowers_serial() {
        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);
        let mut g = GraphBuilder::new();

        struct Case {
            num_polys: usize,
            l_skip: usize,
            is_intt: bool,
            out: BufId,
            expected: Vec<EF>,
        }
        let mut cases = Vec::new();
        let mut seed = 0u64;
        for &num_polys in NUM_POLYS.iter() {
            for &l_skip in L_SKIPS.iter() {
                for is_intt in [false, true] {
                    seed = seed.wrapping_add(1);
                    let m = num_polys << l_skip;
                    // Independent `p` and `q` streams: an incorrect
                    // `[p,q,p,q]` / `[all p][all q]` de-interleave would be
                    // invisible if they were correlated.
                    let ps = seeded_ef(m, 0xF7_0000 + seed);
                    let qs = seeded_ef(m, 0xF8_0000 + seed);
                    let input: Vec<Frac<EF>> =
                        ps.iter().zip(&qs).map(|(&p, &q)| Frac { p, q }).collect();

                    let mut expected = host_batch_ntt_ext(&ps, num_polys, l_skip, is_intt);
                    expected.extend(host_batch_ntt_ext(&qs, num_polys, l_skip, is_intt));

                    let name = format!("frac_p{num_polys}_l{l_skip}_i{}", u8::from(is_intt));
                    let inbuf = const_buf(&mut g, device, &format!("{name}_in"), &input);
                    let got = batch_ntt_small_frac_ext_ir(
                        &mut g, inbuf, num_polys, l_skip, is_intt, device, &name,
                    );
                    let out = export_ef(&mut g, device, &format!("{name}_export"), got, 2 * m);
                    cases.push(Case {
                        num_polys,
                        l_skip,
                        is_intt,
                        out,
                        expected,
                    });
                }
            }
        }

        let ids: Vec<BufId> = cases.iter().map(|c| c.out).collect();
        let got = run_graph_read_bufs(g, &ids, &ctx);
        for (case, bytes) in cases.iter().zip(got) {
            assert_eq!(
                bytes,
                as_bytes(&case.expected),
                "mismatch for (num_polys={}, l_skip={}, is_intt={})",
                case.num_polys,
                case.l_skip,
                case.is_intt,
            );
        }
    }
}
