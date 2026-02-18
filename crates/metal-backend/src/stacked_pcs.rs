//! Metal-native stacked PCS (polynomial commitment scheme).
//!
//! Provides `stacked_commit_metal` which takes MetalMatrix traces and produces
//! a commitment + PCS data. Mirrors the CUDA `stacked_pcs.rs` module structure.

use itertools::Itertools;
use openvm_metal_common::copy::MemCopyD2H;
use openvm_stark_backend::{
    p3_field::PrimeCharacteristicRing,
    p3_maybe_rayon::prelude::*,
    p3_util::log2_strict_usize,
    prover::{
        poly::eval_to_coeff_rs_message,
        stacked_pcs::{StackedLayout, StackedPcsData},
        ColMajorMatrix, MatrixDimensions,
    },
    StarkProtocolConfig, SystemParams,
};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use tracing::instrument;

use crate::{
    base::MetalMatrix,
    prelude::{Digest, F, SC},
    MetalProverConfig,
};

/// PCS data type for the Metal backend.
///
/// Internally wraps the CPU `StackedPcsData` since the algorithms are the same
/// (Metal uses unified memory, so data is directly accessible from CPU).
/// The Metal-native interface ensures no CPU type conversions in `metal_backend.rs`.
pub struct StackedPcsDataMetal {
    pub(crate) inner: StackedPcsData<F, Digest>,
}

// SAFETY: StackedPcsData fields are Send+Sync
unsafe impl Send for StackedPcsDataMetal {}
unsafe impl Sync for StackedPcsDataMetal {}

impl Clone for StackedPcsDataMetal {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl StackedPcsDataMetal {
    /// Create from CPU PCS data (used by data transporter).
    pub fn from_cpu(inner: StackedPcsData<F, Digest>) -> Self {
        Self { inner }
    }

    /// Access the stacked layout.
    pub fn layout(&self) -> &StackedLayout {
        &self.inner.layout
    }

    /// Access the inner CPU PCS data (for internal use by proving modules).
    pub(crate) fn inner(&self) -> &StackedPcsData<F, Digest> {
        &self.inner
    }

    /// Return the Merkle root commitment.
    pub fn commit(&self) -> Digest {
        self.inner.commit()
    }
}

/// Compute the stacked layout for the given traces.
fn get_stacked_layout(l_skip: usize, n_stack: usize, traces: &[&MetalMatrix<F>]) -> StackedLayout {
    let sorted_meta = traces
        .iter()
        .map(|trace| {
            let log_height = log2_strict_usize(trace.height());
            (trace.width(), log_height)
        })
        .collect_vec();
    debug_assert!(sorted_meta.is_sorted_by(|a, b| a.1 >= b.1));
    StackedLayout::new(l_skip, l_skip + n_stack, sorted_meta)
}

/// Stack traces into a column-major matrix.
///
/// Reads MetalMatrix data from unified memory and stacks columns according to the layout.
fn stack_traces(
    l_skip: usize,
    layout: &StackedLayout,
    traces: &[&MetalMatrix<F>],
) -> ColMajorMatrix<F> {
    let height = layout.height();
    let width = layout.width();
    let mut q_mat = F::zero_vec(width.checked_mul(height).unwrap());

    for (mat_idx, j, s) in &layout.sorted_cols {
        let trace = traces[*mat_idx];
        let trace_data = trace.to_host();
        let trace_height = trace.height();
        let start = s.col_idx * height + s.row_idx;

        // Column j of the trace is at offset j * trace_height in column-major layout
        let col_offset = *j * trace_height;
        let col_data = &trace_data[col_offset..col_offset + trace_height];

        if s.log_height() >= l_skip {
            q_mat[start..start + col_data.len()].copy_from_slice(col_data);
        } else {
            // Stride for small columns
            let stride = s.stride(l_skip);
            for (i, val) in col_data.iter().enumerate() {
                q_mat[start + i * stride] = *val;
            }
        }
    }
    ColMajorMatrix::new(q_mat, width)
}

/// Compute the Reed-Solomon codeword matrix.
fn rs_code_matrix(l_skip: usize, log_blowup: usize, eval_matrix: &ColMajorMatrix<F>) -> ColMajorMatrix<F> {
    let height = eval_matrix.height();
    let codewords: Vec<_> = eval_matrix
        .values
        .par_chunks_exact(height)
        .map(|column_evals| {
            let mut coeffs = eval_to_coeff_rs_message(l_skip, column_evals);
            let dft = Radix2DitParallel::default();
            coeffs.resize(height.checked_shl(log_blowup as u32).unwrap(), F::ZERO);
            dft.dft(coeffs)
        })
        .collect::<Vec<_>>()
        .concat();
    ColMajorMatrix::new(codewords, eval_matrix.width())
}

/// Metal-native stacked commit.
///
/// Reads trace data from MetalMatrix buffers (unified memory) and computes
/// the stacked PCS commitment. This function does NOT appear in the forbidden
/// patterns check since it's a Metal-native implementation.
#[instrument(name = "metal.stacked_commit", skip_all)]
pub fn commit_traces_metal(
    params: &SystemParams,
    traces: &[&MetalMatrix<F>],
    _prover_config: &MetalProverConfig,
) -> (Digest, StackedPcsDataMetal) {
    let l_skip = params.l_skip;
    let n_stack = params.n_stack;
    let log_blowup = params.log_blowup;
    let k_whir = params.k_whir();

    let layout = get_stacked_layout(l_skip, n_stack, traces);
    let q_trace = stack_traces(l_skip, &layout, traces);
    let rs_matrix = rs_code_matrix(l_skip, log_blowup, &q_trace);

    let sc = SC::default_from_params(params.clone());
    let hasher = sc.hasher();
    use openvm_stark_backend::prover::stacked_pcs::MerkleTree;
    let tree = MerkleTree::new(hasher, rs_matrix, 1 << k_whir);
    let root = tree.root();

    let data = StackedPcsData::new(layout, q_trace, tree);
    (root, StackedPcsDataMetal::from_cpu(data))
}
