//! Row-major LogupZerocheck implementation.
//!
//! Optimizations over the reference backend:
//! 1. Eliminates full row-major → col-major conversion for round 0 sumcheck
//! 2. Uses batch DFT on extracted row-major blocks, leveraging SIMD through
//!    plonky3's butterfly operations (PackedField in apply_to_rows)
//! 3. Direct row-major access for constraint/interaction evaluation

use std::{
    cmp::max,
    iter::{self, zip},
    mem::take,
    ops::{Add, Mul, Neg, Sub},
};

use itertools::{izip, Itertools};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{
    batch_multiplicative_inverse, ExtensionField, Field, PackedValue, PrimeCharacteristicRing,
    TwoAdicField,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::{debug, info_span, instrument};

use openvm_stark_backend::{
    air_builders::symbolic::{
        symbolic_expression::SymbolicEvaluator,
        symbolic_variable::{Entry, SymbolicVariable},
        SymbolicConstraints, SymbolicExpressionDag, SymbolicExpressionNode,
    },
    calculate_n_logup,
    dft::Radix2BowersSerial,
    interaction::SymbolicInteraction,
    poly_common::{
        eq_sharp_uni_poly, eq_uni_poly, eval_eq_mle, eval_eq_sharp_uni, eval_eq_uni,
        UnivariatePoly,
    },
    proof::{column_openings_by_rot, BatchConstraintProof, GkrProof},
    prover::{
        fractional_sumcheck_gkr::{fractional_sumcheck, Frac},
        poly::evals_eq_hypercubes,
        stacked_pcs::StackedLayout,
        sumcheck::{
            batch_fold_mle_evals, batch_fold_ple_evals, sumcheck_round0_deg,
            sumcheck_round_poly_evals,
        },
        AirProvingContext, ColMajorMatrix, DeviceMultiStarkProvingKey, MatrixDimensions,
        ProverBackend, ProvingContext,
    },
    FiatShamirTranscript, StarkProtocolConfig,
};

use crate::backend::{CpuBackend, RowMajorMatrixWrapper};

// ============================================================================
// Batch DFT helpers for row-major sumcheck
// ============================================================================

/// Extract `2^l_skip` rows from a row-major matrix for hypercube point `x`.
/// Returns a RowMajorMatrix suitable for batch DFT.
///
/// For the hyperprism D_n evaluation, the rows for point `x` are:
///   row[(x << l_skip) + z + offset] for z = 0..2^l_skip
/// where offset=1 for rotation, 0 otherwise.
#[inline]
fn extract_rm_block<F: TwoAdicField>(
    rm: &RowMajorMatrix<F>,
    x: usize,
    l_skip: usize,
    offset: usize,
) -> RowMajorMatrix<F> {
    let height = rm.values.len() / rm.width;
    let w = rm.width;
    let sz = 1usize << l_skip;
    let base = x << l_skip;
    let mut vals = Vec::with_capacity(sz * w);
    for z in 0..sz {
        let r = (base + z + offset) % height;
        let start = r * w;
        vals.extend_from_slice(&rm.values[start..start + w]);
    }
    RowMajorMatrix::new(vals, w)
}

/// Extract `2^l_skip` rows from a ColMajorMatrix for hypercube point `x`.
/// Returns a RowMajorMatrix suitable for batch DFT.
#[inline]
fn extract_cm_block<F: TwoAdicField>(
    cm: &ColMajorMatrix<F>,
    x: usize,
    l_skip: usize,
) -> RowMajorMatrix<F> {
    let h = cm.height();
    let w = cm.width();
    let sz = 1usize << l_skip;
    let base = x << l_skip;
    let mut vals = Vec::with_capacity(sz * w);
    for z in 0..sz {
        let row = (base + z) % h;
        for c in 0..w {
            vals.push(cm.values[c * h + row]);
        }
    }
    RowMajorMatrix::new(vals, w)
}

/// Batch coset DFT: given iDFT'd coefficients in a RowMajorMatrix,
/// evaluate on the coset `shift * D` where D = <omega_skip>.
///
/// Steps: 1. Multiply row i by shift^i (twisting)
///        2. Forward batch DFT
#[inline]
fn batch_coset_dft<F: TwoAdicField>(coeffs: &RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F> {
    let w = coeffs.width;
    let mut mat = coeffs.clone();
    // Twist: multiply row i by shift^i
    let mut s = F::ONE;
    for chunk in mat.values.chunks_exact_mut(w) {
        if s != F::ONE {
            for v in chunk.iter_mut() {
                *v *= s;
            }
        }
        s *= shift;
    }
    Radix2BowersSerial.dft_batch(mat)
}

/// Fold PLE evaluations directly from row-major data, avoiding the O(n*m) transpose.
///
/// Barycentric interpolation: for each column j,
///   result[j] = scaling_factor * sum_i (col_scale[i] * mat[row_i, j])
///
/// Since rows are contiguous in row-major layout, this is a weighted sum of rows
/// — extremely cache-friendly compared to column-major interpolation.
fn fold_ple_evals_rowmajor<F, EF>(
    l_skip: usize,
    rm: &RowMajorMatrix<F>,
    is_rot: bool,
    r: EF,
) -> ColMajorMatrix<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    let height = rm.values.len() / rm.width;
    let width = rm.width;
    let lifted_height = height.max(1 << l_skip);
    let skip_sz = 1usize << l_skip;
    let new_height = lifted_height >> l_skip;
    let offset = usize::from(is_rot);

    // Precompute barycentric weights (same as in p3-interpolation)
    let omega = F::two_adic_generator(l_skip);
    let omega_pows: Vec<F> = omega.powers().take(skip_sz).collect_vec();
    let denoms: Vec<EF> = omega_pows.iter().map(|&x_i| r - EF::from(x_i)).collect_vec();
    let inv_denoms = batch_multiplicative_inverse(&denoms);

    // col_scale[i] = omega^i / (r - omega^i)
    let col_scale: Vec<EF> = omega_pows
        .iter()
        .zip(&inv_denoms)
        .map(|(&sg, &diff_inv)| diff_inv * sg)
        .collect_vec();

    let log_skip = l_skip;
    let point_pow_height = r.exp_power_of_2(log_skip);
    // shift = F::ONE, so shift_pow_height = 1
    let shift_pow_height = EF::ONE;
    let vanishing_polynomial = point_pow_height - shift_pow_height;
    let denominator = shift_pow_height.mul_2exp_u64(log_skip as u64);
    let scaling_factor = vanishing_polynomial * denominator.inverse();

    // Output is col-major: values[col * new_height + x] for each (x, col)
    let values: Vec<EF> = (0..new_height)
        .into_par_iter()
        .flat_map(|x| {
            // For this x-point, compute weighted sum of 2^l_skip rows
            let mut result = vec![EF::ZERO; width];
            for z in 0..skip_sz {
                let row_idx = ((x << l_skip) + z + offset) % height;
                let row_start = row_idx * width;
                let row = &rm.values[row_start..row_start + width];
                let w = col_scale[z];
                for (j, &val) in row.iter().enumerate() {
                    result[j] += w * val;
                }
            }
            // Apply scaling factor
            for v in &mut result {
                *v *= scaling_factor;
            }
            result
        })
        .collect();

    // Convert from row-major output (x varies fastest) to col-major
    let mut cm_values = EF::zero_vec(width * new_height);
    for x in 0..new_height {
        for j in 0..width {
            cm_values[j * new_height + x] = values[x * width + j];
        }
    }
    ColMajorMatrix::new(cm_values, width)
}

/// Run the round-0 sumcheck using batch DFT from row-major data.
///
/// This replaces the pattern of: to_col_major_owned → sumcheck_uni_round0_poly
/// with direct row-major block extraction → batch iDFT → batch coset DFT → callback.
///
/// Key optimizations:
/// 1. No full matrix transpose (saves ~100-200ms allocation + O(n) work)
/// 2. Batch DFT processes all columns at once; butterfly `apply_to_rows` uses PackedField SIMD
/// 3. Row extraction from row-major is contiguous (cache-friendly)
fn sumcheck_uni_round0_batch<F, EF, FN, const WD: usize>(
    l_skip: usize,
    n: usize,
    d: usize,
    rm_mats: &[(&RowMajorMatrix<F>, bool)],
    sels_cm: &ColMajorMatrix<F>,
    w: FN,
) -> [UnivariatePoly<EF>; WD]
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    FN: Fn(F, usize, &[Vec<F>]) -> [EF; WD] + Sync,
{
    if d == 0 {
        return std::array::from_fn(|_| UnivariatePoly::new(vec![]));
    }
    let g = F::GENERATOR;
    let omega_skip = F::two_adic_generator(l_skip);
    let coset_shifts: Vec<F> = g.powers().skip(1).take(d).collect_vec();
    let skip_sz = 1usize << l_skip;

    // Map-Reduce over x ∈ H_n
    let evals = (0..1usize << n).into_par_iter().map(|x| {
        let dft = Radix2BowersSerial;

        // Extract and batch-iDFT selector block (small: 3 columns)
        let sels_block = extract_cm_block(sels_cm, x, l_skip);
        let sels_coeffs = dft.idft_batch(sels_block);
        let sels_cosets: Vec<RowMajorMatrix<F>> = coset_shifts
            .iter()
            .map(|&shift| batch_coset_dft(&sels_coeffs, shift))
            .collect();

        // Extract and batch-iDFT each trace matrix block
        // Each block is 2^l_skip × width, contiguous row extraction from row-major
        let mat_coeffs: Vec<RowMajorMatrix<F>> = rm_mats
            .iter()
            .map(|(rm, is_rot)| {
                let block = extract_rm_block(rm, x, l_skip, usize::from(*is_rot));
                dft.idft_batch(block)
            })
            .collect();
        let mat_cosets: Vec<Vec<RowMajorMatrix<F>>> = mat_coeffs
            .iter()
            .map(|coeffs| {
                coset_shifts
                    .iter()
                    .map(|&shift| batch_coset_dft(coeffs, shift))
                    .collect()
            })
            .collect();

        // Pre-allocate row_parts buffers: reused across all z-points to avoid
        // per-z-point allocation (saves ~1.4GB of allocations for keccakf)
        let sels_w = sels_cosets[0].width;
        let mut row_parts: Vec<Vec<F>> = Vec::with_capacity(1 + mat_cosets.len());
        row_parts.push(vec![F::ZERO; sels_w]);
        for mc in &mat_cosets {
            row_parts.push(vec![F::ZERO; mc[0].width]);
        }

        // Evaluate callback at each z-point across all cosets
        let mut results = Vec::with_capacity(d * skip_sz);
        for (z_idx, z) in omega_skip.powers().take(skip_sz).enumerate() {
            for (ci, &shift) in coset_shifts.iter().enumerate() {
                // Copy into pre-allocated buffers (no new allocation)
                let ss = z_idx * sels_w;
                row_parts[0].copy_from_slice(&sels_cosets[ci].values[ss..ss + sels_w]);

                for (mat_idx, mc) in mat_cosets.iter().enumerate() {
                    let mw = mc[ci].width;
                    let ms = z_idx * mw;
                    row_parts[1 + mat_idx].copy_from_slice(&mc[ci].values[ms..ms + mw]);
                }

                results.push(w(shift * z, x, &row_parts));
            }
        }
        results
    });

    // Reduce: sum over H_n
    let hypercube_sum = |mut acc: Vec<[EF; WD]>, x: Vec<[EF; WD]>| {
        for (acc_i, x_i) in acc.iter_mut().zip(x) {
            for (a, b) in acc_i.iter_mut().zip(x_i) {
                *a += b;
            }
        }
        acc
    };
    cfg_if::cfg_if! {
        if #[cfg(feature = "parallel")] {
            let evals = evals.reduce(
                || vec![[EF::ZERO; WD]; d << l_skip],
                hypercube_sum,
            );
        } else {
            let evals: Vec<_> = evals.collect();
            let evals = evals.into_iter().fold(
                vec![[EF::ZERO; WD]; d << l_skip],
                hypercube_sum,
            );
        }
    }

    // Assemble polynomial from coset evaluations
    std::array::from_fn(|i| {
        let values: Vec<EF> = evals.iter().map(|x| x[i]).collect_vec();
        UnivariatePoly::from_geometric_cosets_evals_idft(RowMajorMatrix::new(values, d), g, g)
    })
}

/// Packed SIMD zerocheck round 0: evaluates the constraint DAG for WIDTH
/// z-points simultaneously using F::Packing, reducing DAG walks by WIDTH×.
///
/// On aarch64/NEON (WIDTH=4): reduces 32 DAG walks to 8 per x-point.
/// On x86/AVX2 (WIDTH=8): reduces 32 DAG walks to 4 per x-point.
///
/// The zerofier `(shift*z)^(2^l_skip) - 1 = shift^(2^l_skip) - 1` is
/// constant per coset (since z^(2^l_skip) = 1 for z in D_{l_skip}),
/// so we precompute zerofier_inv per coset.
fn sumcheck_uni_round0_zerocheck_packed<SC>(
    l_skip: usize,
    n: usize,
    d: usize,
    rm_mats: &[(&RowMajorMatrix<SC::F>, bool)],
    sels_cm: &ColMajorMatrix<SC::F>,
    helper: &RowMajorEvalHelper<'_, SC>,
    eq_xi: &[SC::EF],
    lambda_pows: &[SC::EF],
) -> UnivariatePoly<SC::EF>
where
    SC: StarkProtocolConfig,
    SC::F: TwoAdicField,
    SC::EF: TwoAdicField + ExtensionField<SC::F>,
{
    if d == 0 {
        return UnivariatePoly::new(vec![]);
    }
    let g = SC::F::GENERATOR;
    let coset_shifts: Vec<SC::F> = g.powers().skip(1).take(d).collect_vec();
    let skip_sz = 1usize << l_skip;
    let width = <SC::F as Field>::Packing::WIDTH;

    // Precompute zerofier_inv per coset: shift^(2^l_skip) - 1 is constant per coset
    let zerofier_invs: Vec<SC::F> = coset_shifts
        .iter()
        .map(|&shift| (shift.exp_power_of_2(l_skip) - SC::F::ONE).inverse())
        .collect();

    // Map-Reduce over x ∈ H_n
    let evals = (0..1usize << n).into_par_iter().map(|x| {
        let dft = Radix2BowersSerial;
        let eq = eq_xi[x];

        // Extract and batch-iDFT selector block (3 columns)
        let sels_block = extract_cm_block(sels_cm, x, l_skip);
        let sels_coeffs = dft.idft_batch(sels_block);
        let sels_cosets: Vec<RowMajorMatrix<SC::F>> = coset_shifts
            .iter()
            .map(|&shift| batch_coset_dft(&sels_coeffs, shift))
            .collect();

        // Extract and batch-iDFT each trace matrix block
        let mat_coeffs: Vec<RowMajorMatrix<SC::F>> = rm_mats
            .iter()
            .map(|(rm, is_rot)| {
                let block = extract_rm_block(rm, x, l_skip, usize::from(*is_rot));
                dft.idft_batch(block)
            })
            .collect();
        let mat_cosets: Vec<Vec<RowMajorMatrix<SC::F>>> = mat_coeffs
            .iter()
            .map(|coeffs| {
                coset_shifts
                    .iter()
                    .map(|&shift| batch_coset_dft(coeffs, shift))
                    .collect()
            })
            .collect();

        // Pre-allocate packed row_parts buffers
        let sels_w = sels_cosets[0].width;
        let mut packed_row_parts: Vec<Vec<<SC::F as Field>::Packing>> =
            Vec::with_capacity(1 + mat_cosets.len());
        packed_row_parts.push(vec![<SC::F as Field>::Packing::default(); sels_w]);
        for mc in &mat_cosets {
            packed_row_parts
                .push(vec![<SC::F as Field>::Packing::default(); mc[0].width]);
        }

        // Pre-allocate DAG node buffer (reused across all packed evaluations)
        let mut node_buf: Vec<<SC::F as Field>::Packing> =
            Vec::with_capacity(helper.constraints_dag.nodes.len());

        // Results: d * skip_sz evaluations in EF
        let mut results: Vec<SC::EF> = vec![SC::EF::ZERO; d * skip_sz];

        // Process z-points in packs of WIDTH
        for z_base in (0..skip_sz).step_by(width) {
            let z_count = width.min(skip_sz - z_base);

            for (ci, &zerofier_inv) in zerofier_invs.iter().enumerate() {
                // Pack WIDTH z-points' column values into F::Packing vectors
                // Selectors (3 columns)
                for col in 0..sels_w {
                    packed_row_parts[0][col] =
                        <SC::F as Field>::Packing::from_fn(|lane| {
                            if lane < z_count {
                                sels_cosets[ci].values[(z_base + lane) * sels_w + col]
                            } else {
                                SC::F::ZERO
                            }
                        });
                }

                // Trace matrices
                for (mat_idx, mc) in mat_cosets.iter().enumerate() {
                    let mw = mc[ci].width;
                    for col in 0..mw {
                        packed_row_parts[1 + mat_idx][col] =
                            <SC::F as Field>::Packing::from_fn(|lane| {
                                if lane < z_count {
                                    mc[ci].values[(z_base + lane) * mw + col]
                                } else {
                                    SC::F::ZERO
                                }
                            });
                    }
                }

                // Evaluate DAG once for WIDTH z-points simultaneously
                let evaluator = helper.evaluator_packed(&packed_row_parts);
                eval_nodes_into(&evaluator, &helper.constraints_dag.nodes, &mut node_buf);

                // Unpack lanes: accumulate constraint_eval per lane, multiply by eq * zerofier_inv
                for lane in 0..z_count {
                    let constraint_eval: SC::EF =
                        zip(lambda_pows, &helper.constraints_dag.constraint_idx)
                            .fold(SC::EF::ZERO, |acc, (&lp, &idx)| {
                                acc + lp * node_buf[idx].as_slice()[lane]
                            });
                    let z_idx = z_base + lane;
                    results[z_idx * d + ci] += eq * constraint_eval * zerofier_inv;
                }
            }
        }
        results
    });

    // Reduce: sum over H_n
    let hypercube_sum = |mut acc: Vec<SC::EF>, x: Vec<SC::EF>| {
        for (a, b) in acc.iter_mut().zip(x) {
            *a += b;
        }
        acc
    };
    cfg_if::cfg_if! {
        if #[cfg(feature = "parallel")] {
            let evals = evals.reduce(
                || vec![SC::EF::ZERO; d << l_skip],
                hypercube_sum,
            );
        } else {
            let evals: Vec<_> = evals.collect();
            let evals = evals.into_iter().fold(
                vec![SC::EF::ZERO; d << l_skip],
                hypercube_sum,
            );
        }
    }

    // Assemble polynomial from coset evaluations
    let values: Vec<SC::EF> = evals;
    UnivariatePoly::from_geometric_cosets_evals_idft(RowMajorMatrix::new(values, d), g, g)
}

// ============================================================================
// Constraint evaluator (duplicated from stark-backend since it's pub(super))
// ============================================================================

struct ViewPair<T> {
    local: *const T,
    next: Option<*const T>,
}

// SAFETY: ViewPair is safe to send between threads as long as the underlying
// data (pointed to by the raw pointers) is valid and shared immutably.
unsafe impl<T: Send> Send for ViewPair<T> {}
unsafe impl<T: Sync> Sync for ViewPair<T> {}

impl<T> ViewPair<T> {
    fn new(local: &[T], next: Option<&[T]>) -> Self {
        Self {
            local: local.as_ptr(),
            next: next.map(|nxt| nxt.as_ptr()),
        }
    }

    /// SAFETY: no matrix bounds checks are done.
    unsafe fn get(&self, row_offset: usize, column_idx: usize) -> &T {
        match row_offset {
            0 => &*self.local.add(column_idx),
            1 => &*self.next.unwrap_unchecked().add(column_idx),
            _ => panic!("row offset {row_offset} not supported"),
        }
    }
}

struct ConstraintEvaluator<'a, F, EF> {
    preprocessed: Option<ViewPair<EF>>,
    partitioned_main: Vec<ViewPair<EF>>,
    is_first_row: EF,
    is_last_row: EF,
    is_transition: EF,
    public_values: &'a [F],
}

impl<F: Field, EF: ExtensionField<F>> SymbolicEvaluator<F, EF>
    for ConstraintEvaluator<'_, F, EF>
{
    fn eval_const(&self, c: F) -> EF {
        c.into()
    }
    fn eval_is_first_row(&self) -> EF {
        self.is_first_row
    }
    fn eval_is_last_row(&self) -> EF {
        self.is_last_row
    }
    fn eval_is_transition(&self) -> EF {
        self.is_transition
    }

    fn eval_var(&self, symbolic_var: SymbolicVariable<F>) -> EF {
        let index = symbolic_var.index;
        match symbolic_var.entry {
            Entry::Preprocessed { offset } => unsafe {
                *self
                    .preprocessed
                    .as_ref()
                    .unwrap_unchecked()
                    .get(offset, index)
            },
            Entry::Main { part_index, offset } => unsafe {
                *self.partitioned_main[part_index].get(offset, index)
            },
            Entry::Public => unsafe { EF::from(*self.public_values.get_unchecked(index)) },
            _ => unreachable!("after_challenge not supported"),
        }
    }
}

// ============================================================================
// Packed constraint evaluator (SIMD: evaluates DAG in F::Packing)
// ============================================================================

/// Evaluates the constraint DAG using packed base field (F::Packing), processing
/// WIDTH independent evaluations simultaneously via SIMD.
///
/// On aarch64/NEON: WIDTH=4, on x86/AVX2: WIDTH=8, scalar fallback: WIDTH=1.
/// This reduces the number of DAG walks from N to ceil(N/WIDTH), cutting
/// cache misses on the ~30K-node DAG by WIDTH×.
struct PackedConstraintEvaluator<'a, F: Field> {
    preprocessed: Option<ViewPair<F::Packing>>,
    partitioned_main: Vec<ViewPair<F::Packing>>,
    is_first_row: F::Packing,
    is_last_row: F::Packing,
    is_transition: F::Packing,
    public_values: &'a [F],
}

impl<F: Field> SymbolicEvaluator<F, F::Packing> for PackedConstraintEvaluator<'_, F> {
    fn eval_const(&self, c: F) -> F::Packing {
        F::Packing::from_fn(|_| c)
    }
    fn eval_is_first_row(&self) -> F::Packing {
        self.is_first_row
    }
    fn eval_is_last_row(&self) -> F::Packing {
        self.is_last_row
    }
    fn eval_is_transition(&self) -> F::Packing {
        self.is_transition
    }

    fn eval_var(&self, symbolic_var: SymbolicVariable<F>) -> F::Packing {
        let index = symbolic_var.index;
        match symbolic_var.entry {
            Entry::Preprocessed { offset } => unsafe {
                *self
                    .preprocessed
                    .as_ref()
                    .unwrap_unchecked()
                    .get(offset, index)
            },
            Entry::Main { part_index, offset } => unsafe {
                *self.partitioned_main[part_index].get(offset, index)
            },
            Entry::Public => unsafe {
                F::Packing::from_fn(|_| *self.public_values.get_unchecked(index))
            },
            _ => unreachable!("after_challenge not supported"),
        }
    }
}

/// Evaluate DAG nodes into a pre-allocated buffer, avoiding per-call allocation.
/// The buffer is cleared and reused across calls, saving ~120KB allocation per
/// invocation for typical KeccakAir DAGs (~30K nodes).
#[inline]
fn eval_nodes_into<F, E>(
    evaluator: &impl SymbolicEvaluator<F, E>,
    nodes: &[SymbolicExpressionNode<F>],
    buf: &mut Vec<E>,
) where
    F: Field,
    E: Add<E, Output = E> + Sub<E, Output = E> + Mul<E, Output = E> + Neg<Output = E> + Clone,
{
    buf.clear();
    for node in nodes {
        let val = match *node {
            SymbolicExpressionNode::Variable(var) => evaluator.eval_var(var),
            SymbolicExpressionNode::Constant(c) => evaluator.eval_const(c),
            SymbolicExpressionNode::Add {
                left_idx,
                right_idx,
                ..
            } => buf[left_idx].clone() + buf[right_idx].clone(),
            SymbolicExpressionNode::Sub {
                left_idx,
                right_idx,
                ..
            } => buf[left_idx].clone() - buf[right_idx].clone(),
            SymbolicExpressionNode::Neg { idx, .. } => -buf[idx].clone(),
            SymbolicExpressionNode::Mul {
                left_idx,
                right_idx,
                ..
            } => buf[left_idx].clone() * buf[right_idx].clone(),
            SymbolicExpressionNode::IsFirstRow => evaluator.eval_is_first_row(),
            SymbolicExpressionNode::IsLastRow => evaluator.eval_is_last_row(),
            SymbolicExpressionNode::IsTransition => evaluator.eval_is_transition(),
        };
        buf.push(val);
    }
}

// ============================================================================
// RowMajorEvalHelper
// ============================================================================

/// Evaluation helper for a single AIR, storing preprocessed trace as row-major.
pub(crate) struct RowMajorEvalHelper<'a, SC: StarkProtocolConfig> {
    pub constraints_dag: &'a SymbolicExpressionDag<SC::F>,
    pub interactions: Vec<SymbolicInteraction<SC::F>>,
    pub public_values: Vec<SC::F>,
    pub preprocessed_trace: Option<&'a RowMajorMatrix<SC::F>>,
    pub needs_next: bool,
    pub constraint_degree: u8,
}

impl<'a, SC: StarkProtocolConfig> RowMajorEvalHelper<'a, SC>
where
    SC::F: TwoAdicField,
    SC::EF: TwoAdicField + ExtensionField<SC::F>,
{
    pub fn has_preprocessed(&self) -> bool {
        self.preprocessed_trace.is_some()
    }

    /// Returns list of (&RowMajorMatrix, is_rot) in order:
    /// - (if preprocessed) (preprocessed, false), (preprocessed, true)
    /// - for each cached: (cached_i, false), (cached_i, true)
    /// - (common, false), (common, true)
    pub fn view_mats_rowmaj(
        &self,
        ctx: &'a AirProvingContext<CpuBackend<SC>>,
    ) -> Vec<(&'a RowMajorMatrix<SC::F>, bool)> {
        let base_mats = usize::from(self.has_preprocessed()) + 1 + ctx.cached_mains.len();
        let cap = if self.needs_next {
            2 * base_mats
        } else {
            base_mats
        };
        let mut mats = Vec::with_capacity(cap);
        if let Some(pp) = self.preprocessed_trace {
            mats.push((pp, false));
            if self.needs_next {
                mats.push((pp, true));
            }
        }
        for cd in &ctx.cached_mains {
            mats.push((&cd.trace.inner, false));
            if self.needs_next {
                mats.push((&cd.trace.inner, true));
            }
        }
        mats.push((&ctx.common_main.inner, false));
        if self.needs_next {
            mats.push((&ctx.common_main.inner, true));
        }
        mats
    }

    fn evaluator<FF: ExtensionField<SC::F>>(
        &self,
        row_parts: &[Vec<FF>],
    ) -> ConstraintEvaluator<'_, SC::F, FF> {
        let sels = &row_parts[0];
        let mut view_pairs = if self.needs_next {
            let mut chunks = row_parts[1..].chunks_exact(2);
            let pairs = chunks
                .by_ref()
                .map(|pair| ViewPair::new(&pair[0], Some(&pair[1][..])))
                .collect_vec();
            debug_assert!(chunks.remainder().is_empty());
            pairs
        } else {
            row_parts[1..]
                .iter()
                .map(|part| ViewPair::new(part, None))
                .collect_vec()
        };
        let mut preprocessed = None;
        if self.has_preprocessed() {
            preprocessed = Some(view_pairs.remove(0));
        }
        ConstraintEvaluator {
            preprocessed,
            partitioned_main: view_pairs,
            is_first_row: sels[0],
            is_transition: sels[1],
            is_last_row: sels[2],
            public_values: &self.public_values,
        }
    }

    pub fn acc_constraints<FF: ExtensionField<SC::F>, EF: ExtensionField<FF>>(
        &self,
        row_parts: &[Vec<FF>],
        lambda_pows: &[EF],
    ) -> EF {
        let evaluator = self.evaluator(row_parts);
        let nodes = evaluator.eval_nodes(&self.constraints_dag.nodes);
        zip(lambda_pows, &self.constraints_dag.constraint_idx)
            .fold(EF::ZERO, |acc, (&lambda_pow, &idx)| {
                acc + lambda_pow * nodes[idx]
            })
    }

    /// Create a packed constraint evaluator from packed row_parts.
    /// Each element in row_parts is a Vec<F::Packing> where each packed value
    /// holds WIDTH independent evaluations across different z-points.
    fn evaluator_packed(
        &self,
        row_parts: &[Vec<<SC::F as Field>::Packing>],
    ) -> PackedConstraintEvaluator<'_, SC::F> {
        let sels = &row_parts[0];
        let mut view_pairs = if self.needs_next {
            let mut chunks = row_parts[1..].chunks_exact(2);
            let pairs = chunks
                .by_ref()
                .map(|pair| ViewPair::new(&pair[0], Some(&pair[1][..])))
                .collect_vec();
            debug_assert!(chunks.remainder().is_empty());
            pairs
        } else {
            row_parts[1..]
                .iter()
                .map(|part| ViewPair::new(part, None))
                .collect_vec()
        };
        let mut preprocessed = None;
        if self.has_preprocessed() {
            preprocessed = Some(view_pairs.remove(0));
        }
        PackedConstraintEvaluator {
            preprocessed,
            partitioned_main: view_pairs,
            is_first_row: sels[0],
            is_transition: sels[1],
            is_last_row: sels[2],
            public_values: &self.public_values,
        }
    }

    pub fn acc_interactions<FF, EF>(
        &self,
        row_parts: &[Vec<FF>],
        beta_pows: &[EF],
        eq_3bs: &[EF],
    ) -> [EF; 2]
    where
        FF: ExtensionField<SC::F>,
        EF: ExtensionField<FF> + ExtensionField<SC::F>,
    {
        let interaction_evals = self.eval_interactions(row_parts, beta_pows);
        let mut numer = EF::ZERO;
        let mut denom = EF::ZERO;
        for (&eq_3b, eval) in zip(eq_3bs, interaction_evals) {
            numer += eq_3b * eval.0;
            denom += eq_3b * eval.1;
        }
        [numer, denom]
    }

    pub fn eval_interactions<FF, EF>(
        &self,
        row_parts: &[Vec<FF>],
        beta_pows: &[EF],
    ) -> Vec<(FF, EF)>
    where
        FF: ExtensionField<SC::F>,
        EF: ExtensionField<FF> + ExtensionField<SC::F>,
    {
        let evaluator = self.evaluator(row_parts);
        self.interactions
            .iter()
            .map(|interaction| {
                let b = SC::F::from_u32(interaction.bus_index as u32 + 1);
                let msg_len = interaction.message.len();
                assert!(msg_len <= beta_pows.len());
                let denom = zip(&interaction.message, beta_pows).fold(
                    beta_pows[msg_len] * b,
                    |h_beta, (msg_j, &beta_j)| {
                        let msg_j_eval = evaluator.eval_expr(msg_j);
                        h_beta + beta_j * msg_j_eval
                    },
                );
                let numer = evaluator.eval_expr(&interaction.count);
                (numer, denom)
            })
            .collect()
    }

    /// Build row_parts from row-major matrices for a given row index.
    /// This is the performance-critical row-major access pattern:
    /// each row is contiguous in memory.
    fn build_row_parts(
        mats: &[(&RowMajorMatrix<SC::F>, bool)],
        row_idx: usize,
        height: usize,
    ) -> Vec<Vec<SC::F>> {
        let is_first = SC::F::from_bool(row_idx == 0);
        let is_transition = SC::F::from_bool(row_idx != height - 1);
        let is_last = SC::F::from_bool(row_idx == height - 1);

        let mut row_parts = Vec::with_capacity(mats.len() + 1);
        row_parts.push(vec![is_first, is_transition, is_last]);

        for &(mat, is_rot) in mats {
            let mat_height = mat.values.len() / mat.width;
            let idx = if is_rot {
                (row_idx + 1) % mat_height
            } else {
                row_idx % mat_height
            };
            let start = idx * mat.width;
            // Contiguous row access — the key cache locality optimization
            row_parts.push(mat.values[start..start + mat.width].to_vec());
        }
        row_parts
    }

}

// ============================================================================
// LogupZerocheckRowMajor
// ============================================================================

pub(crate) struct LogupZerocheckRowMajor<'a, SC: StarkProtocolConfig> {
    pub beta_pows: Vec<SC::EF>,

    pub l_skip: usize,
    pub n_logup: usize,

    pub omega_skip_pows: Vec<SC::F>,

    pub interactions_layout: StackedLayout,
    pub(crate) eval_helpers: Vec<RowMajorEvalHelper<'a, SC>>,
    pub constraint_degree: usize,
    pub n_per_trace: Vec<isize>,
    max_num_constraints: usize,

    pub xi: Vec<SC::EF>,
    lambda_pows: Vec<SC::EF>,
    eq_xi_per_trace: Vec<Vec<SC::EF>>,
    eq_3b_per_trace: Vec<Vec<SC::EF>>,
    sels_per_trace_base: Vec<ColMajorMatrix<SC::F>>,
    pub mat_evals_per_trace: Vec<Vec<ColMajorMatrix<SC::EF>>>,
    pub sels_per_trace: Vec<ColMajorMatrix<SC::EF>>,
    pub(crate) zerocheck_tilde_evals: Vec<SC::EF>,
    pub(crate) logup_tilde_evals: Vec<[SC::EF; 2]>,

    pub(crate) prev_s_eval: SC::EF,
    pub(crate) eq_ns: Vec<SC::EF>,
    pub(crate) eq_sharp_ns: Vec<SC::EF>,
}

impl<'a, SC: StarkProtocolConfig> LogupZerocheckRowMajor<'a, SC>
where
    SC::F: TwoAdicField,
    SC::EF: TwoAdicField + ExtensionField<SC::F>,
    CpuBackend<SC>: ProverBackend<Val = SC::F, Matrix = RowMajorMatrixWrapper<SC::F>>,
{
    pub fn new(
        pk: &'a DeviceMultiStarkProvingKey<CpuBackend<SC>>,
        ctx: &ProvingContext<CpuBackend<SC>>,
        n_logup: usize,
        interactions_layout: StackedLayout,
        _alpha_logup: SC::EF,
        beta_logup: SC::EF,
    ) -> Self {
        let l_skip = pk.params.l_skip;
        let omega_skip = SC::F::two_adic_generator(l_skip);
        let omega_skip_pows = omega_skip.powers().take(1 << l_skip).collect_vec();
        let num_airs_present = ctx.per_trace.len();

        let constraint_degree = pk.max_constraint_degree;
        let max_interaction_length = ctx
            .per_trace
            .iter()
            .flat_map(|(air_idx, _)| {
                pk.per_air[*air_idx]
                    .vk
                    .symbolic_constraints
                    .interactions
                    .iter()
                    .map(|i| i.message.len())
            })
            .max()
            .unwrap_or(0);
        let beta_pows = beta_logup
            .powers()
            .take(max_interaction_length + 1)
            .collect_vec();

        let n_per_trace: Vec<isize> = ctx
            .common_main_traces()
            .map(|(_, t)| log2_strict_usize(t.height()) as isize - l_skip as isize)
            .collect_vec();
        let n_max: usize = n_per_trace[0].max(0) as usize;

        let eval_helpers: Vec<RowMajorEvalHelper<SC>> = ctx
            .per_trace
            .iter()
            .map(|(air_idx, trace_ctx)| {
                let pk = &pk.per_air[*air_idx];
                let constraints = &pk.vk.symbolic_constraints.constraints;
                let public_values = trace_ctx.public_values.clone();
                let preprocessed_trace: Option<&RowMajorMatrix<SC::F>> =
                    pk.preprocessed_data.as_ref().map(|cd| &cd.trace.inner);
                // Validate index bounds
                let mut rotation = 0;
                for node in &constraints.nodes {
                    if let SymbolicExpressionNode::Variable(var) = node {
                        match var.entry {
                            Entry::Preprocessed { offset } => {
                                rotation = max(rotation, offset);
                                assert!(
                                    var.index < preprocessed_trace.unwrap().width,
                                    "col_index={} >= preprocessed width={}",
                                    var.index,
                                    preprocessed_trace.unwrap().width
                                );
                            }
                            Entry::Main { part_index, offset } => {
                                rotation = max(rotation, offset);
                                // Get width of the partition
                                let part_width = if part_index < trace_ctx.cached_mains.len() {
                                    trace_ctx.cached_mains[part_index].trace.inner.width
                                } else {
                                    trace_ctx.common_main.inner.width
                                };
                                assert!(
                                    var.index < part_width,
                                    "col_index={} >= main partition {} width={}",
                                    var.index,
                                    part_index,
                                    part_width
                                );
                            }
                            Entry::Public => {
                                assert!(var.index < public_values.len());
                            }
                            _ => unreachable!("after_challenge not supported"),
                        }
                    }
                }
                let needs_next = pk.vk.params.need_rot;
                debug_assert_eq!(needs_next, rotation > 0);
                let symbolic_constraints =
                    SymbolicConstraints::from(&pk.vk.symbolic_constraints);
                RowMajorEvalHelper {
                    constraints_dag: &pk.vk.symbolic_constraints.constraints,
                    interactions: symbolic_constraints.interactions,
                    public_values,
                    preprocessed_trace,
                    needs_next,
                    constraint_degree: pk.vk.max_constraint_degree,
                }
            })
            .collect();

        let max_num_constraints = pk
            .per_air
            .iter()
            .map(|pk| pk.vk.symbolic_constraints.constraints.constraint_idx.len())
            .max()
            .unwrap_or(0);

        let zerocheck_tilde_evals = vec![SC::EF::ZERO; num_airs_present];
        let logup_tilde_evals = vec![[SC::EF::ZERO; 2]; num_airs_present];
        Self {
            beta_pows,
            l_skip,
            n_logup,
            omega_skip_pows,
            interactions_layout,
            constraint_degree,
            max_num_constraints,
            n_per_trace,
            eval_helpers,
            xi: vec![],
            lambda_pows: vec![],
            sels_per_trace_base: vec![],
            eq_xi_per_trace: vec![],
            eq_3b_per_trace: vec![],
            mat_evals_per_trace: vec![],
            sels_per_trace: vec![],
            zerocheck_tilde_evals,
            logup_tilde_evals,
            prev_s_eval: SC::EF::ZERO,
            eq_ns: Vec::with_capacity(n_max + 1),
            eq_sharp_ns: Vec::with_capacity(n_max + 1),
        }
    }

    pub fn sumcheck_uni_round0_polys(
        &mut self,
        ctx: &ProvingContext<CpuBackend<SC>>,
        lambda: SC::EF,
    ) -> Vec<UnivariatePoly<SC::EF>> {
        let n_logup = self.n_logup;
        let l_skip = self.l_skip;
        let xi = &self.xi;
        self.lambda_pows = lambda.powers().take(self.max_num_constraints).collect_vec();

        self.eq_3b_per_trace = self
            .eval_helpers
            .par_iter()
            .zip(&self.n_per_trace)
            .enumerate()
            .map(|(trace_idx, (helper, &n))| {
                let n_lift = n.max(0) as usize;
                if helper.interactions.is_empty() {
                    return vec![];
                }
                let mut b_vec = vec![SC::F::ZERO; n_logup - n_lift];
                (0..helper.interactions.len())
                    .map(|i| {
                        let stacked_idx =
                            self.interactions_layout.get(trace_idx, i).unwrap().row_idx;
                        debug_assert!(stacked_idx.trailing_zeros() as usize >= n_lift + l_skip);
                        let mut b_int = stacked_idx >> (l_skip + n_lift);
                        for b in &mut b_vec {
                            *b = SC::F::from_bool(b_int & 1 == 1);
                            b_int >>= 1;
                        }
                        eval_eq_mle(&xi[l_skip + n_lift..l_skip + n_logup], &b_vec)
                    })
                    .collect_vec()
            })
            .collect::<Vec<_>>();

        self.eq_xi_per_trace = self
            .n_per_trace
            .par_iter()
            .map(|&n| {
                let n_lift = n.max(0) as usize;
                evals_eq_hypercubes(n_lift, xi[l_skip..l_skip + n_lift].iter().rev())
            })
            .collect();

        self.sels_per_trace_base = self
            .n_per_trace
            .iter()
            .map(|&n| {
                let log_height = l_skip.checked_add_signed(n).unwrap();
                let height = 1 << log_height;
                let lifted_height = height.max(1 << l_skip);
                let mut mat = SC::F::zero_vec(3 * lifted_height);
                mat[lifted_height..2 * lifted_height].fill(SC::F::ONE);
                for i in (0..lifted_height).step_by(height) {
                    mat[i] = SC::F::ONE;
                    mat[lifted_height + i + height - 1] = SC::F::ZERO;
                    mat[2 * lifted_height + i + height - 1] = SC::F::ONE;
                }
                ColMajorMatrix::new(mat, 3)
            })
            .collect_vec();

        // Zerocheck round 0 — uses batch DFT from row-major data
        let sp_0_zerochecks = self
            .eval_helpers
            .par_iter()
            .enumerate()
            .map(|(trace_idx, helper)| {
                let trace_ctx = &ctx.per_trace[trace_idx].1;
                let n_lift = log2_strict_usize(trace_ctx.height()).saturating_sub(l_skip);
                let rm_mats = helper.view_mats_rowmaj(trace_ctx);
                let eq_xi = &self.eq_xi_per_trace[trace_idx][(1 << n_lift) - 1..(2 << n_lift) - 1];
                let sels_cm = &self.sels_per_trace_base[trace_idx];

                let constraint_deg = helper.constraint_degree as usize;
                if constraint_deg == 0 {
                    return UnivariatePoly::new(vec![]);
                }
                let num_cosets = constraint_deg - 1;
                let q = sumcheck_uni_round0_zerocheck_packed::<SC>(
                    l_skip,
                    n_lift,
                    num_cosets,
                    &rm_mats,
                    sels_cm,
                    helper,
                    eq_xi,
                    &self.lambda_pows,
                );
                let sp_0_deg = sumcheck_round0_deg(l_skip, constraint_deg);
                let coeffs = (0..=sp_0_deg)
                    .map(|i| {
                        let mut c = -*q.coeffs().get(i).unwrap_or(&SC::EF::ZERO);
                        if i >= 1 << l_skip {
                            c += q.coeffs()[i - (1 << l_skip)];
                        }
                        c
                    })
                    .collect_vec();
                debug_assert_eq!(
                    coeffs.iter().step_by(1 << l_skip).copied().sum::<SC::EF>(),
                    SC::EF::ZERO,
                    "Zerocheck sum is not zero for air_id: {}",
                    ctx.per_trace[trace_idx].0
                );
                UnivariatePoly::new(coeffs)
            })
            .collect::<Vec<_>>();

        // Logup round 0 — uses batch DFT from row-major data
        let sp_0_logups = self
            .eval_helpers
            .par_iter()
            .enumerate()
            .flat_map(|(trace_idx, helper)| {
                if helper.interactions.is_empty() {
                    return [(); 2].map(|_| UnivariatePoly::new(vec![]));
                }
                let trace_ctx = &ctx.per_trace[trace_idx].1;
                let log_height = log2_strict_usize(trace_ctx.height());
                let n_lift = log_height.saturating_sub(l_skip);
                let rm_mats = helper.view_mats_rowmaj(trace_ctx);
                let eq_xi = &self.eq_xi_per_trace[trace_idx][(1 << n_lift) - 1..(2 << n_lift) - 1];
                let eq_3bs = &self.eq_3b_per_trace[trace_idx];
                let sels_cm = &self.sels_per_trace_base[trace_idx];
                let norm_factor_denom = 1 << l_skip.saturating_sub(log_height);
                let norm_factor = SC::F::from_usize(norm_factor_denom).inverse();

                let [mut numer, denom] = sumcheck_uni_round0_batch::<SC::F, SC::EF, _, 2>(
                    l_skip,
                    n_lift,
                    helper.constraint_degree as usize,
                    &rm_mats,
                    sels_cm,
                    |_z, x, row_parts| {
                        let eq = eq_xi[x];
                        let [numer, denom] =
                            helper.acc_interactions(row_parts, &self.beta_pows, eq_3bs);
                        [eq * numer, eq * denom]
                    },
                );
                for p in numer.coeffs_mut() {
                    *p *= norm_factor;
                }
                [numer, denom]
            })
            .collect::<Vec<_>>();

        sp_0_logups.into_iter().chain(sp_0_zerochecks).collect()
    }

    /// Fold PLE evaluations using randomness r_0.
    /// Reads directly from row-major matrices via barycentric interpolation,
    /// avoiding the O(n*m) transpose to col-major (saves ~3s for keccakf).
    pub fn fold_ple_evals(&mut self, ctx: &ProvingContext<CpuBackend<SC>>, r_0: SC::EF) {
        let l_skip = self.l_skip;
        self.mat_evals_per_trace = self
            .eval_helpers
            .par_iter()
            .zip(ctx.per_trace.par_iter())
            .map(|(helper, (_, trace_ctx))| {
                let rm_mats = helper.view_mats_rowmaj(trace_ctx);
                rm_mats
                    .into_iter()
                    .map(|(rm, is_rot)| {
                        fold_ple_evals_rowmajor(l_skip, rm, is_rot, r_0)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        self.sels_per_trace =
            batch_fold_ple_evals(l_skip, take(&mut self.sels_per_trace_base), false, r_0);
        let eq_r0 = eval_eq_uni(l_skip, self.xi[0], r_0);
        let eq_sharp_r0 = eval_eq_sharp_uni(&self.omega_skip_pows, &self.xi[..l_skip], r_0);
        self.eq_ns.push(eq_r0);
        self.eq_sharp_ns.push(eq_sharp_r0);
        self.eq_xi_per_trace.iter_mut().for_each(|eq| {
            if eq.len() > 1 {
                eq.truncate(eq.len() / 2);
            }
        });
    }

    /// After folding, operates on small ColMajorMatrix<EF> — identical to reference.
    pub fn sumcheck_polys_eval(&mut self, round: usize, r_prev: SC::EF) -> Vec<Vec<SC::EF>> {
        let sp_deg = self.constraint_degree;
        let sp_zerocheck_evals: Vec<Vec<SC::EF>> = izip!(
            &self.eval_helpers,
            &mut self.zerocheck_tilde_evals,
            &self.n_per_trace,
            &self.mat_evals_per_trace,
            &self.sels_per_trace,
            &self.eq_xi_per_trace
        )
        .map(|(helper, tilde_eval, &n, mats, sels, eq_xi_tree)| {
            let n_lift = n.max(0) as usize;
            if round > n_lift {
                if round == n_lift + 1 {
                    let parts = iter::once(sels)
                        .chain(mats)
                        .map(|mat| mat.columns().map(|c| c[0]).collect_vec())
                        .collect_vec();
                    let eq_r_acc = *self.eq_ns.last().unwrap();
                    *tilde_eval = eq_r_acc * helper.acc_constraints(&parts, &self.lambda_pows);
                } else {
                    *tilde_eval *= r_prev;
                };
                vec![*tilde_eval]
            } else {
                let log_num_y = n_lift - round;
                let num_y = 1 << log_num_y;
                let eq_xi = &eq_xi_tree[num_y - 1..];
                let parts = iter::once(sels)
                    .chain(mats)
                    .map(|m| m.as_view())
                    .collect_vec();
                let [s] = sumcheck_round_poly_evals(
                    log_num_y + 1,
                    sp_deg,
                    &parts,
                    |_x, y, row_parts| {
                        let eq = eq_xi[y];
                        let constraint_eval = helper.acc_constraints(row_parts, &self.lambda_pows);
                        [eq * constraint_eval]
                    },
                );
                s
            }
        })
        .collect();

        let sp_logup_evals: Vec<Vec<SC::EF>> = izip!(
            &self.eval_helpers,
            &mut self.logup_tilde_evals,
            &self.n_per_trace,
            &self.mat_evals_per_trace,
            &self.sels_per_trace,
            &self.eq_xi_per_trace,
            &self.eq_3b_per_trace
        )
        .flat_map(|(helper, tilde_eval, &n, mats, sels, eq_xi_tree, eq_3bs)| {
            if helper.interactions.is_empty() {
                return [vec![SC::EF::ZERO; sp_deg], vec![SC::EF::ZERO; sp_deg]];
            }
            let n_lift = n.max(0) as usize;
            let norm_factor_denom = 1 << (-n).max(0);
            let norm_factor = SC::F::from_usize(norm_factor_denom).inverse();
            if round > n_lift {
                if round == n_lift + 1 {
                    let parts = iter::once(sels)
                        .chain(mats)
                        .map(|mat| mat.columns().map(|c| c[0]).collect_vec())
                        .collect_vec();
                    let eq_sharp_r_acc = *self.eq_sharp_ns.last().unwrap();
                    *tilde_eval = helper
                        .acc_interactions(&parts, &self.beta_pows, eq_3bs)
                        .map(|x| eq_sharp_r_acc * x);
                    tilde_eval[0] *= norm_factor;
                } else {
                    for x in tilde_eval.iter_mut() {
                        *x *= r_prev;
                    }
                };
                tilde_eval.map(|tilde_eval| vec![tilde_eval])
            } else {
                let parts = iter::once(sels)
                    .chain(mats)
                    .map(|m| m.as_view())
                    .collect_vec();
                let log_num_y = n_lift - round;
                let num_y = 1 << log_num_y;
                let eq_xi = &eq_xi_tree[num_y - 1..];
                let [mut numer, denom] = sumcheck_round_poly_evals(
                    log_num_y + 1,
                    sp_deg,
                    &parts,
                    |_x, y, row_parts| {
                        let eq = eq_xi[y];
                        helper
                            .acc_interactions(row_parts, &self.beta_pows, eq_3bs)
                            .map(|eval| eq * eval)
                    },
                );
                for p in &mut numer {
                    *p *= norm_factor;
                }
                [numer, denom]
            }
        })
        .collect();

        sp_logup_evals
            .into_iter()
            .chain(sp_zerocheck_evals)
            .collect()
    }

    pub fn fold_mle_evals(&mut self, round: usize, r_round: SC::EF) {
        self.mat_evals_per_trace = take(&mut self.mat_evals_per_trace)
            .into_iter()
            .map(|mats| batch_fold_mle_evals(mats, r_round))
            .collect_vec();
        self.sels_per_trace = batch_fold_mle_evals(take(&mut self.sels_per_trace), r_round);
        self.eq_xi_per_trace.par_iter_mut().for_each(|eq| {
            if eq.len() > 1 {
                eq.truncate(eq.len() / 2);
            }
        });
        let xi = self.xi[self.l_skip + round - 1];
        let eq_r = eval_eq_mle(&[xi], &[r_round]);
        self.eq_ns.push(self.eq_ns[round - 1] * eq_r);
        self.eq_sharp_ns.push(self.eq_sharp_ns[round - 1] * eq_r);
    }

    pub fn into_column_openings(&mut self) -> Vec<Vec<Vec<SC::EF>>> {
        let num_airs_present = self.mat_evals_per_trace.len();
        let mut column_openings = Vec::with_capacity(num_airs_present);
        for (helper, mut mat_evals) in self
            .eval_helpers
            .iter()
            .zip(take(&mut self.mat_evals_per_trace))
        {
            let openings_of_air: Vec<Vec<SC::EF>> = if helper.needs_next {
                let common_main_rot = mat_evals.pop().unwrap();
                let common_main = mat_evals.pop().unwrap();
                iter::once(&[common_main, common_main_rot] as &[_])
                    .chain(mat_evals.chunks_exact(2))
                    .map(|pair| {
                        zip(pair[0].columns(), pair[1].columns())
                            .flat_map(|(claim, claim_rot)| {
                                assert_eq!(claim.len(), 1);
                                assert_eq!(claim_rot.len(), 1);
                                [claim[0], claim_rot[0]]
                            })
                            .collect_vec()
                    })
                    .collect_vec()
            } else {
                let common_main = mat_evals.pop().unwrap();
                iter::once(common_main)
                    .chain(mat_evals.into_iter())
                    .map(|mat| {
                        mat.columns()
                            .map(|claim| {
                                assert_eq!(claim.len(), 1);
                                claim[0]
                            })
                            .collect_vec()
                    })
                    .collect_vec()
            };
            column_openings.push(openings_of_air);
        }
        column_openings
    }
}

// ============================================================================
// prove_zerocheck_and_logup
// ============================================================================

#[instrument(level = "info", skip_all)]
pub fn prove_zerocheck_and_logup<SC: StarkProtocolConfig, TS>(
    transcript: &mut TS,
    mpk: &DeviceMultiStarkProvingKey<CpuBackend<SC>>,
    ctx: &ProvingContext<CpuBackend<SC>>,
) -> (GkrProof<SC>, BatchConstraintProof<SC>, Vec<SC::EF>)
where
    TS: FiatShamirTranscript<SC>,
    SC::F: TwoAdicField,
    SC::EF: TwoAdicField + ExtensionField<SC::F>,
    CpuBackend<SC>: ProverBackend<Val = SC::F, Matrix = RowMajorMatrixWrapper<SC::F>>,
{
    let l_skip = mpk.params.l_skip;
    let constraint_degree = mpk.max_constraint_degree;
    let num_traces = ctx.per_trace.len();

    let n_max = log2_strict_usize(ctx.per_trace[0].1.common_main.height()).saturating_sub(l_skip);
    let mut total_interactions = 0u64;
    let interactions_meta: Vec<_> = ctx
        .per_trace
        .iter()
        .map(|(air_idx, trace_ctx)| {
            let pk = &mpk.per_air[*air_idx];
            let num_interactions = pk.vk.symbolic_constraints.interactions.len();
            let height = trace_ctx.common_main.height();
            let log_height = log2_strict_usize(height);
            let log_lifted_height = log_height.max(l_skip);
            total_interactions += (num_interactions as u64) << log_lifted_height;
            (num_interactions, log_lifted_height)
        })
        .collect();
    let n_logup = calculate_n_logup(l_skip, total_interactions);
    debug!(%n_logup);
    let interactions_layout = StackedLayout::new(0, l_skip + n_logup, interactions_meta);

    let logup_pow_witness = transcript.grind(mpk.params.logup.pow_bits);
    let alpha_logup = transcript.sample_ext();
    let beta_logup = transcript.sample_ext();
    debug!(%alpha_logup, %beta_logup);

    let mut prover = LogupZerocheckRowMajor::new(
        mpk,
        ctx,
        n_logup,
        interactions_layout,
        alpha_logup,
        beta_logup,
    );

    // GKR: compute logup input layer using row-major access
    let has_interactions = !prover.interactions_layout.sorted_cols.is_empty();
    let gkr_input_evals = if !has_interactions {
        vec![]
    } else {
        // Per trace: row-major interaction evaluations using contiguous row access
        let unstacked_interaction_evals = prover
            .eval_helpers
            .par_iter()
            .enumerate()
            .map(|(trace_idx, helper)| {
                let trace_ctx = &ctx.per_trace[trace_idx].1;
                let mats = helper.view_mats_rowmaj(trace_ctx);
                let height = trace_ctx.common_main.height();
                (0..height)
                    .into_par_iter()
                    .map(|i| {
                        // Build row_parts from contiguous row-major memory
                        let row_parts =
                            RowMajorEvalHelper::<SC>::build_row_parts(&mats, i, height);
                        helper.eval_interactions(&row_parts, &prover.beta_pows)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut evals = vec![Frac::default(); 1 << (l_skip + n_logup)];
        for (trace_idx, interaction_idx, s) in
            prover.interactions_layout.sorted_cols.iter().copied()
        {
            let pq_evals = &unstacked_interaction_evals[trace_idx];
            let height = pq_evals.len();
            debug_assert_eq!(s.col_idx, 0);
            debug_assert_eq!(1 << s.log_height(), s.len(0));
            debug_assert_eq!(s.len(0) % height, 0);
            let norm_factor_denom = s.len(0) / height;
            let norm_factor = SC::F::from_usize(norm_factor_denom).inverse();
            evals[s.row_idx..s.row_idx + s.len(0)]
                .chunks_exact_mut(height)
                .for_each(|evals| {
                    evals
                        .par_iter_mut()
                        .zip(pq_evals)
                        .for_each(|(pq_eval, evals_at_z)| {
                            let (mut numer, denom) = evals_at_z[interaction_idx];
                            numer *= norm_factor;
                            *pq_eval = Frac::new(numer.into(), denom);
                        });
                });
        }
        evals.par_iter_mut().for_each(|frac| frac.q += alpha_logup);
        evals
    };

    let (frac_sum_proof, mut xi) = fractional_sumcheck::<SC, _>(transcript, &gkr_input_evals, true);

    let n_global = max(n_max, n_logup);
    debug!(%n_global);
    while xi.len() != l_skip + n_global {
        xi.push(transcript.sample_ext());
    }
    debug!(?xi);
    prover.xi = xi;

    // Begin batch sumcheck
    let mut sumcheck_round_polys = Vec::with_capacity(n_max);
    let mut r = Vec::with_capacity(n_max + 1);
    let lambda = transcript.sample_ext();
    debug!(%lambda);

    let sp_0_polys = prover.sumcheck_uni_round0_polys(ctx, lambda);
    let sp_0_deg = sumcheck_round0_deg(l_skip, constraint_degree);
    let s_deg = constraint_degree + 1;
    let s_0_deg = sumcheck_round0_deg(l_skip, s_deg);
    let large_uni_domain = (s_0_deg + 1).next_power_of_two();
    let dft = Radix2BowersSerial;
    let s_0_logup_polys = {
        let eq_sharp_uni = eq_sharp_uni_poly(&prover.xi[..l_skip]);
        let mut eq_coeffs = eq_sharp_uni.into_coeffs();
        eq_coeffs.resize(large_uni_domain, SC::EF::ZERO);
        let eq_evals = dft.dft(eq_coeffs);

        let width = 2 * num_traces;
        let mut sp_coeffs_mat = SC::EF::zero_vec(width * large_uni_domain);
        for (i, coeffs) in sp_0_polys[..2 * num_traces].iter().enumerate() {
            for (j, &c_j) in coeffs.coeffs().iter().enumerate().take(sp_0_deg + 1) {
                unsafe {
                    *sp_coeffs_mat.get_unchecked_mut(j * width + i) = c_j;
                }
            }
        }
        let mut s_evals = dft.dft_batch(RowMajorMatrix::new(sp_coeffs_mat, width));
        for (eq, row) in zip(eq_evals, s_evals.values.chunks_mut(width)) {
            for x in row {
                *x *= eq;
            }
        }
        dft.idft_batch(s_evals)
    };

    let skip_domain_size = SC::F::from_usize(1 << l_skip);
    let (numerator_term_per_air, denominator_term_per_air): (Vec<_>, Vec<_>) = (0..num_traces)
        .map(|trace_idx| {
            let [sum_claim_p, sum_claim_q] = [0, 1].map(|is_denom| {
                (0..=s_0_deg)
                    .step_by(1 << l_skip)
                    .map(|j| unsafe {
                        *s_0_logup_polys
                            .values
                            .get_unchecked(j * 2 * num_traces + 2 * trace_idx + is_denom)
                    })
                    .sum::<SC::EF>()
                    * skip_domain_size
            });
            transcript.observe_ext(sum_claim_p);
            transcript.observe_ext(sum_claim_q);
            (sum_claim_p, sum_claim_q)
        })
        .unzip();

    let mu = transcript.sample_ext();
    debug!(%mu);
    let mu_pows = mu.powers().take(3 * num_traces).collect_vec();

    let s_0_zc_poly = {
        let eq_uni = eq_uni_poly::<SC::F, _>(l_skip, prover.xi[0]);
        let mut eq_coeffs = eq_uni.into_coeffs();
        eq_coeffs.resize(large_uni_domain, SC::EF::ZERO);
        let eq_evals = dft.dft(eq_coeffs);

        let mut sp_coeffs = SC::EF::zero_vec(large_uni_domain);
        let mus = &mu_pows[2 * num_traces..];
        let polys = &sp_0_polys[2 * num_traces..];
        for (j, batch_coeff) in sp_coeffs.iter_mut().enumerate().take(sp_0_deg + 1) {
            for (&mu, poly) in zip(mus, polys) {
                *batch_coeff += mu * *poly.coeffs().get(j).unwrap_or(&SC::EF::ZERO);
            }
        }
        let mut s_evals = dft.dft(sp_coeffs);
        for (eq, x) in zip(eq_evals, &mut s_evals) {
            *x *= eq;
        }
        dft.idft(s_evals)
    };

    let s_0_poly = UnivariatePoly::new(
        zip(
            s_0_logup_polys.values.chunks_exact(2 * num_traces),
            s_0_zc_poly,
        )
        .take(s_0_deg + 1)
        .map(|(logup_row, batched_zc)| {
            let coeff = batched_zc
                + zip(&mu_pows, logup_row)
                    .map(|(&mu_j, &x)| mu_j * x)
                    .sum::<SC::EF>();
            transcript.observe_ext(coeff);
            coeff
        })
        .collect(),
    );

    let r_0 = transcript.sample_ext();
    r.push(r_0);
    debug!(round = 0, r_round = %r_0);
    prover.prev_s_eval = s_0_poly.eval_at_point(r_0);
    debug!("s_0(r_0) = {}", prover.prev_s_eval);

    prover.fold_ple_evals(ctx, r_0);

    // MLE rounds
    let _mle_rounds_span =
        info_span!("prover.batch_constraints.mle_rounds", phase = "prover").entered();
    debug!(%s_deg);
    for round in 1..=n_max {
        let sp_round_evals = prover.sumcheck_polys_eval(round, r[round - 1]);
        let tail_start = prover
            .n_per_trace
            .iter()
            .find_position(|&&n| round as isize > n)
            .map(|(i, _)| i)
            .unwrap_or(num_traces);
        let mut sp_head_zc = vec![SC::EF::ZERO; constraint_degree];
        let mut sp_head_logup = vec![SC::EF::ZERO; constraint_degree];
        let mut sp_tail = SC::EF::ZERO;
        for trace_idx in 0..num_traces {
            let zc_idx = 2 * num_traces + trace_idx;
            let numer_idx = 2 * trace_idx;
            let denom_idx = numer_idx + 1;
            if trace_idx < tail_start {
                for i in 0..constraint_degree {
                    sp_head_zc[i] += mu_pows[zc_idx] * sp_round_evals[zc_idx][i];
                    sp_head_logup[i] += mu_pows[numer_idx] * sp_round_evals[numer_idx][i]
                        + mu_pows[denom_idx] * sp_round_evals[denom_idx][i];
                }
            } else {
                sp_tail += mu_pows[zc_idx] * sp_round_evals[zc_idx][0]
                    + mu_pows[numer_idx] * sp_round_evals[numer_idx][0]
                    + mu_pows[denom_idx] * sp_round_evals[denom_idx][0];
            }
        }
        let mut sp_head_evals = vec![SC::EF::ZERO; s_deg];
        for i in 0..constraint_degree {
            sp_head_evals[i + 1] = prover.eq_ns[round - 1] * sp_head_zc[i]
                + prover.eq_sharp_ns[round - 1] * sp_head_logup[i];
        }
        let xi_cur = prover.xi[l_skip + round - 1];
        {
            let eq_xi_0 = SC::EF::ONE - xi_cur;
            let eq_xi_1 = xi_cur;
            sp_head_evals[0] =
                (prover.prev_s_eval - eq_xi_1 * sp_head_evals[1] - sp_tail) * eq_xi_0.inverse();
        }
        let sp_head = UnivariatePoly::lagrange_interpolate(
            &(0..s_deg).map(SC::F::from_usize).collect_vec(),
            &sp_head_evals,
        );
        let batch_s = {
            let mut coeffs = sp_head.into_coeffs();
            coeffs.push(SC::EF::ZERO);
            let b = SC::EF::ONE - xi_cur;
            let a = xi_cur - b;
            for i in (0..s_deg).rev() {
                coeffs[i + 1] = a * coeffs[i] + b * coeffs[i + 1];
            }
            coeffs[0] *= b;
            coeffs[1] += sp_tail;
            UnivariatePoly::new(coeffs)
        };
        let batch_s_evals = (1..=s_deg)
            .map(|i| batch_s.eval_at_point(SC::EF::from_usize(i)))
            .collect_vec();
        for &eval in &batch_s_evals {
            transcript.observe_ext(eval);
        }
        sumcheck_round_polys.push(batch_s_evals);

        let r_round = transcript.sample_ext();
        debug!(%round, %r_round);
        r.push(r_round);
        prover.prev_s_eval = batch_s.eval_at_point(r_round);

        prover.fold_mle_evals(round, r_round);
    }
    drop(_mle_rounds_span);
    assert_eq!(r.len(), n_max + 1);

    let column_openings = prover.into_column_openings();

    // Observe openings
    for (helper, openings) in prover.eval_helpers.iter().zip(column_openings.iter()) {
        for (claim, claim_rot) in column_openings_by_rot(&openings[0], helper.needs_next) {
            transcript.observe_ext(claim);
            transcript.observe_ext(claim_rot);
        }
    }
    for (helper, openings) in prover.eval_helpers.iter().zip(column_openings.iter()) {
        for part in openings.iter().skip(1) {
            for (claim, claim_rot) in column_openings_by_rot(part, helper.needs_next) {
                transcript.observe_ext(claim);
                transcript.observe_ext(claim_rot);
            }
        }
    }

    let batch_constraint_proof = BatchConstraintProof::<SC> {
        numerator_term_per_air,
        denominator_term_per_air,
        univariate_round_coeffs: s_0_poly.into_coeffs(),
        sumcheck_round_polys,
        column_openings,
    };
    let gkr_proof = GkrProof::<SC> {
        logup_pow_witness,
        q0_claim: frac_sum_proof.fractional_sum.1,
        claims_per_layer: frac_sum_proof.claims_per_layer,
        sumcheck_polys: frac_sum_proof.sumcheck_polys,
    };
    (gkr_proof, batch_constraint_proof, r)
}
