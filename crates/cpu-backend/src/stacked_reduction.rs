//! Stacked opening reduction for [CpuBackend].
//!
//! Delegates to the reference implementation since stacked reduction operates on
//! `StackedPcsData` which is shared between both backends.

use std::{collections::HashMap, iter::zip, mem::take};

use itertools::Itertools;
use openvm_stark_backend::{
    poly_common::{eval_eq_mle, eval_eq_uni, eval_eq_uni_at_one, eval_in_uni, UnivariatePoly},
    prover::{
        poly::evals_eq_hypercube,
        stacked_pcs::{StackedPcsData, StackedSlice},
        stacked_reduction::StackedReductionProver,
        sumcheck::{
            batch_fold_mle_evals, fold_mle_evals, sumcheck_round0_deg,
            sumcheck_round_poly_evals,
        },
        ColMajorMatrix, ColMajorMatrixView, MatrixDimensions, MatrixView, ProverBackend,
    },
    StarkProtocolConfig,
};
use p3_field::{
    batch_multiplicative_inverse, ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::{backend::CpuBackend, device::CpuDevice};

/// Precomputed twiddle factors for inline DFT on 2^l_skip elements.
struct DftTwiddles<F> {
    /// iDFT twiddle factors (DIF: omega_inv powers), per layer.
    idft_tw: Vec<Vec<F>>,
    /// DFT twiddle factors (DIT: omega powers), per layer.
    dft_tw: Vec<Vec<F>>,
    /// 1/N scaling factor for iDFT.
    n_inv: F,
    l_skip: usize,
}

impl<F: TwoAdicField> DftTwiddles<F> {
    fn new(l_skip: usize) -> Self {
        let n = 1usize << l_skip;
        let omega = F::two_adic_generator(l_skip);
        let omega_inv = omega.inverse();
        let mut n_field = F::ONE;
        for _ in 0..l_skip {
            n_field += n_field;
        }
        let n_inv = n_field.inverse();

        // iDFT (DIF): twiddles from omega_inv
        let idft_tw: Vec<Vec<F>> = (0..l_skip)
            .map(|layer| {
                let half = n >> (layer + 1);
                let w = omega_inv.exp_power_of_2(layer);
                w.powers().take(half).collect()
            })
            .collect();

        // DFT (DIT): twiddles from omega
        let dft_tw: Vec<Vec<F>> = (0..l_skip)
            .map(|layer| {
                let half = 1usize << layer;
                let w = omega.exp_power_of_2(l_skip - 1 - layer);
                w.powers().take(half).collect()
            })
            .collect();

        Self {
            idft_tw,
            dft_tw,
            n_inv,
            l_skip,
        }
    }

    /// In-place iDFT (DIF + bit-reverse + scale by 1/N).
    fn idft_inplace(&self, buf: &mut [F]) {
        let n = 1usize << self.l_skip;
        let mut block_size = n;
        for tw in &self.idft_tw {
            let half = block_size >> 1;
            let mut k = 0;
            while k < n {
                for j in 0..half {
                    let u = buf[k + j];
                    let v = buf[k + j + half];
                    buf[k + j] = u + v;
                    buf[k + j + half] = (u - v) * tw[j];
                }
                k += block_size;
            }
            block_size = half;
        }
        // Bit-reverse permutation
        for i in 0..n {
            let j = i.reverse_bits() >> (usize::BITS as u32 - self.l_skip as u32);
            if i < j {
                buf.swap(i, j);
            }
        }
        // Scale by 1/N
        for val in buf.iter_mut() {
            *val *= self.n_inv;
        }
    }

    /// In-place DFT (DIT: bit-reverse input, then butterflies).
    fn dft_inplace(&self, buf: &mut [F]) {
        let n = 1usize << self.l_skip;
        // Bit-reverse permutation
        for i in 0..n {
            let j = i.reverse_bits() >> (usize::BITS as u32 - self.l_skip as u32);
            if i < j {
                buf.swap(i, j);
            }
        }
        // DIT butterflies
        let mut block_size = 2;
        for tw in &self.dft_tw {
            let half = block_size >> 1;
            let mut k = 0;
            while k < n {
                for j in 0..half {
                    let u = buf[k + j];
                    let v = buf[k + j + half] * tw[j];
                    buf[k + j] = u + v;
                    buf[k + j + half] = u - v;
                }
                k += block_size;
            }
            block_size <<= 1;
        }
    }

    /// Coset DFT: evaluate polynomial at shift * omega^k for k=0..N-1.
    /// coeffs are modified in-place (multiplied by shift powers), then DFT applied.
    fn coset_dft_inplace(&self, buf: &mut [F], shift: F) {
        // Multiply coefficients by shift^i
        let mut s = F::ONE;
        for val in buf.iter_mut() {
            *val *= s;
            s *= shift;
        }
        self.dft_inplace(buf);
    }
}

/// Optimized PLE fold: precompute barycentric weights once, then evaluate each element
/// as an inline dot product. Eliminates the 5.5M per-element Vec allocations in the
/// reference `fold_ple_evals` which calls `interpolate_coset_with_precomputation` per element.
fn fold_ple_evals_cpu<F, EF>(
    l_skip: usize,
    mat: &ColMajorMatrix<F>,
    r: EF,
) -> ColMajorMatrix<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    let height = mat.height();
    let n = 1usize << l_skip;
    let lifted_height = height.max(n);
    let width = mat.width();
    let new_height = lifted_height >> l_skip;

    // Precompute barycentric interpolation weights (shared across all elements):
    // col_scale[i] = omega^i / (r - omega^i)
    let omega = F::two_adic_generator(l_skip);
    let omega_pows: Vec<F> = omega.powers().take(n).collect();
    let denoms: Vec<EF> = omega_pows.iter().map(|&w| r - EF::from(w)).collect();
    let inv_denoms = batch_multiplicative_inverse(&denoms);
    let col_scale: Vec<EF> = omega_pows
        .iter()
        .zip(&inv_denoms)
        .map(|(&w, &inv_d)| inv_d * w)
        .collect();

    // scaling_factor = (r^N - 1) / N  where N = 2^l_skip, shift = 1
    let r_pow_n = r.exp_power_of_2(l_skip);
    let vanishing = r_pow_n - EF::ONE;
    let mut n_ef = EF::ONE;
    for _ in 0..l_skip {
        n_ef += n_ef;
    }
    let scaling_factor = vanishing * n_ef.inverse();

    // For each (column j, hypercube point x): dot product of 16 base field evals with col_scale,
    // then multiply by scaling_factor. Zero allocations per element.
    let values: Vec<EF> = mat
        .values
        .par_chunks_exact(height)
        .flat_map(|col| {
            (0..new_height)
                .into_par_iter()
                .map(|x| {
                    let base = x << l_skip;
                    let mut sum = EF::ZERO;
                    for z in 0..n {
                        // col-major: col[base + z] is contiguous (cache-friendly)
                        let f_z = col[(base + z) % height];
                        sum += col_scale[z] * f_z;
                    }
                    sum * scaling_factor
                })
                .collect::<Vec<_>>()
        })
        .collect();
    ColMajorMatrix::new(values, width)
}

pub struct StackedReductionCpuNew<'a, SC: StarkProtocolConfig> {
    l_skip: usize,
    omega_skip: SC::F,

    r_0: SC::EF,
    lambda_pows: Vec<SC::EF>,
    eq_const: SC::EF,

    stacked_per_commit: Vec<&'a StackedPcsData<SC::F, SC::Digest>>,
    trace_views: Vec<TraceViewMeta>,
    ht_diff_idxs: Vec<usize>,

    eq_r_per_lht: HashMap<usize, ColMajorMatrix<SC::EF>>,

    // After round 0:
    k_rot_r_per_lht: HashMap<usize, ColMajorMatrix<SC::EF>>,
    q_evals: Vec<ColMajorMatrix<SC::EF>>,
    eq_ub_per_trace: Vec<SC::EF>,
}

struct TraceViewMeta {
    com_idx: usize,
    slice: StackedSlice,
    lambda_eq_idx: usize,
    lambda_rot_idx: Option<usize>,
}

/// `x_int` is the integer representation of point on H_n.
fn rot_prev(x_int: usize, n: usize) -> usize {
    debug_assert!(x_int < (1 << n));
    if x_int == 0 {
        (1 << n) - 1
    } else {
        x_int - 1
    }
}

impl<'a, SC: StarkProtocolConfig>
    StackedReductionProver<'a, CpuBackend<SC>, CpuDevice<SC>> for StackedReductionCpuNew<'a, SC>
where
    SC::F: TwoAdicField,
    SC::EF: TwoAdicField + ExtensionField<SC::F>,
    CpuBackend<SC>: ProverBackend<
        Val = SC::F,
        Challenge = SC::EF,
        PcsData = StackedPcsData<SC::F, SC::Digest>,
    >,
{
    fn new(
        device: &CpuDevice<SC>,
        stacked_per_commit: Vec<&'a StackedPcsData<SC::F, SC::Digest>>,
        need_rot_per_commit: Vec<Vec<bool>>,
        r: &[SC::EF],
        lambda: SC::EF,
    ) -> Self {
        let l_skip = device.params().l_skip;
        let omega_skip = SC::F::two_adic_generator(l_skip);

        let mut trace_views = Vec::new();
        let mut lambda_idx = 0usize;
        for (com_idx, d) in stacked_per_commit.iter().enumerate() {
            let need_rot_for_commit = &need_rot_per_commit[com_idx];
            debug_assert_eq!(need_rot_for_commit.len(), d.layout.mat_starts.len());
            for &(mat_idx, _col_idx, slice) in &d.layout.sorted_cols {
                let lambda_eq_idx = lambda_idx;
                lambda_idx += 1;
                let lambda_rot_idx = if need_rot_for_commit[mat_idx] {
                    Some(lambda_idx)
                } else {
                    None
                };
                lambda_idx += 1;
                trace_views.push(TraceViewMeta {
                    com_idx,
                    slice,
                    lambda_eq_idx,
                    lambda_rot_idx,
                });
            }
        }
        let lambda_pows = lambda.powers().take(lambda_idx).collect_vec();

        let mut ht_diff_idxs = Vec::new();
        let mut eq_r_per_lht: HashMap<usize, ColMajorMatrix<SC::EF>> = HashMap::new();
        let mut last_height = 0;
        for (i, tv) in trace_views.iter().enumerate() {
            let n_lift = tv.slice.log_height().saturating_sub(l_skip);
            if i == 0 || tv.slice.log_height() != last_height {
                ht_diff_idxs.push(i);
                last_height = tv.slice.log_height();
            }
            eq_r_per_lht
                .entry(tv.slice.log_height())
                .or_insert_with(|| ColMajorMatrix::new(evals_eq_hypercube(&r[1..1 + n_lift]), 1));
        }
        ht_diff_idxs.push(trace_views.len());

        let eq_const = eval_eq_uni_at_one(l_skip, r[0] * omega_skip);
        let eq_ub_per_trace = vec![SC::EF::ONE; trace_views.len()];

        Self {
            l_skip,
            omega_skip,
            r_0: r[0],
            lambda_pows,
            eq_const,
            stacked_per_commit,
            trace_views,
            ht_diff_idxs,
            eq_r_per_lht,
            q_evals: vec![],
            k_rot_r_per_lht: HashMap::new(),
            eq_ub_per_trace,
        }
    }

    fn batch_sumcheck_uni_round0_poly(&mut self) -> UnivariatePoly<SC::EF> {
        let _span = tracing::info_span!("stacked_round0").entered();
        let l_skip = self.l_skip;
        let omega_skip = self.omega_skip;
        let r_0 = self.r_0;
        let eq_const = self.eq_const;
        let n_skip = 1usize << l_skip;
        let d = 2usize;
        let d_n = d << l_skip;
        let s_0_deg = sumcheck_round0_deg(l_skip, d);
        let g = SC::F::GENERATOR;
        let coset_shifts: Vec<SC::F> = g.powers().skip(1).take(d).collect_vec();
        let twiddles = DftTwiddles::new(l_skip);

        // Extract all self-field references before parallel iteration to satisfy borrow checker
        let ht_diff_idxs = &self.ht_diff_idxs;
        let trace_views = &self.trace_views;
        let stacked_per_commit = &self.stacked_per_commit;
        let eq_r_per_lht = &self.eq_r_per_lht;
        let lambda_pows = &self.lambda_pows;

        let s_0_polys: Vec<[UnivariatePoly<SC::EF>; 2]> = ht_diff_idxs
            .par_windows(2)
            .map(|window| {
                let t_window = &trace_views[window[0]..window[1]];
                let log_height = t_window[0].slice.log_height();
                let n = log_height as isize - l_skip as isize;
                let n_lift = n.max(0) as usize;
                let eq_rs = eq_r_per_lht.get(&log_height).unwrap().column(0);
                debug_assert_eq!(eq_rs.len(), 1 << n_lift);

                // Collect column data references (each trace view is a single column)
                let q_cols: Vec<&[SC::F]> = t_window
                    .iter()
                    .map(|tv| {
                        debug_assert_eq!(tv.slice.log_height(), log_height);
                        let q = &stacked_per_commit[tv.com_idx].matrix;
                        let s = tv.slice;
                        &q.column(s.col_idx)[s.row_idx..s.row_idx + s.len(l_skip)]
                    })
                    .collect();

                // Precompute z-dependent scalars (d_n values, independent of x)
                let omega_pows: Vec<SC::F> = omega_skip.powers().take(n_skip).collect();
                let (l, omega_l, r_uni) = if n.is_negative() {
                    (
                        l_skip.wrapping_add_signed(n),
                        omega_skip.exp_power_of_2(-n as usize),
                        r_0.exp_power_of_2(-n as usize),
                    )
                } else {
                    (l_skip, omega_skip, r_0)
                };
                // pre_z[rm_idx] = (pre_eq, pre_rot, pre_1) where rm_idx = z_idx * d + c_idx
                let mut pre_z = Vec::with_capacity(d_n);
                for z_idx in 0..n_skip {
                    for &shift in &coset_shifts {
                        let z_val: SC::F = shift * omega_pows[z_idx];
                        let ind = eval_in_uni(l_skip, n, z_val);
                        let eq_r0: SC::EF = eval_eq_uni(l, z_val.into(), r_uni);
                        let eq_r0_rot: SC::EF =
                            eval_eq_uni(l, z_val.into(), r_uni * omega_l);
                        let eq_1: SC::F = eval_eq_uni_at_one(l_skip, z_val);
                        pre_z.push((eq_r0 * ind, eq_r0_rot * ind, eq_const * eq_1 * ind));
                    }
                }

                // Pre-extract lambda weights per trace
                let lambda_eqs: Vec<SC::EF> = t_window
                    .iter()
                    .map(|tv| lambda_pows[tv.lambda_eq_idx])
                    .collect();
                let lambda_rots: Vec<Option<SC::EF>> = t_window
                    .iter()
                    .map(|tv| tv.lambda_rot_idx.map(|idx| lambda_pows[idx]))
                    .collect();
                let num_traces = q_cols.len();

                // Parallel fold-reduce over x-points in H_{n_lift}.
                // Each thread reuses its own DFT and weighted-sum buffers.
                let evals = (0..1usize << n_lift)
                    .into_par_iter()
                    .fold(
                        || {
                            (
                                vec![[SC::EF::ZERO; 2]; d_n],
                                vec![SC::F::ZERO; n_skip],
                                vec![SC::F::ZERO; n_skip],
                                vec![SC::EF::ZERO; d_n],
                                vec![SC::EF::ZERO; d_n],
                            )
                        },
                        |(mut result, mut coeffs, mut coset, mut w_eq, mut w_rot), x| {
                            // Zero out per-x weighted sums
                            for v in w_eq.iter_mut() {
                                *v = SC::EF::ZERO;
                            }
                            for v in w_rot.iter_mut() {
                                *v = SC::EF::ZERO;
                            }

                            // Phase 1: For each trace, iDFT + coset DFTs, accumulate weighted sums
                            for t in 0..num_traces {
                                let col = q_cols[t];
                                let height = col.len();

                                // Read n_skip evaluations from column at block x
                                for z in 0..n_skip {
                                    coeffs[z] = col[((x << l_skip) + z) % height];
                                }
                                twiddles.idft_inplace(&mut coeffs);

                                let le = lambda_eqs[t];
                                let lr = lambda_rots[t];
                                for (c_idx, &shift) in coset_shifts.iter().enumerate() {
                                    coset.copy_from_slice(&coeffs);
                                    twiddles.coset_dft_inplace(&mut coset, shift);

                                    for z_idx in 0..n_skip {
                                        let rm_idx = z_idx * d + c_idx;
                                        let q = coset[z_idx];
                                        w_eq[rm_idx] += le * q;
                                        if let Some(lr) = lr {
                                            w_rot[rm_idx] += lr * q;
                                        }
                                    }
                                }
                            }

                            // Phase 2: Multiply weighted sums by precomputed z-scalars and
                            // x-dependent eq evaluations
                            let eq_cube = eq_rs[x];
                            let k_rot_cube = eq_rs[rot_prev(x, n_lift)];
                            let delta = k_rot_cube - eq_cube;

                            for rm_idx in 0..d_n {
                                let (pe, pr, p1) = pre_z[rm_idx];
                                result[rm_idx][0] += (pe * eq_cube) * w_eq[rm_idx];
                                result[rm_idx][1] +=
                                    (pr * eq_cube + p1 * delta) * w_rot[rm_idx];
                            }
                            (result, coeffs, coset, w_eq, w_rot)
                        },
                    )
                    .map(|(result, ..)| result)
                    .reduce(
                        || vec![[SC::EF::ZERO; 2]; d_n],
                        |mut a, b| {
                            for (ai, bi) in a.iter_mut().zip(b.iter()) {
                                ai[0] += bi[0];
                                ai[1] += bi[1];
                            }
                            a
                        },
                    );

                // Convert accumulated coset evaluations to polynomial coefficients
                std::array::from_fn(|wd| {
                    let values: Vec<SC::EF> = evals.iter().map(|v| v[wd]).collect();
                    UnivariatePoly::from_geometric_cosets_evals_idft(
                        RowMajorMatrix::new(values, d),
                        g,
                        g,
                    )
                })
            })
            .collect();

        let s_0_coeffs = (0..=s_0_deg)
            .map(|i| {
                s_0_polys
                    .iter()
                    .flat_map(|ps| ps.iter())
                    .map(|p| p.coeffs().get(i).copied().unwrap_or(SC::EF::ZERO))
                    .sum::<SC::EF>()
            })
            .collect_vec();
        UnivariatePoly::new(s_0_coeffs)
    }

    fn fold_ple_evals(&mut self, u_0: SC::EF) {
        let _span = tracing::info_span!("stacked_fold_ple").entered();
        let l_skip = self.l_skip;
        let r_0 = self.r_0;
        let omega_skip = self.omega_skip;
        self.q_evals = self
            .stacked_per_commit
            .iter()
            .map(|d| fold_ple_evals_cpu(l_skip, &d.matrix, u_0))
            .collect_vec();
        let eq_uni_u0r0 = eval_eq_uni(l_skip, u_0, r_0);
        let eq_uni_u0r0_rot = eval_eq_uni(l_skip, u_0, r_0 * omega_skip);
        let eq_uni_u01 = eval_eq_uni_at_one(l_skip, u_0);
        self.k_rot_r_per_lht = self
            .eq_r_per_lht
            .par_iter_mut()
            .map(|(&log_height, mat)| {
                let n = log_height as isize - l_skip as isize;
                let n_lift = n.max(0) as usize;
                debug_assert_eq!(mat.values.len(), 1 << n_lift);
                let ind = eval_in_uni(l_skip, n, u_0);
                let (eq_uni, eq_uni_rot) = if n.is_negative() {
                    let omega = omega_skip.exp_power_of_2(-n as usize);
                    let r = r_0.exp_power_of_2(-n as usize);
                    let l = l_skip.wrapping_add_signed(n);
                    (eval_eq_uni(l, u_0, r), eval_eq_uni(l, u_0, r * omega))
                } else {
                    (eq_uni_u0r0, eq_uni_u0r0_rot)
                };
                let evals: Vec<_> = (0..1 << n_lift)
                    .into_par_iter()
                    .map(|x| {
                        let eq_cube = unsafe { *mat.get_unchecked(x, 0) };
                        let k_rot_cube = unsafe { *mat.get_unchecked(rot_prev(x, n_lift), 0) };
                        ind * (eq_uni_rot * eq_cube
                            + self.eq_const * eq_uni_u01 * (k_rot_cube - eq_cube))
                    })
                    .collect();
                mat.values.par_iter_mut().for_each(|v| {
                    *v *= ind * eq_uni;
                });
                (log_height, ColMajorMatrix::new(evals, 1))
            })
            .collect();
    }

    fn batch_sumcheck_poly_eval(&mut self, round: usize, _u_prev: SC::EF) -> [SC::EF; 2] {
        let _span = tracing::info_span!("stacked_round_eval").entered();
        let l_skip = self.l_skip;
        let s_deg = 2;
        let s_evals: Vec<_> = self
            .ht_diff_idxs
            .par_windows(2)
            .flat_map(|window| {
                let t_views = &self.trace_views[window[0]..window[1]];
                let log_height = t_views[0].slice.log_height();
                let n_lift = log_height.saturating_sub(l_skip);
                let hypercube_dim = n_lift.saturating_sub(round);
                let eq_rs = self.eq_r_per_lht.get(&log_height).unwrap().column(0);
                let k_rot_rs = self.k_rot_r_per_lht.get(&log_height).unwrap().column(0);
                debug_assert_eq!(eq_rs.len(), 1 << n_lift.saturating_sub(round - 1));
                debug_assert_eq!(k_rot_rs.len(), 1 << n_lift.saturating_sub(round - 1));
                let t_cols = t_views
                    .iter()
                    .map(|tv| {
                        debug_assert_eq!(tv.slice.log_height(), log_height);
                        let q = &self.q_evals[tv.com_idx];
                        let s = tv.slice;
                        let row_start = if round <= n_lift {
                            (s.row_idx >> log_height) << (hypercube_dim + 1)
                        } else {
                            (s.row_idx >> (l_skip + round)) << 1
                        };
                        let t_col =
                            &q.column(s.col_idx)[row_start..row_start + (2 << hypercube_dim)];
                        ColMajorMatrixView::new(t_col, 1)
                    })
                    .collect_vec();
                sumcheck_round_poly_evals(hypercube_dim + 1, s_deg, &t_cols, |x, y, evals| {
                    evals
                        .iter()
                        .enumerate()
                        .fold([SC::EF::ZERO; 2], |mut acc, (i, eval)| {
                            let t_idx = window[0] + i;
                            let tv = &self.trace_views[t_idx];
                            let q = eval[0];
                            let mut eq_ub = self.eq_ub_per_trace[t_idx];
                            let (eq, k_rot) = if round > n_lift {
                                let b = (tv.slice.row_idx >> (l_skip + round - 1)) & 1;
                                eq_ub *= eval_eq_mle(&[x], &[SC::F::from_bool(b == 1)]);
                                debug_assert_eq!(y, 0);
                                (eq_rs[0] * eq_ub, k_rot_rs[0] * eq_ub)
                            } else {
                                let eq_r =
                                    eq_rs[y << 1] * (SC::EF::ONE - x) + eq_rs[(y << 1) + 1] * x;
                                let k_rot_r = k_rot_rs[y << 1] * (SC::EF::ONE - x)
                                    + k_rot_rs[(y << 1) + 1] * x;
                                (eq_r * eq_ub, k_rot_r * eq_ub)
                            };
                            acc[0] += self.lambda_pows[tv.lambda_eq_idx] * q * eq;
                            if let Some(rot_idx) = tv.lambda_rot_idx {
                                acc[1] += self.lambda_pows[rot_idx] * q * k_rot;
                            }
                            acc
                        })
                })
            })
            .collect();
        std::array::from_fn(|i| s_evals.iter().map(|evals| evals[i]).sum::<SC::EF>())
    }

    fn fold_mle_evals(&mut self, round: usize, u_round: SC::EF) {
        let _span = tracing::info_span!("stacked_fold_mle").entered();
        let l_skip = self.l_skip;
        self.q_evals = batch_fold_mle_evals(take(&mut self.q_evals), u_round);
        self.eq_r_per_lht = take(&mut self.eq_r_per_lht)
            .into_par_iter()
            .map(|(lht, mat)| (lht, fold_mle_evals(mat, u_round)))
            .collect();
        self.k_rot_r_per_lht = take(&mut self.k_rot_r_per_lht)
            .into_par_iter()
            .map(|(lht, mat)| (lht, fold_mle_evals(mat, u_round)))
            .collect();
        for (tv, eq_ub) in zip(&self.trace_views, &mut self.eq_ub_per_trace) {
            let s = tv.slice;
            let n_lift = s.log_height().saturating_sub(l_skip);
            if round > n_lift {
                let b = (s.row_idx >> (l_skip + round - 1)) & 1;
                *eq_ub *= eval_eq_mle(&[u_round], &[SC::F::from_bool(b == 1)]);
            }
        }
    }

    fn into_stacked_openings(self) -> Vec<Vec<SC::EF>> {
        self.q_evals
            .into_iter()
            .map(|q| {
                debug_assert_eq!(q.height(), 1);
                q.values
            })
            .collect()
    }
}
