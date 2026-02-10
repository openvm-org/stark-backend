//! Single AIR constraint evaluation helpers

use std::iter::zip;

use itertools::Itertools;
use openvm_stark_backend::{
    air_builders::symbolic::{symbolic_expression::SymbolicEvaluator, SymbolicExpressionDag},
    interaction::SymbolicInteraction,
};
use p3_field::{ExtensionField, TwoAdicField};

use crate::prover::{
    logup_zerocheck::evaluator::{ProverConstraintEvaluator, ViewPair},
    AirProvingContextV2, CpuBackendV2, StridedColMajorMatrixView,
};

type SCV2 = crate::baby_bear_poseidon2::BabyBearPoseidon2ConfigV2;

/// For a single AIR
pub struct EvalHelper<'a, F> {
    /// AIR constraints
    pub constraints_dag: &'a SymbolicExpressionDag<F>,
    /// Interactions
    pub interactions: Vec<SymbolicInteraction<F>>,
    pub public_values: Vec<F>,
    pub preprocessed_trace: Option<StridedColMajorMatrixView<'a, F>>,
    // TODO: skip rotation if vk dictates it is never used
    pub needs_next: bool,
    pub constraint_degree: u8,
}

impl<'a> EvalHelper<'a, crate::baby_bear_poseidon2::F> {
    /// Returns list of (ref to column-major matrix, is_rot) pairs in the order:
    /// - (if has_preprocessed) (preprocessed, false), (preprocessed, true)
    /// - (cached_0, false), (cached_0, true), ..., (cached_{m-1}, false), (cached_{m-1}, true)
    /// - (common, false), (common, true)
    ///
    /// Note: currently every matrix returns both non-rotated and rotated versions. This will change
    /// in the future for perf.
    pub fn view_mats(
        &self,
        ctx: &'a AirProvingContextV2<CpuBackendV2<SCV2>>,
    ) -> Vec<(StridedColMajorMatrixView<'a, crate::baby_bear_poseidon2::F>, bool)> {
        let base_mats = usize::from(self.has_preprocessed()) + 1 + ctx.cached_mains.len();
        let mut mats = Vec::with_capacity(if self.needs_next {
            2 * base_mats
        } else {
            base_mats
        });
        if let Some(mat) = self.preprocessed_trace {
            mats.push((mat, false));
            if self.needs_next {
                mats.push((mat, true));
            }
        }
        for cd in ctx.cached_mains.iter() {
            let trace_view = cd.data.mat_view(0);
            mats.push((trace_view, false));
            if self.needs_next {
                mats.push((trace_view, true));
            }
        }
        mats.push((ctx.common_main.as_view().into(), false));
        if self.needs_next {
            mats.push((ctx.common_main.as_view().into(), true));
        }
        mats
    }
}

impl<F: TwoAdicField> EvalHelper<'_, F> {
    pub fn has_preprocessed(&self) -> bool {
        self.preprocessed_trace.is_some()
    }

    /// See [Self::evaluator].
    // Assumes that `z[0] != 1` or `omega_D^{-1}` to avoid handling division by zero.
    pub fn acc_constraints<FF: ExtensionField<F>, EF: ExtensionField<FF>>(
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

    /// See [Self::evaluator].
    ///
    /// Returns sum of ordered list of `interactions`, weighted by `eq(\xi_3, b_{T,\hat\sigma})`
    /// terms as (numerator, denominator) pair.
    ///
    /// Note: the denominator does not include the `alpha` term.
    pub fn acc_interactions<FF, EF>(
        &self,
        row_parts: &[Vec<FF>],
        beta_pows: &[EF],
        eq_3bs: &[EF],
    ) -> [EF; 2]
    where
        FF: ExtensionField<F>,
        EF: ExtensionField<FF> + ExtensionField<F>,
    {
        // PERF[jpw]: no need to collect the vec, but I ran into a lifetime issue returning iterator
        // in `eval_interactions`
        let interaction_evals = self.eval_interactions(row_parts, beta_pows);
        let mut numer = EF::ZERO;
        let mut denom = EF::ZERO; // without alpha term
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
        FF: ExtensionField<F>,
        EF: ExtensionField<FF> + ExtensionField<F>,
    {
        let evaluator = self.evaluator(row_parts);
        self.interactions
            .iter()
            .map(|interaction| {
                let b = F::from_u32(interaction.bus_index as u32 + 1);
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

    // `row_parts` should have separate Vec in following order:
    // - selectors [is_first_row, is_transition, is_last_row]
    // - (if has_preprocessed) preprocessed
    // - (if has_preprocessed) preprocessed_rot
    // - cached_0
    // - cached_0_rot
    // - ...
    // - common
    // - common_rot
    fn evaluator<FF: ExtensionField<F>>(
        &self,
        row_parts: &[Vec<FF>],
    ) -> ProverConstraintEvaluator<'_, F, FF> {
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
        ProverConstraintEvaluator {
            preprocessed,
            partitioned_main: view_pairs,
            is_first_row: sels[0],
            is_transition: sels[1],
            is_last_row: sels[2],
            public_values: &self.public_values,
        }
    }
}
