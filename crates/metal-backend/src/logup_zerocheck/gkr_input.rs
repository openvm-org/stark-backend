use std::cmp::max;

use itertools::Itertools;
use openvm_metal_common::{copy::MemCopyH2D, d_buffer::MetalBuffer};
#[cfg(debug_assertions)]
use openvm_stark_backend::air_builders::symbolic::{
    symbolic_expression::SymbolicEvaluator,
    symbolic_variable::{Entry, SymbolicVariable},
    SymbolicConstraints,
};
use openvm_stark_backend::prover::{
    fractional_sumcheck_gkr::Frac,
    stacked_pcs::{StackedLayout, StackedSlice},
    DeviceMultiStarkProvingKey, MatrixDimensions, ProvingContext,
};
use p3_field::{Field, PrimeCharacteristicRing};
use tracing::instrument;

use super::errors::InteractionMetalError;
#[cfg(debug_assertions)]
use super::rules::{codec::Codec, Rule, RuleWithFlag, Source};
use crate::{
    metal::logup_zerocheck::{
        frac_add_alpha, frac_matrix_vertically_repeat, frac_vector_scalar_multiply_ext_fp,
        logup_gkr_input_eval,
    },
    prelude::{EF, F},
    MetalBackend,
};

const TASK_SIZE: u32 = 65536;

#[allow(dead_code)]
#[derive(Clone)]
pub struct TraceInteractionMeta {
    pub trace_idx: usize,
    pub air_idx: usize,
    pub layout_slices: Vec<StackedSlice>,
}

#[cfg(debug_assertions)]
struct CpuInteractionEvaluator<'a> {
    row: usize,
    height: usize,
    preprocessed: Option<&'a [F]>,
    mains: &'a [Vec<F>],
    public_values: &'a [F],
    challenges: &'a [EF],
}

#[cfg(debug_assertions)]
impl SymbolicEvaluator<F, EF> for CpuInteractionEvaluator<'_> {
    fn eval_const(&self, c: F) -> EF {
        c.into()
    }

    fn eval_var(&self, symbolic_var: SymbolicVariable<F>) -> EF {
        match symbolic_var.entry {
            Entry::Preprocessed { offset } => {
                let matrix = self.preprocessed.expect("missing preprocessed matrix");
                matrix[symbolic_var.index * self.height + (self.row + offset) % self.height].into()
            }
            Entry::Main { part_index, offset } => {
                let matrix = &self.mains[part_index];
                matrix[symbolic_var.index * self.height + (self.row + offset) % self.height].into()
            }
            Entry::Public => self.public_values[symbolic_var.index].into(),
            Entry::Challenge => self.challenges[symbolic_var.index],
            Entry::Permutation { .. } | Entry::Exposed => {
                unreachable!("unsupported symbolic variable")
            }
        }
    }

    fn eval_is_first_row(&self) -> EF {
        EF::from_bool(self.row == 0)
    }

    fn eval_is_last_row(&self) -> EF {
        EF::from_bool(self.row == self.height - 1)
    }

    fn eval_is_transition(&self) -> EF {
        EF::from_bool(self.row != self.height - 1)
    }
}

#[cfg(debug_assertions)]
fn frac_segment_to_vec(buf: &MetalBuffer<Frac<EF>>, offset: usize, len: usize) -> Vec<Frac<EF>> {
    if len == 0 {
        return Vec::new();
    }
    let p_ptr = buf.as_ptr() as *const EF;
    let q_ptr = unsafe { p_ptr.add(buf.len()) };
    (0..len)
        .map(|i| unsafe { Frac::new(*p_ptr.add(offset + i), *q_ptr.add(offset + i)) })
        .collect()
}

#[cfg(debug_assertions)]
#[allow(clippy::too_many_arguments)]
fn eval_source_for_row(
    source: &Source<F>,
    row: usize,
    height: usize,
    intermediates: &[EF],
    preprocessed: Option<&[F]>,
    mains: &[Vec<F>],
    public_values: &[F],
    challenges: &[EF],
) -> EF {
    match source {
        Source::Intermediate(idx) => intermediates[*idx],
        Source::Var(var) => match var.entry {
            Entry::Preprocessed { offset } => preprocessed.expect("missing preprocessed matrix")
                [var.index * height + (row + offset) % height]
                .into(),
            Entry::Main { part_index, offset } => {
                mains[part_index][var.index * height + (row + offset) % height].into()
            }
            Entry::Public => public_values[var.index].into(),
            Entry::Challenge => challenges[var.index],
            Entry::Permutation { .. } | Entry::Exposed => unreachable!("unsupported variable"),
        },
        Source::IsFirst => EF::from_bool(row == 0),
        Source::IsLast => EF::from_bool(row == height - 1),
        Source::IsTransition => EF::from_bool(row != height - 1),
        Source::Constant(c) => (*c).into(),
        Source::TerminalIntermediate => EF::ZERO,
    }
}

#[cfg(debug_assertions)]
#[allow(clippy::too_many_arguments)]
fn eval_rules_row_q(
    encoded_rules: &[u128],
    used_nodes: &[usize],
    pair_idxs: &[u32],
    row: usize,
    height: usize,
    num_interactions: usize,
    buffer_size: usize,
    preprocessed: Option<&[F]>,
    mains: &[Vec<F>],
    public_values: &[F],
    challenges: &[EF],
) -> (Vec<EF>, Vec<EF>) {
    let rules: Vec<RuleWithFlag<F>> = encoded_rules
        .iter()
        .map(|&encoded| RuleWithFlag::decode(encoded))
        .collect();
    let mut p = vec![EF::ZERO; num_interactions];
    let mut q = vec![EF::ZERO; num_interactions];
    let mut intermediates = vec![EF::ZERO; buffer_size.max(1)];
    let mut rules_evaluated = 0usize;

    for (&node_idx, &pair_idx) in used_nodes.iter().zip(pair_idxs.iter()) {
        let result = if node_idx < rules_evaluated {
            match &rules[node_idx].inner {
                Rule::Variable(x) | Rule::BufferVar(x, _) => eval_source_for_row(
                    x,
                    row,
                    height,
                    &intermediates,
                    preprocessed,
                    mains,
                    public_values,
                    challenges,
                ),
                Rule::Add(_, _, z) | Rule::Sub(_, _, z) | Rule::Mul(_, _, z) | Rule::Neg(_, z) => {
                    let z_idx = match z {
                        Source::Intermediate(i) => *i,
                        _ => 0,
                    };
                    intermediates[z_idx]
                }
            }
        } else {
            let mut result = EF::ZERO;
            for (rule_idx, rule) in rules
                .iter()
                .enumerate()
                .take(node_idx + 1)
                .skip(rules_evaluated)
            {
                let node_result = match &rule.inner {
                    Rule::Add(x, y, z) => {
                        let x = eval_source_for_row(
                            x,
                            row,
                            height,
                            &intermediates,
                            preprocessed,
                            mains,
                            public_values,
                            challenges,
                        );
                        let y = eval_source_for_row(
                            y,
                            row,
                            height,
                            &intermediates,
                            preprocessed,
                            mains,
                            public_values,
                            challenges,
                        );
                        let out = x + y;
                        if let Source::Intermediate(i) = z {
                            intermediates[*i] = out;
                        }
                        out
                    }
                    Rule::Sub(x, y, z) => {
                        let x = eval_source_for_row(
                            x,
                            row,
                            height,
                            &intermediates,
                            preprocessed,
                            mains,
                            public_values,
                            challenges,
                        );
                        let y = eval_source_for_row(
                            y,
                            row,
                            height,
                            &intermediates,
                            preprocessed,
                            mains,
                            public_values,
                            challenges,
                        );
                        let out = x - y;
                        if let Source::Intermediate(i) = z {
                            intermediates[*i] = out;
                        }
                        out
                    }
                    Rule::Mul(x, y, z) => {
                        let x = eval_source_for_row(
                            x,
                            row,
                            height,
                            &intermediates,
                            preprocessed,
                            mains,
                            public_values,
                            challenges,
                        );
                        let y = eval_source_for_row(
                            y,
                            row,
                            height,
                            &intermediates,
                            preprocessed,
                            mains,
                            public_values,
                            challenges,
                        );
                        let out = x * y;
                        if let Source::Intermediate(i) = z {
                            intermediates[*i] = out;
                        }
                        out
                    }
                    Rule::Neg(x, z) => {
                        let x = eval_source_for_row(
                            x,
                            row,
                            height,
                            &intermediates,
                            preprocessed,
                            mains,
                            public_values,
                            challenges,
                        );
                        let out = -x;
                        if let Source::Intermediate(i) = z {
                            intermediates[*i] = out;
                        }
                        out
                    }
                    Rule::Variable(x) => eval_source_for_row(
                        x,
                        row,
                        height,
                        &intermediates,
                        preprocessed,
                        mains,
                        public_values,
                        challenges,
                    ),
                    Rule::BufferVar(x, z) => {
                        let out = eval_source_for_row(
                            x,
                            row,
                            height,
                            &intermediates,
                            preprocessed,
                            mains,
                            public_values,
                            challenges,
                        );
                        if let Source::Intermediate(i) = z {
                            intermediates[*i] = out;
                        }
                        out
                    }
                };
                if rule_idx == node_idx {
                    result = node_result;
                }
            }
            rules_evaluated = node_idx + 1;
            result
        };

        let interaction_idx = (pair_idx >> 1) as usize;
        if (pair_idx & 1) == 0 {
            p[interaction_idx] = result;
        } else {
            q[interaction_idx] = result;
        }
    }

    (p, q)
}

#[cfg(debug_assertions)]
#[allow(clippy::too_many_arguments)]
fn eval_rules_row_values(
    encoded_rules: &[u128],
    row: usize,
    height: usize,
    buffer_size: usize,
    preprocessed: Option<&[F]>,
    mains: &[Vec<F>],
    public_values: &[F],
    challenges: &[EF],
) -> (Vec<EF>, Vec<EF>) {
    let rules: Vec<RuleWithFlag<F>> = encoded_rules
        .iter()
        .map(|&encoded| RuleWithFlag::decode(encoded))
        .collect();
    let mut results = vec![EF::ZERO; rules.len()];
    let mut intermediates = vec![EF::ZERO; buffer_size.max(1)];

    for (rule_idx, rule) in rules.iter().enumerate() {
        let node_result = match &rule.inner {
            Rule::Add(x, y, z) => {
                let x = eval_source_for_row(
                    x,
                    row,
                    height,
                    &intermediates,
                    preprocessed,
                    mains,
                    public_values,
                    challenges,
                );
                let y = eval_source_for_row(
                    y,
                    row,
                    height,
                    &intermediates,
                    preprocessed,
                    mains,
                    public_values,
                    challenges,
                );
                let out = x + y;
                if let Source::Intermediate(i) = z {
                    intermediates[*i] = out;
                }
                out
            }
            Rule::Sub(x, y, z) => {
                let x = eval_source_for_row(
                    x,
                    row,
                    height,
                    &intermediates,
                    preprocessed,
                    mains,
                    public_values,
                    challenges,
                );
                let y = eval_source_for_row(
                    y,
                    row,
                    height,
                    &intermediates,
                    preprocessed,
                    mains,
                    public_values,
                    challenges,
                );
                let out = x - y;
                if let Source::Intermediate(i) = z {
                    intermediates[*i] = out;
                }
                out
            }
            Rule::Mul(x, y, z) => {
                let x = eval_source_for_row(
                    x,
                    row,
                    height,
                    &intermediates,
                    preprocessed,
                    mains,
                    public_values,
                    challenges,
                );
                let y = eval_source_for_row(
                    y,
                    row,
                    height,
                    &intermediates,
                    preprocessed,
                    mains,
                    public_values,
                    challenges,
                );
                let out = x * y;
                if let Source::Intermediate(i) = z {
                    intermediates[*i] = out;
                }
                out
            }
            Rule::Neg(x, z) => {
                let x = eval_source_for_row(
                    x,
                    row,
                    height,
                    &intermediates,
                    preprocessed,
                    mains,
                    public_values,
                    challenges,
                );
                let out = -x;
                if let Source::Intermediate(i) = z {
                    intermediates[*i] = out;
                }
                out
            }
            Rule::Variable(x) => eval_source_for_row(
                x,
                row,
                height,
                &intermediates,
                preprocessed,
                mains,
                public_values,
                challenges,
            ),
            Rule::BufferVar(x, z) => {
                let out = eval_source_for_row(
                    x,
                    row,
                    height,
                    &intermediates,
                    preprocessed,
                    mains,
                    public_values,
                    challenges,
                );
                if let Source::Intermediate(i) = z {
                    intermediates[*i] = out;
                }
                out
            }
        };
        results[rule_idx] = node_result;
    }
    (results, intermediates)
}

#[cfg(debug_assertions)]
#[allow(clippy::too_many_arguments)]
fn debug_assert_trace_interactions_match_cpu(
    trace_idx: usize,
    air_idx: usize,
    symbolic_interactions: &[openvm_stark_backend::interaction::SymbolicInteraction<F>],
    encoded_rules: &[u128],
    used_nodes: &[usize],
    pair_idxs: &[u32],
    buffer_size: usize,
    height: usize,
    preprocessed_host: Option<&[F]>,
    partitioned_main_host: &[Vec<F>],
    public_values: &[F],
    challenges: &[EF],
    got: &[Frac<EF>],
) {
    let beta_pows = &challenges[1..];
    for (interaction_idx, interaction) in symbolic_interactions.iter().enumerate() {
        for row in 0..height {
            let evaluator = CpuInteractionEvaluator {
                row,
                height,
                preprocessed: preprocessed_host,
                mains: partitioned_main_host,
                public_values,
                challenges,
            };
            let numer = evaluator.eval_expr(&interaction.count);
            let msg_len = interaction.message.len();
            let bus = F::from_u16(interaction.bus_index + 1);
            let msg_evals = interaction
                .message
                .iter()
                .map(|expr| evaluator.eval_expr(expr))
                .collect_vec();
            let mut denom = beta_pows[msg_len] * bus;
            for (msg_eval, beta) in msg_evals.iter().zip(beta_pows.iter()) {
                denom += *beta * *msg_eval;
            }

            let idx = interaction_idx * height + row;
            let expected = Frac::new(numer, denom);
            if got[idx].p != expected.p || got[idx].q != expected.q {
                let (got_row0_p, expected_row0_p, got_row0_q, expected_row0_q) = if row == 0 {
                    let got_p = (0..symbolic_interactions.len())
                        .map(|i| got[i * height].p)
                        .collect_vec();
                    let got_q = (0..symbolic_interactions.len())
                        .map(|i| got[i * height].q)
                        .collect_vec();
                    let expected_p = symbolic_interactions
                        .iter()
                        .map(|intxn| evaluator.eval_expr(&intxn.count))
                        .collect_vec();
                    let expected_q = symbolic_interactions
                        .iter()
                        .map(|intxn| {
                            let msg_len_i = intxn.message.len();
                            let bus_i = F::from_u16(intxn.bus_index + 1);
                            let msg_evals_i = intxn
                                .message
                                .iter()
                                .map(|expr| evaluator.eval_expr(expr))
                                .collect_vec();
                            let mut denom_i = beta_pows[msg_len_i] * bus_i;
                            for (msg_eval_i, beta_i) in msg_evals_i.iter().zip(beta_pows.iter()) {
                                denom_i += *beta_i * *msg_eval_i;
                            }
                            denom_i
                        })
                        .collect_vec();
                    (Some(got_p), Some(expected_p), Some(got_q), Some(expected_q))
                } else {
                    (None, None, None, None)
                };
                let (
                    rules_row0_p,
                    rules_row0_q,
                    cpu_rule_results_row0,
                    cpu_rule_intermediates_row0,
                    swapped_rules_row0_p,
                    swapped_rules_row0_q,
                ) = if row == 0 {
                    let (p, q) = eval_rules_row_q(
                        encoded_rules,
                        used_nodes,
                        pair_idxs,
                        0,
                        height,
                        symbolic_interactions.len(),
                        buffer_size,
                        preprocessed_host,
                        partitioned_main_host,
                        public_values,
                        challenges,
                    );
                    let (rule_vals, rule_intermediates) = eval_rules_row_values(
                        encoded_rules,
                        0,
                        height,
                        buffer_size,
                        preprocessed_host,
                        partitioned_main_host,
                        public_values,
                        challenges,
                    );
                    (
                        Some(p),
                        Some(q),
                        Some(rule_vals),
                        Some(rule_intermediates),
                        Option::<Vec<EF>>::None,
                        Option::<Vec<EF>>::None,
                    )
                } else {
                    (
                        Option::<Vec<EF>>::None,
                        Option::<Vec<EF>>::None,
                        Option::<Vec<EF>>::None,
                        Option::<Vec<EF>>::None,
                        Option::<Vec<EF>>::None,
                        Option::<Vec<EF>>::None,
                    )
                };
                let denom_shifted = (0..=2)
                    .map(|shift| {
                        if challenges.len() < shift + msg_len + 1 {
                            return None;
                        }
                        let betas = &challenges[shift..];
                        let mut d = betas[msg_len] * bus;
                        for (msg_eval, beta) in msg_evals.iter().zip(betas.iter()) {
                            d += *beta * *msg_eval;
                        }
                        Some(d)
                    })
                    .collect_vec();
                let main0_samples = partitioned_main_host.first().map(|m| {
                    [0usize, 1, 2, 3, height, 2 * height, 3 * height]
                        .into_iter()
                        .filter(|&i| i < m.len())
                        .map(|i| (i, m[i]))
                        .collect_vec()
                });
                panic!(
                    "gkr_input mismatch trace_idx={} air_idx={} interaction_idx={} row={} got={:?} expected={:?} msg_len={} bus={} msg_evals={:?} challenges_head={:?} denom_shifted={:?} main0_samples={:?} got_row0_p={:?} expected_row0_p={:?} got_row0_q={:?} expected_row0_q={:?} rules_row0_p={:?} rules_row0_q={:?} cpu_rule_results_row0={:?} cpu_rule_intermediates_row0={:?} swapped_rules_row0_p={:?} swapped_rules_row0_q={:?}",
                    trace_idx,
                    air_idx,
                    interaction_idx,
                    row,
                    got[idx],
                    expected,
                    msg_len,
                    bus,
                    msg_evals,
                    &challenges[..challenges.len().min(6)],
                    denom_shifted,
                    main0_samples,
                    got_row0_p,
                    expected_row0_p,
                    got_row0_q,
                    expected_row0_q,
                    rules_row0_p,
                    rules_row0_q,
                    cpu_rule_results_row0,
                    cpu_rule_intermediates_row0,
                    swapped_rules_row0_p,
                    swapped_rules_row0_q
                );
            }
        }
    }
}

pub fn collect_trace_interactions(
    pk: &DeviceMultiStarkProvingKey<MetalBackend>,
    ctx: &ProvingContext<MetalBackend>,
    layout: &StackedLayout,
) -> Vec<Option<TraceInteractionMeta>> {
    let mut slices_by_trace: Vec<Vec<(usize, StackedSlice)>> =
        vec![Vec::new(); ctx.per_trace.len()];
    for &(trace_idx, interaction_idx, ref slice) in &layout.sorted_cols {
        if let Some(entries) = slices_by_trace.get_mut(trace_idx) {
            entries.push((interaction_idx, *slice));
        }
    }

    ctx.per_trace
        .iter()
        .enumerate()
        .map(|(trace_idx, (air_idx, _))| {
            let vk = &pk.per_air[*air_idx].vk;
            if !vk.has_interaction() {
                return None;
            }

            let mut layout_entries = vec![None; vk.num_interactions()];
            for (interaction_idx, slice) in &slices_by_trace[trace_idx] {
                if let Some(slot) = layout_entries.get_mut(*interaction_idx) {
                    *slot = Some(*slice);
                }
            }

            let layout_slices = layout_entries
                .into_iter()
                .enumerate()
                .map(|(idx, maybe_slice)| {
                    maybe_slice.unwrap_or_else(|| {
                        panic!(
                            "missing stacked slice for interaction {} of trace {}",
                            idx, trace_idx
                        )
                    })
                })
                .collect_vec();

            Some(TraceInteractionMeta {
                trace_idx,
                air_idx: *air_idx,
                layout_slices,
            })
        })
        .collect()
}

/// Evaluate interactions from trace evaluation matrices to get (p, q) fractional sumcheck input.
#[instrument(name = "prover.rap_constraints.logup_gkr.input_evals", skip_all)]
#[allow(clippy::too_many_arguments)]
pub fn log_gkr_input_evals(
    trace_interactions: &[Option<TraceInteractionMeta>],
    pk: &DeviceMultiStarkProvingKey<MetalBackend>,
    ctx: &ProvingContext<MetalBackend>,
    l_skip: usize,
    alpha_logup: EF,
    d_challenges: &MetalBuffer<EF>,
    total_leaves: usize,
) -> Result<MetalBuffer<Frac<EF>>, InteractionMetalError> {
    if trace_interactions.iter().all(|meta| meta.is_none()) {
        return Ok(MetalBuffer::with_capacity(0));
    }

    let leaves = MetalBuffer::<Frac<EF>>::with_capacity(total_leaves);
    leaves.fill_zero();
    let null_preprocessed = MetalBuffer::<F>::with_capacity(0);

    let mut tmp = MetalBuffer::<Frac<EF>>::with_capacity(0);
    #[cfg(debug_assertions)]
    let challenges_host = d_challenges.to_vec();
    for meta in trace_interactions.iter().flatten() {
        let air_ctx = &ctx.per_trace[meta.trace_idx].1;
        let pk_air = &pk.per_air[meta.air_idx];

        let preprocessed_matrix = pk_air
            .preprocessed_data
            .as_ref()
            .map(|committed| &committed.trace);

        let mut partitioned_main = Vec::with_capacity(air_ctx.cached_mains.len() + 1);
        for committed in &air_ctx.cached_mains {
            partitioned_main.push(&committed.trace);
        }
        partitioned_main.push(&air_ctx.common_main);

        let rules = &pk_air.other_data.interaction_rules;
        let num_interactions = pk_air.vk.symbolic_constraints.interactions.len();
        let main_parts_buffers = partitioned_main.iter().map(|m| m.buffer()).collect_vec();

        let d_preprocessed = preprocessed_matrix
            .as_ref()
            .map(|m| m.buffer())
            .unwrap_or(&null_preprocessed);
        let d_public_values = if air_ctx.public_values.is_empty() {
            MetalBuffer::<F>::with_capacity(0)
        } else {
            air_ctx.public_values.to_device()
        };

        let height = air_ctx.height();
        debug_assert_eq!(height, partitioned_main[0].height());

        let buffer_size = rules.inner.buffer_size;
        let is_global = buffer_size > 10;
        let intermediates = if is_global {
            MetalBuffer::<EF>::with_capacity((TASK_SIZE as usize) * buffer_size as usize)
        } else {
            MetalBuffer::<EF>::with_capacity(1)
        };

        let num_rows_per_tile = height.div_ceil(TASK_SIZE as usize).max(1);

        let slice = meta.layout_slices.first().unwrap();
        if slice.col_idx != 0 {
            return Err(InteractionMetalError::Layout);
        }
        let dst_offset = slice.row_idx;
        let lifted_height = max(height, 1 << l_skip);
        debug_assert_eq!(slice.len(l_skip), lifted_height);
        if height != lifted_height {
            let required = height * num_interactions;
            if required > tmp.len() {
                tmp = MetalBuffer::with_capacity(required);
            }
            unsafe {
                logup_gkr_input_eval(
                    is_global,
                    &tmp,
                    0,
                    d_preprocessed,
                    &main_parts_buffers,
                    &d_public_values,
                    d_challenges,
                    &intermediates,
                    &rules.inner.d_rules,
                    &rules.inner.d_used_nodes,
                    &rules.d_pair_idxs,
                    height as u32,
                    num_rows_per_tile as u32,
                )?;
            }
        } else {
            unsafe {
                logup_gkr_input_eval(
                    is_global,
                    &leaves,
                    dst_offset,
                    d_preprocessed,
                    &main_parts_buffers,
                    &d_public_values,
                    d_challenges,
                    &intermediates,
                    &rules.inner.d_rules,
                    &rules.inner.d_used_nodes,
                    &rules.d_pair_idxs,
                    height as u32,
                    num_rows_per_tile as u32,
                )?;
            }
        }

        #[cfg(debug_assertions)]
        {
            let symbolic = SymbolicConstraints::from(&pk_air.vk.symbolic_constraints);
            let preprocessed_host = preprocessed_matrix
                .as_ref()
                .map(|matrix| matrix.buffer().to_vec());
            let partitioned_main_host = partitioned_main
                .iter()
                .map(|matrix| matrix.buffer().to_vec())
                .collect_vec();
            let got = if height != lifted_height {
                frac_segment_to_vec(&tmp, 0, height * num_interactions)
            } else {
                frac_segment_to_vec(&leaves, dst_offset, height * num_interactions)
            };
            debug_assert_trace_interactions_match_cpu(
                meta.trace_idx,
                meta.air_idx,
                &symbolic.interactions,
                &rules.inner.d_rules.to_vec(),
                &rules.inner.d_used_nodes.to_vec(),
                &rules.d_pair_idxs.to_vec(),
                rules.inner.buffer_size as usize,
                height,
                preprocessed_host.as_deref(),
                &partitioned_main_host,
                &air_ctx.public_values,
                &challenges_host,
                &got,
            );
        }

        if height != lifted_height {
            debug_assert_eq!(lifted_height % height, 0);
            debug_assert!(!tmp.is_empty());
            let norm_factor_denom = lifted_height / height;
            let norm_factor = F::from_usize(norm_factor_denom).inverse();
            unsafe {
                frac_vector_scalar_multiply_ext_fp(&tmp, 0, norm_factor, tmp.len() as u32)?;
                frac_matrix_vertically_repeat(
                    &leaves,
                    dst_offset,
                    &tmp,
                    0,
                    num_interactions as u32,
                    lifted_height as u32,
                    height as u32,
                )?;
            }
        }
    }

    if !leaves.is_empty() {
        unsafe {
            frac_add_alpha(&leaves, alpha_logup)?;
        }
    }

    Ok(leaves)
}
