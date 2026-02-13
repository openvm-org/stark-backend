use std::collections::{BTreeMap, HashMap};

use itertools::Itertools;
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrixView, Matrix};

use super::{BusIndex, SymbolicInteraction};
use crate::air_builders::symbolic::{
    symbolic_expression::SymbolicEvaluator,
    symbolic_variable::{Entry, SymbolicVariable},
};

/// The actual interactions that are sent/received during a single run
/// of trace generation. For debugging purposes only.
#[derive(Default, Clone, Debug)]
pub struct LogicalInteractions<F: Field> {
    /// Bus index => (fields => (air_idx, count))
    #[allow(clippy::type_complexity)]
    pub at_bus: BTreeMap<BusIndex, HashMap<Vec<F>, Vec<(usize, F)>>>,
}

pub fn generate_logical_interactions<F: Field>(
    air_idx: usize,
    all_interactions: &[SymbolicInteraction<F>],
    preprocessed: &Option<RowMajorMatrixView<F>>,
    partitioned_main: &[RowMajorMatrixView<F>],
    public_values: &[F],
    logical_interactions: &mut LogicalInteractions<F>,
) {
    if all_interactions.is_empty() {
        return;
    }

    let height = partitioned_main[0].height();

    for n in 0..height {
        let evaluator = Evaluator {
            preprocessed,
            partitioned_main,
            public_values,
            height,
            local_index: n,
        };
        for interaction in all_interactions {
            let fields = interaction
                .message
                .iter()
                .map(|expr| evaluator.eval_expr(expr))
                .collect_vec();
            let count = evaluator.eval_expr(&interaction.count);
            if count.is_zero() {
                continue;
            }
            logical_interactions
                .at_bus
                .entry(interaction.bus_index)
                .or_default()
                .entry(fields)
                .or_default()
                .push((air_idx, count));
        }
    }
}

struct Evaluator<'a, F: Field> {
    pub preprocessed: &'a Option<RowMajorMatrixView<'a, F>>,
    pub partitioned_main: &'a [RowMajorMatrixView<'a, F>],
    pub public_values: &'a [F],
    pub height: usize,
    pub local_index: usize,
}

impl<F: Field> SymbolicEvaluator<F, F> for Evaluator<'_, F> {
    fn eval_const(&self, c: F) -> F {
        c
    }
    fn eval_var(&self, symbolic_var: SymbolicVariable<F>) -> F {
        let n = self.local_index;
        let height = self.height;
        let index = symbolic_var.index;
        match symbolic_var.entry {
            Entry::Preprocessed { offset } => self
                .preprocessed
                .unwrap()
                .get((n + offset) % height, index)
                .expect("matrix index out of bounds"),
            Entry::Main { part_index, offset } => self.partitioned_main[part_index]
                .get((n + offset) % height, index)
                .expect("matrix index out of bounds"),
            Entry::Public => self.public_values[index],
            _ => unreachable!("There should be no after challenge variables"),
        }
    }
    fn eval_is_first_row(&self) -> F {
        unreachable!()
    }
    fn eval_is_last_row(&self) -> F {
        unreachable!()
    }
    fn eval_is_transition(&self) -> F {
        unreachable!()
    }
}
