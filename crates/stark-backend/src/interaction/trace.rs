use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrixView, Matrix};

use crate::{
    air_builders::symbolic::{
        symbolic_expression::SymbolicEvaluator,
        symbolic_variable::{Entry, SymbolicVariable},
    },
    prover::PairTraceView,
};

pub(super) struct Evaluator<'a, F: Field> {
    pub preprocessed: &'a Option<RowMajorMatrixView<'a, F>>,
    pub partitioned_main: &'a [RowMajorMatrixView<'a, F>],
    pub public_values: &'a [F],
    pub height: usize,
    pub local_index: usize,
}

impl<'a, F: Field> Evaluator<'a, F> {
    pub fn for_local_index(trace_view: &'a PairTraceView<'a, F>, local_index: usize) -> Self {
        Self {
            preprocessed: trace_view.preprocessed,
            partitioned_main: trace_view.partitioned_main,
            public_values: trace_view.public_values,
            height: trace_view.height(),
            local_index,
        }
    }
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
            Entry::Preprocessed { offset } => {
                self.preprocessed.unwrap().get((n + offset) % height, index)
            }
            Entry::Main { part_index, offset } => {
                self.partitioned_main[part_index].get((n + offset) % height, index)
            }
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
