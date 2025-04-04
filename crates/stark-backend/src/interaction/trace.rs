use crate::air_builders::symbolic::{
    symbolic_expression::SymbolicEvaluator,
    symbolic_variable::{Entry, SymbolicVariable},
};
use p3_field::Field;
use p3_matrix::Matrix;

pub(super) struct Evaluator<'a, T, F: Field> {
    pub preprocessed: Option<T>,
    pub partitioned_main: &'a [T],
    pub public_values: &'a [F],
    pub height: usize,
    pub local_index: usize,
}

impl<T: Matrix<F>, F: Field> SymbolicEvaluator<F, F> for Evaluator<'_, T, F> {
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
                .as_ref()
                .unwrap()
                .get((n + offset) % height, index),
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
