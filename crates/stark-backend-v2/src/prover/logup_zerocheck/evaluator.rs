use openvm_stark_backend::air_builders::symbolic::{
    symbolic_expression::SymbolicEvaluator,
    symbolic_variable::{Entry, SymbolicVariable},
};
use p3_field::{ExtensionField, Field};

pub(super) struct ViewPair<T> {
    pub(super) local: *const T,
    pub(super) next: Option<*const T>,
}

impl<T> ViewPair<T> {
    pub fn new(local: &[T], next: Option<&[T]>) -> Self {
        Self {
            local: local.as_ptr(),
            next: next.map(|nxt| nxt.as_ptr()),
        }
    }

    /// SAFETY: no matrix bounds checks are done.
    pub unsafe fn get(&self, row_offset: usize, column_idx: usize) -> &T {
        match row_offset {
            0 => &*self.local.add(column_idx),
            1 => &*self.next.unwrap_unchecked().add(column_idx),
            _ => panic!("row offset {row_offset} not supported"),
        }
    }
}

/// Struct containing partitioned view of one row together with optional "rotated" row.
/// Constraints are evaluated on this struct.
pub(super) struct ProverConstraintEvaluator<'a, F, EF> {
    pub preprocessed: Option<ViewPair<EF>>,
    pub partitioned_main: Vec<ViewPair<EF>>,
    pub is_first_row: EF,
    pub is_last_row: EF,
    pub is_transition: EF,
    pub public_values: &'a [F],
}

impl<F: Field, EF: ExtensionField<F>> SymbolicEvaluator<F, EF>
    for ProverConstraintEvaluator<'_, F, EF>
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

    /// SAFETY: we only use this trait implementation when we have already done
    /// a previous scan to ensure all matrix bounds are satisfied,
    /// so no bounds checks are done here.
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
