use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};

use crate::air_builders::symbolic::{
    symbolic_expression::SymbolicEvaluator,
    symbolic_variable::{Entry, SymbolicVariable},
};

type ViewPair<'a, T> = &'a [(T, T)];

/// Returns the sum `1 + m + ... + m^{2^l - 1}`.
/// Could be done with `if m == 1 { ... } else { num / denom }`,
/// but I don't like divisions of field extension elements.
fn progression_exp_2<EF>(m: EF, l: usize) -> EF
where
    EF: PrimeCharacteristicRing + Copy,
{
    (0..l)
        .fold((m, EF::ONE), |(pow, sum), _| {
            (pow * pow, sum * (EF::ONE + pow))
        })
        .1
}

pub(super) struct VerifierConstraintEvaluator<'a, F, EF> {
    pub preprocessed: Option<ViewPair<'a, EF>>,
    pub partitioned_main: &'a [ViewPair<'a, EF>],
    pub is_first_row: EF,
    pub is_last_row: EF,
    pub is_transition: EF,
    pub public_values: &'a [F],
}

impl<'a, F, EF> VerifierConstraintEvaluator<'a, F, EF>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F>,
{
    pub(super) fn new(
        preprocessed: Option<ViewPair<'a, EF>>,
        partitioned_main: &'a [ViewPair<'a, EF>],
        public_values: &'a [F],
        rs: &'a [EF],
        l_skip: usize,
    ) -> Self {
        let omega = F::two_adic_generator(l_skip);
        let inv = EF::from(F::from_usize(1 << l_skip).inverse());
        let is_first_row = inv
            * progression_exp_2(rs[0], l_skip)
            * rs[1..].iter().fold(EF::ONE, |acc, &x| acc * (EF::ONE - x));
        let is_last_row = inv
            * progression_exp_2(rs[0] * omega, l_skip)
            * rs[1..].iter().fold(EF::ONE, |acc, &x| acc * x);
        let is_transition = if l_skip > 0 {
            let omega_z = EF::from(omega) * rs[0];
            let eq_x = rs[1..].iter().fold(EF::ONE, |acc, &x| acc * x);
            omega_z - eq_x
        } else {
            EF::ZERO
        };
        Self {
            preprocessed,
            partitioned_main,
            is_first_row,
            is_last_row,
            is_transition,
            public_values,
        }
    }
}

impl<F, EF> SymbolicEvaluator<F, EF> for VerifierConstraintEvaluator<'_, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn eval_const(&self, c: F) -> EF {
        EF::from(c)
    }

    fn eval_var(&self, symbolic_var: SymbolicVariable<F>) -> EF {
        let index = symbolic_var.index;
        match symbolic_var.entry {
            Entry::Preprocessed { offset } => match &self.preprocessed {
                Some(vp) => {
                    if offset == 0 {
                        vp[index].0
                    } else {
                        vp[index].1
                    }
                }
                None => panic!(),
            },
            Entry::Main { part_index, offset } => {
                let vp = &self.partitioned_main[part_index];
                if offset == 0 {
                    vp[index].0
                } else {
                    vp[index].1
                }
            }
            Entry::Public => EF::from(self.public_values[index]),
            _ => unimplemented!(),
        }
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
}
