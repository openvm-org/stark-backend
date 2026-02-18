use itertools::{fold, Itertools};
use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    interaction::{BusIndex, InteractionBuilder},
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend},
    PartitionedBaseAir, StarkProtocolConfig,
};

#[derive(Debug, Clone, Copy)]
pub struct SelfInteractionAir {
    pub width: usize,
    pub bus_index: BusIndex,
}

impl<F> BaseAir<F> for SelfInteractionAir {
    fn width(&self) -> usize {
        self.width
    }
}
impl<F> BaseAirWithPublicValues<F> for SelfInteractionAir {}
impl<F> PartitionedBaseAir<F> for SelfInteractionAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for SelfInteractionAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let mut local: Vec<<AB as AirBuilder>::Expr> =
            (*local).iter().map(|v| (*v).into()).collect_vec();
        let mut next: Vec<<AB as AirBuilder>::Expr> =
            (*next).iter().map(|v| (*v).into()).collect_vec();

        let local_sum = fold(&local, AB::Expr::ZERO, |acc, val| acc + val.clone());
        let next_sum = fold(&local, AB::Expr::ZERO, |acc, val| acc + val.clone());

        // Interaction where count is constant
        builder.push_interaction(self.bus_index, local.clone(), AB::Expr::ONE, 1);
        builder.push_interaction(self.bus_index, next.clone(), AB::Expr::NEG_ONE, 1);

        // Interaction where count is an expression + common with interaction below
        builder.push_interaction(self.bus_index, local.clone(), local_sum.clone(), 1);
        builder.push_interaction(self.bus_index, next.clone(), -next_sum.clone(), 1);

        // Interaction where count == fields[0]
        builder.push_interaction(self.bus_index, local.clone(), local[0].clone(), 1);
        builder.push_interaction(self.bus_index, next.clone(), -next[0].clone(), 1);

        local.reverse();
        next.reverse();

        // Interaction where count is an expression + common with interaction above
        builder.push_interaction(self.bus_index, local, local_sum, 1);
        builder.push_interaction(self.bus_index, next, -next_sum, 1);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SelfInteractionChip {
    pub width: usize,
    pub log_height: usize,
}

impl SelfInteractionChip {
    pub fn generate_proving_ctx<SC: StarkProtocolConfig>(
        &self,
    ) -> AirProvingContext<CpuBackend<SC>> {
        assert!(self.width > 0);
        let mut trace = vec![SC::F::ZERO; (1 << self.log_height) * self.width];
        for (row_idx, chunk) in trace.chunks_mut(self.width).enumerate() {
            for (i, val) in chunk.iter_mut().enumerate() {
                *val = SC::F::from_usize((row_idx + i) % self.width);
            }
        }
        let rm = RowMajorMatrix::new(trace, self.width);
        AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&rm))
    }
}
