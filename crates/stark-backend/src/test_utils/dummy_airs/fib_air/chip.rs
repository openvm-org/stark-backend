use std::sync::Arc;

use p3_matrix::Matrix;

use super::trace::generate_trace_rows;
use crate::{
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend},
    test_utils::dummy_airs::fib_air::air::FibonacciAir,
    AirRef, Chip, StarkProtocolConfig,
};

#[derive(Clone, Debug)]
pub struct FibonacciChip {
    /// The 0th number in the fibonacci sequence.
    pub a: u64,
    /// The 1st number in the fibonacci sequence.
    pub b: u64,
    /// Target n-th number in the fibonacci sequence.
    pub n: usize,
}

impl FibonacciChip {
    pub fn new(a: u64, b: u64, n: usize) -> Self {
        assert!(n.is_power_of_two());
        Self { a, b, n }
    }

    pub fn air<SC: StarkProtocolConfig>(&self) -> AirRef<SC> {
        Arc::new(FibonacciAir)
    }
}

impl<SC: StarkProtocolConfig> Chip<(), CpuBackend<SC>> for FibonacciChip {
    fn generate_proving_ctx(&self, _: ()) -> AirProvingContext<CpuBackend<SC>> {
        let common_main = generate_trace_rows::<SC::F>(self.a, self.b, self.n);
        let a = common_main.get(0, 0).expect("matrix index out of bounds");
        let b = common_main.get(0, 1).expect("matrix index out of bounds");
        let last_val = common_main
            .get(self.n - 1, 1)
            .expect("matrix index out of bounds");
        AirProvingContext::simple(
            ColMajorMatrix::from_row_major(&common_main),
            vec![a, b, last_val],
        )
    }
}
