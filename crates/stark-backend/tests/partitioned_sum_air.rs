//! Tests for AIR with partitioned main trace (cached + common).
//!
//! Defines a custom `SumAir` that exercises `PartitionedAirBuilder` with both
//! cached and common main traces, constraining x == sum(y_i). The AIR
//! definition and trace generation are local to this file and specific to this
//! test scenario, so they are not in the shared backend test suite.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

use itertools::Itertools;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder,
    prover::{stacked_pcs::stacked_commit, AirProvingContext, ColMajorMatrix, CommittedTraceData},
    utils::disable_debug_builder,
    PartitionedBaseAir, StarkEngine, StarkProtocolConfig,
};
use openvm_stark_sdk::{config::baby_bear_poseidon2::*, utils::setup_tracing};
use p3_air::{Air, BaseAir, BaseAirWithPublicValues};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// AIR with partitioned main trace: common_main has `x` (width 1), cached_main has `y` (width w).
/// Constrains x == y_0 + ... + y_{w-1}.
struct SumAir(usize);

impl<F> BaseAirWithPublicValues<F> for SumAir {}
impl<F> PartitionedBaseAir<F> for SumAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![self.0]
    }
    fn common_main_width(&self) -> usize {
        1
    }
}
impl<F> BaseAir<F> for SumAir {
    fn width(&self) -> usize {
        self.0 + 1
    }
}

impl<AB: PartitionedAirBuilder> Air<AB> for SumAir {
    fn eval(&self, builder: &mut AB) {
        assert_eq!(builder.cached_mains().len(), 1);

        let x = builder.common_main().row_slice(0).expect("common main row")[0].clone();
        let ys = builder.cached_mains()[0]
            .row_slice(0)
            .expect("cached main row");

        let mut y_sum = AB::Expr::ZERO;
        for y in ys.iter() {
            y_sum += y.clone();
        }
        drop(ys);

        builder.assert_eq(x, y_sum);
    }
}

fn prove_and_verify_sum_air(
    x: Vec<F>,
    ys: Vec<Vec<F>>,
) -> Result<(), openvm_stark_backend::verifier::VerifierError<EF>> {
    assert_eq!(x.len(), ys.len());

    let engine: BabyBearPoseidon2CpuEngine<DuplexSponge> =
        StarkEngine::new(openvm_stark_backend::test_utils::default_test_params_small());
    let config = engine.config();
    let params = config.params();

    let y_width = ys[0].len();
    let air = Arc::new(SumAir(y_width));

    let x_trace = ColMajorMatrix::from_row_major(&RowMajorMatrix::new(x, 1));
    let y_rm = RowMajorMatrix::new(ys.into_iter().flatten().collect_vec(), y_width);
    let y_trace = ColMajorMatrix::from_row_major(&y_rm);

    let (commit, data) = stacked_commit(
        config.hasher(),
        params.l_skip,
        params.n_stack,
        params.log_blowup,
        params.k_whir(),
        &[&y_trace],
    )
    .unwrap();

    let trace_ctx = AirProvingContext {
        cached_mains: vec![CommittedTraceData {
            commitment: commit,
            trace: y_trace,
            data: Arc::new(data),
        }],
        common_main: x_trace,
        public_values: vec![],
    };

    engine.run_test(vec![air], vec![trace_ctx]).map(|_| ())
}

fn generate_random_matrix(mut rng: impl Rng, height: usize, width: usize) -> Vec<Vec<F>> {
    (0..height)
        .map(|_| {
            (0..width)
                .map(|_| F::from_u32(rng.random_range(0..1000)))
                .collect()
        })
        .collect()
}

#[test]
fn test_partitioned_sum_air_happy_path() {
    setup_tracing();
    let rng = StdRng::seed_from_u64(0);
    let n = 1 << 3;
    let ys = generate_random_matrix(rng, n, 5);
    let x: Vec<F> = ys
        .iter()
        .map(|row| row.iter().fold(F::ZERO, |sum, x| sum + *x))
        .collect();
    prove_and_verify_sum_air(x, ys).expect("Verification failed");
}

#[test]
fn test_partitioned_sum_air_neg() {
    setup_tracing();
    let rng = StdRng::seed_from_u64(0);
    let n = 1 << 3;
    let ys = generate_random_matrix(rng, n, 5);
    let mut x: Vec<F> = ys
        .iter()
        .map(|row| row.iter().fold(F::ZERO, |sum, x| sum + *x))
        .collect();
    x[0] = F::ZERO;
    disable_debug_builder();
    let result = catch_unwind(AssertUnwindSafe(|| prove_and_verify_sum_air(x, ys)));
    assert!(result.is_err() || result.unwrap().is_err());
}
