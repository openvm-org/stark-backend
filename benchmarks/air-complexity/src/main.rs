//! Benchmark for measuring proving time vs AIR complexity.
//!
//! Creates configurable synthetic AIRs with:
//! - Zero witness (all trace cells are 0)
//! - Boolean constraints: `x * (x - 1) = 0`
//! - Self-canceling bus interactions: `bus_send([x])` + `bus_receive([x])`
//!
//! Usage (CPU):
//!   cargo run -p openvm-benchmark-air-complexity --release -- \
//!     --num-airs 4 --cols-per-air 100 --constraints-per-col 2 --log-rows-per-air 18
//!
//! Usage (GPU):
//!   cargo run -p openvm-benchmark-air-complexity --release --features cuda -- \
//!     --num-airs 4 --cols-per-air 100 --constraints-per-col 2 --log-rows-per-air 18
//!
//! To also write a metrics.json file:
//!   METRICS_OUTPUT=metrics.json cargo run -p openvm-benchmark-air-complexity --release -- ...

use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    keygen::types::MultiStarkProvingKey,
    proof::Proof,
    prover::{AirProvingContext, ColMajorMatrix, DeviceDataTransporter, ProvingContext},
    AirRef, PartitionedBaseAir, StarkEngine,
};
use openvm_stark_sdk::config::{
    app_params_with_100_bits_security,
    baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2RefEngine},
};
use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues};
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;

type F = BabyBear;
type SC = BabyBearPoseidon2Config;

#[derive(Parser)]
#[command(about = "Benchmark proving time vs AIR complexity")]
struct Args {
    /// Number of AIRs
    #[arg(long, default_value_t = 1)]
    num_airs: usize,

    /// Number of columns per AIR
    #[arg(long, default_value_t = 20)]
    cols_per_air: usize,

    /// Boolean constraints per column (can be fractional, e.g. 0.5 means every
    /// other column gets a constraint)
    #[arg(long, default_value_t = 1.0)]
    constraints_per_col: f64,

    /// Send/receive bus interaction pairs per column (can be fractional, e.g.
    /// 0.25 means a pair on every 8th column). Each pair creates one send and
    /// one receive with the same message, so they cancel out.
    #[arg(long, default_value_t = 0.25)]
    interactions_per_col: f64,

    /// Log2 of the number of rows per AIR.
    #[arg(long, default_value_t = 18)]
    log_rows_per_air: usize,

    /// Log2 of stacked height for the PCS. Default is 24
    /// (MAX_APP_LOG_STACKED_HEIGHT), matching default_app_config() in the
    /// openvm cli crate.
    #[arg(long, default_value_t = 24)]
    log_stacked_height: usize,
}

// ---------------------------------------------------------------------------
// BenchmarkAir
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct BenchmarkAir {
    pub num_columns: usize,
    /// Total boolean constraints in this AIR (spread across columns round-robin).
    pub total_constraints: usize,
    /// Total send+receive pairs in this AIR (spread across columns round-robin).
    pub total_interaction_pairs: usize,
}

impl<F> BaseAir<F> for BenchmarkAir {
    fn width(&self) -> usize {
        self.num_columns
    }
}
impl<F> BaseAirWithPublicValues<F> for BenchmarkAir {}
impl<F> PartitionedBaseAir<F> for BenchmarkAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for BenchmarkAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).unwrap();

        // Boolean constraints spread round-robin across columns
        for i in 0..self.total_constraints {
            builder.assert_bool(local[i % self.num_columns]);
        }

        // Self-canceling bus interaction pairs spread round-robin across columns
        for i in 0..self.total_interaction_pairs {
            let field = vec![local[i % self.num_columns]];
            builder.push_interaction(0, field.clone(), AB::Expr::ONE, 0);
            builder.push_interaction(0, field, AB::Expr::NEG_ONE, 0);
        }
    }
}

// ---------------------------------------------------------------------------
// Backend-specific proving
// ---------------------------------------------------------------------------

#[cfg(not(feature = "cuda"))]
fn prove(
    params: &openvm_stark_backend::SystemParams,
    pk: &MultiStarkProvingKey<SC>,
    cpu_traces: Vec<ColMajorMatrix<F>>,
) -> Proof<SC> {
    let engine: BabyBearPoseidon2RefEngine = StarkEngine::new(params.clone());
    let d_pk = engine.device().transport_pk_to_device(pk);
    let ctx = ProvingContext::new(
        cpu_traces
            .into_iter()
            .enumerate()
            .map(|(i, trace)| (i, AirProvingContext::simple_no_pis(trace)))
            .collect(),
    );
    engine.prove(&d_pk, ctx).unwrap()
}

#[cfg(feature = "cuda")]
fn prove(
    params: &openvm_stark_backend::SystemParams,
    pk: &MultiStarkProvingKey<SC>,
    cpu_traces: Vec<ColMajorMatrix<F>>,
) -> Proof<SC> {
    use openvm_cuda_backend::{prelude::SC as CudaSC, BabyBearPoseidon2GpuEngine, GpuBackend};

    let engine = BabyBearPoseidon2GpuEngine::new(params.clone());
    let device = engine.device();
    let d_pk =
        <_ as DeviceDataTransporter<CudaSC, GpuBackend>>::transport_pk_to_device(device, pk);
    let ctx = ProvingContext::new(
        cpu_traces
            .iter()
            .enumerate()
            .map(|(i, trace)| {
                let d_trace =
                    <_ as DeviceDataTransporter<CudaSC, GpuBackend>>::transport_matrix_to_device(
                        device, trace,
                    );
                (i, AirProvingContext::simple_no_pis(d_trace))
            })
            .collect(),
    );
    engine.prove(&d_pk, ctx).unwrap()
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();

    // run_with_metric_collection sets up tracing + metrics recording.
    // If METRICS_OUTPUT is set, writes a metrics.json on completion.
    openvm_stark_sdk::bench::run_with_metric_collection("METRICS_OUTPUT", || run(&args));
}

fn run(args: &Args) {
    assert!(args.num_airs > 0);
    assert!(args.cols_per_air > 0);

    let trace_height = 1usize << args.log_rows_per_air;
    let total_cells = args.num_airs * args.cols_per_air * trace_height;

    // Compute actual integer counts per AIR from fractional rates
    let constraints_per_air =
        (args.constraints_per_col * args.cols_per_air as f64).round() as usize;
    let interaction_pairs_per_air =
        (args.interactions_per_col * args.cols_per_air as f64 / 2.0).round() as usize;

    let total_constraints = constraints_per_air * args.num_airs;
    let total_bus_interactions = interaction_pairs_per_air * 2 * args.num_airs;
    let constraint_instances = total_constraints * trace_height;
    let bus_interaction_messages = total_bus_interactions * trace_height;

    let backend_name = if cfg!(feature = "cuda") {
        "GPU (CUDA)"
    } else {
        "CPU"
    };

    println!("=== AIR Complexity Benchmark ({backend_name}) ===");
    println!("  num_airs:               {}", args.num_airs);
    println!("  cols_per_air:           {}", args.cols_per_air);
    println!("  constraints_per_col:    {}", args.constraints_per_col);
    println!("  interactions_per_col:   {}", args.interactions_per_col);
    println!("  trace_height:           {trace_height} (2^{})", args.log_rows_per_air);
    println!(
        "  trace_cells:            {total_cells} (2^{} rows * {} AIRs * {} columns / AIR)",
        args.log_rows_per_air, args.num_airs, args.cols_per_air
    );
    println!("  constraints:            {total_constraints}");
    println!("  bus_interactions:       {total_bus_interactions}");
    println!("  constraint_instances:   {constraint_instances}");
    println!("  bus_interaction_msgs:   {bus_interaction_messages}");

    // Create AIRs
    let airs: Vec<AirRef<SC>> = (0..args.num_airs)
        .map(|_| {
            Arc::new(BenchmarkAir {
                num_columns: args.cols_per_air,
                total_constraints: constraints_per_air,
                total_interaction_pairs: interaction_pairs_per_air,
            }) as AirRef<SC>
        })
        .collect();

    let params = app_params_with_100_bits_security(args.log_stacked_height);

    // Keygen (always CPU)
    let keygen_engine: BabyBearPoseidon2RefEngine = StarkEngine::new(params.clone());
    println!("\nKeygen...");
    let start = Instant::now();
    let (pk, vk) = keygen_engine.keygen(&airs);
    println!("  time: {:?}", start.elapsed());

    // Generate zero traces on CPU
    let cpu_traces: Vec<ColMajorMatrix<F>> = (0..args.num_airs)
        .map(|_| {
            ColMajorMatrix::new(
                vec![F::ZERO; trace_height * args.cols_per_air],
                args.cols_per_air,
            )
        })
        .collect();

    // Prove (backend-specific)
    println!("Proving...");
    let start = Instant::now();
    let proof = prove(&params, &pk, cpu_traces);
    let prove_time = start.elapsed();
    println!("  time: {prove_time:?}");

    // Verify (always CPU)
    println!("Verifying...");
    let start = Instant::now();
    keygen_engine.verify(&vk, &proof).unwrap();
    println!("  time: {:?}", start.elapsed());
}
