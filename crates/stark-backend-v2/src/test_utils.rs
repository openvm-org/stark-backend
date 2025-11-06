use std::sync::Arc;

use openvm_stark_backend::{AirRef, prover::Prover};
pub use openvm_stark_sdk::dummy_airs::fib_air::air::FibonacciAir;
use openvm_stark_sdk::{
    any_rap_arc_vec,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Config, setup_tracing},
    dummy_airs::{
        self, fib_selector_air::air::FibonacciSelectorAir,
        interaction::dummy_interaction_air::DummyInteractionAir,
    },
};
use p3_baby_bear::BabyBear;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    BabyBearPoseidon2CpuEngineV2, Digest, F, StarkEngineV2,
    keygen::types::{MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2, SystemParams},
    poseidon2::sponge::{
        DuplexSponge, DuplexSpongeRecorder, FiatShamirTranscript, TranscriptHistory, TranscriptLog,
    },
    proof::Proof,
    prover::{
        AirProvingContextV2, ColMajorMatrix, CpuBackendV2, DeviceDataTransporterV2,
        ProverBackendV2, ProvingContextV2, stacked_pcs::stacked_commit,
    },
};

pub fn transport_proving_ctx_to_device<PB, PD>(
    device: &PD,
    ctx: &ProvingContextV2<CpuBackendV2>,
) -> ProvingContextV2<PB>
where
    PB: ProverBackendV2<Val = F, Commitment = Digest>,
    PD: DeviceDataTransporterV2<PB>,
{
    let per_trace = ctx
        .per_trace
        .iter()
        .map(|(air_idx, air_ctx)| {
            let common_main = device.transport_matrix_to_device(&air_ctx.common_main);
            let cached_mains = air_ctx
                .cached_mains
                .iter()
                .map(|(commit, data)| {
                    (
                        *commit,
                        Arc::new(device.transport_pcs_data_to_device(data.as_ref())),
                    )
                })
                .collect();
            let air_ctx_gpu =
                AirProvingContextV2::new(cached_mains, common_main, air_ctx.public_values.clone());
            (*air_idx, air_ctx_gpu)
        })
        .collect();
    ProvingContextV2::new(per_trace)
}

fn get_fib_number(mut a: u32, mut b: u32, n: usize) -> u32 {
    for _ in 0..n - 1 {
        let c = (a + b) % BabyBear::ORDER_U32;
        a = b;
        b = c;
    }
    b
}

fn get_conditional_fib_number(mut a: u32, mut b: u32, sels: &[bool]) -> u32 {
    for &s in sels[0..sels.len() - 1].iter() {
        if s {
            let c = (a + b) % BabyBear::ORDER_U32;
            a = b;
            b = c;
        }
    }
    b
}

#[derive(derive_new::new)]
pub struct FibFixture {
    pub a: u32,
    pub b: u32,
    pub n: usize,
}

/// Trait for object responsible for generating the collection of AIRs and trace matrices for a
/// single test case.
pub trait TestFixture {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>>;

    fn generate_proving_ctx(&self) -> ProvingContextV2<CpuBackendV2>;

    fn keygen<E: StarkEngineV2>(
        &self,
        engine: &E,
    ) -> (MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2) {
        engine.keygen(&self.airs())
    }

    fn prove<E: StarkEngineV2>(&self, engine: &E, pk: &MultiStarkProvingKeyV2) -> Proof {
        self.prove_from_transcript(engine, pk, &mut E::TS::default())
    }

    /// Prove using CPU tracegen and transport to device.
    fn prove_from_transcript<E: StarkEngineV2>(
        &self,
        engine: &E,
        pk: &MultiStarkProvingKeyV2,
        transcript: &mut E::TS,
    ) -> Proof {
        let ctx = self.generate_proving_ctx();
        let device = engine.device();
        let d_pk = device.transport_pk_to_device(pk);
        let d_ctx = transport_proving_ctx_to_device(device, &ctx);
        let mut prover = engine.prover_from_transcript(transcript.clone());
        let proof = prover.prove(&d_pk, d_ctx);
        *transcript = prover.transcript;
        proof
    }

    fn keygen_and_prove<E: StarkEngineV2>(&self, engine: &E) -> (MultiStarkVerifyingKeyV2, Proof) {
        let (pk, vk) = self.keygen(engine);
        let proof = self.prove(engine, &pk);
        (vk, proof)
    }
}

impl TestFixture for FibFixture {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        vec![Arc::new(FibonacciAir)]
    }

    fn generate_proving_ctx(&self) -> ProvingContextV2<CpuBackendV2> {
        use dummy_airs::fib_air::trace::generate_trace_rows;
        let trace = generate_trace_rows(self.a, self.b, self.n);
        let f_n = get_fib_number(self.a, self.b, self.n);
        let pis = [self.a, self.b, f_n].map(BabyBear::from_canonical_u32);
        let trace = ColMajorMatrix::from_row_major(&trace);

        let single_ctx = AirProvingContextV2::simple(trace, pis.to_vec());
        ProvingContextV2::new(vec![(0, single_ctx)])
    }
}

/// Interactions fixture with 1 sender and 1 receiver
pub struct InteractionsFixture11;

impl TestFixture for InteractionsFixture11 {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let sender_air = DummyInteractionAir::new(1, true, 0);
        let receiver_air = DummyInteractionAir::new(1, false, 0);
        any_rap_arc_vec!(sender_air, receiver_air)
    }

    fn generate_proving_ctx(&self) -> ProvingContextV2<CpuBackendV2> {
        // Default traces
        // Sender (2 columns: Mul, Val):
        //   0    1
        //   7    4
        //   3    5
        // 546  889
        let sender_trace = RowMajorMatrix::new(
            [0, 1, 3, 5, 7, 4, 546, 889]
                .into_iter()
                .map(BabyBear::from_canonical_usize)
                .collect(),
            2,
        );

        // Receiver (2 columns: Mul, Val):
        //   1    5
        //   3    4
        //   4    4
        //   2    5
        //   0  123
        // 545  889
        //   1  889
        //   0  456
        let receiver_trace = RowMajorMatrix::new(
            [1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 889, 1, 889, 0, 456]
                .into_iter()
                .map(BabyBear::from_canonical_usize)
                .collect(),
            2,
        );

        ProvingContextV2::new(
            [sender_trace, receiver_trace]
                .into_iter()
                .enumerate()
                .map(|(air_idx, trace)| {
                    (
                        air_idx,
                        AirProvingContextV2::simple_no_pis(ColMajorMatrix::from_row_major(&trace)),
                    )
                })
                .collect(),
        )
    }
}

/// Dummy interaction AIRs with cached trace: 1 sender, 1 receiver
#[derive(derive_new::new)]
pub struct CachedFixture11 {
    pub params: SystemParams,
}

impl TestFixture for CachedFixture11 {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let sender_air = DummyInteractionAir::new(1, true, 0).partition();
        let receiver_air = DummyInteractionAir::new(1, false, 0).partition();
        any_rap_arc_vec!(sender_air, receiver_air)
    }

    fn generate_proving_ctx(&self) -> ProvingContextV2<CpuBackendV2> {
        // Default traces
        // Sender (2 columns: Mul, Val):
        //   0    1
        //   3    5
        //   7    4
        // 546  889
        let sender_trace = ColMajorMatrix::new(
            [0, 3, 7, 546]
                .into_iter()
                .map(BabyBear::from_canonical_usize)
                .collect(),
            1,
        );
        let sender_cached_trace = ColMajorMatrix::new(
            [1, 5, 4, 889]
                .into_iter()
                .map(BabyBear::from_canonical_usize)
                .collect(),
            1,
        );

        // Receiver (2 columns: Mul, Val):
        //   1    5
        //   3    4
        //   4    4
        //   2    5
        //   0  123
        // 545  889
        //   1  889
        //   0  456
        let receiver_trace = ColMajorMatrix::new(
            [1, 3, 4, 2, 0, 545, 1, 0]
                .into_iter()
                .map(BabyBear::from_canonical_usize)
                .collect(),
            1,
        );
        let receiver_cached_trace = ColMajorMatrix::new(
            [5, 4, 4, 5, 123, 889, 889, 456]
                .into_iter()
                .map(BabyBear::from_canonical_usize)
                .collect(),
            1,
        );

        let params = self.params;
        ProvingContextV2::new(
            [
                (sender_trace, sender_cached_trace),
                (receiver_trace, receiver_cached_trace),
            ]
            .map(|(common, cached)| {
                let (commit, data) = stacked_commit(
                    params.l_skip,
                    params.n_stack,
                    params.log_blowup,
                    params.k_whir,
                    &[&cached],
                );
                AirProvingContextV2 {
                    cached_mains: vec![(commit, Arc::new(data))],
                    common_main: common,
                    public_values: vec![],
                }
            })
            .into_iter()
            .enumerate()
            .collect(),
        )
    }
}

#[derive(derive_new::new)]
pub struct PreprocessedFibFixture {
    pub a: u32,
    pub b: u32,
    pub sels: Vec<bool>,
}

impl TestFixture for PreprocessedFibFixture {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let air = Arc::new(FibonacciSelectorAir::new(self.sels.clone(), false));
        vec![air]
    }

    fn generate_proving_ctx(&self) -> ProvingContextV2<CpuBackendV2> {
        use openvm_stark_sdk::dummy_airs::fib_selector_air::trace::generate_trace_rows;
        let trace = generate_trace_rows(self.a, self.b, &self.sels);
        let f_n = get_conditional_fib_number(self.a, self.b, &self.sels);
        let pis = [self.a, self.b, f_n].map(BabyBear::from_canonical_u32);

        let single_ctx =
            AirProvingContextV2::simple(ColMajorMatrix::from_row_major(&trace), pis.to_vec());
        ProvingContextV2::new(vec![(0, single_ctx)])
    }
}

/// Trace heights cannot exceed 2^{l_skip + n_stack} when using these system params.
pub fn test_system_params_small(l_skip: usize, n_stack: usize, k_whir: usize) -> SystemParams {
    let log_final_poly_len = (n_stack + l_skip) % k_whir;
    test_system_params_small_with_poly_len(l_skip, n_stack, k_whir, log_final_poly_len)
}

pub fn test_system_params_small_with_poly_len(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
    log_final_poly_len: usize,
) -> SystemParams {
    assert!(log_final_poly_len < l_skip + n_stack);
    // Use all different numbers
    SystemParams {
        l_skip,
        n_stack,
        log_blowup: 1,
        k_whir,
        num_whir_queries: 5,
        log_final_poly_len,
        logup_pow_bits: 1,
        whir_pow_bits: 1,
    }
}

pub fn default_test_params_small() -> SystemParams {
    test_system_params_small(2, 8, 3)
}

pub fn test_engine_small() -> BabyBearPoseidon2CpuEngineV2<DuplexSponge> {
    setup_tracing();
    BabyBearPoseidon2CpuEngineV2::new(default_test_params_small())
}

#[derive(Clone)]
pub struct DuplexSpongeValidator {
    pub inner: DuplexSpongeRecorder,
    pub idx: usize,
    log: TranscriptLog,
}

impl DuplexSpongeValidator {
    pub fn new(log: TranscriptLog) -> Self {
        debug_assert_eq!(log.len(), log.samples().len());
        Self {
            inner: Default::default(),
            idx: 0,
            log,
        }
    }
}

impl FiatShamirTranscript for DuplexSpongeValidator {
    fn observe(&mut self, x: F) {
        debug_assert!(self.idx < self.log.len(), "transcript replay overflow");
        assert!(!self.log.samples()[self.idx]);
        let exp_x = self.log[self.idx];
        assert_eq!(x, exp_x);
        self.idx += 1;
        self.inner.observe(x);
    }

    fn sample(&mut self) -> F {
        debug_assert!(self.idx < self.log.len(), "transcript replay overflow");
        assert!(self.log.samples()[self.idx]);
        let x = self.inner.sample();
        let exp_x = self.log[self.idx];
        self.idx += 1;
        assert_eq!(x, exp_x);
        x
    }
}

impl TranscriptHistory for DuplexSpongeValidator {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn into_log(self) -> TranscriptLog {
        debug_assert_eq!(self.inner.len(), self.log.len());
        debug_assert_eq!(
            self.inner.len(),
            self.idx,
            "transcript replay ended with {} of {} entries consumed",
            self.idx,
            self.inner.len()
        );
        debug_assert_eq!(
            self.log.len(),
            self.idx,
            "transcript replay ended with {} of {} entries consumed",
            self.idx,
            self.log.len()
        );
        self.inner.into_log()
    }
}
