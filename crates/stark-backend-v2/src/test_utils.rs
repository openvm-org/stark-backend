use std::sync::Arc;

use itertools::Itertools;
use openvm_stark_backend::{
    interaction::BusIndex,
    prover::{MatrixDimensions, Prover},
    AirRef,
};
pub use openvm_stark_sdk::dummy_airs::fib_air::air::FibonacciAir;
use openvm_stark_sdk::{
    any_rap_arc_vec,
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Config,
        log_up_params::log_up_security_params_baby_bear_100_bits, setup_tracing,
    },
    dummy_airs::{
        self,
        fib_selector_air::air::FibonacciSelectorAir,
        interaction::{
            dummy_interaction_air::DummyInteractionAir,
            self_interaction_air::{SelfInteractionAir, SelfInteractionChip},
        },
    },
};
use p3_baby_bear::BabyBear;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    keygen::types::{MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2},
    poseidon2::sponge::{
        DuplexSponge, DuplexSpongeRecorder, FiatShamirTranscript, TranscriptHistory, TranscriptLog,
    },
    proof::Proof,
    prover::{
        stacked_pcs::stacked_commit, AirProvingContextV2, ColMajorMatrix, CommittedTraceDataV2,
        CpuBackendV2, DeviceDataTransporterV2, DeviceMultiStarkProvingKeyV2, MultiRapProver,
        ProvingContextV2, TraceCommitterV2,
    },
    BabyBearPoseidon2CpuEngineV2, ChipV2, StarkEngineV2, SystemParams, F,
};

#[allow(clippy::type_complexity)]
pub fn prove_up_to_batch_constraints<E: StarkEngineV2>(
    engine: &E,
    transcript: &mut E::TS,
    pk: &DeviceMultiStarkProvingKeyV2<E::PB>,
    ctx: ProvingContextV2<E::PB>,
) -> (
    <E::PD as MultiRapProver<E::PB, E::TS>>::PartialProof,
    <E::PD as MultiRapProver<E::PB, E::TS>>::Artifacts,
) {
    let (_, common_main_pcs_data) = engine.device().commit(
        &ctx.common_main_traces()
            .map(|(_, trace)| trace)
            .collect_vec(),
    );
    engine
        .device()
        .prove_rap_constraints(transcript, pk, &ctx, &common_main_pcs_data)
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
        let d_ctx = device.transport_proving_ctx_to_device(&ctx);
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

pub struct FibFixture {
    pub a: u32,
    pub b: u32,
    pub n: usize,
    pub num_airs: usize,
    pub empty_air_indices: Vec<usize>,
}

impl FibFixture {
    pub fn new(a: u32, b: u32, n: usize) -> Self {
        FibFixture {
            a,
            b,
            n,
            num_airs: 1,
            empty_air_indices: vec![],
        }
    }

    pub fn new_with_num_airs(a: u32, b: u32, n: usize, num_airs: usize) -> Self {
        FibFixture {
            a,
            b,
            n,
            num_airs,
            empty_air_indices: vec![],
        }
    }

    pub fn with_empty_air_indices(mut self, empty_air_indices: impl Into<Vec<usize>>) -> Self {
        self.empty_air_indices = empty_air_indices.into();
        self
    }
}

impl TestFixture for FibFixture {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let air = Arc::new(FibonacciAir);
        vec![air; self.num_airs]
    }

    fn generate_proving_ctx(&self) -> ProvingContextV2<CpuBackendV2> {
        use dummy_airs::fib_air::trace::generate_trace_rows;
        let f_n = get_fib_number(self.a, self.b, self.n);
        let pis = [self.a, self.b, f_n].map(BabyBear::from_canonical_u32);

        ProvingContextV2::new(
            (0..self.num_airs)
                .filter(|i| !self.empty_air_indices.contains(i))
                .map(|i| {
                    (
                        i,
                        AirProvingContextV2::simple(
                            ColMajorMatrix::from_row_major(&generate_trace_rows::<F>(
                                self.a, self.b, self.n,
                            )),
                            pis.to_vec(),
                        ),
                    )
                })
                .collect_vec(),
        )
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
                    params.k_whir(),
                    &[&cached],
                );
                assert_eq!(common.height(), cached.height());
                let cached_data = CommittedTraceDataV2 {
                    commitment: commit,
                    trace: cached,
                    data: Arc::new(data),
                };
                AirProvingContextV2 {
                    cached_mains: vec![cached_data],
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

#[derive(derive_new::new)]
pub struct SelfInteractionFixture {
    pub widths: Vec<usize>,
    pub log_height: usize,
    pub bus_index: BusIndex,
}

impl TestFixture for SelfInteractionFixture {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        self.widths
            .iter()
            .map(|&width| {
                Arc::new(SelfInteractionAir {
                    width,
                    bus_index: self.bus_index,
                }) as AirRef<BabyBearPoseidon2Config>
            })
            .collect_vec()
    }

    fn generate_proving_ctx(&self) -> ProvingContextV2<CpuBackendV2> {
        let per_trace = self
            .widths
            .iter()
            .map(|&width| {
                let chip = SelfInteractionChip {
                    width,
                    log_height: self.log_height,
                };
                chip.generate_proving_ctx(())
            })
            .enumerate()
            .collect_vec();
        ProvingContextV2 { per_trace }
    }
}

pub struct MixtureFixture {
    pub fxs: Vec<MixtureFixtureEnum>,
}

pub enum MixtureFixtureEnum {
    FibFixture(FibFixture),
    InteractionsFixture11(InteractionsFixture11),
    CachedFixture11(CachedFixture11),
    PreprocessedFibFixture(PreprocessedFibFixture),
    SelfInteractionFixture(SelfInteractionFixture),
}

impl MixtureFixtureEnum {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        use crate::test_utils::MixtureFixtureEnum::*;
        match self {
            FibFixture(fx) => fx.airs(),
            InteractionsFixture11(fx) => fx.airs(),
            CachedFixture11(fx) => fx.airs(),
            PreprocessedFibFixture(fx) => fx.airs(),
            SelfInteractionFixture(fx) => fx.airs(),
        }
    }

    fn generate_air_proving_ctxs(&self) -> Vec<AirProvingContextV2<CpuBackendV2>> {
        use crate::test_utils::MixtureFixtureEnum::*;
        let ctx = match self {
            FibFixture(fx) => fx.generate_proving_ctx(),
            InteractionsFixture11(fx) => fx.generate_proving_ctx(),
            CachedFixture11(fx) => fx.generate_proving_ctx(),
            PreprocessedFibFixture(fx) => fx.generate_proving_ctx(),
            SelfInteractionFixture(fx) => fx.generate_proving_ctx(),
        };
        ctx.per_trace
            .into_iter()
            .map(|(_, air_ctx)| air_ctx)
            .collect_vec()
    }
}

impl MixtureFixture {
    pub fn new(fxs: Vec<MixtureFixtureEnum>) -> Self {
        Self { fxs }
    }

    pub fn standard(log_height: usize, params: SystemParams) -> Self {
        let height = 1usize << log_height;
        let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
        let widths = vec![4, 7, 8, 8, 10, 100];
        Self::new(vec![
            MixtureFixtureEnum::FibFixture(FibFixture::new(8, 8, height)),
            MixtureFixtureEnum::InteractionsFixture11(InteractionsFixture11),
            MixtureFixtureEnum::CachedFixture11(CachedFixture11::new(params)),
            MixtureFixtureEnum::PreprocessedFibFixture(PreprocessedFibFixture::new(7, 3, sels)),
            MixtureFixtureEnum::SelfInteractionFixture(SelfInteractionFixture::new(
                widths, log_height, 5,
            )),
        ])
    }
}

impl TestFixture for MixtureFixture {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        self.fxs.iter().flat_map(|fx| fx.airs()).collect_vec()
    }

    fn generate_proving_ctx(&self) -> ProvingContextV2<CpuBackendV2> {
        let per_trace = self
            .fxs
            .iter()
            .flat_map(|fx| fx.generate_air_proving_ctxs())
            .enumerate()
            .collect_vec();
        ProvingContextV2 { per_trace }
    }
}

impl SystemParams {
    /// Parameters for testing traces of height up to `2^log_trace_height` with **toy security
    /// parameters** for faster testing.
    ///
    /// **These parameters should not be used in production!**
    pub fn new_for_testing(log_trace_height: usize) -> Self {
        let l_skip = 4;
        test_system_params_small(4, log_trace_height - l_skip, 4)
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
        logup: log_up_security_params_baby_bear_100_bits(),
        whir_pow_bits: 1,
        max_constraint_degree: 3,
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
