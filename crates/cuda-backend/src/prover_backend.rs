use std::iter::zip;

use itertools::{izip, zip_eq, Itertools};
use openvm_cuda_common::{memory_manager::MemTracker, stream::gpu_metrics_span};
use openvm_stark_backend::{
    air_builders::symbolic::SymbolicConstraints,
    config::{Com, PcsProof, RapPartialProvingKey, RapPhaseSeqPartialProof},
    keygen::view::MultiStarkVerifyingKeyView,
    p3_challenger::{DuplexChallenger, FieldChallenger},
    proof::{OpenedValues, OpeningProof},
    prover::{
        hal::{
            MatrixDimensions, OpeningProver, ProverBackend, ProverDevice, QuotientCommitter,
            RapPartialProver, TraceCommitter,
        },
        types::{
            AirView, DeviceMultiStarkProvingKeyView, DeviceStarkProvingKey, PairView,
            ProverDataAfterRapPhases, RapSinglePhaseView, RapView,
        },
    },
};
use p3_baby_bear::Poseidon2BabyBear;
use p3_commit::PolynomialSpace;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::{
    base::DeviceMatrix,
    gpu_device::GpuDevice,
    lde::{GpuLde, GpuLdeImpl},
    merkle_tree::GpuMerkleTree,
    opener::OpeningProverGpu,
    prelude::*,
    quotient::{QuotientCommitterGpu, QuotientDataGpu},
};

/// Gpu backend implementation for STARK proving system
#[derive(Clone, Copy, Default, Debug)]
pub struct GpuBackend {}

impl ProverBackend for GpuBackend {
    const CHALLENGE_EXT_DEGREE: u8 = 4;

    // Host Types
    type Val = F;
    type Challenge = EF;
    type OpeningProof = OpeningProof<PcsProof<SC>, Self::Challenge>;
    type RapPartialProof = Option<RapPhaseSeqPartialProof<SC>>;
    type Commitment = Com<SC>; // From<[BabyBear; DIGEST_WIDTH]>
    type Challenger = DuplexChallenger<F, Poseidon2BabyBear<WIDTH>, WIDTH, RATE>;

    // Device Types
    type Matrix = DeviceMatrix<F>;
    type PcsData = GpuPcsData;
    type RapPartialProvingKey = RapPartialProvingKey<SC>;
}

#[derive(Clone)]
pub struct GpuPcsData {
    pub data: GpuMerkleTree<GpuLdeImpl>,
    pub log_trace_heights: Vec<u8>,
}

impl ProverDevice<GpuBackend> for GpuDevice {}

impl TraceCommitter<GpuBackend> for GpuDevice {
    #[instrument(level = "debug", skip_all)]
    fn commit(&self, traces: &[DeviceMatrix<F>]) -> (Com<SC>, GpuPcsData) {
        let _mem = MemTracker::start("commit");
        tracing::debug!(
            "trace (size,strong_count): {:?}",
            traces
                .iter()
                .map(|t| (t.buffer().len(), t.strong_count()))
                .collect::<Vec<_>>()
        );
        let traces_with_shifts = traces
            .iter()
            .map(|trace| (trace.clone(), self.config.shift))
            .collect_vec();
        // We drop the trace in Lde because `traces` is passed by reference
        let (log_trace_heights, merkle_tree) =
            self.commit_traces_with_lde(traces_with_shifts, self.config.fri.log_blowup);
        let root = merkle_tree.root();
        let pcs_data = GpuPcsData {
            data: merkle_tree,
            log_trace_heights,
        };

        (root, pcs_data)
    }
}

type GB = GpuBackend;
type GBChallenger = <GB as ProverBackend>::Challenger;
type GBMatrix = <GB as ProverBackend>::Matrix;
type GBVal = <GB as ProverBackend>::Val;
type GBPcsData = <GB as ProverBackend>::PcsData;
type GBCommitment = <GB as ProverBackend>::Commitment;

impl RapPartialProver<GB> for GpuDevice {
    #[instrument(skip_all)]
    fn partially_prove(
        &self,
        challenger: &mut GBChallenger,
        mpk: &DeviceMultiStarkProvingKeyView<'_, GB>,
        trace_views: Vec<AirView<GBMatrix, GBVal>>,
    ) -> (
        Option<RapPhaseSeqPartialProof<SC>>,
        ProverDataAfterRapPhases<GB>,
    ) {
        let mem = MemTracker::start("partially_prove");
        let num_airs = mpk.per_air.len();
        assert_eq!(num_airs, trace_views.len());

        let (constraints_per_air, rap_pk_per_air): (Vec<_>, Vec<_>) = mpk
            .per_air
            .iter()
            .map(|pk| {
                (
                    SymbolicConstraints::from(&pk.vk.symbolic_constraints),
                    &pk.rap_partial_pk,
                )
            })
            .unzip();

        let trace_views = zip(&mpk.per_air, trace_views)
            .map(|(pk, v)| PairView {
                log_trace_height: log2_strict_usize(v.partitioned_main.first().unwrap().height())
                    as u8,
                preprocessed: pk.preprocessed_data.as_ref().map(|p| p.trace.clone()), // DeviceMatrix is smart pointer clone for now
                partitioned_main: v.partitioned_main,
                public_values: v.public_values,
            })
            .collect_vec();

        let (rap_phase_seq_proof, rap_phase_seq_data) =
            info_span!("generate_perm_trace").in_scope(|| {
                self.rap_phase_seq()
                    .partially_prove_gpu(
                        challenger,
                        &constraints_per_air.iter().collect_vec(),
                        &rap_pk_per_air,
                        trace_views,
                    )
                    .map_or((None, None), |(p, d)| (Some(p), Some(d)))
            });
        mem.tracing_info("after perm trace generation");

        // Set up for the final output
        let mvk_view = MultiStarkVerifyingKeyView::new(
            mpk.per_air.iter().map(|pk| &pk.vk).collect(),
            mpk.trace_height_constraints,
            *mpk.vk_pre_hash,
        );

        let mut perm_matrix_idx = 0usize;
        let rap_views_per_phase;
        let perm_trace_per_air = if let Some(phase_data) = rap_phase_seq_data {
            assert_eq!(mvk_view.num_phases(), 1);
            assert_eq!(
                mvk_view.num_challenges_in_phase(0),
                phase_data.challenges.len()
            );
            let perm_views = zip_eq(
                &phase_data.after_challenge_trace_per_air,
                phase_data.exposed_values_per_air,
            )
            .map(|(perm_trace, exposed_values)| {
                let mut matrix_idx = None;
                if perm_trace.is_some() {
                    matrix_idx = Some(perm_matrix_idx);
                    perm_matrix_idx += 1;
                }
                RapSinglePhaseView {
                    inner: matrix_idx,
                    challenges: phase_data.challenges.clone(),
                    exposed_values: exposed_values.unwrap_or_default(),
                }
            })
            .collect_vec();
            rap_views_per_phase = vec![perm_views]; // 1 challenge phase
            phase_data.after_challenge_trace_per_air
        } else {
            assert_eq!(mvk_view.num_phases(), 0);
            rap_views_per_phase = vec![];
            vec![None; num_airs]
        };

        // Commit to permutation traces: this means only 1 challenge round right now
        // One shared commit for all permutation traces (done on GPU)
        let committed_pcs_data_per_phase: Vec<(Com<SC>, GpuPcsData)> =
            gpu_metrics_span("perm_trace_commit_time_ms", || {
                let flattened_traces_with_shifts = perm_trace_per_air
                    .into_iter()
                    .flatten()
                    .map(|trace| (trace, self.config.shift))
                    .collect_vec();
                // Only commit if there are permutation traces
                if !flattened_traces_with_shifts.is_empty() {
                    let (log_trace_heights, merkle_tree) = self.commit_traces_with_lde(
                        flattened_traces_with_shifts,
                        self.config.fri.log_blowup,
                    );
                    let root = merkle_tree.root();
                    let pcs_data = GpuPcsData {
                        data: merkle_tree,
                        log_trace_heights,
                    };

                    Some((root, pcs_data))
                } else {
                    None
                }
            })
            .unwrap()
            .into_iter()
            .collect();
        let prover_view = ProverDataAfterRapPhases {
            committed_pcs_data_per_phase,
            rap_views_per_phase,
        };
        (rap_phase_seq_proof, prover_view)
    }
}

impl QuotientCommitter<GB> for GpuDevice {
    #[instrument(skip_all)]
    fn eval_and_commit_quotient(
        &self,
        challenger: &mut GBChallenger,
        pk_views: &[&DeviceStarkProvingKey<GB>],
        public_values: &[Vec<GBVal>],
        cached_pcs_datas_per_air: &[Vec<GBPcsData>],
        common_main_pcs_data: &GBPcsData,
        prover_data_after: &ProverDataAfterRapPhases<GB>,
    ) -> (GBCommitment, GBPcsData) {
        let mem = MemTracker::start("quotient");
        let alpha: EF = challenger.sample_algebra_element();
        tracing::debug!("alpha: {alpha:?}");
        let qc = QuotientCommitterGpu::new(alpha);

        let mut common_main_idx = 0;
        let per_rap_quotient = gpu_metrics_span("quotient_poly_compute_time_ms", || {
            izip!(pk_views, cached_pcs_datas_per_air, public_values)
                .enumerate()
                .map(|(i, (pk, cached_pcs_datas, pvs))| {
                    // Prepare extended views(for GPU):
                    let quotient_degree = pk.vk.quotient_degree;
                    let log_trace_height = if pk.vk.has_common_main() {
                        common_main_pcs_data.log_trace_heights[common_main_idx]
                    } else {
                        cached_pcs_datas[0].log_trace_heights[0]
                    };
                    let trace_domain = self.natural_domain_for_degree(1usize << log_trace_height);
                    let quotient_domain = trace_domain
                        .create_disjoint_domain(trace_domain.size() * quotient_degree as usize);
                    tracing::debug!("quotient_domain: {:?}", quotient_domain);
                    let preprocessed = pk.preprocessed_data.as_ref().map(|cv| {
                        cv.data.data.leaves[cv.matrix_idx as usize].take_lde(quotient_domain.size())
                    });
                    let mut partitioned_main: Vec<DeviceMatrix<F>> = cached_pcs_datas
                        .iter()
                        .map(|cv| cv.data.leaves[0].take_lde(quotient_domain.size()))
                        .collect();
                    if pk.vk.has_common_main() {
                        partitioned_main.push(
                            common_main_pcs_data.data.leaves[common_main_idx]
                                .take_lde(quotient_domain.size()),
                        );
                        common_main_idx += 1;
                    }
                    let mut per_phase = zip(
                        &prover_data_after.committed_pcs_data_per_phase,
                        &prover_data_after.rap_views_per_phase,
                    )
                    .map(|((_, pcs_data), rap_views)| -> Option<_> {
                        let rap_view = rap_views.get(i)?;
                        let matrix_idx = rap_view.inner?;
                        let extended_matrix =
                            pcs_data.data.leaves[matrix_idx].take_lde(quotient_domain.size());
                        Some(RapSinglePhaseView {
                            inner: Some(extended_matrix),
                            challenges: rap_view.challenges.clone(),
                            exposed_values: rap_view.exposed_values.clone(),
                        })
                    })
                    .collect_vec();
                    while let Some(last) = per_phase.last() {
                        if last.is_none() {
                            per_phase.pop();
                        } else {
                            break;
                        }
                    }
                    let per_phase = per_phase
                        .into_iter()
                        .map(|v| v.unwrap_or_default())
                        .collect();

                    // Compute quotient values
                    let extended_view_gpu = RapView {
                        log_trace_height,
                        preprocessed,
                        partitioned_main,
                        public_values: pvs.to_vec(),
                        per_phase,
                    };
                    let constraints = &pk.vk.symbolic_constraints;
                    let quotient_degree = pk.vk.quotient_degree;
                    qc.single_rap_quotient_values(
                        self,
                        constraints,
                        extended_view_gpu,
                        quotient_degree,
                    )
                })
                .collect()
        })
        .unwrap();

        let quotient_data = QuotientDataGpu {
            inner: per_rap_quotient,
        };

        let quotient_values = quotient_data
            .split()
            .into_iter()
            .map(|q| (q.chunk, self.config.shift / q.domain.shift()))
            .collect_vec();
        mem.tracing_info("before commit");

        // Commit to quotient polynomials. One shared commit for all quotient polynomials
        gpu_metrics_span("quotient_poly_commit_time_ms", || {
            let (log_trace_heights, merkle_tree) =
                self.commit_traces_with_lde(quotient_values, self.config.fri.log_blowup);
            let root = merkle_tree.root();
            let pcs_data = GpuPcsData {
                data: merkle_tree,
                log_trace_heights,
            };

            (root, pcs_data)
        })
        .unwrap()
    }
}

impl OpeningProver<GB> for GpuDevice {
    #[instrument(skip_all)]
    fn open(
        &self,
        challenger: &mut GBChallenger,
        preprocessed: Vec<&GBPcsData>,
        main: Vec<GBPcsData>,
        after_phase: Vec<GBPcsData>,
        quotient_data: GBPcsData,
        quotient_degrees: &[u8],
    ) -> OpeningProof<PcsProof<SC>, EF> {
        let zeta: EF = challenger.sample_algebra_element();
        tracing::debug!("zeta: {zeta:?}");

        let domain = |log_height| self.natural_domain_for_degree(1usize << log_height);

        let preprocessed_iter = preprocessed.iter().map(|v| {
            assert_eq!(v.log_trace_heights.len(), 1);
            let domain = domain(v.log_trace_heights[0]);
            (&v.data, vec![domain])
        });
        let main_iter = main.iter().map(|v| {
            let domains = v
                .log_trace_heights
                .iter()
                .copied()
                .map(domain)
                .collect_vec();
            (&v.data, domains)
        });
        let after_phase_iter = after_phase.iter().map(|v| {
            let domains = v
                .log_trace_heights
                .iter()
                .copied()
                .map(domain)
                .collect_vec();
            (&v.data, domains)
        });
        let mut rounds = preprocessed_iter
            .chain(main_iter)
            .chain(after_phase_iter)
            .map(|(data, domains)| {
                let points_per_mat = domains
                    .iter()
                    .map(|domain| vec![zeta, domain.next_point(zeta).unwrap()])
                    .collect_vec();
                (data, points_per_mat)
            })
            .collect_vec();
        let num_chunks = quotient_degrees.iter().sum::<u8>() as usize;
        let quotient_opening_points = vec![vec![zeta]; num_chunks];
        rounds.push((&quotient_data.data, quotient_opening_points));

        let opener = OpeningProverGpu {};
        let (mut opening_values, opening_proof) =
            info_span!("OpeningProverGpu::open").in_scope(|| opener.open(self, rounds, challenger));

        // Unflatten opening_values
        let mut quotient_openings = opening_values.pop().expect("Should have quotient opening");

        let num_after_challenge = after_phase.len();
        let after_challenge_openings = opening_values
            .split_off(opening_values.len() - num_after_challenge)
            .into_iter()
            .map(|values| opener.collect_trace_openings(values))
            .collect_vec();
        assert_eq!(
            after_challenge_openings.len(),
            num_after_challenge,
            "Incorrect number of after challenge trace openings"
        );

        let main_openings = opening_values
            .split_off(preprocessed.len())
            .into_iter()
            .map(|values| opener.collect_trace_openings(values))
            .collect_vec();
        assert_eq!(
            main_openings.len(),
            main.len(),
            "Incorrect number of main trace openings"
        );

        let preprocessed_openings = opening_values
            .into_iter()
            .map(|values| {
                let mut openings = opener.collect_trace_openings(values);
                openings
                    .pop()
                    .expect("Preprocessed trace should be opened at 1 point")
            })
            .collect_vec();
        assert_eq!(
            preprocessed_openings.len(),
            preprocessed.len(),
            "Incorrect number of preprocessed trace openings"
        );

        // Unflatten quotient openings
        let quotient_openings = quotient_degrees
            .iter()
            .map(|&chunk_size| {
                quotient_openings
                    .drain(..chunk_size as usize)
                    .map(|mut op| {
                        op.pop()
                            .expect("quotient chunk should be opened at 1 point")
                    })
                    .collect_vec()
            })
            .collect_vec();

        OpeningProof {
            proof: opening_proof,
            values: OpenedValues {
                preprocessed: preprocessed_openings,
                main: main_openings,
                after_challenge: after_challenge_openings,
                quotient: quotient_openings,
            },
        }
    }
}
