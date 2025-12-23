use std::{collections::BTreeMap, fmt::Debug, iter};

use itertools::{izip, Itertools};
use openvm_cuda_common::{copy::MemCopyD2H, d_buffer::DeviceBuffer, memory_manager::MemTracker};
use openvm_stark_backend::{
    config::Com,
    p3_challenger::{CanObserve, CanSampleBits, FieldChallenger, GrindingChallenger},
    proof::AdjacentOpenedValues,
    prover::hal::MatrixDimensions,
};
use ops::*;
use p3_baby_bear::Poseidon2BabyBear;
use p3_commit::{BatchOpening, ExtensionMmcs, OpenedValues};
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{dot_product, BasedVectorSpace, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_fri::{CommitPhaseProofStep, FriProof, QueryProof};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_util::{linear_map::LinearMap, log2_strict_usize};
use tracing::{debug_span, info_span};

use crate::{
    base::{DevicePoly, ExtendedLagrangeCoeff},
    gpu_device::GpuDevice,
    lde::GpuLde,
    merkle_tree::GpuMerkleTree,
    prelude::*,
    prover_backend::GpuPcsData,
};

pub(crate) mod ops;

const DIGEST_WIDTH: usize = 8;
type PackedVal = <F as Field>::Packing;
type Perm = Poseidon2BabyBear<16>;
type Hash = PaddingFreeSponge<Perm, WIDTH, 8, DIGEST_WIDTH>;
type Compress = TruncatedPermutation<Perm, 2, DIGEST_WIDTH, WIDTH>;
type ValMmcs = MerkleTreeMmcs<PackedVal, <F as Field>::Packing, Hash, Compress, DIGEST_WIDTH>;
type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;

pub struct OpeningProverGpu {}

#[allow(clippy::type_complexity)]
impl OpeningProverGpu {
    pub fn open<T: GpuLde>(
        &self,
        device: &GpuDevice,
        rounds: Vec<(&GpuMerkleTree<T>, Vec<Vec<EF>>)>,
        challenger: &mut Challenger,
    ) -> (
        OpenedValues<EF>,
        FriProof<EF, ChallengeMmcs, F, Vec<BatchOpening<F, ValMmcs>>>,
    ) {
        let mem = MemTracker::start("open");
        // for each matrix (with columns p_i) and opening point z, we want:
        //         reduced[X] += alpha_offset * inv_denom[X] * [ sum_i [ alpha^i * p_i[X] ] - sum_i
        // [ alpha^i * y[i] ] ]
        let mats_and_points = rounds
            .iter()
            .map(|(data, points)| {
                let mats = data.leaves.iter().collect_vec();
                assert_eq!(mats.len(), points.len());
                (mats, points)
            })
            .collect_vec();
        // height of LDE matrices
        let heights_and_points = mats_and_points
            .iter()
            .map(|(mats, points)| (mats.iter().map(|m| m.height()).collect_vec(), *points))
            .collect_vec();

        let global_max_height = heights_and_points
            .iter()
            .flat_map(|(mats, _)| mats.iter().copied())
            .max()
            .unwrap();
        let log_global_max_height = log2_strict_usize(global_max_height);

        // height of trace matrices
        let trace_heights_and_points = mats_and_points
            .iter()
            .map(|(mats, points)| (mats.iter().map(|m| m.trace_height()).collect_vec(), *points))
            .collect_vec();

        let mut inv_denoms = LinearMap::new();
        let mut last_shift = None;
        let all_opened_values: OpenedValues<EF> = info_span!("evaluate matrix").in_scope(|| {
            mats_and_points
                .iter()
                .map(|(mats, points)| {
                    // matrices that have same shift are grouped together.
                    let mut mats_by_shift: BTreeMap<F, Vec<usize>> = BTreeMap::new();
                    mats.iter().enumerate().for_each(|(i, mat)| {
                        if let Some(indices) = mats_by_shift.get_mut(&mat.shift()) {
                            indices.push(i);
                        } else {
                            mats_by_shift.insert(mat.shift(), vec![i]);
                        }
                    });
                    // BTreeMap guarantees eval_orders is deterministic
                    let eval_orders = mats_by_shift
                        .into_iter()
                        .flat_map(|(_, indices)| indices.into_iter())
                        .collect_vec();

                    let openings = eval_orders
                        .iter()
                        .map(|&idx| {
                            let mat = mats[idx];
                            let points_for_mat = &points[idx];

                            let shift = F::GENERATOR;

                            // if last_shift is not set or not equal to current shift
                            if last_shift.is_none() || last_shift.unwrap() != shift {
                                // for each point and each log height, we will precompute 1/(X - z)
                                // for subgroup of order 2^log_height.
                                // TODO: reduce inv_denoms' size
                                inv_denoms = compute_per_height_inverse_denominators(
                                    trace_heights_and_points.as_slice(),
                                    shift,
                                );
                            }
                            last_shift = Some(shift);

                            // matrix is evaluations on domain shift*H = { shift* g^i }.
                            points_for_mat
                                .iter()
                                .map(|z| {
                                    let trace_height = mat.trace_height();
                                    let log_height = log2_strict_usize(trace_height);
                                    let low_coset_mat = mat.take_lde(trace_height);
                                    let g = F::two_adic_generator(log_height);
                                    let inv_denom =
                                        inv_denoms.get(z).unwrap()[log_height].as_ref().unwrap();
                                    matrix_evaluate(
                                        &low_coset_mat,
                                        inv_denom,
                                        *z,
                                        shift,
                                        g,
                                        trace_height,
                                    )
                                    .unwrap()
                                })
                                .collect_vec()
                        })
                        .collect_vec();

                    let mut original_orders = vec![0; eval_orders.len()];
                    for (reorder_idx, original_idx) in eval_orders.iter().enumerate() {
                        original_orders[*original_idx] = reorder_idx;
                    }

                    original_orders
                        .iter()
                        .map(|reorder_idx| {
                            openings[*reorder_idx].iter().for_each(|ys| {
                                ys.iter()
                                    .for_each(|&y| challenger.observe_algebra_element(y));
                            });
                            openings[*reorder_idx].clone()
                        })
                        .collect_vec()
                })
                .collect()
        });
        drop(inv_denoms);

        // Batch combination challenge
        let alpha: EF = challenger.sample_algebra_element();
        tracing::debug!("alpha sampled in gpu_pcs::open(): {:?}", alpha);

        // For each unique opening point z, we will find the largest degree bound
        // for that point, and precompute 1/(X - z) for the largest subgroup.
        let inv_denoms =
            compute_max_height_inverse_denominators(heights_and_points.as_slice(), F::GENERATOR);

        let mut reduced_openings: [_; 32] = core::array::from_fn(|_| None);
        let mut num_reduced = [0; 32];

        info_span!("build fri inputs").in_scope(|| {
            for ((mats, points), openings_for_round) in
                mats_and_points.iter().zip_eq(all_opened_values.iter())
            {
                for (mat, points_for_mat, openings_for_mat) in
                    izip!(mats.iter(), points.iter(), openings_for_round.iter())
                {
                    let log_height = log2_strict_usize(mat.height());
                    let reduced_opening = reduced_openings[log_height].get_or_insert_with(|| {
                        DevicePoly::new(false, DeviceBuffer::<EF>::with_capacity(mat.height()))
                    });

                    let mat = mat.take_lde(mat.height());

                    for (z, openings) in points_for_mat.iter().zip(openings_for_mat.iter()) {
                        let inv_denom = inv_denoms.get(z).unwrap();
                        let m_z = dot_product(alpha.powers(), openings.iter().copied());
                        reduce_matrix_quotient_acc(
                            reduced_opening,
                            &mat,
                            inv_denom,
                            m_z,
                            alpha,
                            num_reduced[log_height],
                            num_reduced[log_height] == 0,
                        )
                        .unwrap();
                        num_reduced[log_height] += mat.width();
                    }
                }
            }
        });
        let fri_inputs = reduced_openings.into_iter().rev().flatten().collect_vec();
        mem.tracing_info("after fri inputs");
        // codes copied from prover.rs
        let config = device.config.fri;
        assert!(!fri_inputs.is_empty());
        assert!(
            fri_inputs
                .iter()
                .tuple_windows()
                .all(|(l, r)| l.len() >= r.len()),
            "Inputs are not sorted in descending order of length."
        );

        let log_max_height = log2_strict_usize(fri_inputs[0].len());
        let log_min_height = log2_strict_usize(fri_inputs.last().unwrap().len());
        if config.log_final_poly_len > 0 {
            assert!(log_min_height > config.log_final_poly_len + config.log_blowup);
        }

        // commit to the folded polynomials
        let commit_phase_result = commit_phase_on_gpu(device, fri_inputs, challenger);

        let query_pow_witness = challenger.grind(config.query_proof_of_work_bits);

        let extra_query_index_bits = 0;
        let query_proofs = info_span!("query phase").in_scope(|| {
            let query_indices = iter::repeat_with(|| {
                challenger.sample_bits(log_max_height + extra_query_index_bits)
            })
            .take(config.num_queries)
            .collect_vec();

            let mut input_proofs_for_rounds = rounds
                .iter()
                .map(|(tree, _)| {
                    // for each round, query multiple indices at once
                    let log_max_height = log2_strict_usize(tree.get_max_height());
                    let reduced_indices = query_indices
                        .iter()
                        .map(|index| {
                            let bits_reduced = log_global_max_height - log_max_height;
                            index >> bits_reduced
                        })
                        .collect_vec();
                    tree.open_batch_at_multiple_indices(&reduced_indices)
                        .unwrap()
                        .into_iter()
                        .map(|(opened_values, opening_proof)| BatchOpening {
                            opened_values,
                            opening_proof,
                        })
                        .collect_vec()
                })
                .collect_vec();

            let input_proofs_rev: Vec<_> = query_indices
                .iter()
                // reverse the indices to pop
                .rev()
                .map(|_| {
                    // the opening proof for last index comes at the end
                    // therefore we can get it by popping to avoid clone.
                    input_proofs_for_rounds
                        .iter_mut()
                        .map(|input_proofs| input_proofs.pop().unwrap())
                        .collect()
                })
                .collect_vec();

            let commit_phase_openings_rev = answer_batch_queries_on_gpu(
                &commit_phase_result.data,
                query_indices
                    .into_iter()
                    .map(|index| index >> extra_query_index_bits)
                    .collect_vec()
                    .as_slice(),
            );

            input_proofs_rev
                .into_iter()
                .rev()
                .zip(commit_phase_openings_rev.into_iter().rev())
                .map(|(input_proof, commit_phase_openings)| QueryProof {
                    input_proof,
                    commit_phase_openings,
                })
                .collect()
        });

        let fri_proof = FriProof {
            commit_phase_commits: commit_phase_result.commits,
            commit_pow_witnesses: commit_phase_result.pow_witnesses,
            query_proofs,
            final_poly: commit_phase_result.final_poly,
            query_pow_witness,
        };

        (all_opened_values, fri_proof)
    }

    pub fn collect_trace_openings<Challenge: Debug>(
        &self,
        ops: Vec<Vec<Vec<Challenge>>>,
    ) -> Vec<AdjacentOpenedValues<Challenge>> {
        ops.into_iter()
            .map(|op| {
                let [local, next] = op.try_into().expect("Should have 2 openings");
                AdjacentOpenedValues { local, next }
            })
            .collect()
    }
}

pub struct CommitPhaseGPUResult {
    pub commits: Vec<Com<SC>>,
    pub data: Vec<GpuPcsData>,
    pub final_poly: Vec<EF>,
    pub pow_witnesses: Vec<F>,
}

fn commit_phase_on_gpu(
    device: &GpuDevice,
    inputs: Vec<DevicePoly<EF, ExtendedLagrangeCoeff>>,
    challenger: &mut Challenger,
) -> CommitPhaseGPUResult {
    let fri_log_blowup = device.config.fri.log_blowup;
    let fri_log_final_poly_len = device.config.fri.log_final_poly_len;
    let mut inputs_iter = inputs.into_iter().peekable();
    let mut folded = inputs_iter.next().unwrap();
    let mut commits = vec![];
    let mut pow_witnesses = vec![];
    let mut data: Vec<GpuPcsData> = vec![];

    let blowup = 1 << fri_log_blowup;
    let final_poly_len = 1 << fri_log_final_poly_len;

    while folded.len() > blowup * final_poly_len {
        // folded is converted to a matrix over base field with width = 8 and height =
        // folded.len()/2
        let folded_as_matrix = fri_ext_poly_to_base_matrix(&folded).unwrap();
        let (log_trace_heights, merkle_tree) = device.commit_trace(folded_as_matrix);
        let (commit, prover_data) = (
            merkle_tree.root(),
            GpuPcsData {
                data: merkle_tree,
                log_trace_heights,
            },
        );
        challenger.observe(commit);

        commits.push(commit);
        data.push(prover_data);

        let pow_witness = challenger.grind(device.config.fri.commit_proof_of_work_bits);
        pow_witnesses.push(pow_witness);

        let log_folded_len = log2_strict_usize(folded.len());
        let beta: EF = challenger.sample_algebra_element();
        tracing::debug!("beta at gpu pcs (layer = {}): {:?}", log_folded_len, beta);

        let fri_input = inputs_iter.next_if(|v| v.len() == folded.len() / 2);
        let g_inv = EF::two_adic_generator(log_folded_len).inverse();

        folded = fri_fold(folded, fri_input, beta, g_inv).unwrap();
    }

    let mut folded_on_host: Vec<EF> = folded.coeff.to_host().unwrap();
    folded_on_host.truncate(final_poly_len);

    // TODO: For better performance, we could run the IDFT on only the first half
    //       (or less, depending on `log_blowup`) of `final_poly`.
    let final_poly =
        debug_span!("idft final poly").in_scope(|| Radix2Dit::default().idft(folded_on_host));

    // The evaluation domain is "blown-up" relative to the polynomial degree of `final_poly`,
    // so all coefficients after the first final_poly_len should be zero.
    debug_assert!(
        final_poly.iter().skip(final_poly_len).all(|x| x.is_zero()),
        "All coefficients beyond final_poly_len must be zero"
    );
    tracing::debug!("final poly from gpu pcs: {:?}", final_poly);

    // Observe all coefficients of the final polynomial.
    for &x in &final_poly {
        challenger.observe_algebra_element(x);
    }

    CommitPhaseGPUResult {
        commits,
        data,
        final_poly,
        pow_witnesses,
    }
}

fn answer_batch_queries_on_gpu(
    commit_phase_commits: &[GpuPcsData],
    indices: &[usize],
) -> Vec<Vec<CommitPhaseProofStep<EF, ChallengeMmcs>>> {
    let mut proofs_per_phase = commit_phase_commits
        .iter()
        .enumerate()
        .map(|(i, commit)| {
            let pair_indices = indices
                .iter()
                .map(|index| {
                    let index_i = index >> i;
                    index_i >> 1
                })
                .collect::<Vec<_>>();

            commit
                .data
                .open_batch_at_multiple_indices(&pair_indices)
                .unwrap()
                .into_iter()
                .zip(indices.iter())
                .map(|((opened_base_values, opening_proof), index)| {
                    let index_i = index >> i;
                    let index_i_sibling = index_i ^ 1;

                    // copied from commit/src/adapters/extension_mmcs.rs#open_batch
                    let opened_ext_values: Vec<Vec<EF>> = opened_base_values
                        .into_iter()
                        .map(|row| {
                            row.chunks(4)
                                .map(|row| EF::from_basis_coefficients_slice(row).unwrap())
                                .collect()
                        })
                        .collect();
                    let mut opened_rows = opened_ext_values;

                    assert_eq!(opened_rows.len(), 1);

                    let opened_row = opened_rows.pop().unwrap();
                    assert_eq!(opened_row.len(), 2, "Committed data should be in pairs");
                    let sibling_value = opened_row[index_i_sibling % 2];

                    CommitPhaseProofStep {
                        sibling_value,
                        opening_proof,
                    }
                })
                .collect_vec()
        })
        .collect_vec();

    indices
        .iter()
        .rev()
        .map(|_| {
            proofs_per_phase
                .iter_mut()
                .map(|proofs| proofs.pop().unwrap())
                .collect_vec()
        })
        .collect_vec()
}
