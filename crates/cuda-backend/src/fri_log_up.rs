use std::array;

use itertools::Itertools;
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    error::CudaError,
    stream::gpu_metrics_span,
};
use openvm_stark_backend::{
    air_builders::symbolic::{
        symbolic_expression::SymbolicExpression,
        symbolic_variable::{Entry, SymbolicVariable},
        SymbolicConstraints, SymbolicConstraintsDag,
    },
    interaction::{
        fri_log_up::{FriLogUpPartialProof, FriLogUpProvingKey, STARK_LU_NUM_CHALLENGES},
        LogUpSecurityParameters, SymbolicInteraction,
    },
    p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger},
    prover::{hal::MatrixDimensions, types::PairView},
};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

use crate::{
    base::DeviceMatrix,
    cuda::kernels::{permute::*, prefix::*},
    prelude::*,
    transpiler::{codec::Codec, SymbolicRulesOnGpu},
};

// Output format that keeps GPU data as GPU data
#[derive(Debug)]
pub struct GpuRapPhaseResult {
    pub challenges: Vec<EF>,
    pub after_challenge_trace_per_air: Vec<Option<DeviceMatrix<F>>>,
    pub exposed_values_per_air: Vec<Option<Vec<EF>>>,
}

#[derive(Clone, Debug)]
pub struct FriLogUpPhaseGpu {
    log_up_params: LogUpSecurityParameters,
}

impl FriLogUpPhaseGpu {
    pub fn new(log_up_params: LogUpSecurityParameters) -> Self {
        assert!(log_up_params.bits_of_security::<EF>() >= 100);
        Self { log_up_params }
    }

    pub fn partially_prove_gpu(
        &self,
        challenger: &mut Challenger,
        constraints_per_air: &[&SymbolicConstraints<F>],
        params_per_air: &[&FriLogUpProvingKey],
        trace_view_per_air: Vec<PairView<DeviceMatrix<F>, F>>,
    ) -> Option<(FriLogUpPartialProof<F>, GpuRapPhaseResult)> {
        // 1. Check if there are any interactions - if not, we're done
        let has_any_interactions = constraints_per_air
            .iter()
            .any(|constraints| !constraints.interactions.is_empty());

        if !has_any_interactions {
            return None;
        }

        let logup_pow_witness = challenger.grind(self.log_up_params.pow_bits);
        let challenges: [EF; STARK_LU_NUM_CHALLENGES] =
            array::from_fn(|_| challenger.sample_algebra_element::<EF>());

        let (after_challenge_trace_per_air, cumulative_sum_per_air) =
            gpu_metrics_span("generate_perm_trace_time_ms", || {
                self.generate_after_challenge_traces_per_air_gpu(
                    &challenges,
                    constraints_per_air,
                    params_per_air,
                    trace_view_per_air,
                )
            })
            .unwrap();

        // Challenger needs to observe what is exposed (cumulative_sums)
        for cumulative_sum in cumulative_sum_per_air.iter().flatten() {
            let base_slice =
                <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(cumulative_sum);
            challenger.observe_slice(base_slice);
        }

        let exposed_values_per_air = cumulative_sum_per_air
            .iter()
            .map(|csum| csum.map(|csum| vec![csum]))
            .collect_vec();

        Some((
            FriLogUpPartialProof { logup_pow_witness },
            GpuRapPhaseResult {
                challenges: challenges.to_vec(),
                after_challenge_trace_per_air,
                exposed_values_per_air,
            },
        ))
    }
}

impl FriLogUpPhaseGpu {
    fn generate_after_challenge_traces_per_air_gpu(
        &self,
        challenges: &[EF; STARK_LU_NUM_CHALLENGES],
        constraints_per_air: &[&SymbolicConstraints<F>],
        params_per_air: &[&FriLogUpProvingKey],
        trace_view_per_air: Vec<PairView<DeviceMatrix<F>, F>>,
    ) -> (Vec<Option<DeviceMatrix<F>>>, Vec<Option<EF>>) {
        let interaction_partitions = params_per_air
            .iter()
            .map(|&params| params.clone().interaction_partitions())
            .collect_vec();

        constraints_per_air
            .iter()
            .zip(trace_view_per_air)
            .zip(interaction_partitions.iter())
            .map(|((constraints, trace_view), interaction_partitions)| {
                self.generate_after_challenge_trace_row_wise_gpu(
                    &constraints.interactions,
                    trace_view,
                    challenges,
                    interaction_partitions,
                )
            })
            .unzip()
    }

    fn generate_after_challenge_trace_row_wise_gpu(
        &self,
        all_interactions: &[SymbolicInteraction<F>],
        trace_view: PairView<DeviceMatrix<F>, F>,
        permutation_randomness: &[EF; STARK_LU_NUM_CHALLENGES],
        interaction_partitions: &[Vec<usize>],
    ) -> (Option<DeviceMatrix<F>>, Option<EF>) {
        if all_interactions.is_empty() {
            return (None, None);
        }

        let height = trace_view.partitioned_main[0].height();
        debug_assert!(
            trace_view
                .partitioned_main
                .iter()
                .all(|m| m.height() == height),
            "All main trace parts must have same height"
        );

        let alphas_len = 1;
        let &[alpha, beta] = permutation_randomness;
        // Generate betas
        let max_fields_len = all_interactions
            .iter()
            .map(|interaction| interaction.message.len())
            .max()
            .unwrap_or(0);
        let betas = beta.powers().take(max_fields_len + 1).collect_vec();

        // 0. Prepare challenges
        let challenges = std::iter::once(&alpha)
            .chain(betas.iter())
            .cloned()
            .collect_vec();
        let symbolic_challenges: Vec<SymbolicExpression<F>> = (0..challenges.len())
            .map(|index| SymbolicVariable::<F>::new(Entry::Challenge, index).into())
            .collect_vec();

        // 1. Generate interactions message as denom = alpha + sum(beta_i * message_i) + beta_{m} *
        //    b
        // We use SymbolicInteraction to store (message = [denom], multiplicity = numerator) pair
        // symbolically.
        let mut full_interactions: Vec<SymbolicInteraction<F>> = Vec::new();
        for interaction_indices in interaction_partitions {
            full_interactions.extend(
                interaction_indices
                    .iter()
                    .map(|&interaction_idx| {
                        let mut interaction: SymbolicInteraction<F> =
                            all_interactions[interaction_idx].clone();
                        let b = SymbolicExpression::from_u32(interaction.bus_index as u32 + 1);
                        let betas = symbolic_challenges[alphas_len..].to_vec();
                        debug_assert!(interaction.message.len() <= betas.len());
                        let mut fields = interaction.message.iter();
                        let alpha = symbolic_challenges[0].clone();
                        let mut denom = alpha + fields.next().unwrap().clone();
                        for (expr, beta) in fields.zip(betas.iter().skip(1)) {
                            denom += beta.clone() * expr.clone();
                        }
                        denom += betas[interaction.message.len()].clone() * b;
                        interaction.message = vec![denom];
                        interaction
                    })
                    .collect_vec(),
            );
        }

        // 2. Transpile to GPU Rules
        // We use SymbolicConstraints as a way to encode the symbolic interactions as (denom,
        // numerator) pairs to transport to GPU.
        let constraints = SymbolicConstraints {
            constraints: vec![],
            interactions: full_interactions,
        };
        let constraints_dag: SymbolicConstraintsDag<F> = constraints.into();
        let rules = SymbolicRulesOnGpu::new(constraints_dag.clone(), true);
        let encoded_rules = rules.constraints.iter().map(|c| c.encode()).collect_vec();

        // 3. Call GPU module
        let partition_lens = interaction_partitions
            .iter()
            .map(|p| p.len() as u32)
            .collect_vec();
        let perm_width = interaction_partitions.len() + 1;
        let perm_height = height;
        let (device_matrix, sum) = self.permute_trace_gen_gpu(
            perm_width * 4, // the dim of base field matrix
            perm_height,
            trace_view.preprocessed,
            trace_view.partitioned_main,
            &challenges,
            &encoded_rules,
            rules.buffer_size,
            &partition_lens,
            &rules.used_nodes,
        );

        (Some(device_matrix), Some(sum))
    }

    // gpu-module/src/permute.rs
    #[allow(clippy::too_many_arguments)]
    fn permute_trace_gen_gpu(
        &self,
        permutation_width: usize,
        permutation_height: usize,
        preprocessed: Option<DeviceMatrix<F>>,
        partitioned_main: Vec<DeviceMatrix<F>>,
        challenges: &[EF],
        rules: &[u128],
        num_intermediates: usize,
        partition_lens: &[u32],
        used_nodes: &[usize],
    ) -> (DeviceMatrix<F>, EF) {
        assert!(!rules.is_empty(), "No rules provided to permute");

        tracing::debug!(
            "permute gen rules.len() = {}, num_intermediates = {}",
            rules.len(),
            num_intermediates,
        );

        // 1. input data
        let null_buffer = DeviceBuffer::<F>::new();
        let partitioned_main_ptrs = partitioned_main
            .iter()
            .map(|m| m.buffer().as_raw_ptr() as u64)
            .collect_vec();
        let d_partitioned_main = partitioned_main_ptrs.to_device().unwrap();
        let d_preprocessed = preprocessed
            .as_ref()
            .map(|m| m.buffer())
            .unwrap_or(&null_buffer);

        // 2. gpu buffers
        let d_sum = DeviceBuffer::<EF>::with_capacity(1);
        let d_permutation = DeviceMatrix::<F>::with_capacity(permutation_height, permutation_width);
        let d_challenges = challenges.to_device().unwrap();
        let d_rules = rules.to_device().unwrap();
        let d_partition_lens = partition_lens.to_device().unwrap();
        let d_used_nodes = used_nodes.to_device().unwrap();

        // 3. hal function
        let _ = self.hal_permute_trace_gen(
            &d_sum,
            d_permutation.buffer(),
            d_preprocessed,
            &d_partitioned_main,
            &d_challenges,
            &d_rules,
            rules.len(),
            num_intermediates,
            permutation_height,
            permutation_width / 4,
            &d_partition_lens,
            &d_used_nodes,
        );
        // We can drop preprocessed and main traces now that permutation trace is generated.
        // Note these matrices may be smart pointers so they may not be fully deallocated.
        drop(preprocessed);
        drop(partitioned_main);

        // 4. output data
        let h_sum = d_sum.to_host().unwrap()[0];
        (d_permutation, h_sum)
    }

    // gpu-backend/src/cuda.rs
    #[allow(clippy::too_many_arguments)]
    fn hal_permute_trace_gen(
        &self,
        sum: &DeviceBuffer<EF>,
        permutation: &DeviceBuffer<F>,
        preprocessed: &DeviceBuffer<F>,
        main_partitioned: &DeviceBuffer<u64>,
        challenges: &DeviceBuffer<EF>,
        rules: &DeviceBuffer<u128>,
        num_rules: usize,
        num_intermediates: usize,
        permutation_height: usize,
        permutation_width_ext: usize,
        partition_lens: &DeviceBuffer<u32>,
        used_nodes: &DeviceBuffer<usize>,
    ) -> Result<(), CudaError> {
        let task_size = 65536;
        let tile_per_thread = (permutation_height as u32).div_ceil(task_size as u32);

        tracing::debug!("permutation_height = {permutation_height}, task_size = {}, tile_per_thread = {} num_rules = {num_rules}", task_size, tile_per_thread);

        let is_global = num_intermediates > 10;
        let d_intermediates = if is_global {
            DeviceBuffer::<EF>::with_capacity(task_size * num_intermediates)
        } else {
            DeviceBuffer::<EF>::with_capacity(1) // Dummy buffer for register-based version
        };

        let d_cumulative_sums = DeviceBuffer::<EF>::with_capacity(permutation_height);
        unsafe {
            calculate_cumulative_sums(
                is_global,
                permutation,
                &d_cumulative_sums,
                preprocessed,
                main_partitioned,
                challenges,
                &d_intermediates,
                rules,
                used_nodes,
                partition_lens,
                partition_lens.len(),
                permutation_height as u32,
                permutation_width_ext as u32,
                tile_per_thread,
            )
            .unwrap();
        }

        self.poly_prefix_sum_ext(&d_cumulative_sums, permutation_height as u64);

        unsafe {
            permute_update(
                sum,
                permutation,
                &d_cumulative_sums,
                permutation_height as u32,
                permutation_width_ext as u32,
            )
        }
    }

    fn poly_prefix_sum_ext(&self, inout: &DeviceBuffer<EF>, count: u64) {
        // Parameters for the scan
        let acc_per_thread: u64 = 16;
        let tiles_per_block: u64 = 256;
        let element_per_block: u64 = tiles_per_block * acc_per_thread;
        let mut block_num = (count as u32).div_ceil(tiles_per_block as u32) as u64;

        // First round
        let mut round_stride = 1_u64;
        unsafe {
            prefix_scan_block_ext(inout, count, round_stride, block_num).unwrap();
        }

        // Subsequent rounds
        while block_num > 1 {
            block_num = (block_num as u32).div_ceil(element_per_block as u32) as u64;
            round_stride *= element_per_block;
            unsafe {
                prefix_scan_block_ext(inout, count, round_stride, block_num).unwrap();
            }
        }

        // Block downsweep
        while round_stride > element_per_block {
            let low_level_round_stride = round_stride / element_per_block;
            unsafe {
                prefix_scan_block_downsweep_ext(inout, count, round_stride).unwrap();
            }
            round_stride = low_level_round_stride;
        }

        // Epilogue
        unsafe {
            prefix_scan_epilogue_ext(inout, count).unwrap();
        }
    }
}
