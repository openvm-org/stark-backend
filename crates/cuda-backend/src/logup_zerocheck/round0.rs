use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError, stream::GpuDeviceCtx};
use openvm_stark_backend::prover::{fractional_sumcheck_gkr::Frac, DeviceStarkProvingKey};
use p3_field::PrimeCharacteristicRing;
use tracing::{debug, warn};

use super::errors::Round0EvalError;
use crate::{
    cuda::logup_zerocheck::{
        _logup_r0_intermediates_buffer_size, _logup_r0_temp_sums_buffer_size,
        _zerocheck_r0_intermediates_buffer_size, _zerocheck_r0_temp_sums_buffer_size,
        logup_bary_eval_interactions_round0, zerocheck_ntt_eval_constraints,
    },
    gpu_backend::GenericGpuBackend,
    hash_scheme::GpuHashScheme,
    pkey::InteractionRound0Rules,
    prelude::{EF, F},
};

const ROUND0_COSET_PARALLEL_THRESHOLD: u32 = 32768;
const MAX_LOCKSTEP_NUM_COSETS: u32 = 4;

fn uses_round0_coset_parallel(num_x: u32, skip_domain: u32) -> bool {
    num_x.saturating_mul(skip_domain) < ROUND0_COSET_PARALLEL_THRESHOLD
}

/// Shared scratch buffers for the round-0 kernels of one proof. Kernels on a single
/// stream execute serially, so all per-AIR round-0 launches can share one scratch set,
/// sized to the per-AIR maximum, instead of allocating (and stream-ordered freeing)
/// per AIR — which under back-to-back launches would hold every AIR's scratch
/// simultaneously in the memory pool.
pub struct Round0Scratch {
    intermediates: DeviceBuffer<F>,
    zc_temp_sums: DeviceBuffer<EF>,
    logup_temp_sums: DeviceBuffer<Frac<EF>>,
}

impl Round0Scratch {
    pub fn new() -> Self {
        Self {
            intermediates: DeviceBuffer::new(),
            zc_temp_sums: DeviceBuffer::new(),
            logup_temp_sums: DeviceBuffer::new(),
        }
    }

    pub(crate) fn ensure(
        &mut self,
        intermediates: usize,
        zc_temp_sums: usize,
        logup_temp_sums: usize,
        device_ctx: &GpuDeviceCtx,
    ) {
        if self.intermediates.len() < intermediates {
            self.intermediates = DeviceBuffer::with_capacity_on(intermediates, device_ctx);
        }
        if self.zc_temp_sums.len() < zc_temp_sums {
            self.zc_temp_sums = DeviceBuffer::with_capacity_on(zc_temp_sums, device_ctx);
        }
        if self.logup_temp_sums.len() < logup_temp_sums {
            self.logup_temp_sums = DeviceBuffer::with_capacity_on(logup_temp_sums, device_ctx);
        }
    }
}

/// `(intermediates, temp_sums)` scratch capacities for the zerocheck round-0 kernel.
pub fn zerocheck_r0_scratch_capacities(
    buffer_size: u32,
    skip_domain: u32,
    num_x: u32,
    num_cosets: u32,
    max_temp_bytes: usize,
) -> (usize, usize) {
    if num_cosets == 0 {
        return (0, 0);
    }
    unsafe {
        (
            _zerocheck_r0_intermediates_buffer_size(
                buffer_size,
                skip_domain,
                num_x,
                num_cosets,
                max_temp_bytes,
            ),
            _zerocheck_r0_temp_sums_buffer_size(
                buffer_size,
                skip_domain,
                num_x,
                num_cosets,
                max_temp_bytes,
            ),
        )
    }
}

/// `(intermediates, temp_sums)` scratch capacities for the logup round-0 kernel.
pub fn logup_r0_scratch_capacities(
    buffer_size: u32,
    skip_domain: u32,
    num_x: u32,
    num_cosets: u32,
    max_temp_bytes: usize,
) -> (usize, usize) {
    unsafe {
        (
            _logup_r0_intermediates_buffer_size(
                buffer_size,
                skip_domain,
                num_x,
                num_cosets,
                max_temp_bytes,
            ),
            _logup_r0_temp_sums_buffer_size(
                buffer_size,
                skip_domain,
                num_x,
                num_cosets,
                max_temp_bytes,
            ),
        )
    }
}

fn validate_round0_num_cosets(
    num_x: u32,
    skip_domain: u32,
    num_cosets: u32,
) -> Result<(), Round0EvalError> {
    if num_cosets > MAX_LOCKSTEP_NUM_COSETS && !uses_round0_coset_parallel(num_x, skip_domain) {
        return Err(CudaError::new(1).into());
    }
    Ok(())
}

/// Evaluate plain AIR constraints (not interactions) for a single AIR, given prepared trace input.
///
/// `num_cosets` should equal `constraint_degree - 1` because we evaluate the quotient polynomial.
/// See [`crate::logup_zerocheck`] module docs for async-free/peak memory behavior.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_round0_constraints_gpu<HS: GpuHashScheme>(
    pk: &DeviceStarkProvingKey<GenericGpuBackend<HS>>,
    selectors_cube: &DeviceBuffer<F>,
    main_parts: *const *const F,
    public_values: &DeviceBuffer<F>,
    eq_cube: *const EF,
    lambda_pows: &DeviceBuffer<EF>,
    scratch: &mut Round0Scratch,
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: F,
    max_temp_bytes: usize,
    device_ctx: &GpuDeviceCtx,
) -> Result<DeviceBuffer<EF>, Round0EvalError> {
    let constraints_dag = &pk.vk.symbolic_constraints;
    if constraints_dag.constraints.constraint_idx.is_empty() || num_cosets == 0 {
        // No plain AIR constraints, return empty buffer
        return Ok(DeviceBuffer::new());
    }
    validate_round0_num_cosets(num_x, skip_domain, num_cosets)?;

    let stream = device_ctx.stream.as_raw();
    let rules = &pk.other_data.zerocheck_round0;

    let buffer_size: u32 = rules.inner.buffer_size;
    let (intermed_capacity, temp_sums_buffer_capacity) = zerocheck_r0_scratch_capacities(
        buffer_size,
        skip_domain,
        num_x,
        num_cosets,
        max_temp_bytes,
    );
    debug!("zerocheck:intermediates_capacity={intermed_capacity}");
    debug!("zerocheck:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    // Shared scratch: launches on one stream execute serially, so all round-0 kernels of a
    // proof can reuse the same scratch buffers (sized to the per-AIR maximum by the caller).
    scratch.ensure(intermed_capacity, temp_sums_buffer_capacity, 0, device_ctx);
    let intermediates = &mut scratch.intermediates;
    let temp_sums_buffer = &mut scratch.zc_temp_sums;
    let used_temp_bytes =
        intermed_capacity * size_of::<F>() + temp_sums_buffer_capacity * size_of::<EF>();
    if used_temp_bytes > max_temp_bytes {
        // We do not error if the required bytes is greater than the requested max, but this may
        // lead to unexpected peak memory usage.
        warn!("zerocheck used_temp_bytes ({used_temp_bytes}) > max_temp_bytes ({max_temp_bytes})");
    }

    let preprocessed_ptr = pk
        .preprocessed_data
        .as_ref()
        .map(|cd| cd.trace.as_ptr())
        .unwrap_or(std::ptr::null());

    let mut sp_evals = DeviceBuffer::<EF>::with_capacity_on(
        num_cosets as usize * skip_domain as usize,
        device_ctx,
    );
    // SAFETY:
    // - No bounds checks are done in this kernel. It fully assumes that the Rules are trusted and
    //   all nodes are valid.
    unsafe {
        zerocheck_ntt_eval_constraints(
            temp_sums_buffer,
            &mut sp_evals,
            selectors_cube,
            preprocessed_ptr,
            main_parts,
            eq_cube,
            lambda_pows,
            public_values,
            &rules.inner.d_rules,
            &rules.inner.d_used_nodes,
            buffer_size,
            intermediates,
            skip_domain,
            num_x,
            height,
            num_cosets,
            g_shift,
            max_temp_bytes,
            stream,
        )?;
    }

    Ok(sp_evals)
}

/// Computes the challenge-dependent logup round0 weight vectors and denominator init
/// from the keygen-cached [`InteractionRound0Rules`]. Mirrors the weight accumulation
/// previously done alongside the per-prove DAG rebuild.
pub fn compute_logup_round0_weights(
    rules: &InteractionRound0Rules,
    eq_3bs: &[EF],
    beta_pows: &[EF],
) -> (Vec<EF>, Vec<EF>, EF) {
    debug_assert_eq!(
        rules.count_rule_idxs.len(),
        eq_3bs.len(),
        "interaction count must match eq_3bs"
    );
    let mut numer_weights = vec![EF::ZERO; rules.rules_len];
    let mut denom_weights = vec![EF::ZERO; rules.rules_len];
    let mut denom_sum_init = EF::ZERO;
    for (interaction_idx, &count_rule_idx) in rules.count_rule_idxs.iter().enumerate() {
        numer_weights[count_rule_idx as usize] += eq_3bs[interaction_idx];
        let (message_len, bus_index) = rules.interaction_meta[interaction_idx];
        denom_sum_init +=
            eq_3bs[interaction_idx] * beta_pows[message_len as usize] * F::from_u32(bus_index + 1);
        for (message_idx, &message_rule_idx) in
            rules.message_rule_idxs[interaction_idx].iter().enumerate()
        {
            denom_weights[message_rule_idx as usize] +=
                eq_3bs[interaction_idx] * beta_pows[message_idx];
        }
    }
    (numer_weights, denom_weights, denom_sum_init)
}

/// Evaluate interaction constraints (excluding plain AIR constraints) for a single AIR, given
/// prepared trace input.
///
/// The interactions DAG is cached at keygen in [`InteractionRound0Rules`]; the caller
/// computes the challenge-dependent weights via [`compute_logup_round0_weights`] and passes
/// them as device pointers (e.g. staged via `RoundStager`).
/// See [`crate::logup_zerocheck`] module docs for async-free/peak memory behavior.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_round0_interactions_gpu<HS: GpuHashScheme>(
    pk: &DeviceStarkProvingKey<GenericGpuBackend<HS>>,
    selectors_cube: &DeviceBuffer<F>,
    main_parts: *const *const F,
    public_values: &DeviceBuffer<F>,
    eq_cube: *const EF,
    numer_weights: *const EF,
    denom_weights: *const EF,
    denom_sum_init: EF,
    scratch: &mut Round0Scratch,
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: F,
    max_temp_bytes: usize,
    device_ctx: &GpuDeviceCtx,
) -> Result<DeviceBuffer<Frac<EF>>, Round0EvalError> {
    validate_round0_num_cosets(num_x, skip_domain, num_cosets)?;
    let stream = device_ctx.stream.as_raw();
    let large_domain = num_cosets * skip_domain;

    let rules = pk
        .other_data
        .interaction_round0
        .as_ref()
        .expect("AIR must have interactions for logup round0");
    let d_rules = &rules.d_rules;
    let buffer_size: u32 = rules.buffer_size;
    let (intermed_capacity, temp_sums_buffer_capacity) =
        logup_r0_scratch_capacities(buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes);
    debug!("logup_r0:intermediates_capacity={intermed_capacity}");
    debug!("logup_r0:tmp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    scratch.ensure(intermed_capacity, 0, temp_sums_buffer_capacity, device_ctx);
    let intermediates = &mut scratch.intermediates;
    let temp_sums_buffer = &mut scratch.logup_temp_sums;
    let used_temp_bytes =
        intermed_capacity * size_of::<F>() + temp_sums_buffer_capacity * size_of::<Frac<EF>>();
    if used_temp_bytes > max_temp_bytes {
        warn!(
            "logup_round0 used_temp_bytes ({used_temp_bytes}) > max_temp_bytes ({max_temp_bytes})"
        );
    }

    let preprocessed_ptr = pk
        .preprocessed_data
        .as_ref()
        .map(|cd| cd.trace.as_ptr())
        .unwrap_or(std::ptr::null());

    let mut s_evals = DeviceBuffer::<Frac<EF>>::with_capacity_on(large_domain as usize, device_ctx);

    unsafe {
        logup_bary_eval_interactions_round0(
            temp_sums_buffer,
            &mut s_evals,
            selectors_cube,
            preprocessed_ptr,
            main_parts,
            eq_cube,
            public_values,
            numer_weights,
            denom_weights,
            denom_sum_init,
            d_rules,
            buffer_size,
            intermediates,
            skip_domain,
            num_x,
            height,
            num_cosets,
            g_shift,
            max_temp_bytes,
            stream,
        )?;
    }

    Ok(s_evals)
}
