use cuda_kernels::quotient::*;
use cuda_utils::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use itertools::Itertools;
use openvm_stark_backend::{
    air_builders::symbolic::SymbolicExpressionDag, config::Domain, prover::hal::MatrixDimensions,
};
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    base::{DeviceMatrix, DevicePoly, ExtendedLagrangeCoeff},
    prelude::*,
    transpiler::{codec::Codec, SymbolicRulesOnGpu},
};

#[allow(clippy::too_many_arguments)]
#[instrument(
    name = "(GPU) compute single RAP quotient polynomial",
    level = "debug",
    skip_all
)]
pub fn compute_single_rap_quotient_values_gpu(
    constraints: &SymbolicExpressionDag<F>,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    preprocessed_trace_on_quotient_domain: Option<DeviceMatrix<F>>,
    partitioned_main_buffer: Vec<DeviceMatrix<F>>,
    after_challenge_buffer: Vec<DeviceMatrix<F>>,
    // For each challenge round, the challenges drawn
    challenges: Vec<Vec<EF>>,
    alpha: EF,
    public_values: &[F],
    // Values exposed to verifier after challenge round i
    perm_challenge: &[&[EF]],
) -> DevicePoly<EF, ExtendedLagrangeCoeff> {
    // quotient params
    let quotient_size = quotient_domain.size();
    assert!(partitioned_main_buffer
        .iter()
        .all(|m| m.height() >= quotient_size));

    // constraints
    let constraints_len = constraints.num_constraints();
    let rules = SymbolicRulesOnGpu::new(constraints.clone());
    let encoded_rules = rules.constraints.iter().map(|c| c.encode()).collect_vec();

    tracing::debug!(
        constraints = constraints_len,
        encoded = encoded_rules.len(),
        intermediates = rules.num_intermediates,
        "Single RAP quotient: "
    );

    quotient_evaluate_gpu(
        preprocessed_trace_on_quotient_domain,
        partitioned_main_buffer,
        after_challenge_buffer,
        public_values,
        alpha,
        challenges,
        perm_challenge,
        &encoded_rules,
        rules.num_intermediates,
        trace_domain.size(),
        quotient_size,
        quotient_domain.first_point(),
    )
}

#[allow(clippy::too_many_arguments)]
fn quotient_evaluate_gpu(
    preprocessed: Option<DeviceMatrix<F>>,
    main_ldes: Vec<DeviceMatrix<F>>,
    mut perm_ldes: Vec<DeviceMatrix<F>>,
    public_values: &[F],
    alpha: EF,
    mut challenges: Vec<Vec<EF>>,
    exposed_values: &[&[EF]], // exposed_values_after_challenge
    constraints: &[u128],
    num_intermediates: usize,
    trace_size: usize,
    quotient_size: usize,
    coset_shift: F,
) -> DevicePoly<EF, ExtendedLagrangeCoeff> {
    // Create accumulator buffer
    let mut d_accumulators = DeviceBuffer::<EF>::with_capacity(quotient_size);
    if constraints.is_empty() {
        // Since the constraints are empty, return an empty accumulator
        // You may need a utility to zero the buffer if needed
        return DevicePoly::new(true, d_accumulators);
    }

    let qdb_degree = log2_strict_usize(quotient_size) - log2_strict_usize(trace_size);

    tracing::debug!("num partitioned main LDEs = {}", main_ldes.len());
    tracing::debug!("num perm LDEs = {}", perm_ldes.len());
    tracing::debug!("quotient_size = {quotient_size} qdb_degree = {qdb_degree}, ext_degree = 4");

    // main: could be multiple partitioned LDEs
    assert!(!main_ldes.is_empty());
    let main_height = main_ldes[0].height();
    assert!(main_height >= quotient_size);
    // check that all main LDEs have the same height
    assert!(main_ldes.iter().all(|lde| lde.height() == main_height));

    // we support at most one permutation LDE
    assert!(perm_ldes.len() <= 1);
    assert_eq!(perm_ldes.len(), challenges.len());
    let perm_lde = perm_ldes.pop().inspect(|perm_lde| {
        assert!(perm_lde.height() >= quotient_size);
    });

    // Copy challenges to device
    let d_challenge = match challenges.pop() {
        Some(challenges_data) => challenges_data.to_device().unwrap(),
        None => DeviceBuffer::<EF>::with_capacity(64),
    };

    // Copy exposed values to device
    let d_exposed_values = match exposed_values.first() {
        Some(exposed_data) => exposed_data.to_device().unwrap(),
        None => DeviceBuffer::<EF>::with_capacity(64),
    };

    // Copy public values to device
    let d_public_values = if public_values.is_empty() {
        DeviceBuffer::<F>::with_capacity(64)
    } else {
        public_values.to_device().unwrap()
    };

    // alpha
    let alpha_vec = [alpha; 1];
    let d_alpha = alpha_vec.to_device().unwrap();

    // Copy constraints to device
    let d_rules_customgate = constraints.to_device().unwrap();

    // do quotient evaluation
    quotient_evaluate(
        &mut d_accumulators,
        preprocessed.as_ref().map(|lde| lde.buffer()),
        main_ldes.iter().map(|lde| lde.buffer()).collect_vec(),
        perm_lde.as_ref().map(|lde| lde.buffer()),
        d_exposed_values,
        d_public_values,
        d_challenge,
        d_alpha,
        &d_rules_customgate,
        constraints.len(),
        num_intermediates,
        trace_size,
        quotient_size,
        preprocessed.as_ref().map(|lde| lde.height()).unwrap_or(0),
        main_height,
        perm_lde.as_ref().map(|lde| lde.height()).unwrap_or(0),
        coset_shift,
    );

    DevicePoly::new(true, d_accumulators)
}

#[allow(clippy::too_many_arguments)]
fn quotient_evaluate(
    accumulators: &mut DeviceBuffer<EF>,
    preprocessed: Option<&DeviceBuffer<F>>,
    main_partitioned: Vec<&DeviceBuffer<F>>,
    permutation: Option<&DeviceBuffer<F>>,
    exposed_values: DeviceBuffer<EF>,
    public_values: DeviceBuffer<F>,
    challenge: DeviceBuffer<EF>,
    alpha: DeviceBuffer<EF>,
    rules: &DeviceBuffer<u128>,
    num_rules: usize,
    num_intermediates: usize,
    trace_size: usize,
    quotient_size: usize,
    prep_height: usize,
    main_height: usize,
    perm_height: usize,
    quotient_coset: F,
) {
    let task_size = 65536;
    let tile_per_thread = quotient_size.div_ceil(task_size);
    tracing::debug!(
        "quotient_size = {}, task_size = {}, tile_per_thread = {} num_rules = {}",
        quotient_size,
        task_size,
        tile_per_thread,
        num_rules
    );

    let main_matrices_ptr = main_partitioned
        .iter()
        .map(|m| m.as_ptr() as u64)
        .collect::<Vec<_>>();

    // DevicePointer<u8>: size = 8, align = 0x8
    // Allocate `u64` buffer to store main_partitioned[i].as_device_ptr(),
    let main_matrices_ptr_buf = main_matrices_ptr.to_device().unwrap();

    // lagrange selectors
    let lagrange_selectors = lagrange_selectors_on_coset(
        log2_strict_usize(trace_size),
        log2_strict_usize(quotient_size),
        quotient_coset.as_canonical_u32(),
    );
    let is_first_row = &lagrange_selectors[0];
    let is_last_row = &lagrange_selectors[1];
    let is_transition = &lagrange_selectors[2];
    let inv_zeroifier = &lagrange_selectors[3];

    // qdb_degree
    let qdb_degree = log2_strict_usize(quotient_size) - log2_strict_usize(trace_size);

    let is_global = num_intermediates > 10;

    let mut buffer_size = 1;
    if is_global {
        buffer_size = task_size * num_intermediates;
    }

    let d_intermediates = DeviceBuffer::<EF>::with_capacity(buffer_size);

    unsafe {
        quotient_global_or_local(
            is_global,
            accumulators,
            preprocessed.unwrap_or(&DeviceBuffer::<F>::new()),
            &main_matrices_ptr_buf,
            permutation.as_ref().unwrap_or(&&DeviceBuffer::<F>::new()),
            &exposed_values,
            &public_values,
            is_first_row,
            is_last_row,
            is_transition,
            inv_zeroifier,
            &challenge,
            &alpha,
            &d_intermediates,
            rules,
            num_rules as u64,
            quotient_size as u32,
            prep_height as u32,
            main_height as u32,
            perm_height as u32,
            qdb_degree as u64,
            tile_per_thread as u32,
        )
        .unwrap();
    }
}

fn lagrange_selectors_on_coset(
    log_n: usize,
    coset_log_n: usize,
    coset_shift: u32,
) -> Vec<DeviceBuffer<F>> {
    assert!(coset_log_n >= log_n);
    let coset_size = 1 << coset_log_n;

    // alloc memory for lagrange selectors
    let lagrange_selectors = (0..4)
        .map(|_| DeviceBuffer::<F>::with_capacity(coset_size))
        .collect_vec();

    unsafe {
        quotient_selectors(
            &lagrange_selectors[0], // is_first_row
            &lagrange_selectors[1], // is_last_row
            &lagrange_selectors[2], // is_transition
            &lagrange_selectors[3], // inv_zeroifier
            log_n as u64,
            coset_log_n as u64,
            coset_shift,
        )
        .unwrap();
    }

    lagrange_selectors
}
