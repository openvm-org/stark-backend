use std::{ffi::c_void, iter::once, mem::ManuallyDrop, sync::Arc};

use itertools::Itertools;
use openvm_cuda_backend::{
    base::DeviceMatrix,
    cuda::kernels::{fri::split_ext_poly_to_base_col_major_matrix, lde::batch_expand_pad},
    ntt::batch_ntt,
};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D, cuda_memcpy},
    d_buffer::DeviceBuffer,
};
use openvm_stark_backend::prover::MatrixDimensions;
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, TwoAdicField};
use p3_util::log2_strict_usize;
use stark_backend_v2::{
    poseidon2::sponge::FiatShamirTranscript,
    proof::{MerkleProof, WhirProof},
    prover::whir::WhirProver,
};
use tracing::instrument;

use crate::{
    D_EF, Digest, EF, F, GpuBackendV2, GpuDeviceV2, ProverError,
    cuda::{
        poly::{
            algebraic_batch_matrices, batch_eq_hypercube_stage, eq_hypercube_stage_ext,
            eval_poly_ext_at_point_from_base, mle_interpolate_stage_ext,
        },
        sumcheck::fold_mle,
        whir::{
            _whir_sumcheck_required_temp_buffer_size, w_evals_accumulate, whir_sumcheck_mle_round,
        },
    },
    merkle_tree::MerkleTreeGpu,
    poly::{evals_eq_hypercube, mle_evals_to_coeffs_inplace},
    stacked_pcs::StackedPcsDataGpu,
};

impl<TS: FiatShamirTranscript> WhirProver<GpuBackendV2, GpuDeviceV2, TS> for GpuDeviceV2 {
    #[instrument(level = "info", skip_all)]
    fn prove_whir(
        &self,
        transcript: &mut TS,
        common_main_pcs_data: StackedPcsDataGpu<F, Digest>,
        pre_cached_pcs_data_per_commit: Vec<Arc<StackedPcsDataGpu<F, Digest>>>,
        u_cube: &[EF],
    ) -> WhirProof {
        prove_whir_opening_gpu(
            self,
            transcript,
            common_main_pcs_data,
            pre_cached_pcs_data_per_commit,
            u_cube,
        )
        .unwrap()
    }
}

fn prove_whir_opening_gpu<TS: FiatShamirTranscript>(
    device: &GpuDeviceV2,
    transcript: &mut TS,
    common_main_pcs_data: StackedPcsDataGpu<F, Digest>,
    other_pcs_data: Vec<Arc<StackedPcsDataGpu<F, Digest>>>,
    u: &[EF],
) -> Result<WhirProof, ProverError> {
    let params = device.config();
    let k_whir = params.k_whir;
    let log_blowup = params.log_blowup;
    let num_whir_queries = params.num_whir_queries;
    let log_final_poly_len = params.log_final_poly_len;
    let whir_pow_bits = params.whir_pow_bits;
    let l_skip = params.l_skip;

    let height = common_main_pcs_data.matrix.height();
    debug_assert!(other_pcs_data.iter().all(|d| d.matrix.height() == height));
    let mut m = log2_strict_usize(height);
    assert_eq!(m, u.len());
    debug_assert!(m >= l_skip);

    // Sample randomness for algebraic batching.
    // We batch the codewords for \hat{q}_j together _before_ applying WHIR.
    let mu = transcript.sample_ext();
    let num_commits = other_pcs_data.len() + 1;

    // The evaluations of `\hat{f}` in the current WHIR round on the hypercube `H_m`.
    let mut f_evals = DeviceBuffer::<EF>::with_capacity(height);
    // We algebraically batch all matrices together so we only need to interpolate one column vector
    {
        let common_main = common_main_pcs_data.matrix;
        // We only need the mixed PLE buffer for each backing matrix.
        let (mat_ptrs, widths): (Vec<_>, Vec<_>) = once(&common_main)
            .chain(other_pcs_data.iter().map(|d| &d.matrix))
            .map(|mat| (mat.mixed.as_ptr(), mat.width() as u32))
            .unzip();
        let common_main_mixed = ManuallyDrop::new(common_main.mixed);
        let mut mu_idxs = Vec::with_capacity(num_commits);
        let mut total_width = 0u32;
        for &width in &widths {
            mu_idxs.push(total_width);
            total_width += width;
        }
        let mu_powers = mu.powers().take(total_width as usize).collect_vec();
        let d_mat_ptrs = mat_ptrs.to_device()?;
        let d_mu_powers = mu_powers.to_device()?;
        let d_mu_idxs = mu_idxs.to_device()?;
        let d_widths = widths.to_device()?;
        // SAFETY:
        // - `mu_powers` has length `total_width` and `mu_idxs` are in bounds by construction.
        // - `f_evals` has capacity `height`.
        // - We assume `ple_mixed` all have same `height`
        // - `mu_idxs`, `widths`, `ple_mixed` have same length `num_commits`.
        unsafe {
            algebraic_batch_matrices(
                &mut f_evals,
                &d_mat_ptrs,
                &d_mu_powers,
                &d_mu_idxs,
                &d_widths,
                height,
                num_commits,
            )?;
        }
        let _ = ManuallyDrop::into_inner(common_main_mixed);
    } // common_main_pcs_data.matrix has now been freed

    // `f_evals` is currently the 2^l_skip coefficients of `f(Z, \vect x)` for each `\vect x in H_{m
    // - l_skip}`. The univariate coefficient form is the same as the multilinear coefficient
    // form for multilinear polys on H_{l_skip}, so we must now multilinear interpolate `f_evals` to
    // get fully multilinear evaluations on `H_m`.
    for i in 0..l_skip {
        let step = 1u32 << i;
        // SAFETY: `f_evals` has length `2^m` with `m >= l_skip`.
        unsafe {
            // Multilinear Coeff -> Eval
            mle_interpolate_stage_ext(&mut f_evals, step, false)?;
        }
    }

    // We will drop `common_main_tree` after whir round 0
    let mut common_main_tree = Some(common_main_pcs_data.tree);

    debug_assert_eq!((m - log_final_poly_len) % k_whir, 0);
    let num_whir_rounds = (m - log_final_poly_len) / k_whir;
    assert!(num_whir_rounds > 0);

    // We assume `\hat{w}` in a WHIR round is always multilinear and maintain its
    // evaluations on `H_m`.
    let mut w_evals = DeviceBuffer::<EF>::with_capacity(1 << m);
    unsafe {
        evals_eq_hypercube(&mut w_evals, u)?;
    }

    let mut whir_sumcheck_polys: Vec<[EF; 2]> = vec![];
    let mut codeword_commits = vec![];
    let mut ood_values = vec![];
    // per commitment, per whir query, per column
    let mut initial_round_opened_rows: Vec<Vec<Vec<Vec<F>>>> = vec![vec![]; num_commits];
    let mut initial_round_merkle_proofs: Vec<Vec<MerkleProof>> = vec![];
    let mut codeword_opened_values: Vec<Vec<Vec<EF>>> = vec![];
    let mut codeword_merkle_proofs: Vec<Vec<MerkleProof>> = vec![];
    let mut whir_pow_witnesses = vec![];
    let mut rs_tree = None;
    let mut log_rs_domain_size = m + log_blowup;
    let mut final_poly = None;

    let mut d_input_fw_ptrs = DeviceBuffer::<usize>::with_capacity(2);
    let mut d_output_fw_ptrs = DeviceBuffer::<usize>::with_capacity(2);
    let d_widths = [1u32, 1u32].to_device()?;
    let mut d_s_evals = DeviceBuffer::<EF>::with_capacity(2);

    for whir_round in 0..num_whir_rounds {
        let is_last_round = whir_round == num_whir_rounds - 1;
        // Run k_whir rounds of sumcheck on `sum_{x in H_m} \hat{w}(\hat{f}(x), x)`
        for round in 0..k_whir {
            // Do not use f_evals.len() because it might have extra capacity
            let f_height = 1 << (m - round);
            debug_assert!(
                f_evals.len() >= f_height,
                "f_evals has length {}, expected 2^{} for m={m}, round={round}",
                f_evals.len(),
                m - round
            );
            debug_assert!(w_evals.len() >= f_height);
            let output_height = f_height / 2;
            let tmp_buffer_capacity =
                unsafe { _whir_sumcheck_required_temp_buffer_size(f_height as u32) };
            // PERF[jpw]: memory management could be optimized to re-use buffers
            // Currently not ping-ponging buffers to free memory earlier
            let mut new_f_evals =
                DeviceBuffer::<EF>::with_capacity(output_height.max(tmp_buffer_capacity as usize));
            // SAFETY:
            // - `d_s_evals` has length 2
            // - We use `new_f_evals` as the temp buffer, which needs length >=
            //   output_height.div_ceil(threads_per_block)
            unsafe {
                whir_sumcheck_mle_round(
                    &f_evals,
                    &w_evals,
                    &mut d_s_evals,
                    &mut new_f_evals,
                    f_height as u32,
                )?;
            }
            let s_evals = d_s_evals.to_host()?;
            for &eval in &s_evals {
                transcript.observe_ext(eval);
            }
            whir_sumcheck_polys.push(s_evals.try_into().unwrap());

            let alpha = transcript.sample_ext();

            // PERF[jpw]: memory management could be optimized to re-use buffers
            let new_w_evals = DeviceBuffer::<EF>::with_capacity(output_height);
            [f_evals.as_ptr() as usize, w_evals.as_ptr() as usize].copy_to(&mut d_input_fw_ptrs)?;
            [new_f_evals.as_ptr() as usize, new_w_evals.as_ptr() as usize]
                .copy_to(&mut d_output_fw_ptrs)?;
            // Fold `f_evals`, `w_evals` as MLE with respect to `alpha`:
            // SAFETY:
            // - `new_f_evals`, `new_w_evals` are allocated with half the length of `f_evals` and
            //   `w_evals`.
            // - all pointers of type `*const EF` are cast to `usize` are sent to device as
            //   `DeviceBuffer<usize>`.
            // - we treat `f_evals`, `w_evals` as two separate matrices of width 1 each
            unsafe {
                fold_mle(
                    &d_input_fw_ptrs,
                    &d_output_fw_ptrs,
                    &d_widths,
                    2,
                    output_height as u32,
                    alpha,
                )?;
            }
            f_evals = new_f_evals;
            w_evals = new_w_evals;
        }
        // Define g^ = f^(alpha, \cdot) and send matrix commit of RS(g^)
        // f_evals is the evaluations of f^(alpha, \cdot) on hypercube
        let f_height = 1 << (m - k_whir);
        debug_assert!(f_evals.len() >= f_height);
        debug_assert_eq!(size_of::<EF>() / size_of::<F>(), D_EF);
        let mut g_coeffs = DeviceBuffer::<F>::with_capacity(f_height * D_EF);
        // SAFETY: we allocated `f_evals.len() * D_EF` space for `g_coeffs` to do a 1-to-D_EF
        // (1-to-4) split
        unsafe {
            split_ext_poly_to_base_col_major_matrix(
                &g_coeffs,
                &f_evals,
                f_height as u64,
                f_height as u32,
            )?;
        }
        // We convert f from column major in EF to column-major in F.
        // The MLE interpolation is the same since it's linear.
        // PERF[jpw]: it may be more performant to interpolate in EF-form for better memory
        // coalescing, but our batch expand kernel is in the base field so the implementation is
        // currently simpler to go directly to F first.
        mle_evals_to_coeffs_inplace(&mut g_coeffs, m - k_whir)?;
        let (g_tree, z_0) = if !is_last_round {
            let codeword_height = 1 << (log_rs_domain_size - 1);
            // `g: \mathcal{L}^{(2)} \to \mathbb F`
            let g_rs = DeviceBuffer::<F>::with_capacity(D_EF * codeword_height);
            // SAFETY:
            // - g_coeffs is a single EF polynomial, treated as 4 F-polynomials of height
            //   2^{m-k_whir}
            // - We resize each F-poly to RS domain size 2^{log_rs_domain_size - 1}, which is
            //   equivalent to resizing the EF-polynomial
            unsafe {
                batch_expand_pad::<F>(
                    &g_rs,
                    &g_coeffs,
                    D_EF as u32,
                    codeword_height as u32,
                    f_height as u32,
                )?;

                batch_ntt(
                    &g_rs,
                    (log_rs_domain_size - 1) as u32,
                    0u32,
                    D_EF as u32,
                    true,
                    false,
                );
            }

            let g_tree = MerkleTreeGpu::<F, Digest>::new(
                DeviceMatrix::new(Arc::new(g_rs), codeword_height, D_EF),
                1 << k_whir,
            )?;
            let g_commit = g_tree.root();
            transcript.observe_commit(g_commit);
            codeword_commits.push(g_commit);

            let z_0 = transcript.sample_ext();
            // SAFETY:
            // - `g_coeffs` is coefficient form of `\hat{g}`, which is degree `2^{m-k_whir}`.
            // - `g_coeffs` is F-column major matrix.
            let g_opened_value =
                unsafe { eval_poly_ext_at_point_from_base(&g_coeffs, 1 << (m - k_whir), z_0)? };
            transcript.observe_ext(g_opened_value);
            ood_values.push(g_opened_value);

            (Some(g_tree), Some(z_0))
        } else {
            // Observe the final poly
            debug_assert_eq!(log_final_poly_len, m - k_whir);
            let final_poly_len = 1 << log_final_poly_len;
            let base_coeffs = g_coeffs.to_host()?;
            debug_assert_eq!(base_coeffs.len(), D_EF * final_poly_len);
            let mut coeffs = Vec::with_capacity(final_poly_len);
            for i in 0..final_poly_len {
                let coeff = EF::from_base_fn(|j| base_coeffs[j * final_poly_len + i]);
                transcript.observe_ext(coeff);
                coeffs.push(coeff);
            }
            final_poly = Some(coeffs);
            (None, None)
        };

        // omega is generator of RS domain `\mathcal{L}^{(2^k)}`
        let omega = F::two_adic_generator(log_rs_domain_size - k_whir);
        let mut query_indices = Vec::with_capacity(num_whir_queries);
        whir_pow_witnesses.push(transcript.grind(whir_pow_bits));
        // Sample query indices first
        for _ in 0..num_whir_queries {
            // This is the index of the leaf in the Merkle tree
            let index = transcript.sample_bits(log_rs_domain_size - k_whir);
            query_indices.push(index as usize);
        }
        if !is_last_round {
            codeword_opened_values.push(vec![]);
            codeword_merkle_proofs.push(vec![]);
        }
        if whir_round == 0 {
            let common_main_tree = common_main_tree.take().unwrap();
            let trees = once(&common_main_tree)
                .chain(other_pcs_data.iter().map(|d| &d.tree))
                .collect::<Vec<_>>();
            // Get merkle proofs for in-domain samples necessary to evaluate Fold(f, \vec
            // \alpha)(z_i)
            initial_round_merkle_proofs =
                MerkleTreeGpu::batch_query_merkle_proofs(&trees, &query_indices)?;
            let num_rows_per_query = trees[0].rows_per_query;
            initial_round_opened_rows = MerkleTreeGpu::batch_open_rows(&trees, &query_indices)?
                .into_iter()
                .map(|rows_per_commit| {
                    rows_per_commit
                        .into_iter()
                        .map(|rows| {
                            let width = rows.len() / num_rows_per_query;
                            rows.chunks_exact(width).map(|row| row.to_vec()).collect()
                        })
                        .collect()
                })
                .collect();
        } else {
            let tree: &MerkleTreeGpu<F, Digest> = rs_tree.as_ref().unwrap();
            codeword_merkle_proofs[whir_round - 1] =
                MerkleTreeGpu::batch_query_merkle_proofs(&[tree], &query_indices)?
                    .pop()
                    .expect("exactly 1 tree");
            codeword_opened_values[whir_round - 1] =
                MerkleTreeGpu::batch_open_rows(&[tree], &query_indices)?
                    .pop()
                    .unwrap()
                    .into_iter() // We could transmute `rows` here, but we'll keep it safe for now
                    .map(|rows| rows.chunks_exact(D_EF).map(EF::from_base_slice).collect())
                    .collect();
        }
        rs_tree = g_tree;

        // We still sample on the last round to match the verifier, who uses a
        // final gamma to unify some logic. But we do not need to update
        // `w_evals`.
        let gamma = transcript.sample_ext();

        if !is_last_round {
            // Update \hat{w} to
            // ```test
            // \hat{w}'(x) = \hat{w}(x) + \sum_{i=0..=num_queries} γ^{i+1} * eq(x,pow(z_i))
            // ```
            // where z_0 is the OOD point and z_i = \omega^{query_indices[i]}
            // for i >= 1 are the in-domain points.
            let height = f_height;
            let log_height = m - k_whir;
            debug_assert_eq!(log2_strict_usize(height), log_height);
            // PERF[jpw]: num_whir_queries can be 100-200, so this is a substantial amount of
            // memory. However `height` drops off with whir rounds and we have dropped the big
            // round0 merkle tree already, so this is likely not peak memory.
            let mut eq_zs = DeviceBuffer::<F>::with_capacity(height * num_whir_queries);
            // We keep eq(z_0, -) separate since z_0 is in EF while the other z_i are in F.
            let mut eq_z0 = DeviceBuffer::<EF>::with_capacity(height);
            {
                // Build the eq(-,pow(z_i))
                let mut z_0_pow = z_0.unwrap();
                let mut z_pows = query_indices
                    .iter()
                    .map(|&index| omega.exp_u64(index as u64))
                    .collect_vec();
                // initialize first entry of each column with F::ONE
                [EF::ONE].copy_to(&mut eq_z0)?;
                let one = [F::ONE];
                for j in 0..num_whir_queries {
                    // SAFETY:
                    // - eq_zs is in bounds
                    // - H2D copy
                    unsafe {
                        cuda_memcpy::<false, true>(
                            eq_zs.as_mut_ptr().add(j * height) as *mut c_void,
                            one.as_ptr() as *const c_void,
                            size_of::<F>(),
                        )?;
                    }
                }
                for i in 0..log_height {
                    let step = 1 << i;

                    let d_zs = z_pows.to_device()?;
                    // SAFETY:
                    // - eq_z0 has size height
                    // - eq_zs has size height * num_whir_queries
                    // - d_zs has length num_whir_queries
                    // - step < height
                    unsafe {
                        eq_hypercube_stage_ext(eq_z0.as_mut_ptr(), z_0_pow, step)?;
                        batch_eq_hypercube_stage(&mut eq_zs, &d_zs, step, height as u32)?;
                    }
                    z_0_pow = z_0_pow.square();
                    for z in &mut z_pows {
                        *z = z.square();
                    }
                }
            }
            // SAFETY:
            // - `w_evals` has size `height`
            // - `eq_z0` has size `height`
            // - `eq_zs` has size `height * num_whir_queries`
            unsafe {
                w_evals_accumulate(
                    &mut w_evals,
                    &eq_z0,
                    &eq_zs,
                    gamma,
                    num_whir_queries.try_into().unwrap(),
                )?;
            }
        }

        m -= k_whir;
        log_rs_domain_size -= 1;
    }

    Ok(WhirProof {
        whir_sumcheck_polys,
        codeword_commits,
        ood_values,
        whir_pow_witnesses,
        initial_round_opened_rows,
        initial_round_merkle_proofs,
        codeword_opened_values,
        codeword_merkle_proofs,
        final_poly: final_poly.unwrap(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use itertools::Itertools;
    use openvm_stark_sdk::config::setup_tracing_with_log_level;
    use p3_field::FieldAlgebra;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use stark_backend_v2::{
        BabyBearPoseidon2CpuEngineV2, EF, F, SystemParams,
        keygen::types::MultiStarkProvingKeyV2,
        poly_common::Squarable,
        poseidon2::sponge::{DuplexSponge, DuplexSpongeRecorder, TranscriptHistory},
        prover::{
            ColMajorMatrix, CpuBackendV2, DeviceDataTransporterV2, ProvingContextV2, poly::Ple,
            stacked_pcs::stacked_commit, whir::WhirProver,
        },
        test_utils::{DuplexSpongeValidator, FibFixture, TestFixture},
        verifier::whir::{VerifyWhirError, verify_whir},
    };
    use tracing::Level;

    use crate::GpuDeviceV2;

    fn generate_random_z(params: &SystemParams, rng: &mut StdRng) -> (Vec<EF>, Vec<EF>) {
        let z_prism: Vec<_> = (0..params.n_stack + 1)
            .map(|_| EF::from_wrapped_u64(rng.random()))
            .collect();

        let z_cube = {
            let z_cube = z_prism[0]
                .exp_powers_of_2()
                .take(params.l_skip)
                .chain(z_prism[1..].iter().copied())
                .collect_vec();
            debug_assert_eq!(z_cube.len(), params.n_stack + params.l_skip);
            z_cube
        };

        (z_prism, z_cube)
    }

    fn stacking_openings_for_matrix(
        params: &SystemParams,
        z_prism: &[EF],
        matrix: &ColMajorMatrix<F>,
    ) -> Vec<EF> {
        matrix
            .columns()
            .map(|col| {
                Ple::from_evaluations(params.l_skip, col).eval_at_point(
                    params.l_skip,
                    z_prism[0],
                    &z_prism[1..],
                )
            })
            .collect()
    }

    fn run_whir_test(
        params: SystemParams,
        pk: MultiStarkProvingKeyV2,
        ctx: ProvingContextV2<CpuBackendV2>,
    ) -> Result<(), VerifyWhirError> {
        let device = GpuDeviceV2::new(params);
        let mut rng = StdRng::seed_from_u64(0);
        let (z_prism, z_cube) = generate_random_z(&params, &mut rng);

        let (common_main_commit, common_main_pcs_data) = {
            let traces = ctx
                .common_main_traces()
                .map(|(_, trace)| trace)
                .collect_vec();
            stacked_commit(
                params.l_skip,
                params.n_stack,
                params.log_blowup,
                params.k_whir,
                &traces,
            )
        };
        let d_common_main_pcs_data = device.transport_pcs_data_to_device(&common_main_pcs_data);

        let mut stacking_openings = vec![stacking_openings_for_matrix(
            &params,
            &z_prism,
            &common_main_pcs_data.matrix,
        )];
        let mut commits = vec![common_main_commit];
        let mut pre_cached_pcs_data_per_commit = Vec::new();
        for (air_id, air_ctx) in ctx.per_trace {
            let pcs_datas = pk.per_air[air_id]
                .preprocessed_data
                .iter()
                .chain(air_ctx.cached_mains.iter().map(|cd| &cd.data));
            for data in pcs_datas {
                commits.push(data.commit());
                stacking_openings.push(stacking_openings_for_matrix(
                    &params,
                    &z_prism,
                    &data.matrix,
                ));
                let d_data = device.transport_pcs_data_to_device(data);
                pre_cached_pcs_data_per_commit.push(Arc::new(d_data));
            }
        }

        let mut prover_sponge = DuplexSpongeRecorder::default();

        let proof = device.prove_whir(
            &mut prover_sponge,
            d_common_main_pcs_data,
            pre_cached_pcs_data_per_commit,
            &z_cube,
        );

        let mut verifier_sponge = DuplexSpongeValidator::new(prover_sponge.into_log());
        verify_whir(
            &mut verifier_sponge,
            &params,
            &proof,
            &stacking_openings,
            &commits,
            &z_cube,
        )
    }

    fn run_whir_fib_test(params: SystemParams) -> Result<(), VerifyWhirError> {
        let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);
        let fib = FibFixture::new(0, 1, 1 << (params.n_stack + params.l_skip));
        let (pk, _vk) = fib.keygen(&engine);
        let ctx = fib.generate_proving_ctx();
        run_whir_test(params, pk, ctx)
    }

    #[test]
    fn test_whir_single_fib_nstack_0() -> Result<(), VerifyWhirError> {
        setup_tracing_with_log_level(Level::DEBUG);

        let params = SystemParams {
            l_skip: 2,
            n_stack: 0,
            log_blowup: 1,
            k_whir: 1,
            num_whir_queries: 5,
            log_final_poly_len: 0,
            logup_pow_bits: 1,
            whir_pow_bits: 0,
        };
        run_whir_fib_test(params)
    }

    #[test]
    fn test_whir_single_fib_nstack_2() -> Result<(), VerifyWhirError> {
        setup_tracing_with_log_level(Level::DEBUG);

        let params = SystemParams {
            l_skip: 2,
            n_stack: 2,
            log_blowup: 1,
            k_whir: 1,
            num_whir_queries: 5,
            log_final_poly_len: 2,
            logup_pow_bits: 1,
            whir_pow_bits: 0,
        };
        run_whir_fib_test(params)
    }

    #[test]
    fn test_whir_single_fib_kwhir_2() -> Result<(), VerifyWhirError> {
        setup_tracing_with_log_level(Level::DEBUG);

        let params = SystemParams {
            l_skip: 2,
            n_stack: 2,
            log_blowup: 1,
            k_whir: 2,
            num_whir_queries: 5,
            log_final_poly_len: 0,
            logup_pow_bits: 1,
            whir_pow_bits: 0,
        };
        run_whir_fib_test(params)
    }

    #[test]
    fn test_whir_single_fib_kwhir_3() -> Result<(), VerifyWhirError> {
        setup_tracing_with_log_level(Level::DEBUG);

        let params = SystemParams {
            l_skip: 2,
            n_stack: 2,
            log_blowup: 1,
            k_whir: 3,
            num_whir_queries: 5,
            log_final_poly_len: 1,
            logup_pow_bits: 1,
            whir_pow_bits: 0,
        };
        run_whir_fib_test(params)
    }

    #[test]
    fn test_whir_single_fib_kwhir_4() -> Result<(), VerifyWhirError> {
        setup_tracing_with_log_level(Level::DEBUG);

        let params = SystemParams {
            l_skip: 2,
            n_stack: 2,
            log_blowup: 1,
            k_whir: 4,
            num_whir_queries: 5,
            log_final_poly_len: 0,
            logup_pow_bits: 1,
            whir_pow_bits: 0,
        };
        run_whir_fib_test(params)
    }

    #[test]
    fn test_whir_single_fib_log_blowup_2() -> Result<(), VerifyWhirError> {
        setup_tracing_with_log_level(Level::DEBUG);

        let params = SystemParams {
            l_skip: 2,
            n_stack: 2,
            log_blowup: 1,
            k_whir: 4,
            num_whir_queries: 5,
            log_final_poly_len: 0,
            logup_pow_bits: 2,
            whir_pow_bits: 0,
        };
        run_whir_fib_test(params)
    }

    // TODO: test multiple commitments
}
