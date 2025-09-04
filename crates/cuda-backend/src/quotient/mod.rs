use itertools::{multiunzip, Itertools};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::{
    air_builders::symbolic::SymbolicConstraintsDag,
    config::Domain,
    prover::{hal::MatrixDimensions, types::RapView},
};
use p3_commit::PolynomialSpace;
use tracing::instrument;

use self::single::compute_single_rap_quotient_values_gpu;
use crate::{
    base::{DeviceMatrix, DevicePoly, ExtendedLagrangeCoeff},
    cuda::kernels::matrix::split_ext_poly_to_multiple_base_matrix,
    gpu_device::GpuDevice,
    prelude::*,
};

pub(crate) mod single;

pub struct QuotientCommitterGpu {
    alpha: EF,
}

impl QuotientCommitterGpu {
    pub fn new(alpha: EF) -> Self {
        Self { alpha }
    }
}

impl QuotientCommitterGpu {
    /// "GPU quotient should process one RAP at a time using `single_rap_quotient_values`, to
    /// prevent GPU OOM)"
    // @dev: view_gpu is the extended view of the evaluation on quotient domain (subset of LDE
    // domain), which is always larger than trace domain
    #[instrument(name = "compute single RAP quotient values on gpu", level = "debug", skip_all, fields(
        quotient_domain_size = (1usize << view_gpu.log_trace_height) * (quotient_degree as usize),
        num_constraints = constraints.constraints.num_constraints()
    ))]
    pub(super) fn single_rap_quotient_values(
        &self,
        device: &GpuDevice,
        constraints: &SymbolicConstraintsDag<F>,
        view_gpu: RapView<DeviceMatrix<F>, F, EF>,
        quotient_degree: u8,
    ) -> SingleQuotientDataGpu {
        let log_trace_height = view_gpu.log_trace_height;
        let trace_domain = device.natural_domain_for_degree(1usize << log_trace_height);
        let quotient_degree = quotient_degree as usize;
        let quotient_domain =
            trace_domain.create_disjoint_domain(trace_domain.size() * quotient_degree);

        // run quotient evaluator on GPU
        let (after_challenge_lde_on_quotient_domain, challenges, perm_challenge): (
            Vec<_>,
            Vec<_>,
            Vec<_>,
        ) = multiunzip(view_gpu.per_phase.into_iter().map(|view| {
            (
                view.inner
                    .expect("gap in challenge phase not supported yet"),
                view.challenges,
                view.exposed_values,
            )
        }));

        let quotient_values = compute_single_rap_quotient_values_gpu(
            constraints,
            trace_domain,
            quotient_domain,
            view_gpu.preprocessed,
            view_gpu.partitioned_main,
            after_challenge_lde_on_quotient_domain,
            challenges,
            self.alpha,
            &view_gpu.public_values,
            perm_challenge
                .iter()
                .map(|v| v.as_slice())
                .collect::<Vec<_>>()
                .as_slice(),
        );

        SingleQuotientDataGpu {
            quotient_degree,
            quotient_domain,
            quotient_values,
        }
    }
}

/// The quotient polynomials from multiple RAP matrices.
pub(crate) struct QuotientDataGpu {
    pub(crate) inner: Vec<SingleQuotientDataGpu>,
}

impl QuotientDataGpu {
    /// Splits the quotient polynomials from multiple AIRs into chunks of size equal to the trace
    /// domain size.
    #[instrument(name = "split quotient data", level = "debug", skip_all)]
    pub fn split(self) -> impl IntoIterator<Item = QuotientChunkGpu> {
        self.inner.into_iter().flat_map(|data| data.split())
    }
}

/// The quotient polynomial from a single matrix RAP, evaluated on the quotient domain.
pub(crate) struct SingleQuotientDataGpu {
    pub(crate) quotient_degree: usize,
    /// Quotient domain
    pub(crate) quotient_domain: Domain<SC>,
    /// Evaluations of the quotient polynomial on the quotient domain
    pub(crate) quotient_values: DevicePoly<EF, ExtendedLagrangeCoeff>,
}

impl SingleQuotientDataGpu {
    /// The vector of evaluations of the quotient polynomial on the quotient domain,
    /// first flattened from vector of extension field elements to matrix of base field elements,
    /// and then split into chunks of size equal to the trace domain size (quotient domain size
    /// divided by `quotient_degree`).
    pub fn split(self) -> impl IntoIterator<Item = QuotientChunkGpu> {
        let quotient_degree = self.quotient_degree;
        let quotient_domain = self.quotient_domain;

        // Flatten from extension field elements to base field elements
        //   input: single extension field polynomial
        //   output: multiple base field matrices (col-major)
        let quotient_chunks: Vec<_> =
            self.split_ext_poly_into_base_matrices(&self.quotient_values, quotient_degree);
        let qc_domains = quotient_domain.split_domains(quotient_degree);
        qc_domains
            .into_iter()
            .zip_eq(quotient_chunks)
            .map(|(domain, chunk)| QuotientChunkGpu { domain, chunk })
    }

    fn split_ext_poly_into_base_matrices(
        &self,
        ext_poly: &DevicePoly<EF, ExtendedLagrangeCoeff>,
        num_chunks: usize,
    ) -> Vec<DeviceMatrix<F>> {
        let matrices = (0..num_chunks)
            .map(|matrix_idx| {
                // plonky3/matrix/src/lib.rs: fn vertically_strided
                let height = ext_poly.len();
                let full_strides = height / num_chunks;
                let remainder = height % num_chunks;
                let final_stride = matrix_idx < remainder;
                let matrix_height = full_strides + final_stride as usize;
                let matrix_width = 4; // EF::D
                tracing::debug!(
                    "matrix {}: width = {}, height = {}",
                    matrix_idx,
                    matrix_width,
                    matrix_height
                );
                DeviceMatrix::<F>::with_capacity(matrix_height, matrix_width)
            })
            .collect::<Vec<_>>();

        assert_eq!(
            matrices.iter().map(|m| m.height()).sum::<usize>(),
            ext_poly.len()
        );

        // DevicePointer<u8>: size = 8, align = 0x8
        // Allocate `u64` buffer to store matrices_ptr,
        let matrices_ptr = matrices
            .iter()
            .map(|m: &DeviceMatrix<F>| m.buffer().as_ptr() as u64)
            .collect::<Vec<_>>();

        let matrices_ptr_buf = matrices_ptr.to_device().unwrap();

        unsafe {
            split_ext_poly_to_multiple_base_matrix(
                &matrices_ptr_buf,
                &ext_poly.coeff,
                ext_poly.len() as u64,
                num_chunks as u64,
            )
            .unwrap();
        }

        matrices
    }
}

/// The vector of evaluations of the quotient polynomial on the quotient domain,
/// split into chunks of size equal to the trace domain size (quotient domain size
/// divided by `quotient_degree`).
///
/// This represents a single chunk, where the vector of extension field elements is
/// further flattened to a matrix of base field elements.
pub struct QuotientChunkGpu {
    /// Chunk of quotient domain, which is a coset of the trace domain
    pub domain: Domain<SC>,
    /// Matrix with number of rows equal to trace domain size,
    /// and number of columns equal to extension field degree.
    pub chunk: DeviceMatrix<F>,
}
