use std::sync::Arc;

use getset::Getters;
use openvm_metal_common::{
    copy::{MemCopyD2D, MemCopyH2D},
    d_buffer::MetalBuffer,
    error::MetalError,
};
use openvm_stark_backend::prover::MatrixDimensions;
use p3_field::PrimeCharacteristicRing;

use crate::{
    base::MetalMatrix,
    metal::{
        batch_ntt_small::batch_ntt_small,
        mle_interpolate::{
            mle_interpolate_fused_2d, mle_interpolate_shared_2d, mle_interpolate_stage_2d,
            MLE_SHARED_TILE_LOG_SIZE,
        },
        poly::{
            eq_hypercube_interleaved_stage_ext, eq_hypercube_nonoverlapping_stage_ext,
            eq_hypercube_stage_ext, mobius_eq_hypercube_stage_ext,
        },
        sumcheck::fold_mle_column,
        LOG_SIMD_SIZE,
    },
    prelude::{EF, F},
    KernelError,
};

#[derive(derive_new::new, Getters)]
pub struct PleMatrix<F> {
    /// Stores the coefficient form of the univariate polynomial `f(Z, \vect x)` for each `\vect x
    /// in H_n`. Buffer size is the same as `evals`.
    #[getset(get = "pub")]
    pub(crate) mixed: MetalBuffer<F>,
    height: usize,
    width: usize,
}

impl<F> MatrixDimensions for PleMatrix<F> {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }
}

impl PleMatrix<F> {
    /// Creates a `PleMatrix`. This doubles the VRAM footprint to cache the `mixed` buffer.
    pub fn from_evals(l_skip: usize, evals: MetalBuffer<F>, height: usize, width: usize) -> Self {
        let mut mixed = evals;
        if l_skip > 0 {
            // For univariate coordinate, perform inverse NTT for each 2^l_skip chunk per column:
            // (width cols) * (height / 2^l_skip chunks per col). Use natural ordering.
            let num_uni_poly = width * (height >> l_skip);
            unsafe {
                batch_ntt_small(&mut mixed, l_skip, num_uni_poly, true).unwrap();
            }
        }
        Self {
            mixed,
            height,
            width,
        }
    }

    pub fn to_evals(&self, l_skip: usize) -> Result<MetalMatrix<F>, KernelError> {
        let width = self.width();
        let height = self.height();
        // D2D copy so we can do in-place NTT
        let mut evals = self.mixed.device_copy();
        if l_skip > 0 {
            // For univariate coordinate, perform NTT for each 2^l_skip chunk per column: (width
            // cols) * (height / 2^l_skip chunks per col). Use natural ordering.
            let num_uni_poly = width * (height >> l_skip);
            unsafe {
                batch_ntt_small(&mut evals, l_skip, num_uni_poly, false).unwrap();
            }
        }
        Ok(MetalMatrix::new(Arc::new(evals), height, width))
    }
}

/// Assumes that `evals` is column-major matrix of evaluations on a hypercube `H_n`.
/// In-place interpolates `evals` from evaluations to coefficient form.
pub fn mle_evals_to_coeffs_inplace(evals: &mut MetalBuffer<F>, n: usize) -> Result<(), MetalError> {
    if n == 0 {
        return Ok(());
    }
    debug_assert!(evals.len().is_multiple_of(1 << n));
    let width = evals.len() >> n;
    unsafe {
        mle_interpolate_stages(
            evals,
            width.try_into().unwrap(),
            1 << n,
            0,
            0,
            n as u32 - 1,
            true,
            false,
        )
    }
}

/// Helper function to process MLE interpolation stages from `start_log_step` to `end_log_step`
/// (both inclusive).
///
/// Automatically selects the best kernel(s):
/// - SIMD shuffle for steps 1-16 (log_step 0-4) when 2+ stages can be fused
/// - Shared memory for steps up to 2048 (log_step up to 11)
/// - Fallback individual kernels for larger steps
///
/// # Parameters
/// - `log_blowup`: meaningful_count = padded_height >> log_blowup
/// - `bit_reversed`: if true, physical index = tidx << log_blowup (strided access) if false,
///   physical index = tidx (contiguous access)
///
/// # Safety
/// Same requirements as the individual kernel functions.
#[allow(clippy::too_many_arguments)]
pub unsafe fn mle_interpolate_stages(
    buffer: &mut MetalBuffer<F>,
    width: u16,
    padded_height: u32,
    log_blowup: u32,
    start_log_step: u32,
    end_log_step: u32,
    is_eval_to_coeff: bool,
    right_pad: bool,
) -> Result<(), MetalError> {
    if start_log_step > end_log_step {
        return Ok(());
    }

    let mut current_log_step = start_log_step;

    // Phase 1: Use SIMD shuffle for steps where log_step < LOG_SIMD_SIZE (step <= 16)
    // Only if we can fuse at least 2 stages (otherwise shared memory is just as good)
    let simd_end = end_log_step.min(LOG_SIMD_SIZE as u32 - 1);
    let simd_stages = simd_end.saturating_sub(current_log_step) + 1;
    if current_log_step < LOG_SIMD_SIZE as u32 && simd_stages >= 2 {
        let num_stages = simd_stages;
        let start_step = 1u32 << current_log_step;

        mle_interpolate_fused_2d(
            buffer,
            width,
            padded_height,
            log_blowup,
            start_step,
            num_stages,
            is_eval_to_coeff,
            right_pad,
        )?;

        current_log_step = simd_end + 1;
    }

    if current_log_step > end_log_step {
        return Ok(());
    }

    // Phase 2: Use shared memory for steps where log_step < MLE_SHARED_TILE_LOG_SIZE
    if current_log_step < MLE_SHARED_TILE_LOG_SIZE {
        let shared_end = end_log_step.min(MLE_SHARED_TILE_LOG_SIZE - 1);

        mle_interpolate_shared_2d(
            buffer,
            width,
            padded_height,
            log_blowup,
            current_log_step,
            shared_end,
            is_eval_to_coeff,
            right_pad,
        )?;

        current_log_step = shared_end + 1;
    }

    // Phase 3: Fallback to individual kernel launches for very large steps
    assert!(
        current_log_step > end_log_step || !right_pad,
        "bit_reversed mode not supported for log_step >= MLE_SHARED_TILE_LOG_SIZE"
    );
    let height = padded_height >> log_blowup;
    while current_log_step <= end_log_step {
        let step = 1u32 << current_log_step;
        mle_interpolate_stage_2d(buffer, width, height, padded_height, step, is_eval_to_coeff)?;
        current_log_step += 1;
    }

    Ok(())
}

/// Given vector `x` in `F^n`, populates `out` with `eq_n(x, y)` for `y` on hypercube `H_n`.
///
/// Note: This function launches `n` Metal kernels.
pub unsafe fn evals_eq_hypercube(out: &mut MetalBuffer<EF>, xs: &[EF]) -> Result<(), KernelError> {
    let n = xs.len();
    assert!(out.len() >= 1 << n);
    // Use memcpy instead of memset since EF will be in Montgomery form.
    [EF::ONE].copy_to(out).map_err(KernelError::MemCopy)?;

    for (i, &x_i) in xs.iter().enumerate() {
        let step = 1 << i;
        eq_hypercube_stage_ext(out, 0, x_i, step as u32).map_err(KernelError::Kernel)?;
    }
    Ok(())
}

/// Given vector `u_tilde` in `F^n`, populates `out` with `mobius_eq(u_tilde, y)` for `y` on
/// hypercube `H_n`.
pub unsafe fn evals_mobius_eq_hypercube(
    out: &mut MetalBuffer<EF>,
    omega: &[EF],
) -> Result<(), KernelError> {
    let n = omega.len();
    assert!(out.len() >= 1 << n);
    // Use memcpy instead of memset since EF will be in Montgomery form.
    [EF::ONE].copy_to(out).map_err(KernelError::MemCopy)?;

    for (i, &omega_i) in omega.iter().enumerate() {
        let step = 1 << i;
        mobius_eq_hypercube_stage_ext(out, 0, omega_i, step as u32).map_err(KernelError::Kernel)?;
    }
    Ok(())
}

/// For a fixed `x` in `F^max_n`, stores the evaluations of `eq_n(x[..n], -)` on hypercube `H_n` for
/// `n = 0..=max_n`. The evaluations are stored contiguously in a single buffer of length `2^(max_n
/// + 1)`. We define `eq_0 = 1` always.
#[derive(Getters)]
pub struct EqEvalSegments<F> {
    /// Index 0 should never to read, but it will be initialized to zero.
    #[getset(get = "pub")]
    pub(crate) buffer: MetalBuffer<F>,
    #[getset(get_copy = "pub")]
    max_n: usize,
}

impl<F> EqEvalSegments<F> {
    /// Returns start pointer for buffer of length `2^n` corresponding to evaluations of `eq(x[..n],
    /// -)` on `H_n`.
    pub fn get_ptr(&self, n: usize) -> *const F {
        assert!(n <= self.max_n);
        unsafe { self.buffer.as_ptr().add(1 << n) }
    }

    /// Returns a GPU pointer to evaluations of `eq(x[..n], -)` on `H_n`.
    pub fn get_device_ptr(&self, n: usize) -> *const F {
        assert!(n <= self.max_n);
        unsafe { self.buffer.as_device_ptr().add(1 << n) }
    }

    /// # Safety
    /// Caller must ensure that `buffer` has length `2^{max_n + 1}` and is properly initialized.
    pub unsafe fn from_raw_parts(buffer: MetalBuffer<F>, max_n: usize) -> Self {
        Self { buffer, max_n }
    }
}

// Currently only implement kernels for EF.
impl EqEvalSegments<EF> {
    /// Creates a new `EqEvalSegments` instance with `max_n = x.len()`.
    pub fn new(x: &[EF]) -> Result<Self, KernelError> {
        let max_n = x.len();
        let mut buffer = MetalBuffer::with_capacity(2 << max_n);
        // Index 0 should never to be used, but we initialize it to zero.
        // Index 1 is set to eq_0 = 1 for initial state
        [EF::ZERO, EF::ONE]
            .copy_to(&mut buffer)
            .map_err(KernelError::MemCopy)?;
        // At step i, we populate `eq_{i+1}` starting at offset `2^{i+1}`
        for (i, &x_i) in x.iter().enumerate() {
            let step = 1 << i;
            unsafe {
                eq_hypercube_nonoverlapping_stage_ext(
                    &buffer,
                    2 * step,
                    &buffer,
                    step,
                    x_i,
                    step as u32,
                )
            }
            .map_err(KernelError::Kernel)?;
        }
        Ok(Self { buffer, max_n })
    }
}

/// Same as [EqEvalSegments] but keeping segment tree with buffers separated by layer to allow
/// dropping layers.
#[derive(Getters)]
pub struct EqEvalLayers<F> {
    /// Index 0 should never to read, but it will be initialized to zero.
    pub layers: Vec<MetalBuffer<F>>,
}

impl<F> EqEvalLayers<F> {
    /// Returns start pointer for buffer of length `2^n` corresponding to evaluations of `eq(x[..n],
    /// -)` on `H_n`.
    pub fn get_ptr(&self, n: usize) -> *const F {
        debug_assert_eq!(self.layers[n].len(), 1 << n);
        self.layers[n].as_ptr()
    }

    /// Returns a GPU pointer to evaluations of `eq(x[..n], -)` on `H_n`.
    pub fn get_device_ptr(&self, n: usize) -> *const F {
        debug_assert_eq!(self.layers[n].len(), 1 << n);
        self.layers[n].as_device_ptr()
    }
}

// Currently only implement kernels for EF.
impl EqEvalLayers<EF> {
    /// Creates a new `EqEvalLayers` instance with `layers.len() = x.len() + 1`.
    ///
    /// Inserts `x_i` from the front for each layer.
    pub fn new_rev<'a>(n: usize, x: impl IntoIterator<Item = &'a EF>) -> Result<Self, KernelError> {
        let mut layers = Vec::with_capacity(n + 1);
        let layer_0 = [EF::ONE].to_device();
        layers.push(layer_0);
        for (i, &x_i) in x.into_iter().enumerate() {
            let step = 1 << i;
            let buffer = MetalBuffer::with_capacity(2 * step);
            let src = layers.last().unwrap();
            unsafe { eq_hypercube_interleaved_stage_ext(&buffer, 0, src, 0, x_i, step as u32) }
                .map_err(KernelError::Kernel)?;
            layers.push(buffer);
        }
        Ok(Self { layers })
    }

    /// Creates a new `EqEvalLayers` instance with `layers.len() = x.len() + 1`.
    ///
    /// Inserts `x_i` from the back for each layer. This matches behavior of
    /// [`EqEvalSegments::new`].
    pub fn new<'a>(n: usize, x: impl IntoIterator<Item = &'a EF>) -> Result<Self, KernelError> {
        let mut layers = Vec::with_capacity(n + 1);
        let layer_0 = [EF::ONE].to_device();
        layers.push(layer_0);
        for (i, &x_i) in x.into_iter().enumerate() {
            let step = 1 << i;
            let buffer = MetalBuffer::with_capacity(2 * step);
            let src = layers.last().unwrap();
            unsafe { eq_hypercube_nonoverlapping_stage_ext(&buffer, 0, src, 0, x_i, step as u32) }
                .map_err(KernelError::Kernel)?;
            layers.push(buffer);
        }
        Ok(Self { layers })
    }
}

/// Square-root decomposition of hypercube equality buffer for memory optimization.
pub struct SqrtHyperBuffer {
    pub low: MetalBuffer<EF>,
    pub high: MetalBuffer<EF>,
    pub low_capacity: usize,
    pub size: usize,
}

impl SqrtHyperBuffer {
    /// Build a buffer from `xi`. Note that last elements of `xi` correspond to the lowest index
    /// bits.
    pub fn from_xi(xi: &[EF]) -> Result<Self, KernelError> {
        let low = {
            let mut res = MetalBuffer::with_capacity(1 << (xi.len() / 2));
            unsafe { evals_eq_hypercube(&mut res, &xi[..xi.len() / 2])? };
            res
        };
        let high = {
            let mut res = MetalBuffer::with_capacity(1 << xi.len().div_ceil(2));
            unsafe { evals_eq_hypercube(&mut res, &xi[xi.len() / 2..])? };
            res
        };
        Ok(Self {
            low,
            high,
            low_capacity: 1 << (xi.len() / 2),
            size: 1 << xi.len(),
        })
    }

    pub fn fold_columns(&mut self, r: EF) -> Result<(), MetalError> {
        assert!(self.size > 1);
        if self.size > self.low_capacity {
            unsafe {
                fold_mle_column(&mut self.high, self.size / self.low_capacity, r)?;
            }
        } else {
            unsafe {
                fold_mle_column(&mut self.low, self.size, r)?;
            }
        };
        self.size /= 2;
        Ok(())
    }
}

/// Square-root decomposition using pre-computed eq evaluation layers.
pub struct SqrtEqLayers {
    /// Layers for `xi[(n + 1) / 2..]`.
    pub low: EqEvalLayers<EF>,
    /// Layers for `xi[..(n + 1) / 2]`.
    pub high: EqEvalLayers<EF>,
}

impl SqrtEqLayers {
    /// Build layers from `xi` values.
    pub fn from_xi(xi: &[EF]) -> Result<Self, KernelError> {
        let n = xi.len();
        let low_n = n / 2;
        let high_n = n - low_n;

        let low = EqEvalLayers::new(low_n, xi[high_n..].iter().rev())?;
        let high = EqEvalLayers::new(high_n, xi[..high_n].iter().rev())?;

        Ok(Self { low, high })
    }

    pub fn max_n(&self) -> usize {
        self.low_n() + self.high_n()
    }

    pub fn low_n(&self) -> usize {
        self.low.layers.len() - 1
    }

    pub fn high_n(&self) -> usize {
        self.high.layers.len() - 1
    }

    /// Drop the highest layer. Drops from high first, then low.
    pub fn drop_layer(&mut self) {
        if self.high.layers.len() > 1 {
            self.high.layers.pop();
        } else if self.low.layers.len() > 1 {
            self.low.layers.pop();
        }
    }
}
