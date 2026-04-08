use std::{
    collections::BTreeSet,
    sync::{Mutex, OnceLock},
};

use openvm_cuda_common::{
    common::{device_reset_epoch, get_device},
    d_buffer::DeviceBuffer,
    stream::DeviceContext,
};

use crate::{cuda::ntt, prelude::F};

const MAX_LG_DOMAIN_SIZE: usize = ntt::MAX_CUDA_NTT_LOG_DOMAIN_SIZE as usize;
const LG_WINDOW_SIZE: usize = MAX_LG_DOMAIN_SIZE.div_ceil(5);
const WINDOW_SIZE: usize = 1 << LG_WINDOW_SIZE;
const WINDOW_NUM: usize = MAX_LG_DOMAIN_SIZE.div_ceil(LG_WINDOW_SIZE);
const RADIX_TWIDDLES_SIZE: usize = 32 + 64 + 128 + 256 + 512;

static INIT_FORWARD: OnceLock<Mutex<BTreeSet<(i32, u64)>>> = OnceLock::new();
static INIT_INVERSE: OnceLock<Mutex<BTreeSet<(i32, u64)>>> = OnceLock::new();

fn ensure_initialized(inverse: bool) -> Result<(), openvm_cuda_common::error::CudaError> {
    let device_key = (get_device()?, device_reset_epoch());
    let initialized = if inverse {
        &INIT_INVERSE
    } else {
        &INIT_FORWARD
    };
    let initialized = initialized.get_or_init(|| Mutex::new(BTreeSet::new()));
    let mut initialized = initialized.lock().unwrap();
    if initialized.contains(&device_key) {
        return Ok(());
    }

    {
        let device_ctx = DeviceContext::for_device(device_key.0 as u32)?;
        let partial_twiddles =
            DeviceBuffer::<[F; WINDOW_SIZE]>::with_capacity_on(WINDOW_NUM, &device_ctx);
        let twiddles = DeviceBuffer::<F>::with_capacity_on(RADIX_TWIDDLES_SIZE, &device_ctx);
        unsafe {
            // Both CUDA helpers upload into constant memory and synchronize before returning, so
            // these staging buffers are safe to free once the calls complete.
            ntt::generate_all_twiddles(&twiddles, inverse, device_ctx.stream.as_raw())?;
            ntt::generate_partial_twiddles(&partial_twiddles, inverse, device_ctx.stream.as_raw())?;
        }
    }
    initialized.insert(device_key);
    Ok(())
}

struct NttImpl<'a> {
    buffer: &'a DeviceBuffer<F>,
    lg_domain_size: u32,
    padded_poly_size: u32,
    poly_count: u32,
    is_intt: bool,
    stage: u32,
    device_ctx: DeviceContext,
}

impl<'a> NttImpl<'a> {
    fn new(
        buffer: &'a DeviceBuffer<F>,
        lg_domain_size: u32,
        padded_poly_size: u32,
        poly_count: u32,
        is_intt: bool,
        device_ctx: &DeviceContext,
    ) -> Self {
        ensure_initialized(is_intt).expect("failed to initialize CUDA NTT twiddle tables");
        Self {
            buffer,
            lg_domain_size,
            padded_poly_size,
            poly_count,
            is_intt,
            stage: 0,
            device_ctx: device_ctx.clone(),
        }
    }

    fn step(&mut self, iterations: u32) {
        assert!(iterations <= 10);
        let radix = if iterations < 6 { 6 } else { iterations };
        unsafe {
            ntt::ct_mixed_radix_narrow(
                self.buffer,
                radix,
                self.lg_domain_size,
                self.stage,
                iterations,
                self.padded_poly_size,
                self.poly_count,
                self.is_intt,
                self.device_ctx.stream.as_raw(),
            )
            .expect("failed to launch CUDA mixed-radix NTT step");
        }
        self.stage += iterations;
    }
}

/// Performs column-wise batch NTT on `buffer`, where `buffer` is assumed to be column-major with
/// columns of height `2^(log_trace_height + log_blowup)`. The NTT are performed on the first
/// `2^log_trace_height` elements of each column. If `bit_reverse` is true, then the input columns
/// are assumed to be ordered in **natural** ordering, and a bit-reversal permutation is applied for
/// the internal algorithm of the NTT. If `bit_reverse` is false, then the input columns are assumed
/// to be in bit-reverse ordering. If `is_intt` is true, the inverse NTT is performed; otherwise,
/// the forward NTT is performed.
pub fn batch_ntt(
    buffer: &DeviceBuffer<F>,
    log_trace_height: u32,
    log_blowup: u32,
    width: u32,
    bit_reverse: bool,
    is_intt: bool,
    device_ctx: &DeviceContext,
) {
    if log_trace_height == 0 {
        return;
    }
    assert!(
        log_trace_height <= ntt::MAX_CUDA_NTT_LOG_DOMAIN_SIZE,
        "CUDA batch_ntt supports log_trace_height <= {}",
        ntt::MAX_CUDA_NTT_LOG_DOMAIN_SIZE
    );

    let padded_poly_size = 1 << (log_trace_height + log_blowup);

    if bit_reverse {
        unsafe {
            ntt::bit_rev(
                buffer,
                buffer,
                log_trace_height,
                padded_poly_size,
                width,
                device_ctx.stream.as_raw(),
            )
            .expect("failed to launch CUDA bit-reversal permutation");
        }
    }

    let mut _impl = NttImpl::new(
        buffer,
        log_trace_height,
        padded_poly_size,
        width,
        is_intt,
        device_ctx,
    );
    if log_trace_height <= 10 {
        _impl.step(log_trace_height);
    } else if log_trace_height <= 17 {
        let step = log_trace_height / 2;
        _impl.step(step + log_trace_height % 2);
        _impl.step(step);
    } else if log_trace_height <= ntt::MAX_CUDA_NTT_LOG_DOMAIN_SIZE {
        let step = log_trace_height / 3;
        let rem = log_trace_height % 3;
        _impl.step(step);
        _impl.step(step);
        _impl.step(step + rem);
    } else {
        unreachable!("log_trace_height is bounded above");
    }
}
