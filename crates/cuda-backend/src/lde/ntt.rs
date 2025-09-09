use std::sync::OnceLock;

use openvm_cuda_common::d_buffer::DeviceBuffer;

use crate::{cuda::ntt, prelude::F};

const MAX_LG_DOMAIN_SIZE: usize = 27;
const LG_WINDOW_SIZE: usize = (MAX_LG_DOMAIN_SIZE + 4) / 5;
const WINDOW_SIZE: usize = 1 << LG_WINDOW_SIZE;
const WINDOW_NUM: usize = MAX_LG_DOMAIN_SIZE.div_ceil(LG_WINDOW_SIZE);

struct NttParameters {
    twiddles: DeviceBuffer<F>,
    partial_twiddles: DeviceBuffer<[F; WINDOW_SIZE]>,
}

impl NttParameters {
    fn new(inverse: bool) -> Self {
        let partial_twiddles = DeviceBuffer::<[F; WINDOW_SIZE]>::with_capacity(WINDOW_NUM);
        let twiddles = DeviceBuffer::<F>::with_capacity(32 + 64 + 128 + 256 + 512);

        unsafe {
            ntt::generate_all_twiddles(&twiddles, inverse).unwrap();
            ntt::generate_partial_twiddles(&partial_twiddles, inverse).unwrap();
        }
        Self {
            twiddles,
            partial_twiddles,
        }
    }
}

static FORWARD: OnceLock<NttParameters> = OnceLock::new();
static INVERSE: OnceLock<NttParameters> = OnceLock::new();

#[inline]
fn forward_params() -> &'static NttParameters {
    FORWARD.get_or_init(|| NttParameters::new(false))
}

#[inline]
fn inverse_params() -> &'static NttParameters {
    INVERSE.get_or_init(|| NttParameters::new(true))
}

struct NttImpl<'a> {
    buffer: &'a DeviceBuffer<F>,
    lg_domain_size: u32,
    padded_poly_size: u32,
    poly_count: u32,
    is_intt: bool,
    stage: u32,
    ntt_parameters: &'static NttParameters,
}

impl<'a> NttImpl<'a> {
    fn new(
        buffer: &'a DeviceBuffer<F>,
        lg_domain_size: u32,
        padded_poly_size: u32,
        poly_count: u32,
        is_intt: bool,
    ) -> Self {
        let ntt_parameters = if is_intt {
            inverse_params()
        } else {
            forward_params()
        };
        Self {
            buffer,
            lg_domain_size,
            padded_poly_size,
            poly_count,
            is_intt,
            stage: 0,
            ntt_parameters,
        }
    }

    fn step(&mut self, iterations: u32) {
        assert!(iterations <= 10);
        let radix = if iterations < 6 { 6 } else { iterations };
        let twiddles_offset = (1 << (radix - 6)) << 5;
        unsafe {
            ntt::ct_mixed_radix_narrow(
                self.buffer,
                radix,
                self.lg_domain_size,
                self.stage,
                iterations,
                self.padded_poly_size,
                self.poly_count,
                &self.ntt_parameters.partial_twiddles,
                &self.ntt_parameters.twiddles,
                twiddles_offset,
                self.is_intt,
            )
            .unwrap();
        }
        self.stage += iterations;
    }
}

pub(super) fn batch_ntt(
    buffer: &DeviceBuffer<F>,
    log_trace_height: u32,
    log_blowup: u32,
    width: u32,
    bit_reverse: bool,
    is_intt: bool,
) {
    let padded_poly_size = 1 << (log_trace_height + log_blowup);

    if bit_reverse {
        unsafe {
            ntt::bit_rev(buffer, buffer, log_trace_height, padded_poly_size, width).unwrap();
        }
    }

    let mut _impl = NttImpl::new(buffer, log_trace_height, padded_poly_size, width, is_intt);
    if log_trace_height <= 10 {
        _impl.step(log_trace_height);
    } else if log_trace_height <= 17 {
        let step = log_trace_height / 2;
        _impl.step(step + log_trace_height % 2);
        _impl.step(step);
    } else if log_trace_height <= 30 {
        let step = log_trace_height / 3;
        let rem = log_trace_height % 3;
        _impl.step(step);
        _impl.step(step + (if log_trace_height == 29 { 1 } else { 0 }));
        _impl.step(step + (if log_trace_height == 29 { 1 } else { rem }));
    } else if log_trace_height <= 40 {
        let step = log_trace_height / 4;
        let rem = log_trace_height % 4;
        _impl.step(step);
        _impl.step(step + (if rem > 2 { 1 } else { 0 }));
        _impl.step(step + (if rem > 1 { 1 } else { 0 }));
        _impl.step(step + (if rem > 0 { 1 } else { 0 }));
    } else {
        panic!("log_trace_height > 40");
    }
}
