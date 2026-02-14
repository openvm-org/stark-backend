use std::sync::Once;

use openvm_cuda_common::d_buffer::DeviceBuffer;

use crate::{cuda::ntt, prelude::F};

const MAX_LG_DOMAIN_SIZE: usize = 27;
const LG_WINDOW_SIZE: usize = MAX_LG_DOMAIN_SIZE.div_ceil(5);
const WINDOW_SIZE: usize = 1 << LG_WINDOW_SIZE;
const WINDOW_NUM: usize = MAX_LG_DOMAIN_SIZE.div_ceil(LG_WINDOW_SIZE);

static INIT_FORWARD: Once = Once::new();
static INIT_INVERSE: Once = Once::new();

fn ensure_initialized(inverse: bool) {
    let once = if inverse {
        &INIT_INVERSE
    } else {
        &INIT_FORWARD
    };

    once.call_once(|| {
        let partial_twiddles = DeviceBuffer::<[F; WINDOW_SIZE]>::with_capacity(WINDOW_NUM);
        let twiddles = DeviceBuffer::<F>::with_capacity(32 + 64 + 128 + 256 + 512);
        unsafe {
            ntt::generate_all_twiddles(&twiddles, inverse).unwrap();
            ntt::generate_partial_twiddles(&partial_twiddles, inverse).unwrap();
        }
    });
}

struct NttImpl<'a> {
    buffer: &'a DeviceBuffer<F>,
    lg_domain_size: u32,
    padded_poly_size: u32,
    poly_count: u32,
    is_intt: bool,
    stage: u32,
}

impl<'a> NttImpl<'a> {
    fn new(
        buffer: &'a DeviceBuffer<F>,
        lg_domain_size: u32,
        padded_poly_size: u32,
        poly_count: u32,
        is_intt: bool,
    ) -> Self {
        ensure_initialized(is_intt);
        Self {
            buffer,
            lg_domain_size,
            padded_poly_size,
            poly_count,
            is_intt,
            stage: 0,
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
            )
            .unwrap();
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
) {
    if log_trace_height == 0 {
        return;
    }

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
        panic!("log_trace_height > 40 not supported");
    }
}
