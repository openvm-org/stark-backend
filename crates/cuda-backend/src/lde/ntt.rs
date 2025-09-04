
use crate::cuda::DeviceBuffer;

pub(crate)struct NttParameters {
    twiddles: DeviceBuffer<F>,
    partial_twiddles: DeviceBuffer<[F; WINDOW_SIZE]>,
}

impl NttParameters {
    pub(crate) fn new(inverse: bool) -> Self {
        let partial_twiddles = DeviceBuffer::<F>::with_capacity(WINDOW_NUM * WINDOW_SIZE);
        let twiddles = DeviceBuffer::<F>::with_capacity(32 + 64 + 128 + 256 + 512);

        unsafe {
            ntt::generate_all_twiddles(&twiddles, inverse).unwrap();
            ntt::generate_partial_twiddles(&partial_twiddles, inverse).unwrap();
        }
    }
}

pub(crate) struct AllNttParameters {
    forward: NttParameters,
    inverse: NttParameters,
}

impl AllNttParameters {
    pub(crate) fn new() -> Self {
        let forward = NttParameters::new(false);
        let inverse = NttParameters::new(true);
        Self { forward, inverse }
    }
}

pub(super) struct NttLauncher {
    buffer: &DeviceBuffer<F>,
    lg_domain_size: u32,
    padded_poly_size: u32,
    poly_count: u32,
    is_intt: bool,
    stage: i32,
    ntt_parameters: &NttParameters,
}

impl NttImpl {
    pub(super) fn new(buffer: &DeviceBuffer<F>, lg_domain_size: u32, padded_poly_size: u32, poly_count: u32, is_intt: bool, ntt_parameters: &NttParameters) -> Self {
        Self { buffer, lg_domain_size, padded_poly_size, poly_count, is_intt, ntt_parameters }
    }

    pub(super) fn step(&mut self, iterations: i32) {
        assert!(iterations <= 10);
        let radix = if iterations < 6 { 6 } else { iterations };
        let twiddles_offset = (1 << (radix - 6)) << 5;
        unsafe {
            ntt::ct_mixed_radix_narrow(
                radix, 
                self.lg_domain_size, 
                self.stage, 
                iterations,
                self.buffer, 
                self.padded_poly_size, 
                self.ntt_parameters.partial_twiddles, 
                self.ntt_parameters.twiddles, 
                twiddles_offset,
                self.is_intt,
            ).unwrap();
        }
        self.stage += iterations;
    }
}

pub(super) fn batch_interpolate_ntt(buffer: &DeviceBuffer<F>, log_trace_height: u32, log_blowup: u32, width: u32, bit_reverse: bool, inverse_params: &NttParameters) {

    let padded_poly_size = 1 << (log_trace_height + log_blowup);

    let is_intt = true;

    if bit_reverse {
        unsafe {
            ntt::bit_rev(buffer, buffer, log_trace_height, padded_poly_size, width);
        }
    }

    let mut impl = NttImpl::new(buffer, log_trace_height, padded_poly_size, width, is_intt, inverse_params);
    if (log_trace_height <= 10) {
        impl.step(log_trace_height);
    } else if (log_trace_height <= 17) {
        int step = log_trace_height / 2;
        impl.step(step + log_trace_height % 2);
        impl.step(step);
    } else if (log_trace_height <= 30) {
        int step = log_trace_height / 3;
        impl.step(step);
        impl.step(step + (log_trace_height == 29 ? 1 : 0));
        impl.step(step + (log_trace_height == 29 ? 1 : rem));
    } else if (log_trace_height <= 40) {
        int step = log_trace_height / 4;
        impl.step(step);
        impl.step(step + (rem > 2));
        impl.step(step + (rem > 1));
        impl.step(step + (rem > 0));
    } else {
        assert(false);
    }
)


}