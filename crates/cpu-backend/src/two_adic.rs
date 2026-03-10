use p3_field::TwoAdicField;

/// Precomputed radix-2 twiddle factors for size `2^log_n`.
pub(crate) struct DftTwiddles<F> {
    /// iDFT twiddle factors (DIF: omega_inv powers), per layer.
    idft_tw: Vec<Vec<F>>,
    /// DFT twiddle factors (DIT: omega powers), per layer.
    dft_tw: Vec<Vec<F>>,
    /// 1/N scaling factor for iDFT.
    n_inv: F,
    log_n: usize,
}

impl<F: TwoAdicField> DftTwiddles<F> {
    pub(crate) fn new(log_n: usize) -> Self {
        let n = 1usize << log_n;
        let omega = F::two_adic_generator(log_n);
        let omega_inv = omega.inverse();
        let mut n_field = F::ONE;
        for _ in 0..log_n {
            n_field += n_field;
        }
        let n_inv = n_field.inverse();

        let idft_tw = (0..log_n)
            .map(|layer| {
                let half = n >> (layer + 1);
                let w = omega_inv.exp_power_of_2(layer);
                w.powers().take(half).collect()
            })
            .collect();

        let dft_tw = (0..log_n)
            .map(|layer| {
                let half = 1usize << layer;
                let w = omega.exp_power_of_2(log_n - 1 - layer);
                w.powers().take(half).collect()
            })
            .collect();

        Self {
            idft_tw,
            dft_tw,
            n_inv,
            log_n,
        }
    }

    pub(crate) fn size(&self) -> usize {
        1usize << self.log_n
    }

    /// In-place iDFT (DIF + bit-reverse + scale by 1/N).
    pub(crate) fn idft_inplace(&self, buf: &mut [F]) {
        let n = self.size();
        debug_assert_eq!(buf.len(), n);

        let mut block_size = n;
        for tw in &self.idft_tw {
            let half = block_size >> 1;
            let mut k = 0;
            while k < n {
                for j in 0..half {
                    let u = buf[k + j];
                    let v = buf[k + j + half];
                    buf[k + j] = u + v;
                    buf[k + j + half] = (u - v) * tw[j];
                }
                k += block_size;
            }
            block_size = half;
        }

        self.bit_reverse(buf);
        for val in buf.iter_mut() {
            *val *= self.n_inv;
        }
    }

    /// In-place DFT (DIT: bit-reverse input, then butterflies).
    pub(crate) fn dft_inplace(&self, buf: &mut [F]) {
        let n = self.size();
        debug_assert_eq!(buf.len(), n);

        self.bit_reverse(buf);

        let mut block_size = 2;
        for tw in &self.dft_tw {
            let half = block_size >> 1;
            let mut k = 0;
            while k < n {
                for j in 0..half {
                    let u = buf[k + j];
                    let v = buf[k + j + half] * tw[j];
                    buf[k + j] = u + v;
                    buf[k + j + half] = u - v;
                }
                k += block_size;
            }
            block_size <<= 1;
        }
    }

    /// Coset DFT: evaluate polynomial at `shift * omega^k` for `k = 0..N-1`.
    pub(crate) fn coset_dft_inplace(&self, buf: &mut [F], shift: F) {
        debug_assert_eq!(buf.len(), self.size());

        let mut s = F::ONE;
        for val in buf.iter_mut() {
            *val *= s;
            s *= shift;
        }
        self.dft_inplace(buf);
    }

    fn bit_reverse(&self, buf: &mut [F]) {
        let n = self.size();
        for i in 0..n {
            let j = i.reverse_bits() >> (usize::BITS as u32 - self.log_n as u32);
            if i < j {
                buf.swap(i, j);
            }
        }
    }
}
