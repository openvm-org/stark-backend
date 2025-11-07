use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SystemParams {
    pub l_skip: usize,
    pub n_stack: usize,
    pub log_blowup: usize,
    pub k_whir: usize,
    pub num_whir_queries: usize,
    pub log_final_poly_len: usize,
    pub logup_pow_bits: usize,
    pub whir_pow_bits: usize,
}

impl SystemParams {
    #[inline]
    pub fn num_whir_rounds(&self) -> usize {
        (self.n_stack + self.l_skip - self.log_final_poly_len) / self.k_whir
    }

    #[inline]
    pub fn num_whir_sumcheck_rounds(&self) -> usize {
        self.n_stack + self.l_skip - self.log_final_poly_len
    }
}
