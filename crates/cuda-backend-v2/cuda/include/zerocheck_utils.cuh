// Given x and 2^n, computes 1/2^n * (1 + x + ... + x^{2^n - 1}).
__device__ __forceinline__ Fp avg_gp(Fp x, uint32_t n) {
    #ifdef CUDA_DEBUG
        assert(n && !(n & (n - 1)));
    #endif
        Fp res = Fp::one();
        for (uint32_t i = 1; i < n; i <<= 1) {
            res *= Fp::one() + x;
            res = res.halve();
            x *= x;
        }
        return res;
    }
    