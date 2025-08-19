#include "ff/baby_bear.hpp"
#include "ntt/ntt.cuh"

extern "C" int _sppark_init() {
    NTTParametersHolder::all();
    return cudaGetLastError();
}

extern "C" int _batch_NTT(
    fr_t *d_inout,
    uint32_t lg_domain_size,
    uint32_t poly_count
) {
    if (lg_domain_size == 0)
        return cudaGetLastError();

    uint32_t domain_size = 1U << lg_domain_size;

    NTT::NTT_RUN(
        d_inout,
        lg_domain_size,
        domain_size,
        poly_count,
        false,
        NTT::Direction::forward
    );

    return cudaGetLastError();
}

/// batch inverse NTT on polynomials of degree `2^lg_domain_size` but where polynomials
/// are placed in a buffer where each polynomial has `2^{lg_domain_size + lg_blowup}` field
/// elements allocated for it.
extern "C" int _batch_iNTT(
    fr_t *d_inout,
    uint32_t lg_domain_size,
    uint32_t lg_blowup,
    uint32_t poly_count
) {
    if (lg_domain_size == 0)
        return cudaGetLastError();

    // Each poly is allocated with more space than its degree
    uint32_t padded_poly_size = 1U << (lg_domain_size + lg_blowup);

    NTT::NTT_RUN(
        d_inout,
        lg_domain_size,
        padded_poly_size,
        poly_count,
        true,
        NTT::Direction::inverse
    );

    return cudaGetLastError();
}
