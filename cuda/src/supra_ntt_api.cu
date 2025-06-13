// FROM https://github.com/scroll-tech/plonky3-gpu/blob/openvm-v2/gpu-backend/src/cuda/kernels/supra_ntt_api.cu
#include <ff/baby_bear.hpp>
#include <ntt/ntt.cuh>

#ifndef __CUDA_ARCH__

extern "C" RustError::by_value _sppark_init(uint32_t idx = 0) {
    uint32_t lg_domain_size = 1;
    uint32_t domain_size = 1U << lg_domain_size;

    std::vector<fr_t> inout(domain_size);
    inout[0] = fr_t(1);
    inout[1] = fr_t(1);

    const gpu_t &gpu = select_gpu(idx);

    try {
        NTT::Base(
            gpu,
            &inout[0],
            lg_domain_size,
            NTT::InputOutputOrder::NR,
            NTT::Direction::forward,
            NTT::Type::standard
        );
        gpu.sync();
    } catch (const cuda_error &e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }

    return RustError{cudaSuccess};
}

extern "C" RustError::by_value _batch_NTT(
    fr_t *d_inout,
    uint32_t lg_domain_size,
    uint32_t poly_count,
    uint32_t device_idx
) {
    if (lg_domain_size == 0)
        return RustError{cudaSuccess};

    uint32_t domain_size = 1U << lg_domain_size;

    stream_t ntt_stream(device_idx);
    event_t default_stream_done(cudaStreamPerThread);
    try {
        ntt_stream.wait(default_stream_done);

        for (size_t c = 0; c < poly_count; c++) {
            NTT::Base_dev_ptr(
                ntt_stream,
                &d_inout[c * domain_size],
                lg_domain_size,
                NTT::InputOutputOrder::RN,
                NTT::Direction::forward,
                NTT::Type::standard
            );
        }

        event_t ntt_done;
        ntt_stream.record(ntt_done);
        CUDA_OK(cudaStreamWaitEvent(cudaStreamPerThread, ntt_done, 0));
    } catch (const cuda_error &e) {
        ntt_stream.sync();
        return RustError{e.code(), e.what()};
    }

    return RustError{cudaSuccess};
}

/// batch inverse NTT on polynomials of degree `2^lg_domain_size` but where polynomials
/// are placed in a buffer where each polynomial has `2^{lg_domain_size + lg_blowup}` field
/// elements allocated for it.
extern "C" RustError::by_value _batch_iNTT(
    fr_t *d_inout,
    uint32_t lg_domain_size,
    uint32_t lg_blowup,
    uint32_t poly_count,
    uint32_t device_idx
) {
    if (lg_domain_size == 0)
        return RustError{cudaSuccess};

    // Each poly is allocated with more space than its degree
    uint32_t padded_poly_size = 1U << (lg_domain_size + lg_blowup);

    stream_t ntt_stream(device_idx);
    event_t default_stream_done(cudaStreamPerThread);
    try {
        ntt_stream.wait(default_stream_done);

        for (size_t c = 0; c < poly_count; c++) {
            NTT::Base_dev_ptr(
                ntt_stream,
                &d_inout[c * padded_poly_size],
                lg_domain_size,
                NTT::InputOutputOrder::NN,
                NTT::Direction::inverse,
                NTT::Type::standard
            );
        }

        event_t ntt_done;
        ntt_stream.record(ntt_done);
        CUDA_OK(cudaStreamWaitEvent(cudaStreamPerThread, ntt_done, 0));
    } catch (const cuda_error &e) {
        ntt_stream.sync();
        return RustError{e.code(), e.what()};
    }

    return RustError{cudaSuccess};
}

#endif
// END OF FILE gpu-backend/src/cuda/kernels/supra_ntt_api.cu
