#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

extern "C" int _cuda_get_sm_count(uint32_t device_id, uint32_t *out_sm_count) {
    int sm_count = 0;
    cudaError_t err = cudaDeviceGetAttribute(
        &sm_count, cudaDevAttrMultiProcessorCount, static_cast<int>(device_id)
    );
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    *out_sm_count = static_cast<uint32_t>(sm_count);
    return 0;
}
