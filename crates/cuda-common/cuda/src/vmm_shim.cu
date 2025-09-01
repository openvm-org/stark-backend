#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

extern "C" {

// Return 0 on success, else CUresult / cudaError_t code

int _vmm_check_support(int device_ordinal) {
  CUdevice dev;
  CUresult r = cuDeviceGet(&dev, device_ordinal);
  if (r != CUDA_SUCCESS) return (int)r;
  int vmm = 0;
  r = cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev);
  if (r != CUDA_SUCCESS) return (int)r;
  return vmm ? 0 : (int)CUDA_ERROR_NOT_SUPPORTED;
}

int _vmm_min_granularity(int device_ordinal, size_t* out) {
  if (!out) return (int)CUDA_ERROR_INVALID_VALUE;
  CUdevice dev;
  CUresult r = cuDeviceGet(&dev, device_ordinal);
  if (r != CUDA_SUCCESS) return (int)r;

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_ordinal;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

  return (int)cuMemGetAllocationGranularity(out, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
}

int _vmm_reserve(size_t size, size_t align, CUdeviceptr* out_va) {
  if (!out_va) return (int)CUDA_ERROR_INVALID_VALUE;
  return (int)cuMemAddressReserve(out_va, size, align, 0, 0);
}

int _vmm_release_va(CUdeviceptr base, size_t size) {
  return (int)cuMemAddressFree(base, size);
}

int _vmm_create_physical(int device_ordinal, size_t bytes, CUmemGenericAllocationHandle* out_h) {
  if (!out_h) return (int)CUDA_ERROR_INVALID_VALUE;
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_ordinal;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
  // leave win32HandleMetaData/reserved/allocFlags as zero-init
  return (int)cuMemCreate(out_h, bytes, &prop, 0);
}

int _vmm_map_and_set_access(CUdeviceptr va, size_t bytes, CUmemGenericAllocationHandle h, int device_ordinal) {
  CUresult r = cuMemMap(va, bytes, 0, h, 0);
  if (r != CUDA_SUCCESS) return (int)r;
  CUmemAccessDesc acc{};
  acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  acc.location.id = device_ordinal;
  acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return (int)cuMemSetAccess(va, bytes, &acc, 1);
}

int _vmm_unmap_release(CUdeviceptr va, size_t bytes, CUmemGenericAllocationHandle h) {
  CUresult r = cuMemUnmap(va, bytes);
  if (r != CUDA_SUCCESS) return (int)r;
  return (int)cuMemRelease(h);
}

} // extern "C"
