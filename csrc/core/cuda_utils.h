#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDACHECK(cmd)                                                                             \
  do {                                                                                             \
    cudaError_t e = cmd;                                                                           \
    if (e != cudaSuccess) {                                                                        \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));        \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

namespace pplx {
template <typename T> T *mallocZeroBuffer(size_t size) {
  T *ptr;
  CUDACHECK(cudaMalloc(&ptr, size * sizeof(T)));
  cudaMemset(ptr, 0, size * sizeof(T));
  return ptr;
}

inline int get_sm_count() {
  int device;
  CUDACHECK(cudaGetDevice(&device));
  int numSMs;
  CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));

  return numSMs;
}

} // namespace pplx
