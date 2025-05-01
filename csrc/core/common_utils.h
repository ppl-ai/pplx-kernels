#pragma once

#include <climits>
#include <cstdint>

#ifdef __CUDA_ARCH__
#define PPLX_HOST_DEVICE __host__ __device__
#else
#define PPLX_HOST_DEVICE
#endif

namespace pplx {

/// Return the next power of 2 following the given number.
PPLX_HOST_DEVICE inline uint32_t next_pow_2(const uint32_t num) {
  if (num <= 1) {
    return num;
  }
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename T> PPLX_HOST_DEVICE T ceil_div(T x, T y) { return (x + y - 1) / y; }

template <typename T> PPLX_HOST_DEVICE T round_up(T x, T y) { return ceil_div<T>(x, y) * y; }

} // namespace pplx
