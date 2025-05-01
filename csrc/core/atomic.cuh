#pragma once

#include <cstdint>

namespace pplx {

__forceinline__ __device__ void st_flag_volatile(uint32_t *flag_addr, uint32_t flag) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

__forceinline__ __device__ uint32_t ld_flag_volatile(uint32_t *flag_addr) {
  uint32_t flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

__forceinline__ __device__ uint32_t ld_flag_acquire(uint32_t *flag_addr) {
  uint32_t flag;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

__forceinline__ __device__ void st_flag_release(uint32_t *flag_addr, uint32_t flag) {
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

__forceinline__ __device__ uint32_t add_flag_release(uint32_t *addr, uint32_t val) {
  uint32_t flag;
  asm volatile("atom.release.sys.global.add.u32 %0, [%1], %2;" : "=r"(flag) : "l"(addr), "r"(val));
  return flag;
}

} // namespace pplx
