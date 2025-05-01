#pragma once

#include "core/device_utils.cuh"

#include <cstdint>

namespace pplx {

/// A wrapper around the send/recv buffers, offerring access to slices.
class BufferWrapper {
public:
  __device__ BufferWrapper(
      std::byte **ptr,
      size_t numLocalExperts,
      size_t worldSize,
      size_t maxNumTokens,
      size_t perTokenBytes
  )
      : ptr(ptr),
        numLocalExperts(numLocalExperts),
        worldSize(worldSize),
        maxNumTokens(maxNumTokens),
        perTokenBytes(perTokenBytes) {}

  __device__ __forceinline__ uint32_t &getDispatchSyncPtr(unsigned rank) {
    return *reinterpret_cast<uint32_t *>(getBaseDispatchSyncPtr(rank));
  }

  __device__ __forceinline__ uint32_t &getCombineSyncPtr(unsigned rank) {
    return *reinterpret_cast<uint32_t *>(getBaseCombineSyncPtr(rank));
  }

  __device__ __forceinline__ uint32_t &getCountPtr(unsigned rank, unsigned expert) {
    return reinterpret_cast<uint32_t *>(getBaseCounterPtr(rank))[expert];
  }

  __device__ __forceinline__ std::byte *
  getTokenPtr(unsigned rank, unsigned expert, unsigned token) {
    return getBaseTokenPtr(rank) + (expert * maxNumTokens + token) * perTokenBytes;
  }

private:
  __device__ __forceinline__ std::byte *getBaseDispatchSyncPtr(unsigned rank) { return ptr[rank]; }

  __device__ __forceinline__ std::byte *getBaseCombineSyncPtr(unsigned rank) {
    return getBaseDispatchSyncPtr(rank) + sizeof(uint32_t);
  }

  __device__ __forceinline__ std::byte *getBaseCounterPtr(unsigned rank) {
    return getBaseDispatchSyncPtr(rank) + sizeof(int4);
  }

  __device__ __forceinline__ std::byte *getBaseTokenPtr(unsigned rank) {
    return getBaseCounterPtr(rank) +
           round_up<size_t>(numLocalExperts * sizeof(uint32_t), sizeof(int4));
  }

private:
  std::byte **ptr;
  const size_t numLocalExperts;
  const size_t worldSize;
  const size_t maxNumTokens;
  const size_t perTokenBytes;
};

} // namespace pplx
