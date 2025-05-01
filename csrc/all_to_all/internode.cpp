
#include <nvshmem.h>

#include <cassert>
#include <cstdint>

#include "all_to_all/internode.h"
#include "core/utils.h"

using namespace pplx;

AllToAllInterNode::AllToAllInterNode(
    size_t maxNumTokens,
    size_t numExperts,
    size_t expertsPerToken,
    unsigned rank,
    unsigned worldSize,
    unsigned dpSize,
    size_t hiddenDim,
    size_t hiddenDimBytes,
    size_t hiddenDimScaleBytes
)
    : AllToAll(
          maxNumTokens,
          numExperts,
          expertsPerToken,
          rank,
          worldSize,
          dpSize,
          hiddenDim,
          hiddenDimBytes,
          hiddenDimScaleBytes
      ),
      maxBatchTokens(numLocalExperts * numDPGroups * maxNumTokens) {
  // Buffers for token counts.
  numTokensPerDP = mallocZeroBuffer<uint32_t>(numLocalExperts * numDPGroups);

  numTokensBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * numLocalExperts * numDPGroups);
  PPLX_ASSERT(numTokensBuffer != nullptr, "failed to allocate numTokensBuffer");
  cudaMemset(numTokensBuffer, 0, sizeof(uint64_t) * numLocalExperts * numDPGroups);

  numDispatchRecvBuffer =
      (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * numLocalExperts * numDPGroups);
  PPLX_ASSERT(numDispatchRecvBuffer != nullptr, "failed to allocate numDispatchRecvBuffer");
  cudaMemset(numDispatchRecvBuffer, 0, sizeof(uint64_t) * numLocalExperts * numDPGroups);

  combineSignalBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * maxNumTokens);
  PPLX_ASSERT(combineSignalBuffer != nullptr, "failed to allocate combineSignalBuffer");
  cudaMemset(combineSignalBuffer, 0, sizeof(uint64_t) * maxNumTokens);

  combineSyncBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * worldSize);
  PPLX_ASSERT(combineSyncBuffer != nullptr, "failed to allocate combineSyncBuffer");
  cudaMemset(combineSyncBuffer, 0, sizeof(uint64_t) * worldSize);

  // Buffers for dispatch.
  const size_t perTokenBytes =
      round_up<size_t>(hiddenDimBytes + hiddenDimScaleBytes + sizeof(uint32_t), 16);
  xDispatchIn = (std::byte *)nvshmem_malloc(maxNumTokens * perTokenBytes);
  PPLX_ASSERT(xDispatchIn != nullptr, "failed to allocate xDispatchIn");
  xDispatchOut = (std::byte *)nvshmem_malloc(maxBatchTokens * perTokenBytes);
  PPLX_ASSERT(xDispatchOut != nullptr, "failed to allocate xDispatchOut");

  // Buffers for combine. The allocations are a bit wider to accommodate all
  // possible data types (primarily float for testing and bfloat16 for prod).
  xCombineIn = (std::byte *)nvshmem_malloc(maxBatchTokens * hiddenDim * sizeof(float));
  PPLX_ASSERT(xCombineIn != nullptr, "failed to allocate xCombineIn");
  xCombineOut = (std::byte *)nvshmem_malloc(maxNumTokens * numExperts * hiddenDim * sizeof(float));
  PPLX_ASSERT(xCombineOut != nullptr, "failed to allocate xCombineOut");

  // Buffers for token tracking.
  sourceIndex = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceExpert = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceOffset = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceGroup = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceToken = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  tokenIndex = mallocZeroBuffer<uint32_t>(1);
}

AllToAllInterNode::~AllToAllInterNode() {
  CUDACHECK(cudaFree(numTokensPerDP));
  nvshmem_free(numTokensBuffer);
  nvshmem_free(numDispatchRecvBuffer);
  nvshmem_free(combineSignalBuffer);
  nvshmem_free(combineSyncBuffer);
  nvshmem_free(xDispatchIn);
  nvshmem_free(xDispatchOut);
  nvshmem_free(xCombineIn);
  nvshmem_free(xCombineOut);

  CUDACHECK(cudaFree(sourceIndex));
  CUDACHECK(cudaFree(sourceExpert));
  CUDACHECK(cudaFree(sourceOffset));
  CUDACHECK(cudaFree(sourceGroup));
  CUDACHECK(cudaFree(sourceToken));
  CUDACHECK(cudaFree(tokenIndex));
}
