#include "all_to_all/intranode.h"

#include "core/distributed.h"
#include "core/utils.h"

#include <cassert>
#include <cstdint>

using namespace pplx;

AllToAllIntraNode::AllToAllIntraNode(
    size_t maxNumTokens,
    size_t numExperts,
    size_t expertsPerToken,
    unsigned rank,
    unsigned worldSize,
    unsigned dpSize,
    size_t hiddenDim,
    size_t hiddenDimBytes,
    size_t hiddenDimScaleBytes,
    std::shared_ptr<Distributed> distributed
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
      ) {

  // Determine the per-token buffer size. Allocate extra storage for the index.
  // Pad to a 16-byte alignment to allow loads via int4.
  size_t bufferSize;
  {
    size_t dispatchBufferSize = 0;
    {
      const size_t metaSize = sizeof(uint32_t);
      const size_t dispatchPerTokenBytes =
          round_up<size_t>(hiddenDimBytes + hiddenDimScaleBytes + metaSize, 16);

      dispatchBufferSize += sizeof(int4);
      dispatchBufferSize += round_up<size_t>(numLocalExperts * sizeof(uint32_t), sizeof(int4));
      dispatchBufferSize += maxNumTokens * numLocalExperts * dispatchPerTokenBytes;
    }

    size_t combineBufferSize = 0;
    {
      const size_t combinePerTokenBytes = round_up<size_t>(hiddenDim * sizeof(float), 16);
      combineBufferSize += sizeof(int4);
      combineBufferSize += round_up<size_t>(numLocalExperts * sizeof(uint32_t), sizeof(int4));
      combineBufferSize += maxNumTokens * numLocalExperts * combinePerTokenBytes;
    }

    // The size of the buffer is the max of dispatch and combine.
    bufferSize = std::max(dispatchBufferSize, combineBufferSize);
  }

  // Allocate pointers to the buffer collections.
  CUDACHECK(cudaMalloc(&sendBuffersPtr, sizeof(std::byte *) * worldSize));
  CUDACHECK(cudaMalloc(&recvBuffersPtr, sizeof(std::byte *) * worldSize));

  // On the current rank, allocate a buffer to communicate with every other rank.
  // Synchronize via the distributed group. Create indirect pointer arrays to buffers.
  {
    std::vector<cudaIpcMemHandle_t> srcHandlesHost;
    for (unsigned i = 0; i < worldSize; i++) {
      auto &ptr = sendBuffers.emplace_back();
      auto &handle = srcHandlesHost.emplace_back();
      CUDACHECK(cudaMalloc(&ptr, bufferSize));
      CUDACHECK(cudaMemset(ptr, 0, bufferSize));
      CUDACHECK(cudaIpcGetMemHandle(&handle, ptr));
    }

    auto dstHandlesHost = distributed->allToAll(srcHandlesHost);
    for (unsigned i = 0; i < worldSize; i++) {
      auto &ptr = recvBuffers.emplace_back();
      if (i == rank) {
        ptr = sendBuffers[i];
      } else {
        CUDACHECK(
            cudaIpcOpenMemHandle((void **)&ptr, dstHandlesHost[i], cudaIpcMemLazyEnablePeerAccess)
        );
      }
    }

    CUDACHECK(cudaMemcpy(
        sendBuffersPtr, sendBuffers.data(), sizeof(std::byte *) * worldSize, cudaMemcpyHostToDevice
    ));

    CUDACHECK(cudaMemcpy(
        recvBuffersPtr, recvBuffers.data(), sizeof(std::byte *) * worldSize, cudaMemcpyHostToDevice
    ));
  }

  // Allocate the local buffer for dispatch counts.
  CUDACHECK(cudaMalloc(&localRecvCountPtr, sizeof(uint32_t) * maxNumTokens));
  CUDACHECK(cudaMemset(localRecvCountPtr, 0, sizeof(uint32_t) * maxNumTokens));
  CUDACHECK(cudaMalloc(&countBuffersPtr, sizeof(uint32_t *) * worldSize));
  {
    cudaIpcMemHandle_t countHandle;
    CUDACHECK(cudaIpcGetMemHandle(&countHandle, localRecvCountPtr));
    auto countHandlesHost = distributed->allGather(countHandle);

    countBuffers.resize(worldSize);
    for (unsigned i = 0; i < worldSize; i++) {
      if (i == rank) {
        countBuffers[i] = localRecvCountPtr;
      } else {
        CUDACHECK(cudaIpcOpenMemHandle(
            (void **)&countBuffers[i], countHandlesHost[i], cudaIpcMemLazyEnablePeerAccess
        ));
      }
    }

    CUDACHECK(cudaMemcpy(
        countBuffersPtr, countBuffers.data(), sizeof(uint32_t *) * worldSize, cudaMemcpyHostToDevice
    ));
  }

  // Allocate the local buffers.
  tokenCount = mallocZeroBuffer<uint32_t>(numExperts);
  numTokensPerRank = mallocZeroBuffer<uint32_t>(numLocalExperts * worldSize);

  // Buffers for token tracking.
  const size_t maxBatchTokens = numLocalExperts * maxNumTokens * worldSize;
  sourceIndex = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceExpert = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceOffset = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceRank = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceToken = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceRoute = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  tokenIndex = mallocZeroBuffer<uint32_t>(1);
}

AllToAllIntraNode::~AllToAllIntraNode() {
  for (unsigned i = 0; i < worldSize; i++) {
    CUDACHECK(cudaFree(sendBuffers[i]));
    if (i != rank) {
      CUDACHECK(cudaIpcCloseMemHandle(recvBuffers[i]));
      CUDACHECK(cudaIpcCloseMemHandle(countBuffers[i]));
    }
  }

  CUDACHECK(cudaFree(recvBuffersPtr));
  CUDACHECK(cudaFree(sendBuffersPtr));
  CUDACHECK(cudaFree(countBuffersPtr));
  CUDACHECK(cudaFree(localRecvCountPtr));

  CUDACHECK(cudaFree(tokenCount));
  CUDACHECK(cudaFree(numTokensPerRank));

  CUDACHECK(cudaFree(sourceIndex));
  CUDACHECK(cudaFree(sourceExpert));
  CUDACHECK(cudaFree(sourceOffset));
  CUDACHECK(cudaFree(sourceRank));
  CUDACHECK(cudaFree(sourceToken));
  CUDACHECK(cudaFree(tokenIndex));
}
