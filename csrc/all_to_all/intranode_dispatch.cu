#include "all_to_all/intranode.cuh"
#include "core/atomic.cuh"
#include "core/device_utils.cuh"
#include "core/utils.h"
#include "intranode.h"

#include <cooperative_groups.h>
#include <cuda.h>
#include <nvtx3/nvToolsExt.h>

using namespace pplx;

namespace {

template <unsigned NUM_WARPS, bool DO_SEND, bool DO_RECV>
__global__ __launch_bounds__(NUM_WARPS * 32, 1) void dispatchKernel(
    int32_t *outNumTokensPerExpert,
    size_t outNumTokensPerExpertStrideElem,
    std::byte *expertX,
    size_t expertXStrideElem,
    size_t expertXStrideRow,
    float *expertXScale,
    size_t expertXScaleStrideElem,
    size_t expertXScaleStrideRow,
    size_t expertXScaleStrideCol,
    const std::byte *dpX,
    size_t dpXStrideElem,
    const float *dpXScale,
    size_t dpXScaleStrideElem,
    size_t dpXScaleStrideRow,
    const uint32_t *indices,
    size_t indicesStrideElem,
    size_t indicesStrideRow,
    size_t maxNumTokens,
    size_t numExperts,
    unsigned rank,
    unsigned worldSize,
    unsigned dpSize,
    size_t hiddenDim,
    size_t hiddenDimScale,
    size_t numExpertsPerToken,
    unsigned *boundM,
    unsigned m,
    std::byte **sendBuffersPtr,
    std::byte **recvBuffersPtr,
    uint32_t *tokenCount,
    uint32_t *numTokensPerRank,
    uint32_t *sourceExpert,
    uint32_t *sourceIndex,
    uint32_t *sourceOffset,
    uint32_t *sourceRank,
    uint32_t *sourceToken,
    uint32_t &globalTokenIndex
) {
  extern __shared__ std::byte sharedMemory[];

  // Determine the rank, DP rank and per-rank constants.
  const size_t numLocalExperts = numExperts / worldSize;
  const unsigned dpRank = rank % dpSize;
  const size_t tokenDim = hiddenDim + hiddenDimScale;
  const size_t tokenStride = round_up<size_t>(tokenDim + sizeof(uint32_t), sizeof(int4));

  // Determine the number of tokens populated which are to be sent.
  const unsigned numSendTokens = boundM ? __ldg(boundM) : m;
  PPLX_DEVICE_ASSERT(numSendTokens <= maxNumTokens);
  PPLX_DEVICE_ASSERT(
      hiddenDimScale == 0 || numSendTokens == 0 || (expertXScale != nullptr && dpXScale != nullptr)
  );
  PPLX_DEVICE_ASSERT(tokenStride % sizeof(int4) == 0);

  BufferWrapper localBuffer(sendBuffersPtr, numLocalExperts, worldSize, maxNumTokens, tokenStride);
  BufferWrapper remoteBuffer(recvBuffersPtr, numLocalExperts, worldSize, maxNumTokens, tokenStride);

  // Create wrappers around the send/recv buffers.
  if constexpr (DO_SEND) {
    // Clear some output buffers.
    if (blockIdx.x == 0) {
      for (unsigned i = threadIdx.x; i < numLocalExperts; i += blockDim.x) {
        outNumTokensPerExpert[i] = 0;
      }
    }

    // Wait for the combine step to finish.
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < worldSize;
         i += blockDim.x * gridDim.x) {
      auto *pollPtr = &localBuffer.getCombineSyncPtr(i);
      while (ld_flag_volatile(pollPtr) != 0)
        ;
    }
    cooperative_groups::this_grid().sync();
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < worldSize;
         i += blockDim.x * gridDim.x) {
      st_flag_volatile(&remoteBuffer.getDispatchSyncPtr(i), 1);
    }

    // Copy the token to shared memory, then to the remote buffers.
    for (unsigned i = blockIdx.x * dpSize + dpRank; i < numSendTokens; i += gridDim.x * dpSize) {
      const int4 *srcX = (int4 *)(dpX + i * dpXStrideElem);
      const float *srcXScale = dpXScale + i * dpXScaleStrideRow;

      // Copy the token to shared memory.
      std::byte *sharedTokenPtr = sharedMemory;
      {
        int4 *dstPtr = (int4 *)sharedTokenPtr;
        int4 *srcPtr = (int4 *)srcX;
        dstPtr += threadIdx.x;
        srcPtr += threadIdx.x;

        const unsigned n = hiddenDim / sizeof(int4);
        for (unsigned d = threadIdx.x; d < n; d += blockDim.x) {
          *dstPtr = __ldg(srcPtr);
          dstPtr += blockDim.x;
          srcPtr += blockDim.x;
        }

        if (hiddenDimScale > 0) {
          std::byte *sharedScalePtr = sharedTokenPtr + hiddenDim;
          for (unsigned d = threadIdx.x; d * sizeof(float) < hiddenDimScale; d += blockDim.x) {
            ((float *)sharedScalePtr)[d] = __ldg(&srcXScale[d * dpXScaleStrideElem]);
          }
        }
      }

      // Wait for the token to be copied to shared memory.
      __syncthreads();

      // Send the token to the other ranks, one send per warp.
      const unsigned WARP_SIZE = 32;
      const unsigned warpId = threadIdx.x / WARP_SIZE;
      const unsigned laneId = threadIdx.x % WARP_SIZE;
      for (unsigned j = warpId; j < numExpertsPerToken; j += NUM_WARPS) {
        const uint32_t dstExpert = __ldg(&indices[i * numExpertsPerToken + j]);
        const uint32_t dstRank = dstExpert / numLocalExperts;
        const uint32_t dstLocalExpert = dstExpert % numLocalExperts;

        // Increment the number of tokens dispatched from the current rank.
        // Determine the index within the expert buffer where the token will be placed.
        unsigned index;
        if (laneId == 0) {
          index = atomicAdd(&tokenCount[dstExpert], 1);
        } else {
          index = 0;
        }
        index = __shfl_sync(0xffffffff, index, 0);

        // Copy the token to the shared buffer.
        std::byte *buffer = remoteBuffer.getTokenPtr(dstRank, dstLocalExpert, index);
        if (laneId == 0) {
          *((uint32_t *)(buffer + tokenDim)) = i;
        }

        const unsigned n = tokenDim / sizeof(int4);
        int4 *dstPtr = (int4 *)buffer;
        int4 *srcPtr = (int4 *)sharedTokenPtr;
        dstPtr += laneId;
        srcPtr += laneId;

        unsigned d = laneId;
        while (d + 4 * WARP_SIZE < n) {
          int4 r[4];
#pragma unroll
          for (unsigned i = 0; i < 4; i++) {
            r[i] = *srcPtr;
            srcPtr += WARP_SIZE;
          }
#pragma unroll
          for (unsigned i = 0; i < 4; i++) {
            *dstPtr = r[i];
            dstPtr += WARP_SIZE;
          }
          d += WARP_SIZE * 4;
        }

        while (d < n) {
          *dstPtr = *srcPtr;
          dstPtr += WARP_SIZE;
          srcPtr += WARP_SIZE;
          d += WARP_SIZE;
        }
      }
    }

    // Wait for all blocks to finish copying tokens to the shared buffers.
    cooperative_groups::this_grid().sync();

    // Post the token counts to the other ranks and reset send buffers.
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < numExperts;
         i += blockDim.x * gridDim.x) {
      unsigned dstRank = i / numLocalExperts;
      unsigned dstLocalExpert = i % numLocalExperts;
      st_flag_release(&remoteBuffer.getCountPtr(dstRank, dstLocalExpert), tokenCount[i] + 1);
      tokenCount[i] = 0;
    }
  }

  if constexpr (DO_RECV) {
    // Wait for the token counts to be sent.
    const size_t numExpertsAndRanks = numLocalExperts * worldSize;
    const size_t expertsPerBlock = ceil_div<size_t>(numExpertsAndRanks, gridDim.x);
    uint32_t *sharedExpert = reinterpret_cast<uint32_t *>(sharedMemory);
    uint32_t *sharedToken = sharedExpert + expertsPerBlock;

    unsigned firstGroup = blockIdx.x * expertsPerBlock;
    unsigned lastGroup = std::min(firstGroup + expertsPerBlock, numExpertsAndRanks);

    for (unsigned group = firstGroup + threadIdx.x; group < lastGroup;
         group += gridDim.x * expertsPerBlock) {
      const uint32_t srcRank = group / numLocalExperts;
      const uint32_t srcLocalExpert = group % numLocalExperts;

      // Fetch the token counts from all incoming ranks.
      auto *counterPtr = &localBuffer.getCountPtr(srcRank, srcLocalExpert);
      uint32_t counter;
      while ((counter = ld_flag_acquire(counterPtr)) == 0)
        ;
      st_flag_volatile(counterPtr, 0);

      size_t numTokens = counter - 1;
      numTokensPerRank[group] = numTokens;
      sharedExpert[group - firstGroup] =
          atomicAdd(&outNumTokensPerExpert[srcLocalExpert], numTokens);
      sharedToken[group - firstGroup] = atomicAdd(&globalTokenIndex, numTokens);
    }

    __syncthreads();

    for (unsigned group = firstGroup; group < lastGroup; group++) {
      const uint32_t srcRank = group / numLocalExperts;
      const uint32_t srcLocalExpert = group % numLocalExperts;

      const size_t numTokens = numTokensPerRank[group];
      auto expertStart = sharedExpert[group - firstGroup];
      auto tokenStart = sharedToken[group - firstGroup];

      for (unsigned i = threadIdx.x; i < numTokens; i += blockDim.x) {
        const std::byte *buffer = localBuffer.getTokenPtr(srcRank, srcLocalExpert, i);
        const uint32_t *metaPtr = (uint32_t *)(buffer + tokenDim);

        uint32_t token = tokenStart + i;
        sourceExpert[token] = srcLocalExpert;
        sourceOffset[token] = expertStart + i;
        sourceRank[token] = srcRank;
        sourceToken[token] = i;
        sourceIndex[token] = metaPtr[0];
      }
    }

    cooperative_groups::this_grid().sync();

    unsigned numRecvTokens = globalTokenIndex;

    for (unsigned i = blockIdx.x; i < numRecvTokens; i += gridDim.x) {
      auto expertLoc = sourceOffset[i];
      auto expert = sourceExpert[i];

      std::byte *dstXExpert = expertX + expert * expertXStrideRow;
      float *dstXScaleExpert = expertXScale + expert * expertXScaleStrideCol;

      std::byte *buffer = localBuffer.getTokenPtr(sourceRank[i], expert, sourceToken[i]);

      const int4 *srcX = (int4 *)buffer;
      int4 *dstX = (int4 *)(dstXExpert + expertLoc * expertXStrideElem);
      for (unsigned k = threadIdx.x; k * sizeof(int4) < hiddenDim; k += blockDim.x) {
        dstX[k] = srcX[k];
      }

      // Copy the scale to the output buffer.
      if (hiddenDimScale > 0) {
        const float *srcXScale = (float *)(buffer + hiddenDim);
        float *dstXScale = dstXScaleExpert + expertLoc * expertXScaleStrideRow;
        for (unsigned k = threadIdx.x; k * sizeof(float) < hiddenDimScale; k += blockDim.x) {
          dstXScale[k * expertXScaleStrideElem] = srcXScale[k];
        }
      }
    }

    // Signal that the current device has finished dispatching tokens.
    cooperative_groups::this_grid().sync();
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < worldSize;
         i += blockDim.x * gridDim.x) {
      st_flag_volatile(&remoteBuffer.getDispatchSyncPtr(i), 0);
    }
  }
}

} // namespace

void AllToAllIntraNode::dispatch(
    const Strided1D<int32_t> &outNumTokensPerExpert,
    const Strided2D<std::byte> &expertX,
    const Strided3D<float> &expertXScale,
    const Strided1D<std::byte> &dpX,
    const Strided2D<float> &dpXScale,
    const Strided2D<uint32_t> &indices,
    unsigned m,
    const unsigned *boundM,
    SplitMode splitMode,
    cudaStream_t stream
) {
  constexpr unsigned NUM_WARPS = 16;
  const unsigned numBlocks = std::min(
      std::max(
          ceil_div<unsigned>(numExperts, NUM_WARPS), (unsigned)(maxNumTokens * expertsPerToken)
      ),
      static_cast<unsigned>(numSMs)
  );
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(NUM_WARPS * 32, 1, 1);

  const size_t expertsPerBlock = ceil_div<size_t>(numLocalExperts * worldSize, numBlocks);
  const size_t sharedMemorySend =
      round_up<size_t>(hiddenDimBytes + hiddenDimScaleBytes + sizeof(uint32_t), sizeof(int4));
  const size_t sharedMemoryRecv = sizeof(uint32_t) * expertsPerBlock * 2;

  void *args[] = {
      const_cast<int32_t **>(&outNumTokensPerExpert.data),
      const_cast<size_t *>(&outNumTokensPerExpert.strideElem),
      const_cast<std::byte **>(&expertX.data),
      const_cast<size_t *>(&expertX.strideElem),
      const_cast<size_t *>(&expertX.strideRow),
      const_cast<float **>(&expertXScale.data),
      const_cast<size_t *>(&expertXScale.strideElem),
      const_cast<size_t *>(&expertXScale.strideRow),
      const_cast<size_t *>(&expertXScale.strideCol),
      const_cast<std::byte **>(&dpX.data),
      const_cast<size_t *>(&dpX.strideElem),
      const_cast<float **>(&dpXScale.data),
      const_cast<size_t *>(&dpXScale.strideElem),
      const_cast<size_t *>(&dpXScale.strideRow),
      const_cast<uint32_t **>(&indices.data),
      const_cast<size_t *>(&indices.strideElem),
      const_cast<size_t *>(&indices.strideRow),
      const_cast<size_t *>(&maxNumTokens),
      const_cast<size_t *>(&numExperts),
      const_cast<unsigned *>(&rank),
      const_cast<unsigned *>(&worldSize),
      const_cast<unsigned *>(&dpSize),
      const_cast<size_t *>(&hiddenDimBytes),
      const_cast<size_t *>(&hiddenDimScaleBytes),
      const_cast<size_t *>(&expertsPerToken),
      const_cast<unsigned **>(&boundM),
      &m,
      &sendBuffersPtr,
      &recvBuffersPtr,
      &tokenCount,
      &numTokensPerRank,
      &sourceExpert,
      &sourceIndex,
      &sourceOffset,
      &sourceRank,
      &sourceToken,
      &tokenIndex};

  nvtxRangePush("dispatch");
  switch (splitMode) {
  case SplitMode::SEND:
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&dispatchKernel<NUM_WARPS, true, false>,
        dimGrid,
        dimBlock,
        args,
        sharedMemorySend,
        stream
    ));
    break;
  case SplitMode::RECV:
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&dispatchKernel<NUM_WARPS, false, true>,
        dimGrid,
        dimBlock,
        args,
        sharedMemoryRecv,
        stream
    ));
    break;
  case SplitMode::NONE:
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&dispatchKernel<NUM_WARPS, true, true>,
        dimGrid,
        dimBlock,
        args,
        std::max(sharedMemorySend, sharedMemoryRecv),
        stream
    ));
    break;
  default:
    PPLX_UNREACHABLE("invalid split mode");
  }
  nvtxRangePop();
}
