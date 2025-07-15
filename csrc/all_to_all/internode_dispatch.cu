#include <cooperative_groups.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvtx3/nvToolsExt.h>

#include "all_to_all/internode.h"
#include "core/device_utils.cuh"
#include "core/utils.h"

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
    std::byte *dpX,
    size_t dpXStrideElem,
    float *dpXScale,
    size_t dpXScaleStrideElem,
    size_t dpXScaleStrideRow,
    uint32_t *indices,
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
    uint32_t *numTokensPerDP,
    uint32_t *sourceExpert,
    uint32_t *sourceIndex,
    uint32_t *sourceOffset,
    uint32_t *sourceGroup,
    uint32_t *sourceToken,
    uint64_t *numTokensBuffer,
    uint64_t *numRecvBuffer,
    uint32_t &globalTokenIndex,
    std::byte *xBufferIn,
    std::byte *xBufferOut
) {
  // Determine the rank, DP rank and per-rank constants.
  const unsigned numLocalExperts = numExperts / worldSize;
  const unsigned numDPGroups = worldSize / dpSize;
  const unsigned dpGroup = rank / dpSize;
  const unsigned dpRank = rank % dpSize;
  const unsigned tokenDim = hiddenDim + hiddenDimScale;
  const unsigned tokenStride = round_up<unsigned>(tokenDim + sizeof(uint32_t), sizeof(int4));
  const unsigned WARP_SIZE = 32;
  const unsigned warpId = threadIdx.x / WARP_SIZE;
  const unsigned laneId = threadIdx.x % WARP_SIZE;

  // Determine the number of tokens populated which are to be sent.
  const unsigned numSendTokens = boundM ? __ldg(boundM) : m;
  PPLX_DEVICE_ASSERT(numSendTokens <= maxNumTokens);
  PPLX_DEVICE_ASSERT(
      hiddenDimScale == 0 || numSendTokens == 0 || (expertXScale != nullptr && dpXScale != nullptr)
  );

  // Zero out the shared memory buffer.
  extern __shared__ std::byte sharedMemory[];
  if constexpr (DO_SEND) {
    uint32_t *tokenIndex = reinterpret_cast<uint32_t *>(sharedMemory);
    for (uint32_t i = threadIdx.x; i < numExperts; i += blockDim.x) {
      tokenIndex[i] = 0;
    }
    __syncthreads();

    if (warpId + 1 == NUM_WARPS) {
      // The experts are split across the available blocks.
      // The warp counts the number of tokens assigned to each expert.
      for (unsigned dstExpert = blockIdx.x * dpSize + dpRank; dstExpert < numExperts;
           dstExpert += gridDim.x * dpSize) {
        const uint32_t dstRank = dstExpert / numLocalExperts;
        const uint32_t dstLocalExpert = dstExpert % numLocalExperts;

        unsigned count = 0;

#pragma unroll
        for (uint32_t i = laneId; i < numSendTokens * numExpertsPerToken; i += WARP_SIZE) {
          unsigned expert = __ldg(&indices[i]);
          if (expert == dstExpert) {
            count += 1;
          }
        }

        unsigned numTokensPerExpert = device::warp_sum(count);
        uint64_t *dstCount = &numTokensBuffer[dstLocalExpert * numDPGroups + dpGroup];

        if (laneId == 0) {
          nvshmemx_signal_op(dstCount, numTokensPerExpert + 1, NVSHMEM_SIGNAL_SET, dstRank);
        }
      }

      // Clear out some buffers.
      if (blockIdx.x == 0) {
        for (uint32_t i = laneId; i < numLocalExperts; i += WARP_SIZE) {
          outNumTokensPerExpert[i] = 0;
        }
      }
    } else {
      // Send the tokens to the destination ranks through RDMA.
      const unsigned numGroupWarps = NUM_WARPS - 1;
      const unsigned numGroupThreads = numGroupWarps * WARP_SIZE;
      for (unsigned i = 0; i < numSendTokens; i++) {
        // Replicate the token count calculation across all blocks.
        if (threadIdx.x < numExpertsPerToken) {
          uint32_t dstExpert = __ldg(&indices[i * numExpertsPerToken + threadIdx.x]);
          tokenIndex[dstExpert]++;
        }
        // If the token is assigned to this block, handle it.
        if (i % (gridDim.x * dpSize) == (blockIdx.x * dpSize + dpRank)) {
          // Copy the token to the symmetric buffer.
          std::byte *xInPtr = xBufferIn + i * tokenStride;
          const int4 *srcX = (int4 *)(dpX + i * dpXStrideElem);
          for (unsigned d = threadIdx.x; d * sizeof(int4) < hiddenDim; d += numGroupThreads) {
            ((int4 *)xInPtr)[d] = srcX[d];
          }

          std::byte *xInScalePtr = xInPtr + hiddenDim;
          const float *srcXScale = dpXScale + i * dpXScaleStrideRow;
          for (unsigned d = threadIdx.x; d * sizeof(float) < hiddenDimScale; d += numGroupThreads) {
            ((float *)xInScalePtr)[d] = srcXScale[d * dpXScaleStrideElem];
          }

          if (threadIdx.x == 0) {
            *((uint32_t *)(xInPtr + tokenDim)) = i;
          }

          // Synchronize the warps within this warp group.
          asm volatile("bar.sync 1, %0;" ::"r"(numGroupThreads));

          // Send the token to the other ranks, one send per warp.
          for (unsigned j = warpId; j < numExpertsPerToken; j += numGroupWarps) {
            const uint32_t dstExpert = __ldg(&indices[i * numExpertsPerToken + j]);
            const uint32_t dstRank = dstExpert / numLocalExperts;
            const uint32_t dstLocalExpert = dstExpert % numLocalExperts;

            const uint32_t index = tokenIndex[dstExpert] - 1;
            const uint32_t group = dstLocalExpert * numDPGroups + dpGroup;
            const unsigned loc = group * maxNumTokens + index;

            std::byte *destPointer = xBufferOut + loc * tokenStride;
            nvshmemx_putmem_signal_nbi_warp(
                destPointer,
                xInPtr,
                tokenStride,
                &numRecvBuffer[group],
                1,
                NVSHMEM_SIGNAL_ADD,
                dstRank
            );
          }
        }
      }
    }

    if (DO_RECV) {
      cooperative_groups::this_grid().sync();
    }
  }

  if constexpr (DO_RECV) {
    // Wait for the token counts to be sent.
    const size_t numExpertsAndGroups = numLocalExperts * numDPGroups;
    const size_t expertsPerBlock = ceil_div<size_t>(numExpertsAndGroups, gridDim.x);
    uint32_t *sharedExpert = reinterpret_cast<uint32_t *>(sharedMemory);
    uint32_t *sharedToken = sharedExpert + expertsPerBlock;

    unsigned firstGroup = blockIdx.x * expertsPerBlock;
    unsigned lastGroup = std::min(firstGroup + expertsPerBlock, numExpertsAndGroups);

    for (unsigned group = firstGroup + threadIdx.x; group < lastGroup; group += blockDim.x) {
      const uint32_t expert = group / numDPGroups;

      // Fetch the token count per DP, which is non-zero to indicate receipt.
      // Afterwards, wait for exactly that many tokens to be sent to us.
      nvshmem_uint64_wait_until(&numTokensBuffer[group], NVSHMEM_CMP_NE, 0);
      size_t numTokens = numTokensBuffer[group] - 1;
      nvshmem_uint64_wait_until(&numRecvBuffer[group], NVSHMEM_CMP_EQ, numTokens);

      numTokensPerDP[group] = numTokens;
      numTokensBuffer[group] = 0;
      numRecvBuffer[group] = 0;
      sharedExpert[group - firstGroup] = atomicAdd(&outNumTokensPerExpert[expert], numTokens);
      sharedToken[group - firstGroup] = atomicAdd(&globalTokenIndex, numTokens);
    }

    __syncthreads();

    for (unsigned group = firstGroup; group < lastGroup; group++) {
      const uint32_t expert = group / numDPGroups;
      const uint32_t dp = group % numDPGroups;
      const size_t numTokens = numTokensPerDP[group];
      auto expertStart = sharedExpert[group - firstGroup];
      auto tokenStart = sharedToken[group - firstGroup];

      for (unsigned i = threadIdx.x; i < numTokens; i += blockDim.x) {
        std::byte *xTokenBuffer = xBufferOut + (group * maxNumTokens + i) * tokenStride;
        uint32_t token = tokenStart + i;
        sourceIndex[token] = *((uint32_t *)(xTokenBuffer + tokenDim));
        sourceExpert[token] = expert;
        sourceOffset[token] = expertStart + i;
        sourceGroup[token] = dp;
        sourceToken[token] = i;
      }
    }

    cooperative_groups::this_grid().sync();
    unsigned numRecvTokens = globalTokenIndex;

    for (unsigned i = blockIdx.x; i < numRecvTokens; i += gridDim.x) {
      auto expertLoc = sourceOffset[i];
      auto expert = sourceExpert[i];
      auto group = expert * numDPGroups + sourceGroup[i];

      std::byte *xTokenBuffer = xBufferOut + (group * maxNumTokens + sourceToken[i]) * tokenStride;
      std::byte *dstXExpert = expertX + expert * expertXStrideRow;
      float *dstXScaleExpert = expertXScale + expert * expertXScaleStrideCol;

      const int4 *srcX = (int4 *)xTokenBuffer;
      int4 *dstX = (int4 *)(dstXExpert + expertLoc * expertXStrideElem);
      for (unsigned k = threadIdx.x; k * sizeof(int4) < hiddenDim; k += blockDim.x) {
        dstX[k] = srcX[k];
      }

      // Copy the scale to the output buffer.
      if (hiddenDimScale > 0) {
        const float *srcXScale = (float *)(xTokenBuffer + hiddenDim);
        float *dstXScale = dstXScaleExpert + expertLoc * expertXScaleStrideRow;
        for (unsigned k = threadIdx.x; k * sizeof(float) < hiddenDimScale; k += blockDim.x) {
          dstXScale[k * expertXScaleStrideElem] = srcXScale[k];
        }
      }
    }
  }
}

} // namespace

void AllToAllInterNode::dispatch(
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
  constexpr unsigned NUM_WARPS = 10;
  const unsigned numBlocks = std::min(
      std::max(
          ceil_div<unsigned>(numExperts, NUM_WARPS), (unsigned)(maxNumTokens * expertsPerToken)
      ),
      static_cast<unsigned>(numSMs)
  );
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(NUM_WARPS * 32, 1, 1);

  const size_t expertsPerBlock = ceil_div<size_t>(numLocalExperts * numDPGroups, numBlocks);
  const size_t sharedMemorySend = sizeof(uint32_t) * numExperts;
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
      &numTokensPerDP,
      &sourceExpert,
      &sourceIndex,
      &sourceOffset,
      &sourceGroup,
      &sourceToken,
      &numTokensBuffer,
      &numDispatchRecvBuffer,
      &tokenIndex,
      &xDispatchIn,
      &xDispatchOut,
  };

  nvtxRangePush("dispatch");
  switch (splitMode) {
  case SplitMode::SEND:
    CUDACHECK(cudaLaunchKernel(
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
