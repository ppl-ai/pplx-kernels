#include "all_to_all/intranode.cuh"
#include "core/atomic.cuh"
#include "core/device_utils.h"
#include "core/utils.h"
#include "intranode.h"

#include <cassert>

#include <cooperative_groups.h>
#include <nvtx3/nvToolsExt.h>

using namespace pplx;

namespace {

template <typename T, typename U, size_t NUM_WARPS, bool DO_SEND, bool DO_RECV>
__global__ __launch_bounds__(NUM_WARPS * 32, 1) void combineKernel(
    U *outTokens,
    size_t outTokensStrideElem,
    uint32_t *indices,
    size_t indicesStrideElem,
    size_t indicesStrideRow,
    float *weights,
    size_t weightsStrideElem,
    size_t weightsStrideRow,
    T *expertX,
    size_t expertXStrideElem,
    size_t expertXStrideRow,
    size_t expertsPerToken,
    size_t maxNumTokens,
    size_t numExperts,
    unsigned rank,
    unsigned worldSize,
    unsigned dpSize,
    size_t hiddenDim,
    unsigned *boundM,
    unsigned m,
    std::byte **sendBuffersPtr,
    std::byte **recvBuffersPtr,
    const uint32_t *sourceExpert,
    const uint32_t *sourceIndex,
    const uint32_t *sourceOffset,
    const uint32_t *sourceRank,
    uint32_t &globalTokenIndex
) {
  const unsigned numLocalExperts = numExperts / worldSize;
  const size_t tokenDim = hiddenDim * sizeof(T);
  const size_t tokenStride = device::round_up<size_t>(tokenDim, sizeof(int4));
  constexpr unsigned WARP_SIZE = 32;
  uint32_t warpId = threadIdx.x / WARP_SIZE;
  const unsigned laneId = threadIdx.x % WARP_SIZE;

  BufferWrapper remoteBuffer(recvBuffersPtr, numLocalExperts, worldSize, maxNumTokens, tokenStride);
  BufferWrapper localBuffer(sendBuffersPtr, numLocalExperts, worldSize, maxNumTokens, tokenStride);

  if (DO_SEND) {
    size_t numSendTokens = globalTokenIndex;

    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < worldSize;
         i += blockDim.x * gridDim.x) {
      while (ld_flag_volatile(&localBuffer.getDispatchSyncPtr(i)) != 0)
        ;
      st_flag_volatile(&remoteBuffer.getCombineSyncPtr(i), 1);
    }

    // Dispatch the tokens from the expert to the DP groups.
    for (uint32_t token = blockIdx.x; token < numSendTokens; token += gridDim.x) {
      const uint32_t expert = __ldg(&sourceExpert[token]);
      const uint32_t index = __ldg(&sourceIndex[token]);
      const uint32_t offset = __ldg(&sourceOffset[token]);
      const uint32_t rank = __ldg(&sourceRank[token]);

      const uint32_t dstLocalExpert = expert % numLocalExperts;

      const T *source = expertX + expert * expertXStrideRow + offset * expertXStrideElem;
      const unsigned n = tokenDim / sizeof(float4);

      auto copy = [&](unsigned rank, unsigned start, unsigned step) {
        std::byte *buffer = remoteBuffer.getTokenPtr(rank, dstLocalExpert, index);
        float4 *srcPtr = (float4 *)source;
        float4 *dstPtr = (float4 *)buffer;

        srcPtr += start;
        dstPtr += start;

#pragma unroll(4)
        for (unsigned j = start; j < n; j += step) {
          *dstPtr = *srcPtr;
          dstPtr += step;
          srcPtr += step;
        }
      };

      if (dpSize == 1) {
        copy(rank, threadIdx.x, blockDim.x);
      } else {
        for (unsigned i = warpId; i < dpSize; i += NUM_WARPS) {
          copy((rank / dpSize) * dpSize + i, laneId, WARP_SIZE);
        }
      }
    }

    cooperative_groups::this_grid().sync();

    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < worldSize;
         i += blockDim.x * gridDim.x) {
      st_flag_release(&remoteBuffer.getCountPtr(i, 0), 1);
    }
  }

  // Synchronize the grid to ensure that tokens routed within the rank are
  // correctly transported from one block to another.
  if (DO_RECV) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < worldSize;
         i += blockDim.x * gridDim.x) {
      while (ld_flag_acquire(&localBuffer.getCountPtr(i, 0)) == 0)
        ;
      st_flag_volatile(&localBuffer.getCountPtr(i, 0), 0);
    }

    cooperative_groups::this_grid().sync();
    globalTokenIndex = 0;

    // Compute the weighed sum of the input tokens.
    const size_t numRecvTokens = boundM ? __ldg(boundM) : m;
    for (unsigned i = blockIdx.x; i < numRecvTokens; i += gridDim.x) {
      U *dstPtr = outTokens + i * outTokensStrideElem;
      constexpr unsigned VEC_SIZE = 8;
      for (unsigned j = threadIdx.x * VEC_SIZE; j < hiddenDim; j += blockDim.x * VEC_SIZE) {
        float sum[VEC_SIZE];

#pragma unroll
        for (unsigned l = 0; l < VEC_SIZE; ++l) {
          sum[l] = 0.0f;
        }

        for (unsigned k = 0; k < expertsPerToken; ++k) {
          const uint32_t expert = indices[i * expertsPerToken + k];
          const uint32_t dstRank = expert / numLocalExperts;
          const uint32_t dstLocalExpert = expert % numLocalExperts;
          const float weight = weights[i * weightsStrideRow + k];

#pragma unroll
          for (unsigned l = 0; l < VEC_SIZE; ++l) {
            std::byte *buffer = localBuffer.getTokenPtr(dstRank, dstLocalExpert, i);
            sum[l] += weight * (float)((T *)buffer)[j + l];
          }
        }

#pragma unroll
        for (unsigned l = 0; l < VEC_SIZE; ++l) {
          dstPtr[j + l] = sum[l];
        }
      }
    }

    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < worldSize;
         i += blockDim.x * gridDim.x) {
      st_flag_volatile(&remoteBuffer.getCombineSyncPtr(i), 0);
    }
  }
}
} // namespace

template <typename T, typename U>
void AllToAllIntraNode::combine(
    const Strided1D<U> &outTokens,
    const Strided2D<uint32_t> &indices,
    const Strided2D<float> &weights,
    const Strided2D<T> &expertX,
    unsigned m,
    const unsigned *boundM,
    SplitMode splitMode,
    cudaStream_t stream
) {
  constexpr size_t NUM_WARPS = 32;

  const size_t numLocalExperts = numExperts / worldSize;
  const size_t numDPGroups = worldSize / dpSize;
  const size_t batchNumTokens = numLocalExperts * numDPGroups * maxNumTokens;
  const size_t numBlocks = std::min(static_cast<size_t>(numSMs), batchNumTokens);

  assert(hiddenDimBytes % 16 == 0);

  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(NUM_WARPS * 32, 1, 1);

  void *args[] = {
      const_cast<U **>(&outTokens.data),
      const_cast<size_t *>(&outTokens.strideElem),
      const_cast<uint32_t **>(&indices.data),
      const_cast<size_t *>(&indices.strideElem),
      const_cast<size_t *>(&indices.strideRow),
      const_cast<float **>(&weights.data),
      const_cast<size_t *>(&weights.strideElem),
      const_cast<size_t *>(&weights.strideRow),
      const_cast<T **>(&expertX.data),
      const_cast<size_t *>(&expertX.strideElem),
      const_cast<size_t *>(&expertX.strideRow),
      const_cast<size_t *>(&expertsPerToken),
      const_cast<size_t *>(&maxNumTokens),
      const_cast<size_t *>(&numExperts),
      const_cast<unsigned *>(&rank),
      const_cast<unsigned *>(&worldSize),
      const_cast<unsigned *>(&dpSize),
      const_cast<size_t *>(&hiddenDim),
      const_cast<unsigned **>(&boundM),
      &m,
      &sendBuffersPtr,
      &recvBuffersPtr,
      &sourceExpert,
      &sourceIndex,
      &sourceOffset,
      &sourceRank,
      &tokenIndex,
  };

  nvtxRangePush("combine");
  switch (splitMode) {
  case SplitMode::SEND:
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&combineKernel<T, U, NUM_WARPS, true, false>, dimGrid, dimBlock, args, 0, stream
    ));
    break;
  case SplitMode::RECV:
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&combineKernel<T, U, NUM_WARPS, false, true>, dimGrid, dimBlock, args, 0, stream
    ));
    break;
  case SplitMode::NONE:
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&combineKernel<T, U, NUM_WARPS, true, true>, dimGrid, dimBlock, args, 0, stream
    ));
    break;
  default:
    ROSE_UNREACHABLE("invalid split mode");
  }
  nvtxRangePop();
}

#define INSTANTIATE_COMBINE(T, U)                                                                  \
  template void AllToAllIntraNode::combine<T, U>(                                                  \
      const Strided1D<U> &outTokens,                                                               \
      const Strided2D<uint32_t> &indices,                                                          \
      const Strided2D<float> &weights,                                                             \
      const Strided2D<T> &expertX,                                                                 \
      unsigned m,                                                                                  \
      const unsigned *boundM,                                                                      \
      SplitMode splitMode,                                                                         \
      cudaStream_t stream                                                                          \
  );

INSTANTIATE_COMBINE(float, nv_bfloat16)
INSTANTIATE_COMBINE(float, half)
INSTANTIATE_COMBINE(half, nv_bfloat16)
INSTANTIATE_COMBINE(half, half)
INSTANTIATE_COMBINE(nv_bfloat16, nv_bfloat16)
INSTANTIATE_COMBINE(nv_bfloat16, half)
