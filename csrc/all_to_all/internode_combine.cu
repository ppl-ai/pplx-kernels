#include <cuda.h>
#include <nvshmem.h>
#include <nvtx3/nvToolsExt.h>

#include "all_to_all/internode.h"
#include "core/nvshmem_utils.h"
#include "core/utils.h"

using namespace pplx;

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
    uint32_t *sourceExpert,
    const uint32_t *sourceIndex,
    const uint32_t *sourceOffset,
    const uint32_t *sourceGroup,
    uint64_t *combineSignalBuffer,
    uint64_t *combineSyncBuffer,
    std::byte *xBufferIn,
    std::byte *xBufferOut
) {
  const unsigned numLocalExperts = numExperts / worldSize;
  const unsigned numDPGroups = worldSize / dpSize;
  const unsigned maxBatchTokens = numLocalExperts * numDPGroups * maxNumTokens;
  const size_t stride = hiddenDim * sizeof(T);
  uint32_t warpId = threadIdx.x / 32;
  const unsigned laneId = threadIdx.x % 32;
  const unsigned numWarps = blockDim.x / 32;

  if (DO_SEND) {
    for (unsigned i = blockIdx.x * numWarps + warpId; i < worldSize; i += gridDim.x * numWarps) {
      if (laneId == 0) {
        nvshmemx_signal_op(&combineSyncBuffer[rank], 1, NVSHMEM_SIGNAL_SET, i);
      }
    }

    // Dispatch the tokens from the expert to the DP groups.
    for (uint32_t token = blockIdx.x; token < maxBatchTokens; token += gridDim.x) {
      const uint32_t expertPlusOne = sourceExpert[token];
      __syncthreads();
      if (expertPlusOne == 0) {
        break;
      }
      sourceExpert[token] = 0;

      const uint32_t source = __ldg(&sourceIndex[token]);
      const uint32_t offset = __ldg(&sourceOffset[token]);
      const uint32_t dp = __ldg(&sourceGroup[token]);
      const uint32_t expert = expertPlusOne - 1;

      // Copy the token to shared memory for send.
      const int4 *expertXTokenPtr =
          (int4 *)(expertX + expert * expertXStrideRow + offset * expertXStrideElem);
      int4 *xTokenPtr = (int4 *)(xBufferIn + token * stride);
      for (unsigned j = threadIdx.x; j * sizeof(int4) < stride; j += blockDim.x) {
        xTokenPtr[j] = expertXTokenPtr[j];
      }
      __syncthreads();

      const uint32_t dstExpert = rank * numLocalExperts + expert;

      for (unsigned i = warpId; i < dpSize; i += numWarps) {
        const int dstRank = dp * dpSize + i;
        const unsigned index = dstExpert * maxNumTokens + source;
        std::byte *dstPtr = xBufferOut + index * stride;
        nvshmemx_putmem_signal_nbi_warp(
            dstPtr, xTokenPtr, stride, &combineSignalBuffer[source], 1, NVSHMEM_SIGNAL_ADD, dstRank
        );
      }
    }
  }

  // Synchronize the grid to ensure that tokens routed within the rank are
  // correctly transported from one block to another.
  if (DO_RECV) {
    if (DO_SEND) {
      cg::this_grid().sync();
    }

    // Compute the weighed sum of the input tokens.
    const size_t localNumTokens = boundM ? __ldg(boundM) : m;
    for (unsigned i = blockIdx.x; i < localNumTokens; i += gridDim.x) {
      nvshmem_uint64_wait_until(&combineSignalBuffer[i], NVSHMEM_CMP_EQ, expertsPerToken);
      __syncthreads();
      combineSignalBuffer[i] = 0;

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
          const float weight = weights[i * weightsStrideRow + k];

#pragma unroll
          for (unsigned l = 0; l < VEC_SIZE; ++l) {
            std::byte *xDstPtr = xBufferOut + (expert * maxNumTokens + i) * stride;
            sum[l] += weight * (float)((T *)xDstPtr)[j + l];
          }
        }

#pragma unroll
        for (unsigned l = 0; l < VEC_SIZE; ++l) {
          dstPtr[j + l] = sum[l];
        }
      }
    }

    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < worldSize;
         i += gridDim.x * blockDim.x) {
      nvshmem_uint64_wait_until(&combineSyncBuffer[i], NVSHMEM_CMP_EQ, 1);
      combineSyncBuffer[i] = 0;
    }
  }
}

template <typename T, typename U>
void AllToAllInterNode::combine(
    const Strided1D<U> &outTokens,
    const Strided2D<uint32_t> &indices,
    const Strided2D<float> &weights,
    const Strided2D<T> &expertX,
    unsigned m,
    const unsigned *boundM,
    SplitMode splitMode,
    cudaStream_t stream
) {

  constexpr size_t NUM_WARPS = 8;

  const size_t numLocalExperts = numExperts / worldSize;
  const size_t numDPGroups = worldSize / dpSize;
  const size_t batchNumTokens = numLocalExperts * numDPGroups * maxNumTokens;
  const size_t numBlocks = std::min(132ul, batchNumTokens);

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
      &sourceExpert,
      &sourceIndex,
      &sourceOffset,
      &sourceGroup,
      &combineSignalBuffer,
      &combineSyncBuffer,
      &xCombineIn,
      &xCombineOut};

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

#define INSTANTIATE_COMBINE(T, U)                                                       \
  template void AllToAllInterNode::combine<T, U>(                                       \
      const Strided1D<U> &outTokens,                                                    \
      const Strided2D<uint32_t> &indices,                                               \
      const Strided2D<float> &weights,                                                  \
      const Strided2D<T> &expertX,                                                      \
      unsigned m,                                                                       \
      const unsigned *boundM,                                                           \
      SplitMode splitMode,                                                              \
      cudaStream_t stream                                                               \
  );

INSTANTIATE_COMBINE(float, nv_bfloat16)
INSTANTIATE_COMBINE(half, nv_bfloat16)
INSTANTIATE_COMBINE(nv_bfloat16, nv_bfloat16)
INSTANTIATE_COMBINE(float, half)
INSTANTIATE_COMBINE(half, half)
INSTANTIATE_COMBINE(nv_bfloat16, half)
