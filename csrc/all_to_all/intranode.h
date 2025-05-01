#pragma once

#include <cstddef>
#include <cstdint>

#include <memory>
#include <vector>

#include <cuda_bf16.h>

#include "all_to_all/all_to_all.h"
#include "core/buffer.h"

namespace pplx {
class Distributed;

/// @brief All-to-all broadcast kernel.
///
/// This kernel keeps track of the intermediary buffers used by
/// the all-to-all broadcast and wraps around the kernel launcher.
class AllToAllIntraNode final : public AllToAll {
public:
  AllToAllIntraNode(
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
  );

  ~AllToAllIntraNode();

  /// Launches the all-to-all dispatch kernel.
  void dispatch(
      const Strided1D<int32_t> &outTokensPerExpert,
      const Strided2D<std::byte> &expertX,
      const Strided3D<float> &expertXScale,
      const Strided1D<std::byte> &dpX,
      const Strided2D<float> &dpXScale,
      const Strided2D<uint32_t> &indices,
      unsigned m,
      const unsigned *boundM,
      SplitMode splitMode,
      cudaStream_t stream
  );

  /// Launches the all-to-all combine kernel.
  template <typename T, typename U>
  void combine(
      const Strided1D<U> &outTokens,
      const Strided2D<uint32_t> &indices,
      const Strided2D<float> &weights,
      const Strided2D<T> &expertX,
      unsigned m,
      const unsigned *boundM,
      SplitMode splitMode,
      cudaStream_t stream
  );

private:
  /// @section Peer-to-Peer shared buffers.
  std::vector<std::byte *> sendBuffers;
  std::byte **sendBuffersPtr;
  std::vector<std::byte *> recvBuffers;
  std::byte **recvBuffersPtr;

  /// Buffer to synchronize multiple senders with a receiver in dispatch.
  uint32_t *localRecvCountPtr;
  std::vector<uint32_t *> countBuffers;
  uint32_t **countBuffersPtr;

  /// @section Global buffers for use within kernels.
  uint32_t *numTokensPerRank;
  uint32_t *tokenCount;

  /// @section Internal buffers communicating between dispatch and combine.
  uint32_t *sourceIndex;
  uint32_t *sourceExpert;
  uint32_t *sourceOffset;
  uint32_t *sourceRank;
  uint32_t *sourceToken;
  uint32_t *sourceRoute;
  uint32_t *tokenIndex;
};

} // namespace pplx
