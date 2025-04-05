#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>

#include "all_to_all/all_to_all.h"
#include "core/buffer.h"

namespace pplx {

/// @brief All-to-all broadcast kernel.
///
/// This kernel keeps track of the intermediary buffers used by
/// the all-to-all broadcast and wraps around the kernel launcher.
class AllToAllInterNode final : public AllToAll {
public:
  AllToAllInterNode(
      size_t maxNumTokens,
      size_t numExperts,
      size_t expertsPerToken,
      unsigned rank,
      unsigned worldSize,
      unsigned dpSize,
      size_t hiddenDim,
      size_t hiddenDimBytes,
      size_t hiddenDimScaleBytes
  );

  ~AllToAllInterNode();

  /// Launches the all-to-all broadcast kernel.
  ///
  /// @param outTokensPerExpert An output array which contains the number of
  /// tokens routed to the experts. Shape: [numLocalExperts].
  ///
  /// @param expertX The output buffer for per-expert tokens.
  /// Shape: [numLocalExperts, maxNumTokens * numDPGroups, hiddenDim].
  ///
  /// @param expertXScale The output buffer for per-expert scale of X.
  /// Shape: [numLocalExperts, maxNumTokens * numDPGroups, hiddenDimScale].

  /// @param dpX The input tokens X.
  /// Shape: [m, hiddenDim].
  ///
  /// @param dpXScale The input scale of X.
  /// Shape: [m, hiddenDimScale].
  ///
  /// @param indices The input indices of size [numExperts, maxNumTokens].
  /// Shape: [numExperts, maxNumTokens].
  ///
  /// @param m The size of the buffers allocated for dpX/dpXScale.
  ///
  /// @param boundM The dynamic number of tokens populated in dpX/dpXScale.
  /// Shape: [1].
  ///
  /// @param doSend Whether to send the data to other ranks (for overlapping).
  ///
  /// @param doRecv Whether to receive the data from other ranks (for
  /// overlapping).
  ///
  /// @param stream The CUDA stream to launch the kernel on.
  void dispatch(
      const Strided1D<int32_t> &outTokensPerExpert,
      const Strided2D<std::byte> &expertX,
      const Strided2D<std::byte> &expertXScale,
      const Strided1D<std::byte> &dpX,
      const Strided1D<std::byte> &dpXScale,
      const Strided2D<uint32_t> &indices,
      unsigned m,
      const unsigned *boundM,
      SplitMode splitMode,
      cudaStream_t stream
  );

  /// Launches the all-to-all combine kernel.
  ///
  /// @param outTokens The output tokens.
  /// Shape: [numExperts, maxNumTokens].
  ///
  /// @param indices The input indices of size [numExperts, maxNumTokens].
  /// Shape: [numExperts, maxNumTokens].
  ///
  /// @param weights The input weights of size [numExperts, maxNumTokens].
  /// Shape: [numExperts, maxNumTokens].
  ///
  /// @param expertX The input tokens X.
  /// Shape: [numLocalExperts, maxNumTokens * numDPGroups, hiddenDim].
  ///
  /// @param numTokensPerExpert The number of tokens per expert.
  /// Shape: [numLocalExperts].
  ///
  /// @param m The size of the buffers allocated for dpX/dpXScale.
  ///
  /// @param boundM The dynamic number of tokens populated in dpX/dpXScale.
  /// Shape: [1].
  ///
  /// @param stream The CUDA stream to launch the kernel on.
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
  /// @section Pre-allocated symmetric shared memory workspace.
  uint64_t *numTokensBuffer = nullptr;
  uint64_t *numDispatchRecvBuffer = nullptr;
  uint64_t *combineSignalBuffer = nullptr;
  uint64_t *combineSyncBuffer = nullptr;
  std::byte *xDispatchIn = nullptr;
  std::byte *xDispatchOut = nullptr;
  std::byte *xCombineIn = nullptr;
  std::byte *xCombineOut = nullptr;
};

} // namespace pplx
