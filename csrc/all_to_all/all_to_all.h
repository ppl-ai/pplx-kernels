#pragma once

#include <cstdint>
#include <cstdlib>

namespace pplx {

/// Specifies which part of a send-and-recv kernel to launch.
enum class SplitMode {
  NONE,
  SEND,
  RECV,
};

/// Base class for all-to-all broadcast kernels.
class AllToAll {
public:
  /// @brief Initializes the all-to-all broadcast kernel.
  ///
  /// @param maxNumTokens The maximum number of tokens per DP group.
  /// @param numExperts The total number of experts spread across all ranks.
  /// @param expertsPerToken The number of experts per token.
  /// @param rank The rank of the current process.
  /// @param worldSize The number of processes in the world.
  /// @param dpSize The size of a DP group.
  /// @param hiddenDimBytes The hidden dimension of X, in bytes.
  /// @param hiddenDimScaleBytes The hidden dimension of the scale of X, in
  /// bytes.
  AllToAll(
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

  virtual ~AllToAll();

  /// @brief Returns the number of experts each token is routed to.
  size_t getNumExpertsPerToken() const { return expertsPerToken; }

protected:
  /// The maximum number of tokens per DP group.
  const size_t maxNumTokens;
  /// The total number of experts spread across all ranks.
  const size_t numExperts;
  /// The number of local experts.
  const size_t numLocalExperts;
  /// The number of DP groups.
  const size_t numDPGroups;
  /// The number of experts per token.
  const size_t expertsPerToken;
  /// The hidden dimension of X, in elements.
  const size_t hiddenDim;
  /// The hidden dimension of X, in bytes.
  const size_t hiddenDimBytes;
  /// The hidden dimension scale of X, in bytes.
  const size_t hiddenDimScaleBytes;
  /// The rank of the current process.
  const unsigned rank;
  /// The number of processes in the world.
  const unsigned worldSize;
  /// The size of a DP group.
  const unsigned dpSize;
  /// The number of streaming multiprocessors (SMs) on the device.
  const int numSMs;
};

} // namespace pplx
