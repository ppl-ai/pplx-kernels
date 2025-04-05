// All-to-all kernel test

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <iostream>
#include <random>

#include "all_to_all/internode.h"
#include "all_to_all/test_utils.h"
#include "core/buffer.h"
#include "core/cuda_utils.h"
#include "core/mpi_utils.h"
#include "core/utils.h"

using namespace pplx;

template <typename T, typename U, typename Kernel>
bool testDispatchCombine(
    cudaStream_t stream,
    unsigned dpRank,
    unsigned dpSize,
    unsigned epRank,
    unsigned epSize,
    uint32_t numExperts = 8,
    size_t expertsPerToken = 3,
    size_t hiddenDim = 16,
    unsigned seed = 0xdeadbeef,
    size_t minNumTokens = 5,
    size_t maxNumTokens = 10,
    size_t blockSize = 2
) {
  const uint32_t numDPGroups = epSize / dpSize;

  if (epRank == 0) {
    std::cout << std::endl << "Starting the broadcast test" << std::endl;
  }

  // Generate the same test data on all ranks.
  // Compute the expected values for the local experts.
  std::vector<RankTestData<T>> rankTestData;
  std::vector<unsigned> expectedExpertIndptr(numExperts);
  std::vector<std::vector<unsigned>> expectedNumTokens(numExperts);
  std::mt19937 gen(seed);
  for (unsigned i = 0; i < numDPGroups; ++i) {
    auto &rank = rankTestData.emplace_back(
        gen, maxNumTokens, numExperts, expertsPerToken, hiddenDim, blockSize
    );

    for (unsigned j = 0; j < numExperts; ++j) {
      auto m = rank.numRouted[j];
      expectedExpertIndptr[j] += m;
      expectedNumTokens[j].push_back(m);
    }

    if (epRank == 0) {
      std::cout << "DP Rank #" << i << ":" << std::endl;
      std::cout << rank << std::endl;
    }
  }

  auto &rank = rankTestData[dpRank];
  DeviceBuffer<T> xDevice(rank.x);
  DeviceBuffer<float> xScaleDevice(rank.xScale);
  DeviceBuffer<uint32_t> indicesDevice(rank.indices);
  DeviceBuffer<float> weightsDevice(rank.weights);

  const unsigned expertsPerRank = numExperts / epSize;
  DeviceBuffer<int32_t> outTokensPerExpertDevice(expertsPerRank);
  DeviceBuffer<T> outExpertDevice(expertsPerRank * maxNumTokens * numDPGroups * rank.hiddenDim);
  DeviceBuffer<float> outExpertScaleDevice(
      expertsPerRank * maxNumTokens * numDPGroups * rank.hiddenDimScale
  );
  DeviceBuffer<U> outTokensDevice(maxNumTokens * hiddenDim);

  const size_t hiddenDimBytes = rank.hiddenDim * sizeof(T);
  const size_t hiddenDimScaleBytes = rank.hiddenDimScale * sizeof(float);

  Kernel allToAll(
      maxNumTokens,
      numExperts,
      expertsPerToken,
      epRank,
      epSize,
      dpSize,
      rank.hiddenDim,
      hiddenDimBytes,
      hiddenDimScaleBytes
  );

  allToAll.dispatch(
      Strided1D<int32_t>(outTokensPerExpertDevice, 1),
      Strided2D<std::byte>(
          outExpertDevice, hiddenDimBytes, hiddenDimBytes * maxNumTokens * numDPGroups
      ),
      Strided2D<std::byte>(
          outExpertScaleDevice,
          hiddenDimScaleBytes,
          hiddenDimScaleBytes * maxNumTokens * numDPGroups
      ),
      Strided1D<std::byte>(xDevice, hiddenDimBytes),
      Strided1D<std::byte>(xScaleDevice, hiddenDimScaleBytes),
      Strided2D<uint32_t>(indicesDevice, 1, expertsPerToken),
      rank.m,
      nullptr,
      SplitMode::NONE,
      stream
  );
  CUDACHECK(cudaStreamSynchronize(stream));

  allToAll.combine(
      Strided1D<U>(outTokensDevice, hiddenDim),
      Strided2D<uint32_t>(indicesDevice, 1, expertsPerToken),
      Strided2D<float>(weightsDevice, 1, expertsPerToken),
      Strided2D<T>(outExpertDevice, hiddenDim, hiddenDim * maxNumTokens * numDPGroups),
      rank.m,
      nullptr,
      SplitMode::NONE,
      stream
  );
  CUDACHECK(cudaStreamSynchronize(stream));

  HostBuffer<int32_t> outNumTokensPerExpertHost(outTokensPerExpertDevice);
  HostBuffer<T> outExpertHost(outExpertDevice);
  HostBuffer<float> outExpertScaleHost(outExpertScaleDevice);
  HostBuffer<U> outTokensHost(outTokensDevice);

  // Print the results.
  for (unsigned i = 0; i < epSize; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);

    // Print per-expert results.
    if (i == epRank) {
      for (size_t j = 0; j < expertsPerRank; ++j) {
        unsigned expert = i * expertsPerRank + j;
        const size_t indptr = outNumTokensPerExpertHost[j];

        std::cout << "Expert #" << expert << " (" << indptr << "): " << std::flush << std::endl;

        unsigned token = 0;
        size_t offset = j * maxNumTokens * numDPGroups;
        size_t offsetScale = j * maxNumTokens * numDPGroups;
        for (unsigned dp = 0; dp < numDPGroups; ++dp) {
          auto numTokens = rankTestData[dp].numRouted[expert];
          for (unsigned index = 0; index < numTokens; ++index) {
            auto rankM = rankTestData[dp].m;
            std::cout << "#" << token << " (from " << dp << ")" << std::endl;
            std::cout << "    ";
            for (size_t l = 0; l < rank.hiddenDim; ++l) {
              std::cout << (float)outExpertHost[(offset + token) * rank.hiddenDim + l] << " ";
            }
            std::cout << std::flush << std::endl;
            std::cout << "    ";
            for (size_t l = 0; l < rank.hiddenDimScale; ++l) {
              std::cout << outExpertScaleHost[(offsetScale + token) * rank.hiddenDimScale + l]
                        << " ";
            }
            std::cout << std::flush << std::endl;

            ++token;
          }
        }
      }
      std::cout << std::flush << std::endl;
    }

    // Print DP group results.
    if (i == epRank && dpRank * dpSize == epRank) {
      std::cout << "DP Group #" << dpRank << ": " << std::endl;
      auto &dpRankData = rankTestData[dpRank];
      for (size_t j = 0; j < dpRankData.m; ++j) {
        std::cout << "#" << j << ": ";
        for (size_t k = 0; k < expertsPerToken; ++k) {
          const float e = dpRankData.indices[j * expertsPerToken + k];
          const float w = dpRankData.weights[j * expertsPerToken + k];
          if (k > 0) {
            std::cout << " + ";
          }
          std::cout << e << " * " << w;
        }
        std::cout << std::endl;

        std::cout << "r = ";
        for (size_t l = 0; l < hiddenDim; ++l) {
          std::cout << (float)outTokensHost[j * hiddenDim + l] << " ";
        }
        std::cout << std::endl;

        std::cout << "e = ";
        for (size_t l = 0; l < hiddenDim; ++l) {
          float sum = 0.0f;
          for (size_t k = 0; k < expertsPerToken; ++k) {
            const float w = dpRankData.weights[j * expertsPerToken + k];
            sum += w * (float)dpRankData.x[j * hiddenDim + l];
          }
          std::cout << sum << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::flush << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Verify the results.
  bool failed = false;
  for (unsigned i = 0; i < epSize; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t j = 0; j < expertsPerRank; ++j) {
      const unsigned expert = epRank * expertsPerRank + j;
      const auto indptr = outNumTokensPerExpertHost[j];

      const auto expectedIndptr = expectedExpertIndptr[expert];
      if (!failed && indptr != expectedIndptr) {
        std::cerr << "expert_indptr[" << expert << "]:" << indptr << " != " << expectedIndptr
                  << std::endl;
        failed = true;
        continue;
      }

      unsigned token = 0;
      size_t offset = j * maxNumTokens * numDPGroups;
      size_t offsetScale = j * maxNumTokens * numDPGroups;
      for (unsigned dp = 0; dp < numDPGroups; ++dp) {
        auto &rankData = rankTestData[dp];

        for (unsigned t = 0; t < rankData.m; ++t) {
          bool found = false;
          for (unsigned l = 0; l < expertsPerToken; ++l) {
            if (rankData.indices[t * expertsPerToken + l] == expert) {
              found = true;
              break;
            }
          }
          if (!found) {
            continue;
          }

          for (size_t l = 0; l < rank.hiddenDim; ++l) {
            auto x1 = outExpertHost[(offset + token) * rank.hiddenDim + l];
            auto x2 = rankTestData[dp].x[t * rank.hiddenDim + l];
            if (!failed && x1 != x2) {
              std::cerr << "Token mismatch at " << expert << " " << dp << " " << token << std::endl;
              failed = true;
              continue;
            }
          }
          for (size_t l = 0; l < rank.hiddenDimScale; ++l) {
            auto x1 = outExpertScaleHost[(offsetScale + token) * rank.hiddenDimScale + l];
            auto x2 = rankTestData[dp].xScale[t * rank.hiddenDimScale + l];
            if (!failed && x1 != x2) {
              std::cerr << "Scale mismatch at " << expert << " " << dp << " " << token << std::endl;
              failed = true;
              continue;
            }
          }

          ++token;
        }
      }
    }

    auto &dpRankData = rankTestData[dpRank];
    for (size_t j = 0; j < dpRankData.m; ++j) {
      for (size_t l = 0; l < hiddenDim; ++l) {
        const float x = (float)outTokensHost[j * hiddenDim + l];

        float sum = 0.0f;
        for (size_t k = 0; k < expertsPerToken; ++k) {
          const float w = dpRankData.weights[j * expertsPerToken + k];
          sum += w * (float)dpRankData.x[j * hiddenDim + l];
        }

        if (abs(x - sum) > 5e1 - 1) {
          std::cerr << "Result mismatch at " << dpRank << " " << j << " " << l << ": " << x
                    << "!=" << sum << std::endl;
          failed = true;
          continue;
        }
      }
    }
  }

  if (failed) {
    std::cout << "Failed" << std::flush << std::endl;
  }

  return !failed;
}

int main(int argc, char **argv) {
  // Set up MPI.
  int world_size, rank;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
  if (world_size < 4) {
    std::cout << "This test requires at least 4 workers" << std::endl;
    MPICHECK(MPI_Finalize());
    return EXIT_FAILURE;
  }

  // Set up NVSHMEM.
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  int currentPE = nvshmem_my_pe();
  int numPEs = nvshmem_n_pes();

  // Set up the current rank.
  int deviceId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  CUDACHECK(cudaSetDevice(deviceId));
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // Run the tests.
  int exit_code = EXIT_SUCCESS;
  if (!testDispatchCombine<float, nv_bfloat16, AllToAllInterNode>(
          stream, rank / 2, 2, rank, world_size
      )) {
    exit_code = EXIT_FAILURE;
  }
  if (!testDispatchCombine<nv_bfloat16, nv_bfloat16, AllToAllInterNode>(
          stream, rank / 2, 2, rank, world_size
      )) {
    exit_code = EXIT_FAILURE;
  }
  if (!testDispatchCombine<float, half, AllToAllInterNode>(stream, rank / 2, 2, rank, world_size)) {
    exit_code = EXIT_FAILURE;
  }
  if (!testDispatchCombine<nv_bfloat16, half, AllToAllInterNode>(
          stream, rank / 2, 2, rank, world_size
      )) {
    exit_code = EXIT_FAILURE;
  }

  // Cleanup.
  CUDACHECK(cudaStreamDestroy(stream));
  nvshmem_barrier_all();
  nvshmem_finalize();
  MPICHECK(MPI_Finalize());
  return exit_code;
}
