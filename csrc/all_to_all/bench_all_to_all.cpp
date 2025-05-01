// All-to-all dispatch / combine benchmark

#include "all_to_all/internode.h"
#include "all_to_all/intranode.h"
#include "all_to_all/test_utils.h"
#include "core/cuda_utils.h"
#include "core/distributed.h"
#include "core/kernels.h"
#include "core/mpi_utils.h"

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_profiler_api.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <nvtx3/nvToolsExt.h>

#include <array>
#include <iomanip>
#include <iostream>

using namespace pplx;

namespace {

struct BenchConfig {
  int numTokens;
  int numExperts;
  int expertsPerToken;
  int hiddenDim;
  int blockSize;
};

struct Time {
  float mean;
  float stddev;
};

std::ostream &operator<<(std::ostream &os, const Time &time) {
  os << std::setw(8) << std::fixed << std::setprecision(2) << time.mean << "us ± " << std::setw(5)
     << time.stddev << "us";
  return os;
}

Time average(const std::vector<float> &timesUs) {
  float sum = 0.0f, sumSquared = 0.0f;
  for (const float time : timesUs) {
    sum += time;
    sumSquared += time * time;
  }
  const float mean = sum / timesUs.size();
  const float stddev = std::sqrt(sumSquared / timesUs.size() - mean * mean);
  return {mean * 1000, stddev * 1000};
}

float duration(cudaEvent_t start, cudaEvent_t end) {
  float ms = 0.0f;
  CUDACHECK(cudaEventElapsedTime(&ms, start, end));
  return ms;
}

template <typename Kernel, typename T, typename U, bool HAS_SCALE, typename... Args>
std::tuple<Time, Time, Time, Time, Time, Time> benchConfig(
    const BenchConfig &config,
    unsigned repeat,
    unsigned currentPE,
    unsigned numPEs,
    cudaStream_t stream,
    Args &&...args
) {
  // Generate test data.
  std::mt19937 gen(currentPE);
  RankTestData<T> data(
      gen,
      config.numTokens,
      config.numExperts,
      config.expertsPerToken,
      config.hiddenDim,
      config.blockSize
  );

  const unsigned expertsPerRank = config.numExperts / numPEs;
  DeviceBuffer<int32_t> outTokensPerExpertDevice(expertsPerRank);
  DeviceBuffer<T> outExpertDevice(expertsPerRank * config.numTokens * numPEs * config.hiddenDim);
  DeviceBuffer<nv_bfloat16> inExpertDevice(
      expertsPerRank * config.numTokens * numPEs * config.hiddenDim
  );
  DeviceBuffer<float> outExpertScaleDevice(
      expertsPerRank * config.numTokens * numPEs * data.hiddenDimScale
  );
  DeviceBuffer<nv_bfloat16> outTokensDevice(config.numTokens * data.hiddenDim);
  DeviceBuffer<T> xDevice(data.x);
  DeviceBuffer<float> xScaleDevice(data.xScale);
  DeviceBuffer<uint32_t> indicesDevice(data.indices);
  DeviceBuffer<float> weightsDevice(data.weights);

  const size_t hiddenDimBytes = data.hiddenDim * sizeof(T);
  const size_t hiddenDimScaleBytes = HAS_SCALE ? data.hiddenDimScale * sizeof(float) : 0;

  Kernel allToAll(
      config.numTokens,
      config.numExperts,
      config.expertsPerToken,
      currentPE,
      numPEs,
      1,
      config.hiddenDim,
      hiddenDimBytes,
      hiddenDimScaleBytes,
      std::forward<Args>(args)...
  );

  MPI_Barrier(MPI_COMM_WORLD);

  constexpr size_t numSamples = 10;

  std::array<std::array<cudaEvent_t, 12>, numSamples> events;
  for (size_t i = 0; i < numSamples; ++i) {
    for (auto &event : events[i]) {
      CUDACHECK(cudaEventCreate(&event));
    }
  }

  auto dispatch = [&](SplitMode mode) {
    allToAll.dispatch(
        Strided1D<int32_t>(outTokensPerExpertDevice, 1),
        Strided2D<std::byte>(
            outExpertDevice, hiddenDimBytes, hiddenDimBytes * config.numTokens * numPEs
        ),
        HAS_SCALE ? Strided3D<float>(
                        outExpertScaleDevice,
                        1,
                        data.hiddenDimScale,
                        data.hiddenDimScale * config.numTokens * numPEs
                    )
                  : Strided3D<float>(nullptr, 0, 0, 0),
        Strided1D<std::byte>(xDevice, hiddenDimBytes),
        HAS_SCALE ? Strided2D<float>(xScaleDevice, 1, data.hiddenDimScale)
                  : Strided2D<float>(nullptr, 0, 0),
        Strided2D<uint32_t>(indicesDevice, 1, config.expertsPerToken),
        data.m,
        nullptr,
        mode,
        stream
    );
  };

  auto combine = [&](SplitMode mode) {
    allToAll.combine(
        Strided1D<nv_bfloat16>(outTokensDevice, config.hiddenDim),
        Strided2D<uint32_t>(indicesDevice, 1, config.expertsPerToken),
        Strided2D<float>(weightsDevice, 1, config.expertsPerToken),
        Strided2D<U>(
            inExpertDevice, config.hiddenDim, config.hiddenDim * config.numTokens * numPEs
        ),
        data.m,
        nullptr,
        mode,
        stream
    );
  };

  // Warmup
  std::vector<float> dispatchTimeUs, dispatchSendTimeUs, dispatchRecvTimeUs;
  std::vector<float> combineTimeUs, combineSendTimeUs, combineRecvTimeUs;
  auto run = [&](bool warmup) {
    nvshmemx_barrier_all_on_stream(stream);
    // Dispatch.
    for (size_t i = 0; i < numSamples; i++) {
      nvshmemx_barrier_all_on_stream(stream);

      CUDACHECK(cudaEventRecord(events[i][0], stream));
      dispatch(SplitMode::NONE);
      CUDACHECK(cudaEventRecord(events[i][1], stream));

      CUDACHECK(cudaEventRecord(events[i][2], stream));
      combine(SplitMode::NONE);
      CUDACHECK(cudaEventRecord(events[i][3], stream));

      nvshmemx_barrier_all_on_stream(stream);

      CUDACHECK(cudaEventRecord(events[i][4], stream));
      dispatch(SplitMode::SEND);
      CUDACHECK(cudaEventRecord(events[i][5], stream));

      sleepOnStream(0.0001, stream);
      nvshmemx_barrier_all_on_stream(stream);

      CUDACHECK(cudaEventRecord(events[i][6], stream));
      dispatch(SplitMode::RECV);
      CUDACHECK(cudaEventRecord(events[i][7], stream));

      CUDACHECK(cudaEventRecord(events[i][8], stream));
      combine(SplitMode::SEND);
      CUDACHECK(cudaEventRecord(events[i][9], stream));

      sleepOnStream(0.0001, stream);
      nvshmemx_barrier_all_on_stream(stream);

      CUDACHECK(cudaEventRecord(events[i][10], stream));
      combine(SplitMode::RECV);
      CUDACHECK(cudaEventRecord(events[i][11], stream));
    }

    CUDACHECK(cudaStreamSynchronize(stream));

    if (!warmup) {
      for (size_t i = 0; i < numSamples; i++) {
        dispatchTimeUs.push_back(duration(events[i][0], events[i][1]));
        combineTimeUs.push_back(duration(events[i][2], events[i][3]));
        dispatchSendTimeUs.push_back(duration(events[i][4], events[i][5]));
        dispatchRecvTimeUs.push_back(duration(events[i][6], events[i][7]));
        combineSendTimeUs.push_back(duration(events[i][8], events[i][9]));
        combineRecvTimeUs.push_back(duration(events[i][10], events[i][11]));
      }
    }
  };

  MPI_Barrier(MPI_COMM_WORLD);
  nvtxRangePush("warmup");
  for (int i = 0; i < 10; i++) {
    run(true);
  }
  nvtxRangePop();

  MPI_Barrier(MPI_COMM_WORLD);
  nvtxRangePush("benchmark");
  for (int i = 0; i < repeat; i++) {
    run(false);
  }
  nvtxRangePop();

  for (auto &sample : events) {
    for (auto &event : sample) {
      CUDACHECK(cudaEventDestroy(event));
    }
  }

  return {
      average(dispatchTimeUs),
      average(dispatchSendTimeUs),
      average(dispatchRecvTimeUs),
      average(combineTimeUs),
      average(combineSendTimeUs),
      average(combineRecvTimeUs)};
}

template <typename Kernel, typename T, typename U, bool HAS_SCALE, typename... Args>
void benchmark(
    const std::vector<BenchConfig> &configs,
    unsigned repeat,
    unsigned currentPE,
    unsigned numPEs,
    cudaStream_t stream,
    Args &&...args
) {
  for (const auto &config : configs) {
    auto [dispatch, dispatchSend, dispatchRecv, combine, combineSend, combineRecv] =
        benchConfig<Kernel, T, U, HAS_SCALE>(
            config, repeat, currentPE, numPEs, stream, std::forward<Args>(args)...
        );

    if (currentPE == 0) {
      const size_t d = config.hiddenDim;
      const size_t bytesPerToken =
          d * sizeof(T) + (HAS_SCALE ? sizeof(float) * ceil_div<size_t>(d, 128) : 0);

      const size_t m = config.numTokens * config.expertsPerToken;

      const size_t dispatchBits = bytesPerToken * m;
      const float dispatchBW = dispatchBits / dispatch.mean * 1e-3;

      const size_t combineBits = config.hiddenDim * sizeof(U) * m;
      const float combineBW = combineBits / combine.mean * 1e-3;

      std::cout << std::setw(3) << config.numTokens << " " << std::setw(3) << config.numExperts
                << " " << std::setw(3) << config.expertsPerToken << " " << std::setw(4) << d
                << " | " << dispatch << " " << dispatchSend << " " << dispatchRecv << " "
                << std::setw(4) << (unsigned)dispatchBW << "GB/s"
                << " | " << combine << " " << combineSend << " " << combineRecv << " "
                << std::setw(4) << (unsigned)combineBW << "GB/s" << std::endl;
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  // Set up MPI.
  int worldSize, rank;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
  if (worldSize < 2) {
    std::cout << "This test requires at least 2 workers" << std::endl;
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
  cudaProfilerStart();

  // Set up configurations for benchmarking.
  std::vector<BenchConfig> configs = {
      {1, 8, 4, 128, 16},
      {4, 8, 6, 2048, 128},
  };
  for (int numExperts : {8, 16, 64, 256}) {
    for (int numTokens : {1, 4, 16, 64, 128}) {
      configs.push_back({numTokens, numExperts, 8, 7168, 128});
    }
  }

  std::shared_ptr<Distributed> distributed = std::make_shared<DistributedNVSHMEM>(rank, worldSize);

  if (currentPE == 0) {
    std::cout << "Intra-Node FP8" << std::endl;
  }
  benchmark<AllToAllIntraNode, __nv_fp8_storage_t, nv_bfloat16, true>(
      configs, 10, currentPE, numPEs, stream, distributed
  );

  if (currentPE == 0) {
    std::cout << "Inter-Node FP8" << std::endl;
  }
  benchmark<AllToAllInterNode, __nv_fp8_storage_t, nv_bfloat16, true>(
      configs, 10, currentPE, numPEs, stream
  );

  if (currentPE == 0) {
    std::cout << "Intra-Node BF16" << std::endl;
  }
  benchmark<AllToAllIntraNode, nv_bfloat16, nv_bfloat16, false>(
      configs, 10, currentPE, numPEs, stream, distributed
  );

  if (currentPE == 0) {
    std::cout << "Inter-Node BF16" << std::endl;
  }
  benchmark<AllToAllInterNode, nv_bfloat16, nv_bfloat16, false>(
      configs, 10, currentPE, numPEs, stream
  );

  // Cleanup.
  CUDACHECK(cudaStreamDestroy(stream));
  nvshmem_barrier_all();
  nvshmem_finalize();
  cudaProfilerStop();
  MPICHECK(MPI_Finalize());
  return EXIT_SUCCESS;
}
