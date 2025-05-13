#include "core/distributed.h"
#include "core/cuda_utils.h"
#include "core/utils.h"

using namespace pplx;

Distributed::Distributed(unsigned rank, unsigned worldSize)
    : rank(rank),
      worldSize(worldSize) {}

Distributed::~Distributed() {}

DistributedNVSHMEM::DistributedNVSHMEM(unsigned rank, unsigned worldSize)
    : Distributed(rank, worldSize) {}

void DistributedNVSHMEM::allToAllImpl(const void *input, void *output, size_t size, size_t count) {
  PPLX_ASSERT(count == worldSize, "count must be equal to world size");

  void *srcBuffer = nvshmem_malloc(size * count);
  PPLX_ASSERT(srcBuffer != nullptr, "Failed to allocate src buffer");
  void *dstBuffer = nvshmem_malloc(size * count);
  PPLX_ASSERT(dstBuffer != nullptr, "Failed to allocate dst buffer");

  CUDACHECK(cudaMemcpy(srcBuffer, input, size * count, cudaMemcpyHostToDevice));

  nvshmem_alltoallmem(NVSHMEM_TEAM_WORLD, dstBuffer, srcBuffer, size);
  nvshmem_quiet();

  CUDACHECK(cudaMemcpy(output, dstBuffer, size * count, cudaMemcpyDeviceToHost));

  nvshmem_free(dstBuffer);
  nvshmem_free(srcBuffer);
}
