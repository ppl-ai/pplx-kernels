#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <cstdint>
#include <cstdlib>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <string>
#include <torch/library.h>
#include <vector>

#include "bindings/nvshmem.h"
#include "core/nvshmem_utils.h"

namespace {

at::Tensor get_unique_id() {
  nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
  nvshmemx_get_uniqueid(&uid);
  return at::from_blob(&uid, sizeof(uid), at::kByte).clone();
}

int64_t unique_id_size() { return sizeof(nvshmemx_uniqueid_t); }

int64_t init(at::Tensor uid, int64_t rank, int64_t world_size) {
  TORCH_CHECK(uid.device().is_cpu(), "uid must be a CPU tensor");
  TORCH_CHECK(uid.scalar_type() == at::kByte, "uid must be a byte tensor");
  TORCH_CHECK(
      uid.numel() == sizeof(nvshmemx_uniqueid_t),
      "Invalid unique id size (expected ",
      sizeof(nvshmemx_uniqueid_t),
      ", got ",
      uid.numel(),
      ")"
  );
  nvshmemx_uniqueid_t id;
  std::memcpy(&id, uid.data_ptr(), sizeof(id));
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  nvshmemx_set_attr_uniqueid_args(rank, world_size, &id, &attr);
  return nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
}

void finalize() { nvshmem_finalize(); }

int64_t my_pe() { return nvshmem_my_pe(); }

int64_t n_pes() { return nvshmem_n_pes(); }

at::Tensor
malloc_tensor(const std::vector<int64_t> &shape, c10::ScalarType dtype, const c10::Device &device) {
  size_t size = c10::elementSize(dtype) * c10::multiply_integers(shape);
  void *ptr = nvshmem_malloc(size);
  if (ptr == nullptr) {
    AT_ERROR("nvshmem_malloc failed. size: ", size);
  }
  return at::from_blob(
      ptr,
      shape,
      [](void *ptr) { nvshmem_free(ptr); },
      at::TensorOptions().dtype(dtype).device(device)
  );
}

void barrier_all() { nvshmem_barrier_all(); }

void barrier_all_on_current_stream() {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  nvshmemx_barrier_all_on_stream(stream);
}

void alltoall(at::Tensor dest, at::Tensor source) {
  TORCH_CHECK(dest.is_contiguous(), "dest must be contiguous");
  TORCH_CHECK(source.is_contiguous(), "source must be contiguous");

  size_t nbytes = dest.numel() * dest.itemsize() / dest.size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  NVSHMEMCHECK(nvshmemx_alltoallmem_on_stream(
      NVSHMEM_TEAM_WORLD, (uint8_t *)dest.data_ptr(), (uint8_t *)source.data_ptr(), nbytes, stream
  ));
}

void fake_alltoall(at::Tensor dest, at::Tensor source) {}

} // namespace

void pplx::register_nvshmem_ops(torch::Library &m) {
  m.def("nvshmem_get_unique_id", &get_unique_id);
  m.def("nvshmem_unique_id_size", &unique_id_size);
  m.def("nvshmem_init", &init);
  m.def("nvshmem_finalize", &finalize);
  m.def("nvshmem_my_pe", &my_pe);
  m.def("nvshmem_n_pes", &n_pes);
  m.def("nvshmem_malloc", &malloc_tensor);
  m.def("nvshmem_barrier_all", &barrier_all);
  m.def("nvshmem_barrier_all_on_current_stream", &barrier_all_on_current_stream);
  m.def("nvshmem_alltoall(Tensor! dest, Tensor src) -> ()");
  m.impl("nvshmem_alltoall", c10::kCUDA, &alltoall);
  m.impl("nvshmem_alltoall", c10::kMeta, &fake_alltoall);
}
