#include "all_to_all/internode.h"
#include "all_to_all/intranode.h"
#include "core/distributed.h"

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

using namespace pplx;

using fptr_t = int64_t;

namespace {

#define _CHECK_TENSOR(ndim, x)                                                                     \
  do {                                                                                             \
    TORCH_CHECK(x.is_cuda(), "Tensor " #x " must be on GPU");                                      \
    TORCH_CHECK(x.is_contiguous(), "Tensor " #x " must be contiguous");                            \
    TORCH_CHECK(x.dim() == (ndim), "Tensor " #x " must be ", (ndim), " dimensional");              \
  } while (0)

class DistributedTorch : public Distributed {
public:
  DistributedTorch(const c10::intrusive_ptr<c10d::ProcessGroup> &group)
      : Distributed(group->getRank(), group->getSize()),
        group(group) {}

private:
  void allToAllImpl(const void *input, void *output, size_t size, size_t count) override {
    unsigned n = group->getSize();
    TORCH_CHECK(n == count, "Group size must be equal to count");

    at::Tensor inputTensor = at::from_blob(
        const_cast<void *>(input), {(int)count, (int)size}, at::TensorOptions().dtype(at::kByte)
    );

    at::Tensor outputTensor =
        at::from_blob(output, {(int)count, (int)size}, at::TensorOptions().dtype(at::kByte));

    std::vector<int64_t> counts(n, 1);

    c10d::AllToAllOptions opts;
    auto work = group->alltoall_base(outputTensor, inputTensor, counts, counts, opts);
    work->wait();
  }

  c10::intrusive_ptr<c10d::ProcessGroup> group;
};

fptr_t create_internode(
    int64_t maxNumTokens,
    int64_t numExperts,
    int64_t expertsPerToken,
    int64_t rank,
    int64_t worldSize,
    int64_t dpSize,
    int64_t hiddenDim,
    int64_t hiddenDimBytes,
    int64_t hiddenDimScaleBytes
) {
  auto *ptr = new AllToAllInterNode(
      maxNumTokens,
      numExperts,
      expertsPerToken,
      rank,
      worldSize,
      dpSize,
      hiddenDim,
      hiddenDimBytes,
      hiddenDimScaleBytes
  );

  // Needed to use host-side initialization information in device APIs.
  nvshmem_init();

  return (fptr_t)ptr;
}

fptr_t create_intranode(
    int64_t maxNumTokens,
    int64_t numExperts,
    int64_t expertsPerToken,
    int64_t rank,
    int64_t worldSize,
    int64_t dpSize,
    int64_t hiddenDim,
    int64_t hiddenDimBytes,
    int64_t hiddenDimScaleBytes,
    const std::string &group_name
) {
  auto group = c10d::resolve_process_group(group_name);
  std::shared_ptr<Distributed> distributed = std::make_shared<DistributedTorch>(group);
  auto *ptr = new AllToAllIntraNode(
      maxNumTokens,
      numExperts,
      expertsPerToken,
      rank,
      worldSize,
      dpSize,
      hiddenDim,
      hiddenDimBytes,
      hiddenDimScaleBytes,
      distributed
  );
  return (fptr_t)ptr;
}

void destroy(fptr_t ptr) { delete (AllToAll *)ptr; }

SplitMode getSplitMode(bool doSend, bool doRecv) {
  if (doSend && doRecv) {
    return SplitMode::NONE;
  } else if (doSend) {
    return SplitMode::SEND;
  } else if (doRecv) {
    return SplitMode::RECV;
  } else {
    TORCH_CHECK(false, "Invalid split mode");
  }
}

template <typename Kernel>
void dispatch(
    fptr_t ptr,
    at::Tensor &outExpertNumTokens,
    at::Tensor &outExpertX,
    const std::optional<at::Tensor> &outExpertXScale,
    const at::Tensor &dpX,
    const std::optional<at::Tensor> &dpXScale,
    const at::Tensor &indices,
    const std::optional<at::Tensor> &boundM,
    bool doSend,
    bool doRecv
) {
  _CHECK_TENSOR(1, outExpertNumTokens);
  _CHECK_TENSOR(3, outExpertX);
  if (outExpertXScale.has_value()) {
    TORCH_CHECK(outExpertXScale->is_cuda(), "Tensor outExpertXScale must be on GPU");
    TORCH_CHECK(outExpertXScale->dim() == 3, "Tensor outExpertXScale must be 3 dimensional");
  }
  _CHECK_TENSOR(2, dpX);
  if (dpXScale.has_value()) {
    TORCH_CHECK(dpXScale->is_cuda(), "Tensor dpXScale must be on GPU");
    TORCH_CHECK(dpXScale->dim() == 2, "Tensor dpXScale must be 2 dimensional");
    TORCH_CHECK(dpXScale->scalar_type() == at::kFloat, "dpXScale must be of type Float");
  }
  _CHECK_TENSOR(2, indices);
  TORCH_CHECK(indices.scalar_type() == at::kUInt32, "indices must be of type UInt32");
  if (boundM.has_value()) {
    _CHECK_TENSOR(1, boundM.value());
    TORCH_CHECK(boundM->scalar_type() == at::kUInt32, "boundM must be of type UInt32");
    TORCH_CHECK(boundM->numel() == 1, "boundM must be a scalar tensor");
  }

  auto *all_to_all = (Kernel *)ptr;

  TORCH_CHECK(indices.size(0) == dpX.size(0), "indices.size(0) must be equal to dpX.size(0)");
  TORCH_CHECK(
      indices.size(1) == all_to_all->getNumExpertsPerToken(),
      "indices.size(1) must be equal to the experts per token"
  );

  at::cuda::OptionalCUDAGuard const device_guard(device_of(indices));

  all_to_all->dispatch(
      Strided1D<int32_t>(
          outExpertNumTokens.data_ptr<int32_t>(), (size_t)outExpertNumTokens.stride(0)
      ),
      Strided2D<std::byte>(
          (std::byte *)outExpertX.data_ptr(),
          (size_t)outExpertX.stride(1) * outExpertX.element_size(),
          (size_t)outExpertX.stride(0) * outExpertX.element_size()
      ),
      outExpertXScale.has_value() ? Strided3D<float>(
                                        outExpertXScale->data_ptr<float>(),
                                        (size_t)outExpertXScale->stride(2),
                                        (size_t)outExpertXScale->stride(1),
                                        (size_t)outExpertXScale->stride(0)
                                    )
                                  : Strided3D<float>(nullptr, 0, 0, 0),
      Strided1D<std::byte>((std::byte *)dpX.data_ptr(), (size_t)dpX.stride(0) * dpX.element_size()),
      dpXScale.has_value() ? Strided2D<float>(
                                 (float *)dpXScale->data_ptr(),
                                 (size_t)dpXScale->stride(1),
                                 (size_t)dpXScale->stride(0)
                             )
                           : Strided2D<float>(nullptr, 0, 0),
      Strided2D<uint32_t>(
          indices.data_ptr<uint32_t>(), (size_t)indices.stride(1), (size_t)indices.stride(0)
      ),
      indices.size(0),
      boundM.has_value() ? boundM->data_ptr<unsigned>() : nullptr,
      getSplitMode(doSend, doRecv),
      at::cuda::getCurrentCUDAStream()
  );
}

void fake_dispatch(
    fptr_t ptr,
    at::Tensor &outExpertNumTokens,
    at::Tensor &outExpertX,
    const std::optional<at::Tensor> &outExpertXScale,
    const at::Tensor &dpX,
    const std::optional<at::Tensor> &dpXScale,
    const at::Tensor &indices,
    const std::optional<at::Tensor> &boundM,
    bool doSend,
    bool doRecv
) {}

template <typename Kernel, typename T, typename U>
void combineImpl(
    Kernel *all_to_all,
    at::Tensor &outTokens,
    const at::Tensor &indices,
    const at::Tensor &weights,
    const at::Tensor &expertY,
    const std::optional<at::Tensor> &boundM,
    bool doSend,
    bool doRecv
) {
  all_to_all->combine(
      Strided1D<U>((U *)outTokens.data_ptr(), (size_t)outTokens.stride(0)),
      Strided2D<uint32_t>(
          indices.data_ptr<uint32_t>(), (size_t)indices.stride(1), (size_t)indices.stride(0)
      ),
      Strided2D<float>(
          weights.data_ptr<float>(), (size_t)weights.stride(1), (size_t)weights.stride(0)
      ),
      Strided2D<T>((T *)expertY.data_ptr(), (size_t)expertY.stride(1), (size_t)expertY.stride(0)),
      indices.size(0),
      boundM.has_value() ? boundM->data_ptr<unsigned>() : nullptr,
      getSplitMode(doSend, doRecv),
      at::cuda::getCurrentCUDAStream()
  );
}

template <typename Kernel>
void combine(
    fptr_t ptr,
    at::Tensor &outTokens,
    const at::Tensor &indices,
    const at::Tensor &weights,
    const at::Tensor &expertY,
    const std::optional<at::Tensor> &boundM,
    bool doSend,
    bool doRecv
) {
  _CHECK_TENSOR(2, outTokens);
  _CHECK_TENSOR(2, indices);
  TORCH_CHECK(indices.scalar_type() == at::kUInt32, "indices must be of type UInt32");
  _CHECK_TENSOR(2, weights);
  TORCH_CHECK(weights.scalar_type() == at::kFloat, "weights must be of type Float");
  _CHECK_TENSOR(3, expertY);
  if (boundM.has_value()) {
    _CHECK_TENSOR(1, boundM.value());
    TORCH_CHECK(boundM->scalar_type() == at::kUInt32, "boundM must be of type UInt32");
    TORCH_CHECK(boundM->numel() == 1, "boundM must be a scalar tensor");
  }

  auto *all_to_all = (Kernel *)ptr;

  at::cuda::OptionalCUDAGuard const device_guard(device_of(indices));

  switch (expertY.scalar_type()) {
  case at::kFloat: {
    switch (outTokens.scalar_type()) {
    case at::kBFloat16:
      return combineImpl<Kernel, float, nv_bfloat16>(
          all_to_all, outTokens, indices, weights, expertY, boundM, doSend, doRecv
      );
    case at::kHalf:
      return combineImpl<Kernel, float, half>(
          all_to_all, outTokens, indices, weights, expertY, boundM, doSend, doRecv
      );
    default:
      TORCH_CHECK(false, "Unsupported dtype for outTokens");
    }
  }
  case at::kBFloat16: {
    switch (outTokens.scalar_type()) {
    case at::kBFloat16:
      return combineImpl<Kernel, nv_bfloat16, nv_bfloat16>(
          all_to_all, outTokens, indices, weights, expertY, boundM, doSend, doRecv
      );
    case at::kHalf:
      return combineImpl<Kernel, nv_bfloat16, half>(
          all_to_all, outTokens, indices, weights, expertY, boundM, doSend, doRecv
      );
    default:
      TORCH_CHECK(false, "Unsupported dtype for outTokens");
    }
  }
  case at::kHalf: {
    switch (outTokens.scalar_type()) {
    case at::kBFloat16:
      return combineImpl<Kernel, half, nv_bfloat16>(
          all_to_all, outTokens, indices, weights, expertY, boundM, doSend, doRecv
      );
    case at::kHalf:
      return combineImpl<Kernel, half, half>(
          all_to_all, outTokens, indices, weights, expertY, boundM, doSend, doRecv
      );
    default:
      TORCH_CHECK(false, "Unsupported dtype for outTokens");
    }
  }
  default:
    TORCH_CHECK(false, "Unsupported dtype for expertY");
  }
}

void fake_combine(
    fptr_t ptr,
    at::Tensor &outTokens,
    const at::Tensor &indices,
    const at::Tensor &weights,
    const at::Tensor &expertY,
    const std::optional<at::Tensor> &boundM,
    bool doSend,
    bool doRecv
) {}

#undef _CHECK_TENSOR

} // namespace

namespace pplx {
void register_all_to_all_ops(torch::Library &m) {
  m.def("all_to_all_destroy", &destroy);

  m.def("all_to_all_internode_create", &create_internode);

  m.def("all_to_all_internode_dispatch("
        "  int fptr,"
        "  Tensor! out_expert_num_tokens,"
        "  Tensor! out_expert_x,"
        "  Tensor!? out_expert_x_scale,"
        "  Tensor dp_x,"
        "  Tensor? dp_x_scale,"
        "  Tensor indices,"
        "  Tensor? bound_m,"
        "  bool do_send,"
        "  bool do_recv"
        ") -> ()");
  m.impl("all_to_all_internode_dispatch", c10::kCUDA, &dispatch<AllToAllInterNode>);
  m.impl("all_to_all_internode_dispatch", c10::kMeta, &fake_dispatch);

  m.def("all_to_all_internode_combine("
        "  int fptr,"
        "  Tensor! out_tokens,"
        "  Tensor indices,"
        "  Tensor weights,"
        "  Tensor expert_y,"
        "  Tensor? bound_m,"
        "  bool do_send,"
        "  bool do_recv"
        ") -> ()");
  m.impl("all_to_all_internode_combine", c10::kCUDA, &combine<AllToAllInterNode>);
  m.impl("all_to_all_internode_combine", c10::kMeta, &fake_combine);

  m.def("all_to_all_intranode_create", &create_intranode);

  m.def("all_to_all_intranode_dispatch("
        "  int fptr,"
        "  Tensor! out_expert_num_tokens,"
        "  Tensor! out_expert_x,"
        "  Tensor!? out_expert_x_scale,"
        "  Tensor dp_x,"
        "  Tensor? dp_x_scale,"
        "  Tensor indices,"
        "  Tensor? bound_m,"
        "  bool do_send,"
        "  bool do_recv"
        ") -> ()");
  m.impl("all_to_all_intranode_dispatch", c10::kCUDA, &dispatch<AllToAllIntraNode>);
  m.impl("all_to_all_intranode_dispatch", c10::kMeta, &fake_dispatch);

  m.def("all_to_all_intranode_combine("
        "  int fptr,"
        "  Tensor! out_tokens,"
        "  Tensor indices,"
        "  Tensor weights,"
        "  Tensor expert_y,"
        "  Tensor? bound_m,"
        "  bool do_send,"
        "  bool do_recv"
        ") -> ()");
  m.impl("all_to_all_intranode_combine", c10::kCUDA, &combine<AllToAllIntraNode>);
  m.impl("all_to_all_intranode_combine", c10::kMeta, &fake_combine);
}

} // namespace pplx
