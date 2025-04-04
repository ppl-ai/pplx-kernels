#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include "all_to_all/internode.h"
#include "all_to_all_ops.h"

using namespace pplx;

using fptr_t = int64_t;

namespace {

#define _CHECK_TENSOR(ndim, x)                                                                     \
  do {                                                                                             \
    TORCH_CHECK(x.is_cuda(), "Tensor " #x " must be on GPU");                                      \
    TORCH_CHECK(x.is_contiguous(), "Tensor " #x " must be contiguous");                            \
    TORCH_CHECK(x.dim() == (ndim), "Tensor " #x " must be ", (ndim), " dimensional");              \
  } while (0)

fptr_t create(
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
  return (fptr_t)ptr;
}

void destroy(fptr_t ptr) { delete (AllToAllInterNode *)ptr; }

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
    _CHECK_TENSOR(3, outExpertXScale.value());
    TORCH_CHECK(
        outExpertXScale->scalar_type() == at::kFloat, "outExpertXScale must be of type Float"
    );
  }
  _CHECK_TENSOR(2, dpX);
  if (dpXScale.has_value()) {
    _CHECK_TENSOR(2, dpXScale.value());
    TORCH_CHECK(dpXScale->scalar_type() == at::kFloat, "dpXScale must be of type Float");
  }
  _CHECK_TENSOR(2, indices);
  TORCH_CHECK(indices.scalar_type() == at::kUInt32, "indices must be of type UInt32");
  if (boundM.has_value()) {
    _CHECK_TENSOR(1, boundM.value());
    TORCH_CHECK(boundM->scalar_type() == at::kUInt32, "boundM must be of type UInt32");
    TORCH_CHECK(boundM->numel() == 1, "boundM must be a scalar tensor");
  }

  auto *all_to_all = (AllToAllInterNode *)ptr;
  all_to_all->dispatch(
      Strided1D<int32_t>(
          outExpertNumTokens.data_ptr<int32_t>(), (size_t)outExpertNumTokens.stride(0)
      ),
      Strided2D<std::byte>(
          (std::byte *)outExpertX.data_ptr(),
          (size_t)outExpertX.stride(1) * outExpertX.element_size(),
          (size_t)outExpertX.stride(0) * outExpertX.element_size()
      ),
      outExpertXScale.has_value()
          ? Strided2D<std::byte>(
                (std::byte *)outExpertXScale->data_ptr(),
                (size_t)outExpertXScale->stride(1) * outExpertXScale->element_size(),
                (size_t)outExpertXScale->stride(0) * outExpertXScale->element_size()
            )
          : Strided2D<std::byte>(nullptr, 0, 0),
      Strided1D<std::byte>((std::byte *)dpX.data_ptr(), (size_t)dpX.stride(0) * dpX.element_size()),
      dpXScale.has_value() ? Strided1D<std::byte>(
                                 (std::byte *)dpXScale->data_ptr(),
                                 (size_t)dpXScale->stride(0) * dpXScale->element_size()
                             )
                           : Strided1D<std::byte>(nullptr, 0),
      Strided2D<uint32_t>(
          indices.data_ptr<uint32_t>(), (size_t)indices.stride(1), (size_t)indices.stride(0)
      ),
      indices.size(0),
      boundM.has_value() ? boundM->data_ptr<unsigned>() : nullptr,
      getSplitMode(doSend, doRecv),
      at::cuda::getCurrentCUDAStream()
  );
}

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
  TORCH_CHECK(outTokens.scalar_type() == at::kBFloat16 || outTokens.scalar_type() == at::kHalf, "outTokens must be of type BFloat16 or Float16");
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

  auto *all_to_all = (AllToAllInterNode *)ptr;
  auto run = [&]<typename T, typename U>() {
    all_to_all->combine<T, U>(
        Strided1D<U>((U*)outTokens.data_ptr(), (size_t)outTokens.stride(0)),
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
  };

  auto out_type_switch = [&]<typename T>(at::ScalarType const& out_dtype) {
    switch(out_dtype) {
      case at::kBFloat16:
        run.operator()<T, nv_bfloat16>();
        break;
      case at::kHalf:
        run.operator()<T, half>();
        break;
      default:
        TORCH_CHECK(false, "Unsupported dtype for outTokens");
    }
  };

  switch (expertY.scalar_type()) {
  case at::kFloat:
    out_type_switch.operator()<float>(outTokens.scalar_type());
    break;
  case at::kBFloat16:
    out_type_switch.operator()<nv_bfloat16>(outTokens.scalar_type());
    break;
  case at::kHalf:
    out_type_switch.operator()<half>(outTokens.scalar_type());
    break;
  default:
    TORCH_CHECK(false, "Unsupported dtype for expertY");
  }
}

#undef _CHECK_TENSOR

} // namespace

void pplx::register_all_to_all_ops(torch::Library &m) {
  m.def("all_to_all_create", &create);
  m.def("all_to_all_destroy", &destroy);
  m.def("all_to_all_dispatch", &dispatch);
  m.def("all_to_all_combine", &combine);
}
