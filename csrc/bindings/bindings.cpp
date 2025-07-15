#include <torch/library.h>

#include "bindings/all_to_all_ops.h"
#include "core/registration.h"

using namespace pplx;

TORCH_LIBRARY(pplx_kernels, m) { register_all_to_all_ops(m); }

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)