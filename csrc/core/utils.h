#pragma once

#include "core/common_utils.h"

#include <unistd.h>

#define _PPLX_ASSERT_MSG(msg)                                                                      \
  do {                                                                                             \
    ssize_t ret = write(2, msg, sizeof(msg));                                                      \
    (void)ret;                                                                                     \
  } while (false);

#define _PPLX_STRINGIFY(x) #x
#define _PPLX_EXPAND_AND_STRINGIFY(x) _PPLX_STRINGIFY(x)

#define PPLX_ASSERT(cond, msg)                                                                     \
  if (!(cond)) {                                                                                   \
    _PPLX_ASSERT_MSG(__FILE__ "(" _PPLX_EXPAND_AND_STRINGIFY(__LINE__) "): " msg "\n");            \
    __builtin_trap();                                                                              \
  }

#define PPLX_UNREACHABLE(msg)                                                                      \
  _PPLX_ASSERT_MSG(__FILE__ "(" _PPLX_EXPAND_AND_STRINGIFY(__LINE__) "): " msg "\n");              \
  __builtin_trap();
