# Core library with common kernels.

add_library(core_lib STATIC
    kernels.cu
    distributed.cpp
)
target_link_libraries(core_lib PUBLIC
    CUDA::cudart
)
target_link_libraries(core_lib INTERFACE
    nvshmem::nvshmem_host
)
target_include_directories(core_lib PRIVATE ${NVSHMEM_INCLUDE_DIR})
set_cuda_compile_options(core_lib)
