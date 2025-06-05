cmake_minimum_required(VERSION 3.22)
project(PPLXKernels
        VERSION 0.0.1
        DESCRIPTION "PPLX Kernels"
        LANGUAGES CXX CUDA)

# === Configuration options ===
option(WITH_TESTS "Build tests" OFF)
option(WITH_BENCHMARKS "Build benchmarks" OFF)
set(CMAKE_CUDA_ARCHITECTURES 90a CACHE STRING "CUDA architecture to target")

# === CMake configuration ===
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# === Dependencies ===
include(FetchContent)
find_package(CUDAToolkit REQUIRED)  # Modern replacement for find_package(CUDA)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)
find_package(NVSHMEM REQUIRED)

if(WITH_TESTS)
    enable_testing()
    find_package(MPI REQUIRED)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(NCCL nccl)
endif()

# Create imported target for PyTorch
add_library(torch_imported INTERFACE)
add_library(torch::py_limited ALIAS torch_imported)
target_include_directories(torch_imported SYSTEM INTERFACE ${TORCH_INCLUDE_DIRS})
# NOTE(lequn): We don't link against all ${TORCH_LIBRARIES} because we use py_limited_api.
# See: https://github.com/pytorch/pytorch/blob/9017becf1d895999a1c819c9d35b8139c090e7a9/torch/utils/cpp_extension.py#L1256-L1270
target_link_libraries(torch_imported INTERFACE c10 torch torch_cpu c10_cuda torch_cuda CUDA::cudart)

# === Compiler and linker flags ===
add_compile_options(-Wno-deprecated-declarations)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
add_compile_definitions(Py_LIMITED_API=0x03090000)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# CUDA-specific compile options function
function(set_cuda_compile_options target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--threads=32 -O3>)
endfunction()

# === Library targets ===
add_subdirectory(all_to_all)
add_subdirectory(core)

# Main shared library
add_library(pplx_kernels SHARED
    bindings/all_to_all_ops.cpp
    bindings/bindings.cpp
    bindings/nvshmem.cpp
)
target_link_libraries(pplx_kernels PUBLIC
    all_to_all_internode_lib
    all_to_all_intranode_lib
    core_lib
    torch::py_limited
    Python::Module
    CUDA::cuda_driver
    CUDA::cudart
    nvshmem::nvshmem
)
set_target_properties(pplx_kernels PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/pplx_kernels
    CUDA_SEPARABLE_COMPILATION ON
)
