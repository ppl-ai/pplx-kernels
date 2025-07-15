# All-to-All library

add_library(all_to_all_common STATIC
    all_to_all.cpp
)

target_link_libraries(all_to_all_common PUBLIC
    CUDA::cudart
)

add_library(all_to_all_intranode_lib STATIC
    intranode_combine.cu
    intranode_dispatch.cu
    intranode.cpp
)
target_link_libraries(all_to_all_intranode_lib PUBLIC
    all_to_all_common
    CUDA::cudart
)
target_link_libraries(all_to_all_intranode_lib INTERFACE
    nvshmem::nvshmem_host
)
target_include_directories(all_to_all_intranode_lib PRIVATE ${NVSHMEM_INCLUDE_DIR})
set_cuda_compile_options(all_to_all_intranode_lib)

add_library(all_to_all_internode_lib STATIC
    internode_combine.cu
    internode_dispatch.cu
    internode.cpp
)
target_link_libraries(all_to_all_internode_lib PUBLIC
    all_to_all_common
    CUDA::cudart
)
target_link_libraries(all_to_all_internode_lib INTERFACE
    nvshmem::nvshmem_host
)
target_include_directories(all_to_all_internode_lib PRIVATE ${NVSHMEM_INCLUDE_DIR})
set_cuda_compile_options(all_to_all_internode_lib)

if(WITH_TESTS)
    # All-to-All test
    add_executable(test_all_to_all
        test_all_to_all.cpp
    )
    target_link_libraries(test_all_to_all PUBLIC
        all_to_all_intranode_lib
        all_to_all_internode_lib
        core_lib
        CUDA::cudart
        CUDA::cuda_driver
        MPI::MPI_CXX
        nvshmem::nvshmem_host
    )
    set_cuda_compile_options(test_all_to_all)
    add_test(NAME AllToAllTest
             COMMAND ${MPIEXEC_EXECUTABLE} -np 4 $<TARGET_FILE:test_all_to_all>)
    set_tests_properties(AllToAllTest PROPERTIES ENVIRONMENT "NVSHMEM_REMOTE_TRANSPORT=None")
endif()

if (WITH_BENCHMARKS)
    add_executable(bench_all_to_all
        bench_all_to_all.cpp
    )
    target_link_libraries(bench_all_to_all PUBLIC
        all_to_all_intranode_lib
        all_to_all_internode_lib
        core_lib
        CUDA::cudart
        CUDA::cuda_driver
        MPI::MPI_CXX
        nvshmem::nvshmem_host
    )
endif()
