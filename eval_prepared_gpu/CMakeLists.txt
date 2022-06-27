cmake_minimum_required(VERSION 3.16)

project(eval_prepared_gpu C CXX CUDA)
find_package( CUDAToolkit )
set(CMAKE_CXX_FLAGS "-O2 -Wall -std=c++14")

file(GLOB GTEST_SOURCE ../test/gtest/gtest-all.cc)
file(GLOB GTEST_HEADER ../test/gtest/gtest.h)
include_directories(${TACO_TEST_DIR} ${TACO_SRC_DIR})
add_library(gtest ${GTEST_HEADER} ${GTEST_SOURCE})

add_executable(sddmm sddmm_csr_gpu_taco.cu)
#target_include_directories(sddmm PRIVATE ../../cnpy)
target_include_directories(sddmm PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(sddmm PRIVATE CUDA::cusparse CUDA::cublas CUDA::curand CUDA::cudart taco gtest)
