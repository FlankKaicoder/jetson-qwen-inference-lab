cmake_minimum_required(VERSION 3.22)
project(phase8_3b_qwen3_rmsnorm_integration LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu REQUIRED)
find_library(TENSORRT_LIBRARY nvinfer
    PATHS /usr/lib/aarch64-linux-gnu /lib/aarch64-linux-gnu REQUIRED)

add_executable(qwen3_rmsnorm_integration qwen3_rmsnorm_integration.cpp)
target_include_directories(qwen3_rmsnorm_integration PRIVATE
    ${TENSORRT_INCLUDE_DIR})
target_link_libraries(qwen3_rmsnorm_integration PRIVATE
    ${TENSORRT_LIBRARY} CUDA::cudart ${CMAKE_DL_LIBS})

add_library(rmsnorm_qwen3_plugin SHARED
    ../rmsnorm_plugin.cpp
    ../rmsnorm_plugin_creator.cpp
    ../rmsnorm_plugin_kernel.cu
    ../../cuda-kernel/rmsnorm/rmsnorm_v2.cu)
target_include_directories(rmsnorm_qwen3_plugin PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ../../cuda-kernel/rmsnorm)
target_compile_options(rmsnorm_qwen3_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(rmsnorm_qwen3_plugin PRIVATE
    ${TENSORRT_LIBRARY} CUDA::cudart)
