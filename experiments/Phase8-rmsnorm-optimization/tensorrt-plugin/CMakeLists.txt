cmake_minimum_required(VERSION 3.22)
project(phase8_rmsnorm_tensorrt_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu
    REQUIRED)
find_library(TENSORRT_LIBRARY nvinfer
    PATHS /usr/lib/aarch64-linux-gnu /lib/aarch64-linux-gnu
    REQUIRED)

add_library(rmsnorm_trt_plugin SHARED
    rmsnorm_plugin.cpp
    rmsnorm_plugin_creator.cpp
    rmsnorm_plugin_kernel.cu
    ../cuda-kernel/rmsnorm/rmsnorm_v2.cu
)
target_include_directories(rmsnorm_trt_plugin PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ../cuda-kernel/rmsnorm
)
target_compile_options(rmsnorm_trt_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_link_libraries(rmsnorm_trt_plugin PRIVATE
    ${TENSORRT_LIBRARY}
    CUDA::cudart
)

add_executable(rmsnorm_plugin_demo rmsnorm_plugin_demo.cpp)
target_include_directories(rmsnorm_plugin_demo PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(rmsnorm_plugin_demo PRIVATE
    ${TENSORRT_LIBRARY}
    CUDA::cudart
    ${CMAKE_DL_LIBS}
)
