cmake_minimum_required(VERSION 3.22)
project(jetson_qwen_rmsnorm_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(rmsnorm_benchmark
    benchmark.cpp
    rmsnorm_v0.cu
    rmsnorm_v1.cu
    rmsnorm_v2.cu
)

target_compile_options(rmsnorm_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(rmsnorm_benchmark PRIVATE CUDA::cudart)
