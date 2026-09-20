cmake_minimum_required(VERSION 3.18)
project(phase8_8a_cuda_fusion LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(fusion_microbenchmark fusion_microbenchmark.cu)
target_compile_features(fusion_microbenchmark PRIVATE cxx_std_17)
set_target_properties(fusion_microbenchmark PROPERTIES
    CUDA_ARCHITECTURES 87
    CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(fusion_microbenchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)
target_link_libraries(fusion_microbenchmark PRIVATE CUDA::cudart CUDA::cublas)
