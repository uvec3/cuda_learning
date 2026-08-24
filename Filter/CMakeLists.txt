cmake_minimum_required(VERSION 4.2)
project(Filter CUDA)

#set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)


add_executable(Filter main.cu filter_device.cu filter_device.cuh)


set_target_properties(Filter PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


target_compile_options(Filter PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>: -lineinfo> # include source code
        $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda> # allow lambda with __device__
)

# switch to the standard conforming preprocessor for msvc compiler
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor")