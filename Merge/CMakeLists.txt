cmake_minimum_required(VERSION 4.2)
project(Merge CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(Merge main.cu device_merge.cu device_merge.hpp)

set_target_properties(Merge PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(Merge PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: -lineinfo> # include source code
)

# switch to the standard conforming preprocessor for msvc compiler
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor")