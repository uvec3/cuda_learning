cmake_minimum_required(VERSION 3.20)
project(Reduction LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#tell MSVC to use the standards-conforming preprocessor
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor")

add_executable(reduction main.cu)

set_target_properties(reduction PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(reduction
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20013>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

