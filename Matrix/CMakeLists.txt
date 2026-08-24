cmake_minimum_required(VERSION 3.20)
project(MATRIX LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#tell MSVC to use the standards-conforming preprocessor
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor")

include(FetchContent)
FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm
        GIT_TAG 8d1fd52e5ab5590e2c81768ace50c72bae28f2ed
        GIT_SHALLOW ON
)

FetchContent_MakeAvailable(glm)

# GoogleTest
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG v1.14.0
        GIT_SHALLOW ON
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(matrices main.cu)

target_link_libraries(matrices glm::glm)

set_target_properties(matrices PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)


target_compile_options(matrices
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20013>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

# Tests
add_executable(matrix_tests Tests/test_matrices.cu)

set_target_properties(matrix_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(matrix_tests
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20013>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

target_include_directories(matrix_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(matrix_tests PRIVATE gtest_main glm::glm)

add_test(NAME matrix_tests COMMAND matrix_tests)
