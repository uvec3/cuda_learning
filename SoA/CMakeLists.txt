cmake_minimum_required(VERSION 4.2)
project(SoA CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

add_executable(SoA main.cu soa_vector.cpp soa_vector.hpp)

set_target_properties(SoA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


include(FetchContent)
FetchContent_Declare(
        Boost
        URL https://github.com/boostorg/boost/releases/download/boost-1.92.0/boost-1.92.0-cmake.tar.gz
)
FetchContent_MakeAvailable(Boost )

target_link_libraries(SoA Boost::headers Boost::pfr)