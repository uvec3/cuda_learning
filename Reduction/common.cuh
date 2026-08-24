#pragma once

#include <cuda/std/bit>

#define CUDA_CHECK(call)                                             \
do {                                                                 \
    cudaError_t err = call;                                          \
    if (err != cudaSuccess) {                                        \
        std::cerr << "CUDA Error:\n"                                 \
        << "  File: " << __FILE__ << ":" << __LINE__ << "\n"         \
        << "  Function: " << #call << "\n"                           \
        << "  Name: " << cudaGetErrorName(err) << "\n"               \
        << "  Message: " << cudaGetErrorString(err) << "\n";         \
        cudaDeviceReset();                                           \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while (0)


__host__ __device__
inline bool compare_floats(float a, float b, int max_difference)
{
    if (isnan(a)&&isnan(b))
        return true;

    auto a_bits=cuda::std::bit_cast<uint32_t>(a);
    auto b_bits=cuda::std::bit_cast<uint32_t>(b);
    const uint32_t sign_bit=1u<<31;

    if (a_bits & sign_bit)
    {
        a_bits= ~a_bits;
    }
    else
    {
        a_bits|=sign_bit;
    }

    if (b_bits & sign_bit)
    {
        b_bits= ~b_bits;
    }else
    {
        b_bits|=sign_bit;
    }

    if (a_bits>b_bits)
        return a_bits-b_bits<=max_difference;
    return b_bits-a_bits<=max_difference;
}