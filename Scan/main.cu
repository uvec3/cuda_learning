#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <chrono>
#include "kernels.cuh"
// #include "kernel_scan_sample.cuh"
// #include <iostream>
#include <cuda_profiler_api.h>
#include <thrust/scan.h>

void host_scan(int* output, int* input, uint32_t n)
{
    output[0]=input[0];
    for (uint32_t i=1;i<n;++i)
    {
        output[i]=output[i-1]+input[i];
    }
}

bool compare(int * v1, int * v2, uint32_t n)
{
    for (uint32_t i=0; i<n;++i)
    {
        if (v1[i]!=v2[i])
            return false;
    }
    return true;
}

void print_vector(int* v, uint32_t n)
{
    for (uint32_t i = 0;i<n;++i)
    {
        std::cout<<i<<")\t\t"<<v[i]<<"\n";
    }
}

int main(int argc, char** argv)
{
    int n=10000;
    if (argc>1)
    {
        n=std::stoi(argv[1]);
    }
    thrust::host_vector<int> vector_h(n);

    for (int i=0;i<n;++i)
    {
        vector_h[i]=i;
    }

    thrust::device_vector<int> vector_input_d = vector_h;
    thrust::device_vector<int> vector_output_d(vector_input_d.size());

    my_scan(vector_output_d.data().get(),vector_input_d.data().get(),vector_input_d.size());
    scan_kogge_stone_warp(vector_output_d.data().get(),vector_input_d.data().get(),vector_input_d.size());
    // preScan(vector_output_d.data().get(),vector_input_d.data().get(),vector_input_d.size());
    scan_kogge_stone_warp_coarse_reg(vector_output_d.data().get(),vector_input_d.data().get(),vector_input_d.size());
    scan_kogge_stone_warp_coarse(vector_output_d.data().get(),vector_input_d.data().get(),vector_input_d.size());
    scan_unidirectional(vector_output_d.data().get(),vector_input_d.data().get(),vector_input_d.size());

    thrust::inclusive_scan(vector_input_d.begin(), vector_input_d.end(), vector_output_d.begin());

    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::host_vector<int> vector_output_h=vector_output_d;
    if (n<=10000)
        print_vector(vector_output_h.data(),vector_output_h.size());


    cudaProfilerStop();

    return 0;
}