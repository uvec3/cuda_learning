#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <chrono>
#include "kernels.cuh"
// #include <iostream>
#include <cuda_profiler_api.h>



int main(int argc, char** argv)
{
    int n=4098;
    if (argc>1)
    {
        n=std::stoi(argv[1]);
    }
    thrust::host_vector<float> vector_h(n);

    for (int i=0;i<n;++i)
    {
        vector_h[i]=i/1000;
    }

    thrust::device_vector<float> vector_d = vector_h;
    auto add_d=[]__device__(float a,float b){return a+b;};

    std::cout<<reduce(vector_d.data().get(),vector_d.size(),add_d)<<"\n";
    std::cout<<reduce_coalesced(vector_d.data().get(),vector_d.size(),add_d)<<"\n";
    std::cout<<reduce_coarsened(vector_d.data().get(),vector_d.size(),add_d)<<"\n";

    cudaProfilerStop();

    return 0;
}