#pragma once

#include <cuda/std/functional>

#include "common.cuh"

template<typename TR,typename TA, typename TB,typename OP>
__global__
void kernel_vec_op(TR *r, TA * a, TB* b, uint32_t N, OP op)
{
    auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N)
        r[i]= op(a[i],b[i]);
}


template<typename T,typename OP>
__global__
void reduce_kernel(T* output,const T* a, int N, OP op)
{
    //STAGE 1
    auto threads=blockDim.x;
    auto elements_per_thread=(N-1)/threads+1;
    if (elements_per_thread<2)
        elements_per_thread=2;

    auto x=elements_per_thread*threads-N;
    uint32_t start;
    if (threadIdx.x<x)
    {
        elements_per_thread-=1;
        start=threadIdx.x*elements_per_thread;
    }
    else
    {
        start=threadIdx.x*elements_per_thread-x;
    }

    // printf("thread %d: starts at %d and takes %d \n", threadIdx.x,start, elements_per_thread);

    T s;
    if (start<N)
    {
        s=a[start];
    }
    for (int i=1;i<elements_per_thread;++i)
    {
        if (start+i<N)
            s=op(s,a[start+i]);
    }

    //at the end of stage 1 each thread writes result into shared memory
    __shared__ T shared[2048];
    shared[threadIdx.x]=s;
    __syncthreads();

    //STAGE 2
    auto n=threads;
    while (n>2)
    {
        T a;
        if (threadIdx.x*2<n)
        {
            a=shared[threadIdx.x*2];

            if (threadIdx.x*2+1<n)
            {
                T b=shared[threadIdx.x*2+1];
                a=op(a,b);
            }
        }

        __syncthreads();
        if (threadIdx.x*2<n)
            shared[threadIdx.x]=a;
        __syncthreads();
        n=n/2+n%2;
    }

    //after stage 2 only 2 elements should remain
    if (threadIdx.x==0)
        *output=op(shared[0],shared[1]);
}

template<typename T,typename OP>
T reduce( const T* a, int N, OP op)
{
    int max_block_size=1024;
    int block_size=std::min(max_block_size,(N-1)/2+1);

    T* output;
    cudaMalloc(&output,sizeof(T));
    reduce_kernel<<<1,block_size>>>(output,a,N,op);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    T result;
    CUDA_CHECK(cudaMemcpy(&result,output,sizeof(T),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(output));
    return result;
}


bool compare(float* v1, float* v2, int N)
{
    bool* comparison;CUDA_CHECK(cudaMalloc(&comparison,sizeof(bool)*N));

    const int block_size=256;
    int blocks=(N-1)/block_size+1;
    auto eq_lambda=[]__device__(float a,float b){return compare_floats(a,b,32);};
    kernel_vec_op<<<blocks,block_size>>>(comparison,v1,v2,N,eq_lambda);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bool result=reduce(comparison,N,[]__device__(bool a,bool b){return a&&b;});
    CUDA_CHECK(cudaFree(comparison));
    return result;
}



template<typename T,typename OP>
__global__
void reduce_coalesced_kernel(T* output,const T* a, int N, OP op)
{
    T s;
    if (threadIdx.x<N)
    {
        s=a[threadIdx.x];
    }
    for (auto i=threadIdx.x+blockDim.x;i<N;i+=blockDim.x)
    {
        s=op(s,a[i]);
    }

    __shared__ T shared[2048];
    shared[threadIdx.x]=s;
    __syncthreads();

    //STAGE 2
    auto n=blockDim.x;
    while (n>2)
    {
        T a;
        if (threadIdx.x*2<n)
        {
            a=shared[threadIdx.x*2];

            if (threadIdx.x*2+1<n)
            {
                T b=shared[threadIdx.x*2+1];
                a=op(a,b);
            }
        }

        __syncthreads();
        if (threadIdx.x*2<n)
            shared[threadIdx.x]=a;
        __syncthreads();
        n=n/2+n%2;
    }

    //after stage 2 only 2 elements should remain
    if (threadIdx.x==0)
        *output=op(shared[0],shared[1]);
}

template<typename T,typename OP>
T reduce_coalesced( const T* a, int N, OP op)
{
    int max_block_size=1024;
    int block_size=std::min(max_block_size,(N-1)/2+1);

    T* output;
    cudaMalloc(&output,sizeof(T));
    reduce_coalesced_kernel<<<1,block_size>>>(output,a,N,op);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    T result;
    CUDA_CHECK(cudaMemcpy(&result,output,sizeof(T),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(output));
    return result;
}


#define COARSENING 4

template<typename T,typename OP>
__global__
void reduce_coarsened_kernel(T* output,const T* a, int N, OP op)
{
    T s;
    if (threadIdx.x<N)
    {
        s=a[threadIdx.x];
    }
    for (auto i=threadIdx.x+blockDim.x;i<N;i+=blockDim.x)
    {
        s=op(s,a[i]);
    }

    __shared__ T shared[2048];
    T* inBuffer=shared;
    T* outBuffer=shared+1024;
    inBuffer[threadIdx.x]=s;

    //STAGE 2
    auto n=blockDim.x;
    while (true)
    {
        __syncthreads();
        T a;
        if (threadIdx.x*COARSENING<n)
        {
            a=inBuffer[threadIdx.x];

            for (auto i=1;i<COARSENING;++i)
            {
                if (COARSENING*i+threadIdx.x<n)
                {
                    T b=inBuffer[COARSENING*i+threadIdx.x];
                    a=op(a,b);
                }
            }
            outBuffer[threadIdx.x]=a;
        }


        if (n<=COARSENING)
            break;

        n=n/COARSENING+n%COARSENING;
        T* tmp=inBuffer;
        inBuffer=outBuffer;
        outBuffer=tmp;
    }

    if (threadIdx.x==0)
        *output=inBuffer[0];
}

template<typename T,typename OP>
T reduce_coarsened( const T* a, int N, OP op)
{
    int max_block_size=1024;
    int block_size=std::min(max_block_size,(N-1)/COARSENING+1);

    T* output;
    cudaMalloc(&output,sizeof(T));
    reduce_coarsened_kernel<<<1,block_size>>>(output,a,N,op);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    T result;
    CUDA_CHECK(cudaMemcpy(&result,output,sizeof(T),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(output));
    return result;
}