#include "device_merge.hpp"

#include <cstdio>

#define BLOCK_SIZE 128
#define COARSENING 13

__device__ __host__ bool less(float l, float r)
{
    //compare only integer parts to observe merge stability
    return std::floor(l)<std::floor(r);
}



__device__ __host__ uint32_t corank(const float* a, uint32_t m,const float* b, uint32_t n, uint32_t k)
{
    auto a_min=k>n?k-n:0;
    auto a_max=k<m?k:m;


    while (true)
    {
        auto i=(a_min+a_max)/2;
        auto j=k-i;

        if (i<a_max && !less(b[j-1],a[i]))//a[i]<=b[j-1] (b[j-1] no less than a[i] => b[j-1]>=a[i] => a[i]<=b[j-1])
        {
            a_min=i+1;
        }
        else if (i>a_min && less(b[j],a[i-1]))//b[j]<a[i-1]
        {
            a_max=i;
        }
        else
        {
            return i;
        }
    }
}

__device__ __host__ inline const float* sequential_merge(float* result, const float* a,const float* a_end,
    const float* b , const float* b_end ,uint32_t n)
{
    // #pragma unroll
    for (int k=0;k<n;++k)
    {
        if (b==b_end || (a!=a_end && !less(*b,*a)))//(*a)<=(*b)
        {
            result[k]= *a;
            ++a;
        }
        else
        {
            result[k]= *b;
            ++b;
        }
    }

    //return  pointer to a, this can be used to determine how many elements of each array was consumed
    return a;
}

__global__ void merge_kernel(float* out, const float* a, uint32_t m,const float* b, uint32_t n)
{
    auto block_offset=blockDim.x*COARSENING*blockIdx.x;
    // __shared__ uint32_t a_block_start;
    // if (threadIdx.x==0)
    // {
    //     a_block_start=corank(a,m,b,n,block_offset);
    // }
    // __syncthreads();


    uint32_t thread_output_start=block_offset+threadIdx.x*COARSENING;
    if (thread_output_start<m+n)
    {
        auto i=corank(a,m,b,n,thread_output_start);
        auto j=thread_output_start-i;

        sequential_merge(out+thread_output_start,a+i,a+m,b+j,b+n,COARSENING);
    }
}

void merge(float* out,const float* a, uint32_t m,const float* b, uint32_t n)
{
    auto number_of_blocks=(m+n-1)/(BLOCK_SIZE*COARSENING)+1;
    merge_kernel<<<number_of_blocks,BLOCK_SIZE>>>(out,a,m,b,n);
}


__device__ void block_memcpy(float* dst, const float* src, uint32_t size)
{
    for (uint32_t i=threadIdx.x;i<size;i+=blockDim.x)
    {
        dst[i]=src[i];
    }
}

__global__ void merge_coalesced_kernel(float* out, const float* a, uint32_t m,const float* b, uint32_t n)
{
    auto block_offset=blockDim.x*COARSENING*blockIdx.x;
    auto block_end=block_offset+blockDim.x*COARSENING;
    if (block_end>m+n)
        block_end=m+n;

    __shared__ uint32_t a_block_start;
    __shared__ uint32_t a_block_end;
    if (threadIdx.x==0)//warp 0
    {
        a_block_start=corank(a,m,b,n,block_offset);
        // printf("block %d start A at %d\n",blockIdx.x,a_block_start);
    }
    else if (threadIdx.x==32)// warp 1
    {
        a_block_end=corank(a,m,b,n,block_end);
        // printf("block %d end A at %d\n",blockIdx.x,a_block_end);
    }
    __syncthreads();

    auto section_size=block_end-block_offset;


    //split the shard memory between a and b according to their sizes
    __shared__ float a_s[COARSENING*BLOCK_SIZE];
    float* b_s=a_s+(a_block_end-a_block_start);


    //load a and b sections to the shared memory
    block_memcpy(a_s,a+a_block_start,a_block_end-a_block_start);
    block_memcpy(b_s,b+block_offset-a_block_start,section_size-(a_block_end-a_block_start));

    __syncthreads();

    //find the starting point for each thread within the block segment
    uint32_t thread_output_start=threadIdx.x*COARSENING;
    auto i=corank(a_s,a_block_end-a_block_start,b_s,section_size-(a_block_end-a_block_start),
        thread_output_start);
    auto j=thread_output_start-i;

    //every thread saves its portion of array A to the registers
    float reg[COARSENING];

    //do sequential scan on shared memory and save the result in the registers
    sequential_merge(reg,a_s+i,b_s,b_s+j,a_s+section_size,COARSENING);

    // printf("block = %d, thread %d, start=%d, i=%d(%d), j=%d(%d) val=%f=min(%f, %f)\n",
        // blockIdx.x, threadIdx.x, thread_output_start,i,i+a_block_start,j,j+block_offset-a_block_start,reg[0],a_s[i],b_s[j]);

    //wait for all threads to finish their scans before overwriting the shared memory
    __syncthreads();

    //write registers to shared memory
    for (int k=0;k<COARSENING;++k)
    {
        a_s[thread_output_start+k]=reg[k];
    }

    //transfer data from shared to global memory in a coalesced way
    __syncthreads();
    block_memcpy(out+block_offset,a_s,section_size);
}

void merge_coalesced(float* out,const float* a, uint32_t m,const float* b, uint32_t n)
{
    auto number_of_blocks=(m+n-1)/(BLOCK_SIZE*COARSENING)+1;
    merge_coalesced_kernel<<<number_of_blocks,BLOCK_SIZE>>>(out,a,m,b,n);
}