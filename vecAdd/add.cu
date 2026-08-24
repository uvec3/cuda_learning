#include <iostream>
#include <string.h>
#include <chrono>
#include <cstdint>
#include <sstream>


#define CHECK_CUDA_ERROR(e) if(e!=cudaSuccess)\
{ \
   throw std::runtime_error("Error:"+std::to_string(__LINE__)); \
   cudaError_t last_error = cudaGetLastError();\
}


__device__ __host__ float operation(float a, float b)
{
   return a+b;
}


void hostAdd(float* result, float* a, float* b,uint64_t N)
{
   for (uint64_t i=0;i<N;++i)
   {
      result[i]= operation(a[i],b[i]);
   }
}


__global__ void kernelAdd(float* result, float* a, float* b,uint64_t N)
{
   auto i = blockIdx.x*blockDim.x+threadIdx.x;
   if (i<N)
   {
      //printf("%d executed\n",i);
      result[i]=operation(a[i],b[i]);
   }
   else
   {
      //printf("%d skipped\n",i);
   }
}


void deviceAdd(float* result_h, float* a_h, float* b_h,uint64_t N)
{
   const uint64_t blockSize=128;

   auto vecSizeBytes=sizeof(result_h[0])*N;

   cudaError resultCode;
   float* a_d; resultCode = cudaMalloc(&a_d,vecSizeBytes);
   CHECK_CUDA_ERROR(resultCode)
   float* b_d; resultCode = cudaMalloc(&b_d,vecSizeBytes);
   CHECK_CUDA_ERROR(resultCode)
   float* result_d; resultCode = cudaMalloc(&result_d,vecSizeBytes);
   CHECK_CUDA_ERROR(resultCode)

   resultCode=cudaMemcpy(a_d,a_h,vecSizeBytes,cudaMemcpyHostToDevice);
   resultCode=cudaMemcpy(b_d,b_h,vecSizeBytes,cudaMemcpyHostToDevice);

   kernelAdd<<<(N-1)/blockSize+1,blockSize>>>(result_d,a_d,b_d,N);

   resultCode=cudaDeviceSynchronize();

   resultCode=cudaMemcpy(result_h,result_d,vecSizeBytes,cudaMemcpyDeviceToHost);

   resultCode=cudaFree(result_d);
   resultCode=cudaFree(a_d);
   resultCode=cudaFree(b_d);
}

__global__ void kernelAddBy16(float* result, float* a, float* b,uint64_t N)
{
   const float4* a4=reinterpret_cast<const float4*>(a);
   const float4* b4=reinterpret_cast<const float4*>(b);
   float4* result4=reinterpret_cast<float4*>(result);
   auto i = (blockIdx.x*blockDim.x+threadIdx.x);
   if (i<N/4)
   {
      float4 vecA=a4[i];
      float4 vecB=b4[i];
      result4[i]={vecA.x+vecB.x,vecA.y+vecB.y,vecA.z+vecB.z,vecA.w+vecB.w};
   }
   else
   {
      for (i=i*4;i<N;++i)
      result[i]=operation(a[i],b[i]);
   }
}


void deviceAdd16(float* result_h, float* a_h, float* b_h,uint64_t N)
{
   const uint64_t blockSize=128;

   auto vecSizeBytes=sizeof(result_h[0])*N;

   cudaError resultCode;
   float* a_d; resultCode = cudaMalloc(&a_d,vecSizeBytes);
   CHECK_CUDA_ERROR(resultCode)
   float* b_d; resultCode = cudaMalloc(&b_d,vecSizeBytes);
   CHECK_CUDA_ERROR(resultCode)
   float* result_d; resultCode = cudaMalloc(&result_d,vecSizeBytes);
   CHECK_CUDA_ERROR(resultCode)

   resultCode=cudaMemcpy(a_d,a_h,vecSizeBytes,cudaMemcpyHostToDevice);
   resultCode=cudaMemcpy(b_d,b_h,vecSizeBytes,cudaMemcpyHostToDevice);

   kernelAddBy16<<<((N-1)/4)/blockSize+1,blockSize>>>(result_d,a_d,b_d,N);

   resultCode=cudaDeviceSynchronize();

   resultCode=cudaMemcpy(result_h,result_d,vecSizeBytes,cudaMemcpyDeviceToHost);

   resultCode=cudaFree(result_d);
   resultCode=cudaFree(a_d);
   resultCode=cudaFree(b_d);
}

#include <cuda/barrier>
#include <cooperative_groups.h>
#define CHUNK_SIZE 1024
__global__ void kernelAddAsyncLoad(float* result, float* a, float* b,uint64_t N)
{
   __shared__ float shared_buffer[CHUNK_SIZE*2];
   __shared__ cuda::barrier<cuda::thread_scope_block> bar;

   // Get the current thread block group
   cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
   // Initialize the barrier for the number of threads in the block participating
   if (block.thread_rank() == 0) {
      init(&bar, block.size());
   }
   block.sync(); // Ensure barrier is initialized before use

   auto block_offset=blockIdx.x*CHUNK_SIZE;
   // Collectively initiate the asynchronous copy of the entire memory chunk
   // Every thread in the block must call this collective function

   int elements_to_read=(N-block_offset)<CHUNK_SIZE?(N-block_offset):CHUNK_SIZE;
   cuda::memcpy_async(block, shared_buffer, a+block_offset, sizeof(float) * elements_to_read, bar);
   cuda::memcpy_async(block, shared_buffer+CHUNK_SIZE, b+block_offset, sizeof(float) * elements_to_read, bar);
   // if (threadIdx.x==0)
   // {
   //    printf("block %d reading %d elements\n",blockIdx.x, elements_to_read);
   // }

   // Threads can do independent compute here while data transfers...
   auto coarsening=CHUNK_SIZE / blockDim.x;


   // A barrier is required to wait until the asynchronous transfer is fully complete
   auto token = bar.arrive();
   bar.wait(std::move(token));

   #pragma unroll
   for (int k=0;k<coarsening;++k)
   {
      if (block_offset+blockDim.x*k+threadIdx.x<N)
         result[block_offset+blockDim.x*k+threadIdx.x]=shared_buffer[blockDim.x*k+threadIdx.x]+shared_buffer[CHUNK_SIZE+blockDim.x*k+threadIdx.x];
   }
}

void deviceAddAsyncLoad(float* result_h, float* a_h, float* b_h,uint64_t N)
{
   const uint64_t blockSize=256;

   auto vecSizeBytes=sizeof(result_h[0])*N;

   cudaError resultCode;
   float* a_d; resultCode = cudaMalloc(&a_d,vecSizeBytes);
   CHECK_CUDA_ERROR(resultCode)
   float* b_d; resultCode = cudaMalloc(&b_d,vecSizeBytes);
   CHECK_CUDA_ERROR(resultCode)
   float* result_d; resultCode = cudaMalloc(&result_d,vecSizeBytes);
   CHECK_CUDA_ERROR(resultCode)

   resultCode=cudaMemcpy(a_d,a_h,vecSizeBytes,cudaMemcpyHostToDevice);
   resultCode=cudaMemcpy(b_d,b_h,vecSizeBytes,cudaMemcpyHostToDevice);

   kernelAddAsyncLoad<<<(N-1)/CHUNK_SIZE+1,blockSize>>>(result_d,a_d,b_d,N);

   resultCode=cudaDeviceSynchronize();

   resultCode=cudaMemcpy(result_h,result_d,vecSizeBytes,cudaMemcpyDeviceToHost);

   resultCode=cudaFree(result_d);
   resultCode=cudaFree(a_d);
   resultCode=cudaFree(b_d);
}



void initVec(float* vec, uint64_t N)
{
   for (uint64_t i=0;i<N;++i)
   {
      vec[i]=i;
   }
}

int main(int argc, char ** argv)
{
   std::cout<<"Starting: \n"<<argv[0]<<'\n';

   uint64_t N=99;
   if (argc>1)
   {
      auto ss=std::istringstream(argv[1]);
      ss>>N;
   }


   float* a= (float*) malloc(N* sizeof(float));
   float* b= (float*) malloc(N* sizeof(float));
   float* result= (float*) malloc(N* sizeof(float));
   float* result_from_device = (float*) malloc(N* sizeof(float));

   initVec(a,N);
   initVec(b,N);
   memset(result,0,N*sizeof(float));

   auto start= std::chrono::high_resolution_clock::now();
   hostAdd(result,a,b,N);
   auto dur=std::chrono::high_resolution_clock::now()-start;
   double time_host=dur.count()/ 1'000'000'000.0;


   start= std::chrono::high_resolution_clock::now();
   deviceAdd(result_from_device,a,b,N);
   deviceAdd16(result_from_device,a,b,N);
   deviceAddAsyncLoad(result_from_device,a,b,N);
   dur=std::chrono::high_resolution_clock::now()-start;
   double time_device=dur.count()/ 1'000'000'000.0;


   if (N<100)
   {
      std::cout<<"a: ";
      for (auto i=0;i<N;++i)
      {
         std::cout<<a[i]<<"\t";
      }

      std::cout<<"\nb: ";
      for (auto i=0;i<N;++i)
      {
         std::cout<<b[i]<<"\t";
      }

      std::cout<<"\nh: ";
      for (auto i=0;i<N;++i)
      {
         std::cout<<result[i]<<"\t";
      }

      std::cout<<"\nd: ";
      for (auto i=0;i<N;++i)
      {
         std::cout<<result_from_device[i]<<"\n";
      }
   }

   std::cout<<"\nTime host:"<<time_host;
   std::cout<<"\nTime device:"<<time_device;

   free(a);
   free(b);
   free(result);
   free(result_from_device);
}