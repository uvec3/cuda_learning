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
   return a*b;
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
   const uint64_t blockSize=32;

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

   uint64_t N=10;
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
   dur=std::chrono::high_resolution_clock::now()-start;
   double time_device=dur.count()/ 1'000'000'000.0;

   // std::cout<<"a: ";
   // for (auto i=0;i<N;++i)
   // {
   //    std::cout<<a[i]<<"\t";
   // }
   //
   // std::cout<<"\nb: ";
   // for (auto i=0;i<N;++i)
   // {
   //    std::cout<<b[i]<<"\t";
   // }
   //
   // std::cout<<"\nh: ";
   // for (auto i=0;i<N;++i)
   // {
   //    std::cout<<result[i]<<"\t";
   // }
   //
   // std::cout<<"\nd: ";
   // for (auto i=0;i<N;++i)
   // {
   //    std::cout<<result[i]<<"\t";
   // }

   std::cout<<"\nTime host:"<<time_host;
   std::cout<<"\nTime device:"<<time_device;

   free(a);
   free(b);
   free(result);
   free(result_from_device);
}