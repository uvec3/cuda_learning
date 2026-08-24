#pragma once


void merge(float* out,const float* a, uint32_t m,const float* b, uint32_t n);
void merge_coalesced(float* out,const float* a, uint32_t m,const float* b, uint32_t n);
__device__ __host__ uint32_t corank(const float* a, uint32_t m,const float* b, uint32_t n, uint32_t k);