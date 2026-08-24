#pragma once

#include "common.cuh"
#include "matrix.cuh"




template <Layout result_layout, Layout left_layout, Layout right_layout>
void check_dimensions(Matrix<Location::Device,result_layout>& result,Matrix<Location::Device,left_layout>& left,Matrix<Location::Device,right_layout>& right )
{
    if (result.rows!=left.rows||result.cols!=right.cols)
    {
        std::cerr<<"Result matrix dimensions are incompatible with the operands!\n";
        exit(1);
    }
}

// Naive multiplication without optimization
template <Layout result_layout, Layout left_layout, Layout right_layout>
__global__
void kernel_multiply_naive(MatrixView<result_layout> result, MatrixView<left_layout> left, MatrixView<right_layout> right)
{
    glm::uvec2 out_cell;
    out_cell[0]=blockIdx.y*blockDim.y+threadIdx.y;
    out_cell[1]=blockIdx.x*blockDim.x+threadIdx.x;

    if (out_cell[0]<result.dims[0] && out_cell[1]<result.dims[1])
    {
        float sum=0;
        for (int i=0;i<left.dims[1];++i)
        {
            sum+=left[{out_cell[0],i}] * right[{i,out_cell[1]}];
        }

        result[out_cell]=sum;
    }
}

template <Layout result_layout, Layout left_layout, Layout right_layout>
bool mul_GPU_naive(Matrix<Location::Device,result_layout>& result,Matrix<Location::Device,left_layout>& left,
    Matrix<Location::Device,right_layout>& right,glm::uvec2 block_size)
{
    check_dimensions(result,left,right);
    glm::uvec2 grid_size=(glm::uvec2{right.cols,left.rows} - 1u)/block_size+1u;
    kernel_multiply_naive<<<{grid_size.x,grid_size.y},{block_size.x,block_size.y}>>>(result.view(),left.view(),right.view());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

//block to row
template <Layout result_layout, Layout left_layout, Layout right_layout>
__global__
void kernel_multiply_row_block(MatrixView<result_layout> result, MatrixView<left_layout> left, MatrixView<right_layout> right)
{
    glm::uvec2 out_cell;
    out_cell[0]=blockIdx.y;
    out_cell[1]=blockIdx.x*blockDim.x+threadIdx.x;

    if (out_cell[0]<result.dims[0] && out_cell[1]<result.dims[1])
    {
        float sum=0;
        for (int i=0;i<left.dims[1];++i)
        {
            sum+=left[{out_cell[0],i}] * right[{i,out_cell[1]}];
        }

        result[out_cell]=sum;
    }
}

template <Layout result_layout, Layout left_layout, Layout right_layout>
bool multiply_row_block(Matrix<Location::Device,result_layout>& result,Matrix<Location::Device,left_layout>& left,Matrix<Location::Device,right_layout>& right, int block_size=1024 )
{
    check_dimensions(result,left,right);

    if (result.cols<block_size)
        block_size=result.cols;
    int blocks_per_row=(result.cols-1)/block_size+1;

    kernel_multiply_naive<<<{blocks_per_row,result.rows},block_size>>>(result.view(),left.view(),right.view());

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}


//row per thread
template <Layout result_layout, Layout left_layout, Layout right_layout>
__global__
void kernel_multiply_row_per_thread(MatrixView<result_layout> result, MatrixView<left_layout> left, MatrixView<right_layout> right)
{
    auto row=blockIdx.x*blockDim.x+threadIdx.x;
    if (row<result.dims[0])
    {
        for (int col=0;col<result.dims[1];++col)
        {
            float sum=0;
            for (int i=0;i<left.dims[1];++i)
            {
                sum+=left[{row,i}] * right[{i,col}];
            }
            result[{row,col}]=sum;
        }
    }
}

template <Layout result_layout, Layout left_layout, Layout right_layout>
bool multiply_row_per_thread(Matrix<Location::Device,result_layout>& result,Matrix<Location::Device,left_layout>& left,Matrix<Location::Device,right_layout>& right,int block_size=32 )
{
    check_dimensions(result,left,right);

    auto blocks=(result.rows-1)/block_size+1;

    kernel_multiply_row_per_thread<<<blocks,block_size>>>(result.view(),left.view(),right.view());

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}


//column per thread
template <Layout result_layout, Layout left_layout, Layout right_layout>
__global__
void kernel_multiply_column_per_thread(MatrixView<result_layout> result, MatrixView<left_layout> left, MatrixView<right_layout> right)
{
    auto col=blockIdx.x*blockDim.x+threadIdx.x;
    if (col<result.dims[1])
    {
        for (int row=0;row<result.dims[1];++row)
        {
            float sum=0;
            for (int i=0;i<left.dims[1];++i)
            {
                sum+=left[{row,i}] * right[{i,col}];
            }
            result[{row,col}]=sum;
        }
    }
}

template <Layout result_layout, Layout left_layout, Layout right_layout>
bool multiply_column_per_thread(Matrix<Location::Device,result_layout>& result,Matrix<Location::Device,left_layout>& left,Matrix<Location::Device,right_layout>& right ,int block_size=32)
{
    check_dimensions(result,left,right);


    auto blocks=(result.rows-1)/block_size+1;

    kernel_multiply_column_per_thread<<<blocks,block_size>>>(result.view(),left.view(),right.view());

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}


// Tiled multiplication
template <Layout result_layout, Layout left_layout, Layout right_layout>
__global__
void kernel_multiply_tiled(MatrixView<result_layout> result, MatrixView<left_layout> left, MatrixView<right_layout> right,int phases,uint32_t TILE_SIZE)
{
    extern __shared__ float shared_memory[];

    float* left_tile=shared_memory;
    float* right_tile=shared_memory+TILE_SIZE*TILE_SIZE;
    
    auto row=blockIdx.y*blockDim.y+threadIdx.y;
    auto col=blockIdx.x*blockDim.x+threadIdx.x;
    float sum=0;
    for (int phase=0;phase<phases;++phase)
    {
        //write-after-read(false) barrier
        __syncthreads();

         glm::uvec2 leftIndx={row, phase*TILE_SIZE+threadIdx.x};
         glm::uvec2 rightIndx={phase*TILE_SIZE+threadIdx.y,col};
         //load
         if (left.valid_index(leftIndx))
             left_tile[threadIdx.y*TILE_SIZE+threadIdx.x]=left[leftIndx];
         else
             left_tile[threadIdx.y*TILE_SIZE+threadIdx.x]=0;

         if (right.valid_index(rightIndx))
             right_tile[threadIdx.y*TILE_SIZE+threadIdx.x]=right[rightIndx];
         else
             right_tile[threadIdx.y*TILE_SIZE+threadIdx.x]=0;

        //read-after-write(true) barrier
        __syncthreads();

        //read
        for (int i=0;i<TILE_SIZE;++i)
        {
            sum+=left_tile[threadIdx.y*TILE_SIZE+i]*right_tile[i*TILE_SIZE+threadIdx.x];
        }
    }

    //write
    if (result.valid_index({row,col}))
        result[{row,col}]=sum;
}

template <Layout result_layout, Layout left_layout, Layout right_layout>
bool mul_GPU_tiled(Matrix<Location::Device,result_layout>& result,Matrix<Location::Device,left_layout>& left,
    Matrix<Location::Device,right_layout>& right,uint32_t TILE_SIZE)
{
    check_dimensions(result,left,right);
    glm::uvec2 grid_size=(glm::uvec2{right.cols,left.rows} - 1u)/TILE_SIZE+1u;

    int phases=(left.cols-1)/TILE_SIZE+1;
    kernel_multiply_tiled<<<{grid_size.x,grid_size.y},{TILE_SIZE,TILE_SIZE},2*TILE_SIZE*TILE_SIZE*sizeof(float)>>>(result.view(),left.view(),right.view(),phases,TILE_SIZE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

// Tiled with corner rotation reads
template <Layout result_layout, Layout left_layout, Layout right_layout>
__global__
void kernel_multiply_tiled_corner(MatrixView<result_layout> result, MatrixView<left_layout> left, MatrixView<right_layout> right,int phases,uint32_t TILE_SIZE)
{
    extern __shared__ float shared_memory[];

    float* left_tile_s=shared_memory;
    float* right_tile_s=shared_memory+TILE_SIZE*(TILE_SIZE+1);

    auto row=blockIdx.y*blockDim.y+threadIdx.y;
    auto col=blockIdx.x*blockDim.x+threadIdx.x;
    float sum=0;
    for (int phase=0;phase<phases;++phase)
    {
        //write-after-read(false) barrier
        __syncthreads();

        glm::uvec2 leftIdxRead;
        uint32_t leftIdxWrite;
        if constexpr (left_layout==Layout::RowMajor)
        {
            leftIdxRead={row, phase*TILE_SIZE+threadIdx.x};
            leftIdxWrite=threadIdx.y*(TILE_SIZE+1)+threadIdx.x;
        }
        else
        {
            leftIdxRead={blockIdx.y*TILE_SIZE+threadIdx.x, phase*TILE_SIZE+threadIdx.y};
            leftIdxWrite=threadIdx.x*(TILE_SIZE+1)+threadIdx.y;
        }

        //load left tile
        if (left.valid_index(leftIdxRead))
            left_tile_s[leftIdxWrite]=left[leftIdxRead];
        else
            left_tile_s[leftIdxWrite]=0;

        glm::uvec2 rightIdxRead;
        uint32_t rightIdxWrite;
        if constexpr (right_layout==Layout::RowMajor)
        {
            rightIdxRead={phase*TILE_SIZE+threadIdx.y,col};
            rightIdxWrite=threadIdx.y*(TILE_SIZE+1)+threadIdx.x;
        }
        else
        {
            rightIdxRead={phase*TILE_SIZE+threadIdx.x,blockIdx.x*TILE_SIZE+threadIdx.y};
            rightIdxWrite=threadIdx.x*(TILE_SIZE+1)+threadIdx.y;
        }

        //load right tile
        if (right.valid_index(rightIdxRead))
            right_tile_s[rightIdxWrite]=right[rightIdxRead];
        else
            right_tile_s[rightIdxWrite]=0;

        //read-after-write(true) barrier
        __syncthreads();

        //read
        for (int i=0;i<TILE_SIZE;++i)
        {
            sum+=left_tile_s[threadIdx.y*(TILE_SIZE+1)+i]*right_tile_s[i*(TILE_SIZE+1)+threadIdx.x];
        }
    }

    //write
    if (result.valid_index({row,col}))
        result[{row,col}]=sum;
}


template <Layout result_layout, Layout left_layout, Layout right_layout>
bool mul_GPU_tiled_corners(Matrix<Location::Device,result_layout>& result,Matrix<Location::Device,left_layout>& left,
    Matrix<Location::Device,right_layout>& right,uint32_t TILE_SIZE)
{
    check_dimensions(result,left,right);
    glm::uvec2 grid_size=(glm::uvec2{right.cols,left.rows} - 1u)/TILE_SIZE+1u;

    int phases=(left.cols-1)/TILE_SIZE+1;
    kernel_multiply_tiled_corner<<<{grid_size.x,grid_size.y},{TILE_SIZE,TILE_SIZE},2*TILE_SIZE*(TILE_SIZE+1)*sizeof(float)>>>(result.view(),left.view(),right.view(),phases,TILE_SIZE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

#define COARSENING_FACTOR  4
template <Layout result_layout, Layout left_layout, Layout right_layout>
__global__
void kernel_multiply_tiled_coarsening( MatrixView<result_layout> result, MatrixView<left_layout> left, MatrixView<right_layout> right,int phases, uint32_t TILE_SIZE )
{
    extern __shared__ float shared_memory[];

    float* left_tile_s=shared_memory;
    float* right_tile_s=shared_memory+TILE_SIZE*TILE_SIZE;

    float results[COARSENING_FACTOR];
    for (int k=0;k<COARSENING_FACTOR;++k)
    {
        results[k]=0;
    }

    auto row=blockIdx.y*blockDim.y+threadIdx.y;
    auto col=blockIdx.x*blockDim.x*COARSENING_FACTOR+threadIdx.x;

    for (int phase=0;phase<phases;++phase)
    {
        glm::uvec2 leftIndx={row, phase*TILE_SIZE+threadIdx.x};
        //load
        if (left.valid_index(leftIndx))
            left_tile_s[threadIdx.y*TILE_SIZE+threadIdx.x]=left[leftIndx];
        else
            left_tile_s[threadIdx.y*TILE_SIZE+threadIdx.x]=0;

        for (int k=0;k<COARSENING_FACTOR;++k)
        {
            glm::uvec2 rightIndx={phase*TILE_SIZE+threadIdx.y,col+k*TILE_SIZE};

            if (right.valid_index(rightIndx))
                right_tile_s[threadIdx.y*TILE_SIZE+threadIdx.x]=right[rightIndx];
            else
                right_tile_s[threadIdx.y*TILE_SIZE+threadIdx.x]=0;

            //read-after-write(true) barrier
            __syncthreads();
            //read
            for (int i=0;i<TILE_SIZE;++i)
            {
                 results[k]+=left_tile_s[threadIdx.y*TILE_SIZE+i]*right_tile_s[i*TILE_SIZE+threadIdx.x];
            }
            //write-after-read(false) barrier
            __syncthreads();
        }
    }

    for (int k=0;k<COARSENING_FACTOR;++k)
    {
        if (result.valid_index({row,col+k*TILE_SIZE}))
            result[{row,col+k*TILE_SIZE}]=results[k];
    }
}


template <Layout result_layout, Layout left_layout, Layout right_layout>
bool mul_GPU_tiled_coarsening(Matrix<Location::Device,result_layout>& result,Matrix<Location::Device,left_layout>& left,
    Matrix<Location::Device,right_layout>& right,uint32_t TILE_SIZE)
{
    check_dimensions(result,left,right);
    glm::uvec2 grid_size=(glm::uvec2{right.cols,left.rows} - 1u)/glm::uvec2(TILE_SIZE*COARSENING_FACTOR,TILE_SIZE)+1u;

    int phases=(left.cols-1)/TILE_SIZE+1;
    kernel_multiply_tiled_coarsening<<<{grid_size[0],grid_size[1]},{TILE_SIZE,TILE_SIZE},2*TILE_SIZE*TILE_SIZE*sizeof(float)>>>(result.view(),left.view(),right.view(),phases,TILE_SIZE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}