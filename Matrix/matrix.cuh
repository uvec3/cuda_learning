#pragma once
#include <stdexcept>
#include <format>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <glm/glm.hpp>

#include "common.cuh"
#include "kernels.cuh"


template <Location location=Location::Host,Layout layout=Layout::RowMajor>
class Matrix
{
    using T=float;
public:
    int rows=1;
    int cols=1;
    T* data=nullptr;

    Matrix (int rows, int cols):rows{rows},cols{cols}
    {
        if constexpr (location==Location::Device)
        {
            CUDA_CHECK(cudaMalloc((void**)&data,rows*cols*sizeof(T)));
        }
        else
        {
            CUDA_CHECK(cudaMallocHost((void**)&data,rows*cols*sizeof(T)));
            std::memset(data,0,rows*cols*sizeof(T));
        }
    }

    template<Location other_location,Layout other_layout>
    Matrix (const Matrix<other_location,other_layout>& other ):Matrix(other.rows,other.cols)
    {
        copyFrom(other);
    }


    Matrix (Matrix<location,layout>&& other ):rows{other.rows},cols{other.cols},data(other.data)
    {
        if constexpr (location==Location::Device)
        {
            CUDA_CHECK(cudaFree(other.data));
        }
        else
        {
            CUDA_CHECK(cudaFreeHost(other.data));
        }
        other.data=nullptr;
    }

    template<Location other_location,Layout other_layout>
    Matrix& operator=(const Matrix<other_location,other_layout>& other)
    {
        rows=other.rows;
        cols=other.cols;
        if constexpr (location==Location::Device)
        {
            CUDA_CHECK(cudaFree(data));
            CUDA_CHECK(cudaMalloc((void**)&data,rows*cols*sizeof(T)));
        }
        else
        {
            CUDA_CHECK(cudaFreeHost(data));
            CUDA_CHECK(cudaMallocHost((void**)&data,rows*cols*sizeof(T)));
            std::memset(data,0,rows*cols*sizeof(T));
        }

        copyFrom(other);
        return *this;
    }

    Matrix& operator=(Matrix<location,layout>&& other)
    {
        rows=other.rows;
        cols=other.cols;
        data=other.data;

        if constexpr (location==Location::Device)
        {
            CUDA_CHECK(cudaFree(other.data));
        }
        else
        {
            CUDA_CHECK(cudaFreeHost(other.data));
        }
        other.data=nullptr;

        return *this;
    }


    Matrix<location,layout> Transposed() const
    {
        Matrix<location,layout>transposed{cols,rows};

        if constexpr (location==Location::Host)
        {
            for (int r=0;r<rows;++r)
            {
                for (int c=0;c<=cols;++c)
                {
                    transposed[{c,r}]=operator[](glm::uvec2{r,c});
                }
            }
        }
        else
        {
            uint32_t block_width=16;
            transpose_kernel<<<{(cols-1)/block_width+1,(rows-1)/block_width+1},{block_width,block_width}>>>(transposed.view(),view());
        }


        return transposed;
    }

    template <Location other_location,Layout other_layout>
    void copyFrom(const Matrix<other_location,other_layout>& other,bool async=false
        )
    {
        if (other.cols!=cols || other.rows!= rows)
            throw std::runtime_error(std::format("Dimensions mismatch: ({},{}) and ({},{})",
                rows,cols,other.rows,other.cols));


        int kind=(location==Location::Device) | ((other_location==Location::Device)<<1);

        //static_assert(layout==other_layout);

        if (async)
        {
            if constexpr (layout==other_layout)
                CUDA_CHECK(cudaMemcpyAsync(data,other.data,rows*cols*sizeof(T),static_cast<enum cudaMemcpyKind>(kind)));
            else
                CUDA_CHECK(cudaMemcpyAsync(data,other.Transposed().data,rows*cols*sizeof(T),static_cast<enum cudaMemcpyKind>(kind)));
        }else
        {
            if constexpr (layout==other_layout)
                CUDA_CHECK(cudaMemcpy(data,other.data,rows*cols*sizeof(T),static_cast<enum cudaMemcpyKind>(kind)));
            else
                CUDA_CHECK(cudaMemcpy(data,other.Transposed().data,rows*cols*sizeof(T),static_cast<enum cudaMemcpyKind>(kind)));
        }
    }

    Matrix<Location::Device,layout> copyToDevice(bool async=true) const
    {
        Matrix<Location::Device,layout> copy{rows,cols};
        copy.copyFrom(*this, async);
        return copy;
    }

    Matrix<Location::Host,layout> copyToHost(bool async=false) const
    {
        Matrix<Location::Host,layout> copy{rows,cols};
        copy.copyFrom(*this, async);
        return copy;
    }


    T& operator[](glm::uvec2 index)
    {
        if constexpr (layout==Layout::RowMajor)
        {
            auto* ptr= reinterpret_cast<T*>(data);
            return  ptr[index[0]*cols+index[1]];
        }
        else
        {
            auto* ptr= reinterpret_cast<T*>(data);
            return  ptr[index[1]*rows+index[0]];
        }
    }


    T operator[](glm::uvec2 index)const
    {
        if constexpr (layout==Layout::RowMajor)
        {
            auto* ptr= reinterpret_cast<T*>(data);
            return  ptr[index[0]*cols+index[1]];
        }
        else
        {
            auto* ptr= reinterpret_cast<T*>(data);
            return  ptr[index[1]*rows+index[0]];
        }
    }

    std::string to_string()
    {
        //if on device copy to host first
        if constexpr (location==Location::Device)
        {
            return copyToHost(false).to_string();
        }

        std::ostringstream ss;

        for (int j=0;j<rows;++j)
        {
            for (int i=0;i<cols;++i)
            {
                ss<<(*this)[{j,i}]<<"\t";
            }
            ss<<"\n";
        }

        return ss.str();
    }


    MatrixView<layout> view() const
    {
        return {static_cast<float*>(data),{rows,cols}};
    }

    void reset()
    {
        if constexpr (location==Location::Device)
            cudaMemset(data,0,sizeof(T)*rows*cols);
        else
            std::memset(data,0,sizeof(T)*rows*cols);
    }


    ~Matrix()
    {
        if constexpr  (location==Location::Device)
            cudaFree(data);
        else
            cudaFreeHost(data);

        // if constexpr (location==Location::Device)
        //     std::cout<<std::format("Device matrix {}x{} destroyed\n",rows,cols)<<std::endl;
        // else
        //     std::cout<<std::format("Host matrix {}x{} destroyed\n",rows,cols)<<std::endl;
    }



};



template <Layout left_layout, Layout right_layout>
bool operator==(Matrix<Location::Host,left_layout>& left_h,Matrix<Location::Host,right_layout>& right_h )
{
    if (left_h.rows!=right_h.rows|| left_h.cols!=right_h.cols)
        return false;

    for (int r=0;r<left_h.rows;++r)
    {
        for (int c=0;c<left_h.cols;++c)
        {
            auto lv=left_h[{r,c}];
            auto rv=right_h[{r,c}];
            if (!compare_floats(lv,rv,10))
            {
                return false;
            }

        }
    }

    return true;
}

template <Layout left_layout, Layout right_layout>
bool operator==(Matrix<Location::Host,left_layout>& left_h,Matrix<Location::Device,right_layout>& right_d )
{
    auto right_h=right_d.copyToHost();
    return left_h==right_h;
}


template <Layout left_layout, Layout right_layout>
bool operator==(Matrix<Location::Device,left_layout>& left_d,Matrix<Location::Host,right_layout>& right_h )
{
    auto left_h=left_d.copyToHost();
    return left_h==right_h;
}

template <Layout left_layout, Layout right_layout>
bool operator==(Matrix<Location::Device,left_layout>& left_d,Matrix<Location::Device,right_layout>& right_d )
{
    if (left_d.rows!=right_d.rows|| left_d.cols!=right_d.cols)
        return false;
    return compare(left_d.data,right_d.data,left_d.cols*left_d.rows);
}


template <Layout result_layout,Layout left_layout, Layout right_layout>
bool mul_cpu_naive(Matrix<Location::Host,result_layout>& result, Matrix<Location::Host,left_layout>& left,Matrix<Location::Host,right_layout>& right )
{
    if (result.rows!=left.rows||result.cols!=right.cols)
    {
        std::cerr<<"Result matrix dimensions are incompatible with the operands!\n";
        exit(1);
    }


    for (int row=0;row<result.rows;++row)
    {
        for (int col = 0 ; col<result.cols;++col)
        {
            float dot=0;
            for (int k=0;k<left.cols;++k)
            {
                dot+=left[{row,k}]*right[{k,col}];
            }
            result[{row,col}]=dot;

        }
    }
    return true;
}


template <Layout left_layout, Layout right_layout>
Matrix<Location::Host,Layout::RowMajor> operator*(Matrix<Location::Host,left_layout>& left,Matrix<Location::Host,right_layout>& right )
{
    if (left.cols!=right.rows)
        throw std::runtime_error(
            std::format("dimensions mismatch: ({} {}) x ({} {})",left.rows,
            left.cols,right.rows,right.cols));

    Matrix<Location::Host,Layout::RowMajor> result{left.rows,right.cols};
    mul_cpu_naive(result,left,right);
    return result;
}

template <Layout result_layout, Layout left_layout, Layout right_layout>
bool mul_GPU_naive(Matrix<Location::Device,result_layout>& result,Matrix<Location::Device,left_layout>& left,
    Matrix<Location::Device,right_layout>& right,glm::uvec2 block_size=glm::uvec2{32});

template <Layout left_layout, Layout right_layout>
Matrix<Location::Device> operator*(Matrix<Location::Device,left_layout>& left,Matrix<Location::Device,right_layout>& right )
{
    if (left.cols!=right.rows)
        throw std::runtime_error(
            std::format("dimensions mismatch: ({} {}) x ({} {})",left.rows,
            left.cols,right.rows,right.cols));

    Matrix<Location::Device> result{left.rows,right.cols};
    mul_GPU_naive(result,left,right,glm::uvec2(32,32));
    return result;
}