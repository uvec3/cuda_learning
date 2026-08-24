#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <chrono>

#include "matrix.cuh"
#include "multiplication_kernels.cuh"
#include "kernels.cuh"
#include <iostream>

#include <cuda_profiler_api.h>


template<Layout layout>
Matrix<Location::Host,layout> generate_random_matrix(int rows, int cols)
{
    Matrix<Location::Host,layout> m{rows,cols};
    for (int r=0;r<rows;++r)
    {
        for (int c=0;c<cols;++c)
        {
            m[{r,c}]=static_cast<float>((rand()%100)/100.00);
            // m[{r,c}]=static_cast<float>(rand())/RAND_MAX*20.0-10.0;
        }
    }

    return m;
}


# define TEST_OP(header,op_fun,result,reference,print)\
{\
    std::cout<<header<<": ";\
    result.reset();\
    cudaDeviceSynchronize();\
    auto start=std::chrono::high_resolution_clock::now();\
    op_fun;\
    cudaDeviceSynchronize();\
    auto duration= std::chrono::high_resolution_clock::now()-start;\
    double time_ms=duration.count()/1'000'000.0;\
    std::cout<<time_ms<<" ms  ";\
    if (result==reference)\
    {\
        std::cout<<"Passed!\n";\
    }\
    else\
    {\
        std::cout<<"Failed!\n";\
    }\
    if (print)\
        std::cout<<"result: \n"<<result.to_string()<<std::endl;\
}


inline std::string to_string(Layout layout)
{
    if (layout==Layout::RowMajor)
        return "RowMajor";
    return "ColumnMajor";
}


template< Layout result_layout, Layout left_layout, Layout right_layout>
void test_multiplications( Matrix<Location::Host,left_layout>& M, Matrix<Location::Host,right_layout>& N, bool gpu_oly )
{
    std::string header=std::format("{}({}x{}) = {}({}x{}) x {}({}x{}))",
        to_string(result_layout),M.rows,N.cols,
        to_string(left_layout),M.rows,M.cols,
        to_string(right_layout),N.rows,N.cols);

    std::cout<<header<<":"<<std::endl;

    bool print=false;




    if (!gpu_oly)
    {
        //Host pass:
        auto reference=M*N;
        Matrix<Location::Host, result_layout> res_h{M.rows,N.cols};
        TEST_OP("\tcpu naive",  mul_cpu_naive(res_h,M,N) ,res_h, reference,print);
    }

    // //Device pass:

    Matrix<Location::Device, result_layout> res_d{M.rows,N.cols};
    auto M_d=M.copyToDevice(false);
    auto N_d=N.copyToDevice(false);
    Matrix<Location::Device,result_layout> reference_d=M_d*N_d;

    TEST_OP("\tgpu naive",  mul_GPU_naive(res_d,M_d,N_d) ,res_d, reference_d,print);
    TEST_OP("\tgpu row per thread",  multiply_row_per_thread(res_d,M_d,N_d) ,res_d, reference_d,print);
    TEST_OP("\tgpu column per thread",  multiply_column_per_thread(res_d,M_d,N_d) ,res_d, reference_d,print);
    TEST_OP("\tgpu tiled 32",  mul_GPU_tiled(res_d,M_d,N_d,32) ,res_d, reference_d,print);
    TEST_OP("\tgpu tiled corners 32",  mul_GPU_tiled_corners(res_d,M_d,N_d,32) ,res_d, reference_d,print);
    TEST_OP("\tgpu tiled coarsening 32",  mul_GPU_tiled_coarsening(res_d,M_d,N_d,32) ,res_d, reference_d,print);

    TEST_OP("\tgpu tiled 8",  mul_GPU_tiled(res_d,M_d,N_d,8) ,res_d, reference_d,print);
    TEST_OP("\tgpu tiled corners 8",  mul_GPU_tiled_corners(res_d,M_d,N_d,8) ,res_d, reference_d,print);
    TEST_OP("\tgpu tiled coarsening 8",  mul_GPU_tiled_coarsening(res_d,M_d,N_d,8) ,res_d, reference_d,print);

}

void test_size_in_all_layouts(int i, int j, int k)
{
    Matrix<Location::Host,Layout::RowMajor> MR=generate_random_matrix<Layout::RowMajor>(i,j);
    Matrix<Location::Host,Layout::RowMajor> NR=generate_random_matrix<Layout::RowMajor>(j,k);
    Matrix<Location::Host,Layout::ColumnMajor> MC=generate_random_matrix<Layout::ColumnMajor>(i,j);
    Matrix<Location::Host,Layout::ColumnMajor> NC=generate_random_matrix<Layout::ColumnMajor>(j,k);

    bool gpu_only=true;
    test_multiplications<Layout::RowMajor>(MR,NR,gpu_only);
    test_multiplications<Layout::ColumnMajor>(MR,NR,gpu_only);

    test_multiplications<Layout::RowMajor>(MC,NR,gpu_only);
    test_multiplications<Layout::ColumnMajor>(MC,NR,gpu_only);

    test_multiplications<Layout::RowMajor>(MR,NC,gpu_only);
    test_multiplications<Layout::ColumnMajor>(MR,NC,gpu_only);

    test_multiplications<Layout::RowMajor>(MC,NC,gpu_only);
    test_multiplications<Layout::ColumnMajor>(MC,NC,gpu_only);
}

int main(int argc, char** argv)
{
    auto m=generate_random_matrix<Layout::RowMajor>(200,500);
    auto mT_h=m.Transposed();
    auto m_d=m.copyToDevice();
    auto mT_d=m_d.Transposed();
    if (mT_d==mT_h)
        std::cout<<"Equal!\n";

    try
    {
        test_size_in_all_layouts(10  ,10,10);
        std::cout<<"\n";
        test_size_in_all_layouts(10  ,10,100);
        std::cout<<"\n";
        test_size_in_all_layouts(100  ,10,100);
        std::cout<<"\n";
        test_size_in_all_layouts(300  ,300,200);
        std::cout<<"\n";
        test_size_in_all_layouts(1000,1000,1000);

        std::cout<<"\n";
        // test_size_in_all_layouts(10000,1000,10000);
    }
    catch (std::exception& e)
    {
        std::cout<<e.what();
    }

    cudaProfilerStop();

    return 0;
}