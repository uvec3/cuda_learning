#include <iostream>
#include "device_merge.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/merge.h>


thrust::host_vector<float> generate_random_sorted_array(uint32_t size, float add=0)
{
    thrust::host_vector<float> result(size);
    int val=0;
    for (int i= 0; i<size;++i)
    {
        result[i]=val+add;
        val+=rand()%5;
    }
    return result;
}

void print_vector(const thrust::host_vector<float>& vec)
{
    for (int i=0;i<vec.size();++i)
    {
        printf("%d)\t\t%f\n",i,vec[i]);
    }
}

void compare_vectors(const thrust::host_vector<float>& etalon, const thrust::host_vector<float>& vector)
{
    if (etalon.size()!=vector.size())
    {
        printf("Vector sizes mismatch!\n");
    }

    for (int i=0;i<etalon.size();++i)
    {
        if (etalon[i]!=vector[i])
        {
            printf("Values differ at position %d : %f!=%f\n",i,etalon[i],vector[i]);
            return;

        }
    }
    printf("Vectors are equal!\n");
}

int main(int argc, char** argv)
{
    uint32_t m=100000;
    uint32_t n=100000;

    if (argc>2)
    {
        m=std::atoi(argv[1]);
        n=std::atoi(argv[2]);
    }
    auto A_h=generate_random_sorted_array(m);
    auto B_h=generate_random_sorted_array(n,0.5);//add 0.5 to distinguish values from a and b
    // std::swap(A_h,B_h);

    printf("A (%d elements)\n",m);
    if (m<1000)
    {
        print_vector(A_h);
        printf("\n");
    }

    printf("B (%d elements)\n",n);
    if (n<1000)
    {
        print_vector(B_h);
        printf("\n");
    }

    thrust::device_vector<float> A_d=A_h;
    thrust::device_vector<float> B_d=B_h;
    thrust::device_vector<float> C_d(m+n);
    thrust::device_vector<float> etalon_d(m+n);

    thrust::merge(A_d.begin(), A_d.end(), B_d.begin(), B_d.end(),etalon_d.begin());


    merge(C_d.data().get(),A_d.data().get(), A_d.size(), B_d.data().get(), B_d.size());
    merge_coalesced(C_d.data().get(),A_d.data().get(), A_d.size(), B_d.data().get(), B_d.size());

    if (auto err=cudaDeviceSynchronize();err!=cudaSuccess)
    {
        printf("Error: %s\n",cudaGetErrorName(err));
        printf("Error message: %s\n",cudaGetErrorString(err));
        exit(err);
    }

    compare_vectors(etalon_d,C_d);


    printf("C (%d elements)\n",m+n);
    if (m+n<2000)
    {
        print_vector(C_d);
        printf("\n");
    }


    // for (int k=1;k<=20;++k)
    // {
    //     auto i=corank(A_h.data(),A_h.size(),B_h.data(),B_h.size(),k);
    //     auto val=i-corank(A_h.data(),A_h.size(),B_h.data(),B_h.size(),k-1)>0?A_h[i-1]:B_h[k-i-1];
    //     printf("k=%d, i=%d, j=%d last value=%f\n",k,i,k-i,val);
    // }

    return 0;
}




