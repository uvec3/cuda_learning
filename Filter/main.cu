#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "filter_device.cuh"

thrust::host_vector<float> generate_vector(uint32_t size, uint32_t positive)
{
    thrust::host_vector<float> result(size);
    for (int i=0;i<size;++i)
    {
        if (rand()/(float)RAND_MAX*size<positive)
        {
            result[i]=i;
        }
        else
        {
            result[i]=-i;
        }
    }


    return result;
}


int main()
{
    uint32_t n=100;

    auto vec=generate_vector(n,n/2);

    printf("Number of elements: %d\n", n);
    if (n<1000)
    {
        for (int i=0;i<n;++i)
        {
            printf("%d)\t\t %f\n",i, vec[i]);
        }
    }


    thrust::device_vector<float> vec_d=vec;
    thrust::device_vector<float> vec_filtered_d(n);


    thrust::copy_if(vec.begin(), vec.end(),vec_filtered_d.begin(),[]__device__(float v){return v>=0;});



    printf("Number of elements after filtering: %d\n", );
    if (n<1000)
    {
        for (int i=0;i<n;++i)
        {
            printf("%d)\t\t %f\n",i, vec[i]);
        }
    }

    return 0;
}
