#include <iostream>
#include "soa_vector.hpp"

struct NestedStruct
{
    int* a;
    float* b;
};

struct RayView
{
    std::vector<int*> array;
    std::pair<float*,double*> pair;
    double* dd;
    int* id;
    float* x;
    float* y;
    int i;
    float* z;
    NestedStruct nested_struct;
    int global;
};

int main()
{

    std::cout << "Hello, World!" << std::endl;
    std::vector<>

    soa_vector<RayView> rays(21);
    // rays.id[0]=0;

    return 0;
}
