#include <iostream>

#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <assert.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/copy.h>

#include "../src/meanfunc.hpp"

typedef double REAL;

void test_const_meanfunc()
{
    const size_t N=5;
    vec x(5);
    x << 1.0, 2.0, 3.0, 4.0, 5.0;

    vec params(1);
    params << 4.4;


    ConstMeanFunc* mf = new ConstMeanFunc();
    vec mean = mf->mean_f(x,params);
    vec deriv = mf->mean_deriv(x,params);
    vec inputderiv = mf->mean_inputderiv(x,params);

    std::cout<<" mean: ";
    for (int i=0; i<N; i++)
        std::cout << mean [i] << " ";
    std::cout << "\n deriv: ";
    for (int i=0; i<N; i++)
        std::cout << deriv [i] << " ";
    std::cout << "\n inputderiv: ";
    for (int i=0; i<N; i++)
        std::cout << inputderiv [i] << " ";
    std::cout << "\n";

    delete mf;
}



int main(void)
{
    test_const_meanfunc();
    return 0;
}
