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
    for (unsigned int i=0; i<N; i++)
        std::cout << mean [i] << " ";
    std::cout << "\n deriv: ";
    for (unsigned int i=0; i<N; i++)
        std::cout << deriv [i] << " ";
    std::cout << "\n inputderiv: ";
    for (unsigned int i=0; i<N; i++)
        std::cout << inputderiv [i] << " ";
    std::cout << "\n";

    delete mf;
}

void test_poly_meanfunc()
{
  // 1D case - 2nd order polynomial
  // mean f should be 2.2 + 4.4*x + 3.3*x^2
    const size_t N=5;
    vec x(5);
    x << 1.0, 2.0, 3.0, 4.0, 5.0;

    vec params(3);
    params << 4.4, 3,3, 2.2;

    std::vector< std::pair<int, int> > dims_powers;
    dims_powers.push_back(std::make_pair<int, int>(0,1));
    dims_powers.push_back(std::make_pair<int, int>(0,2));
    PolyMeanFunc* mf = new PolyMeanFunc(dims_powers);
    vec mean = mf->mean_f(x,params);
    vec deriv = mf->mean_deriv(x,params);
    vec inputderiv = mf->mean_inputderiv(x,params);

    std::cout<<" mean: ";
    for (unsigned int i=0; i<N; i++)
        std::cout << mean [i] << " ";
    std::cout << "\n deriv: ";
    for (unsigned int i=0; i<N; i++)
        std::cout << deriv [i] << " ";
    std::cout << "\n inputderiv: ";
    for (unsigned int i=0; i<N; i++)
        std::cout << inputderiv [i] << " ";
    std::cout << "\n";

    delete mf;
}

void test_clone_meanfunc() {
    std::vector< std::pair<int, int> > dims_powers;
    dims_powers.push_back(std::make_pair<int, int>(0,1));
    dims_powers.push_back(std::make_pair<int, int>(0,2));
//    PolyMeanFunc* mf = new PolyMeanFunc(dims_powers);
    PolyMeanFunc mf(dims_powers);

    PolyMeanFunc new_mf = mf.clone2();
    const size_t N=5;
    vec x(5);
    x << 1.0, 2.0, 3.0, 4.0, 5.0;

    vec params(3);
    params << 4.4, 3,3, 2.2;

    vec mean = new_mf.mean_f(x,params);
    std::cout<<" mean: ";
    for (unsigned int i=.0; i<N; i++)
        std::cout << mean [i] << " ";
    std::cout<<std::endl;
}



int main(void)
{
    test_const_meanfunc();
    test_poly_meanfunc();
  //  test_clone_meanfunc();
    return 0;
}
