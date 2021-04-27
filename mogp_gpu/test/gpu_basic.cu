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

#include "../src/cov_gpu.hpp"
#include "../src/gp_gpu.hpp"

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

typedef double REAL;

void test_device_vector_copy()
{
    const size_t N=5;
    // std::vector<REAL> a{1.0, 2.0, 3.0, 4.0, 5.0};
    const double a[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    thrust::device_vector<REAL> a_d(N);

    std::vector<REAL> result(N);
    thrust::device_vector<REAL> result_d(N);

    thrust::copy(a, a + N, a_d.begin());

    thrust::fill(result_d.begin(), result_d.end(), 0.0);
    thrust::transform(a_d.begin(), a_d.end(), a_d.begin(), result_d.begin(), thrust::plus<float>());

    thrust::copy(result_d.begin(), result_d.end(), result.begin());

    for (int i=0; i<N; i++)
        std::cout << result[i] << " ";
    std::cout << "\n";
}

void test_cov_deriv()
{
    const size_t N=2;
    const size_t Ninput=3;
    const size_t Ntheta=Ninput+1;

    std::vector<REAL> x{1.0, 2.0, 3.0};
    std::vector<REAL> y{4.0, 5.0, 6.0};

    std::vector<REAL> result(Ntheta);

    thrust::device_vector<REAL> result_d(Ntheta, 0.0);
    thrust::device_vector<REAL> x_d(x);
    thrust::device_vector<REAL> y_d(y);
    thrust::device_vector<REAL> theta_d(Ntheta, -1.0);

    // x, y
    cov_deriv_theta_batch_gpu(dev_ptr(result_d), Ninput, 1, 1, dev_ptr(x_d), dev_ptr(y_d), dev_ptr(theta_d));
    thrust::copy(result_d.begin(), result_d.end(), result.begin());

    for (size_t i=0; i<Ntheta; i++)
        std::cout << result[i] << " ";
    std::cout << "\n";

    // x, x
    cov_deriv_theta_batch_gpu(dev_ptr(result_d), Ninput, 1, 1, dev_ptr(x_d), dev_ptr(x_d), dev_ptr(theta_d));
    thrust::copy(result_d.begin(), result_d.end(), result.begin());

    for (size_t i=0; i<Ntheta; i++)
        std::cout << result[i] << " ";
    std::cout << "\n";
}

void test_deriv_x() {
  int Ninput = 1;
  int Nx = 1;
  int Ny = 1;
  thrust::device_vector<REAL> result_d(Ninput, 0.);
  thrust::device_vector<REAL> x_d(Nx, 2.);
  thrust::device_vector<REAL> y_d(Ny, 3.);
  std::vector<REAL> theta{-1.0, -1.0};
  thrust::device_vector<REAL> theta_d(theta);


    cov_deriv_x_batch_gpu(
			  dev_ptr(result_d), Ninput, Nx, Ny, dev_ptr(x_d), dev_ptr(y_d), dev_ptr(theta_d)
			  );
    std::cout<<"result of test_deriv_x is "<<result_d[0]<<std::endl;
}


int main(void)
{
    test_device_vector_copy();
    test_cov_deriv();
    test_deriv_x();

    return 0;
}
