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

#include "../src/kernel.hpp"
#include "../src/densegp_gpu.hpp"

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

typedef double REAL;

void test_cov(kernel_type kernel)
{
  BaseKernel* kern(0);
  if (kernel == SQUARED_EXPONENTIAL)
    kern = new SquaredExponentialKernel();
  else if (kernel == MATERN52)
    kern = new Matern52Kernel();
  const size_t N=3;
  const size_t Ninput=1;
  const size_t Ntheta=Ninput+1;
  std::vector<REAL> x{1.0, 2.0, 3.0};
  std::vector<REAL> result(N*N);

  thrust::device_vector<REAL> result_d(N*N, 0.0);
  thrust::device_vector<REAL> x_d(x);
  thrust::device_vector<REAL> theta_d(Ntheta, -1.0);
  /// calculate the covariance matrix between the inputs x
  kern->cov_batch_gpu(dev_ptr(result_d), N, N, Ninput, dev_ptr(x_d),
                dev_ptr(x_d), dev_ptr(theta_d));

  thrust::copy(result_d.begin(), result_d.end(), result.begin());
  std::cout<<"Covariance matrix: "<<std::endl;
  for (size_t i=0; i<result.size(); i++)
    std::cout << result[i] << " ";
  std::cout << "\n";
  delete kern;
}

void test_cov_deriv(kernel_type kernel)
{
  BaseKernel* kern(0);
    if (kernel == SQUARED_EXPONENTIAL)
      kern = new SquaredExponentialKernel();
    else if (kernel == MATERN52)
      kern = new Matern52Kernel();
    const size_t Ninput=3;
    const size_t Ntheta=Ninput+1;
    /// Test data - single points, 3 dimensions.
    std::vector<REAL> x{1.0, 2.0, 3.0};
    std::vector<REAL> y{4.0, 5.0, 6.0};

    std::vector<REAL> result(Ntheta);

    thrust::device_vector<REAL> result_d(Ntheta, 0.0);
    thrust::device_vector<REAL> x_d(x);
    thrust::device_vector<REAL> y_d(y);
    thrust::device_vector<REAL> theta_d(Ntheta, -1.0);

    // x, y

    kern->cov_deriv_theta_batch_gpu(dev_ptr(result_d), Ninput, 1, 1, dev_ptr(x_d), dev_ptr(y_d), dev_ptr(theta_d));
    thrust::copy(result_d.begin(), result_d.end(), result.begin());

    for (size_t i=0; i<Ntheta; i++)
        std::cout << result[i] << " ";
    std::cout << "\n";

    // x, x
    kern->cov_deriv_theta_batch_gpu(dev_ptr(result_d), Ninput, 1, 1, dev_ptr(x_d), dev_ptr(x_d), dev_ptr(theta_d));
    thrust::copy(result_d.begin(), result_d.end(), result.begin());

    for (size_t i=0; i<Ntheta; i++)
        std::cout << result[i] << " ";
    std::cout << "\n";
    delete kern;
}

void test_deriv_x(kernel_type kernel) {
  BaseKernel* kern(0);
  if (kernel == SQUARED_EXPONENTIAL)
    kern = new SquaredExponentialKernel();
  else if (kernel == MATERN52)
    kern = new Matern52Kernel();
  int Ninput = 2;
  int N = 3;
  mat x(N, Ninput);
  // input x = [[1,2],[2,4],[3,6]]
  for (int i=0; i<N; i++) {
    int j = 2*(i+1);
    x(i,0) = float(i+1);
    x(i,1) = float(j);
  }
  std::cout<<" x: "<<x<<std::endl;
  std::vector<REAL> result(N*N*Ninput);

  thrust::device_vector<REAL> result_d(N*N*Ninput, 0.);
  thrust::device_vector<REAL> x_d(x.data(), x.data() + Ninput * N);

  std::vector<REAL> theta{-1.0, -1.0, -1.0};
  thrust::device_vector<REAL> theta_d(theta);

  kern->cov_deriv_x_batch_gpu(
    dev_ptr(result_d), Ninput, N, N, dev_ptr(x_d), dev_ptr(x_d), dev_ptr(theta_d)
  );
  thrust::copy(result_d.begin(), result_d.end(), result.begin());
  std::cout<<"drdx: "<<std::endl;
  for (size_t i=0; i<result.size(); i++)
    std::cout << result[i] << " ";
  std::cout << "\n";
  delete kern;
}



int main(void)
{
  test_cov(MATERN52);
  test_cov_deriv(MATERN52);
  test_deriv_x(MATERN52);
  return 0;
}
