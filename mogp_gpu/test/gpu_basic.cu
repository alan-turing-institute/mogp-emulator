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



void test_sum_log_diag()
{

  size_t n=3;
  std::vector<REAL> x{1.0, 2.0, 3.0, 4., 5., 6., 7., 8., 9.};
  thrust::device_vector<REAL> x_d(x);
  // determine the size of the buffer for cub::device reduce


  // buffer for cub::DeviceReduce::Sum
  thrust::device_vector<REAL> sum_buffer_d;

    // size of sum_buffer_d
  size_t sum_buffer_size_bytes;
  // The following call determines the size of the cub::DeviceReduce::Sum workspace:
  // sum_buffer_size_bytes is 0 before this call, and the size of sum_buffer_d afterwards.
  // The end iterators are supplied but are not used.
  cub::DeviceReduce::Sum(dev_ptr(sum_buffer_d), sum_buffer_size_bytes,
			 sum_buffer_d.end(), sum_buffer_d.end(), n);
  sum_buffer_d.resize(sum_buffer_size_bytes);

  double result;
  thrust::device_vector<double> result_d(1);
  // call sum_log_diag
  sum_log_diag(n, dev_ptr(x_d), dev_ptr(result_d), dev_ptr(sum_buffer_d), sum_buffer_size_bytes);
  thrust::copy(result_d.begin(), result_d.end(), &result);

  std::cout<<"Result of sum_log_diag is "<<result<<std::endl;
}

void test_trace()
{

  size_t n=3;
  std::vector<REAL> x{1.0, 2.0, 3.0, 4., 5., 6., 7., 8., 9.};
  thrust::device_vector<REAL> x_d(x);
  // determine the size of the buffer for cub::device reduce

  // buffer for cub::DeviceReduce::Sum
  thrust::device_vector<REAL> sum_buffer_d;

    // size of sum_buffer_d
  size_t sum_buffer_size_bytes;
  // The following call determines the size of the cub::DeviceReduce::Sum workspace:
  // sum_buffer_size_bytes is 0 before this call, and the size of sum_buffer_d afterwards.
  // The end iterators are supplied but are not used.
  cub::DeviceReduce::Sum(dev_ptr(sum_buffer_d), sum_buffer_size_bytes,
			 sum_buffer_d.end(), sum_buffer_d.end(), n);
  sum_buffer_d.resize(sum_buffer_size_bytes);

  double result;
  thrust::device_vector<double> result_d(1);
  // call trace
  trace(n, dev_ptr(x_d), dev_ptr(result_d), dev_ptr(sum_buffer_d), sum_buffer_size_bytes);
  thrust::copy(result_d.begin(), result_d.end(), &result);

  std::cout<<"Result of trace is "<<result<<std::endl;
}


int main(void)
{
    test_device_vector_copy();
    test_sum_log_diag();
    test_trace();
    return 0;
}
