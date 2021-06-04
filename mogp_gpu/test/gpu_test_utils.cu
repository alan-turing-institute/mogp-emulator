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

#include "../src/util.hpp"

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

typedef double REAL;

template <typename T>
T *dev_ptr(thrust::device_vector<T>& dv)
{
    return dv.data().get();
}


void test_add_diagonal()
{
    const size_t N=3;
    /// create 3x3 matrix and copy to device
    const double a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    thrust::device_vector<REAL> a_d(N*N);
    thrust::copy(a, a + N*N, a_d.begin());
    // add b to the diagonal
    const REAL b = 0.5;

    add_diagonal(N, b, dev_ptr(a_d));
    std::vector<double> expected = { 1.5, 2.0, 3.0, 4.0, 5.5, 6.0, 7.0, 8.0, 9.5};
    for (unsigned int i=0; i< a_d.size(); ++i) {
      std::cout<<"Element "<<i<<" is "<<a_d[i]<<std::endl;
      assert (a_d[i] == expected[i]);
    }
}



int main(void)
{
    test_add_diagonal();

    return 0;
}
