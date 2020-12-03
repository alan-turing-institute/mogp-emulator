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

int main(void)
{
    test_device_vector_copy();

    return 0;

}
