/// Example of using cuSOLVER Cholesky decomposition, based on StackOverflow
/// https://stackoverflow.com/questions/29196139/cholesky-decomposition-with-cuda

#include<iostream>
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

template <typename T>
T *dev_ptr(thrust::device_vector<T>& dv)
{
    return dv.data().get();
}


/************************************/
/*
   Create test matrix (from Wikipedia)
      4  12 -16       2 0 0     2 6 -8
    ( 12 37 -43 ) = ( 6 1 0 ) ( 0 1  5 )
     -16 -43 98      -8 5 3     0 0 3
 */
/************************************/

void setTest3x3Matrix(double * __restrict h_A) {
  h_A[0] = 4.0; h_A[1] = 12.0; h_A[2] = -16.0;
  h_A[3] = 12.0; h_A[4] = 37.; h_A[5] = -43.0;
  h_A[6] = 16.0; h_A[7] = -43.0; h_A[8] = 98.0;
}


void testCholesky()
{

    // --- CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    // --- CUBLAS initialization
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    /**********************
    Use 3x3 example from wikipedia
    ************************/

    const int N = 3;
    double *A_h = (double *)malloc(N * N * sizeof(double));
    setTest3x3Matrix(A_h);

    // copy to device
    thrust::device_vector<REAL> A_d(N * N);

    thrust::copy(A_h, A_h + N*N, A_d.begin());

    /****************************************/
    /* COMPUTING THE CHOLESKY DECOMPOSITION */
    /****************************************/
    // --- cuSOLVE input/output parameters/arrays
    int work_size = 0;
    int info_h;
    thrust::device_vector<int> info_d(1);
    //   int *devInfo;           cudaMalloc(&devInfo, sizeof(int));

    // --- CUDA CHOLESKY initialization
    cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_LOWER, N, dev_ptr(A_d), N, &work_size);

    std::cout<<"Buffer size is "<<work_size<<std::endl;
    // make the buffer and resize it to the right size
    thrust::device_vector<REAL> potrf_buffer_d;
    potrf_buffer_d.resize(work_size);

    // --- CUDA POTRF execution
    cusolverStatus_t status = cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, N, dev_ptr(A_d), N, dev_ptr(potrf_buffer_d), work_size, dev_ptr(info_d));

    // check the status
    thrust::copy(info_d.begin(), info_d.end(), &info_h);

    if (status != CUSOLVER_STATUS_SUCCESS || info_h < 0) {
      std::string msg;
      std::stringstream smsg(msg);
      smsg << "Error in potrf: return code " << status << ", info " << info_h;
      throw std::runtime_error(smsg.str());

    }

    std::vector<double> expected = { 2.0, 6.0, -8.0, 12.0, 1.0, 5.0, 16.0, -43.0, 3.0};
    for (unsigned int i=0; i< A_d.size(); ++i) {
      std::cout<<"element "<<i<<" is "<<A_d[i]<<std::endl;
      assert (A_d[i] == expected[i]);
    }

}

int main() {
  testCholesky();
  return 0;
}