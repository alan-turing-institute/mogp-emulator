#ifndef UTIL_HPP
#define UTIL_HPP

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
#include <cub/cub.cuh>

#include <Eigen/Dense>

#include "types.hpp"
#include "strided_range.hpp"

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// ----------------------------------------------
// ----   Utility functions ---------------------


/// Fail if a recent cusolver call did not succeed
inline void check_cusolver_status(cusolverStatus_t status, int info_h)
{
    if (status || info_h) {
	std::string msg;
	std::stringstream smsg(msg);
	smsg << "Error in potrf: return code " << status << ", info " << info_h;
	throw std::runtime_error(smsg.str());
    }
}

/// Can a usable CUDA capable device be found?
inline bool have_compatible_device(void)
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    return (err == cudaSuccess && device_count > 0);

}
// ----------------------------------------------


void identity_device(int N, double *A);
void add_diagonal(int N, double b, double *A);
void sum_log_diag(int N, double *A, double *result, double *work, size_t work_size);
void trace(int N, double *A, double *result, double *work, size_t work_size);


#endif
