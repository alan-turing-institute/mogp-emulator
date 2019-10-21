#ifndef GP_GPU_HPP
#define GP_GPU_HPP

#include <iostream>

#include <algorithm>
#include <string>
#include <assert.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "cov_gpu.hpp"

template <typename T>
T *dev_ptr(thrust::device_vector<T>& dv)
{
    return dv.data().get();
}

typedef int obs_kind;
// typedef ivec vec_obs_kind;

// The GP class
// By convention, members ending "_d" are allocated on the device
class DenseGP_GPU {
    // batch size (for allocation)
    const size_t xnew_size = 10000;

    // Number of training points, dimension of input
    unsigned int N, Ninput;
    
    // handle for CUBLAS calls
    cublasHandle_t cublasHandle;

    // inverse covariance matrix on device
    thrust::device_vector<REAL> invC_d;

    // hyperparameters on the device
    thrust::device_vector<REAL> theta_d;

    // xs, on the device, row major order
    thrust::device_vector<REAL> xs_d;
    
    // precomputed product, used in the prediction
    thrust::device_vector<REAL> invCts_d;

    // preallocated work array (length N) on the CUDA device
    thrust::device_vector<REAL> work_d;

    // more work arrays
    thrust::device_vector<REAL> xnew_d, work_mat_d, result_d;

public:
    int data_length(void) const
    {
        return N;
    }

    // length of xnew assumed to be Ninput
    double predict(REAL *xnew)
    {
        thrust::device_vector<REAL> xnew_d(xnew, xnew + Ninput);
         cov_all_gpu(dev_ptr(work_d), N, Ninput, dev_ptr(xnew_d), dev_ptr(xs_d),
                     dev_ptr(theta_d));
        
        REAL result = -1.0;
        CUBLASDOT(cublasHandle, N, dev_ptr(work_d), 1, dev_ptr(invCts_d), 1,
                  &result);

        return double(result);
    }

    // void predict_batch(Col<REAL> &result, Mat<REAL> xnew) const
    // {
    //     REAL alpha = 1.0, beta = 0.0;

    //     cudaMemcpy(xnew_d, xnew.memptr(), Ninput*xnew.n_cols*sizeof(REAL),
    //                cudaMemcpyHostToDevice);

    //     cov_batch_gpu(work_mat_d, xnew.n_cols, N, Ninput, xnew_d, xs_d,
    //                   theta_d);

    //     cublasStatus_t status =
    //         CUBLASGEMV(cublasHandle, CUBLAS_OP_N, xnew.n_cols, N, &alpha,
    //                    work_mat_d, xnew.n_cols, invCts_d, 1, &beta, result_d,
    //                    1);

    //     cudaDeviceSynchronize();

    //     cudaError_t err =
    //         cudaMemcpy(result.memptr(), result_d, result.n_rows*sizeof(REAL),
    //                    cudaMemcpyDeviceToHost);

    //     if (err != cudaSuccess)
    //     {
    //         printf("predict_batch: A CUDA Error occured: %s\n",
    //                cudaGetErrorString(err));
    //     }
    // }

 
    // Update the hyperparameters and invQt which depends on them.
    void update_theta(const double *invQ, const double *theta,
                      const double *invQt)
    {
        thrust::copy(invQ, invQ + N * N, invC_d.begin());
        thrust::copy(theta, theta + N + 1, theta_d.begin());
        thrust::copy(invQt, invQt + N, invCts_d.begin());
    }

    
    DenseGP_GPU(unsigned int N_, unsigned int Ninput_, const double *theta_,
                const double *xs_, const double *ts_, const double *Q_,
                const double *invQ_, const double *invQt_)
        : N(N_)
        , Ninput(Ninput_)
        , invC_d(invQ_, invQ_ + N_ * N_)
        , theta_d(theta_, theta_ + Ninput_ + 1)
        , xs_d(xs_, xs_ + Ninput_ * N_)
        , invCts_d(invQt_, invQt_ + N_)
        , xnew_d(Ninput_ * xnew_size, 0.0)
        , work_d(N_, 0.0)
        , work_mat_d(N_ * xnew_size, 0.0)
        , result_d(xnew_size, 0.0)
    {
        std::cout << "N = " << N << std::endl;
        std::cout << "xnew_size = " << xnew_size << std::endl;
        
        cublasStatus_t status;
        status = cublasCreate(&cublasHandle);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error("CUBLAS initialization error\n");
        }
    }
};

#endif
