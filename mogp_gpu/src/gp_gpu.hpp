#ifndef GP_GPU_HPP
#define GP_GPU_HPP

#include <iostream>

#include <armadillo>
#include <algorithm>
#include <string>
#include <assert.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cov_gpu.hpp"

using arma::ivec;
using arma::vec;
using arma::Col;
using arma::rowvec;
using arma::mat;
using arma::Mat;
using arma::arma_ascii;
using arma::arma_binary;

typedef int obs_kind;
typedef ivec vec_obs_kind;

// The GP class
// By convention, members ending "_d" are allocated on the device
class DenseGP_GPU {
    // hyperparameters on the host
    vec theta;

    // hyperparameters on the device
    REAL *theta_d;

    unsigned int N, Ninput;

    // training inputs (rows)
    mat xs;
    vec ts;
    // vec_obs_kind Ts;

    // xs, on the device, row major order
    REAL *xs_d;

    // covariance matrix and its inverse
    mat C, invC;

    // precomputed product, used in the prediction
    vec invCts;
    // same, but on the CUDA device
    REAL *invCts_d;

    // preallocated work array (length N) on the CUDA device
    REAL *work_d;

    // work arrays
    REAL *xnew_d, *work_mat_d, *alpha_d, *result_d;

    // batch size (for allocation)
    const int xnew_size = 2000;

    // handle for CUBLAS calls
    cublasHandle_t cublasHandle;

public:
    int data_length(void) const
    {
        return N;
    }

    vec get_hypers(void) const
    {
        return theta;
    }
    
    // length of xnew assumed to be Ninput
    double predict(REAL *xnew) const
    {
        std::cout << "Prediction on GPU" << std::endl;
        std::cout << "N = " << N << std::endl;
        std::cout << "Ninput = " << Ninput << std::endl;
        std::cout << "xs = " << xs << std::endl;
        std::cout << "xnew = ";
        for (int i = 0; i < Ninput; i++) {
            std::cout << xnew[i] << " ";
        }
        std::cout << std::endl;
        REAL *xnew_d;
        // Col<REAL> xnewf = arma::conv_to<Col<REAL> >::from(xnew);
        if (cudaMalloc((void**)(&xnew_d), Ninput*sizeof(REAL))
            != cudaSuccess)
        {
            throw std::runtime_error("Device allocation failure (xnew_d)");
        }

        cudaMemcpy(xnew_d, xnew, Ninput*sizeof(REAL), cudaMemcpyHostToDevice);

        cov_all_gpu(work_d, N, xs.n_cols, xnew_d, xs_d, theta_d);

        REAL result;
        CUBLASDOT(cublasHandle, N, work_d, 1, invCts_d, 1, &result);

        std::cout << "C = " << C << std::endl;
        std::cout << "invC = " << invC << std::endl;
        std::cout << "invCts = " << invCts << std::endl;
        std::cout << "result = " << result << std::endl;
        
        return double(result);
    }

    void predict_batch(Col<REAL> &result, Mat<REAL> xnew) const
    {
        REAL alpha = 1.0, beta = 0.0;

        cudaMemcpy(xnew_d, xnew.memptr(), Ninput*xnew.n_cols*sizeof(REAL),
                   cudaMemcpyHostToDevice);

        cov_batch_gpu(work_mat_d, xnew.n_cols, N, Ninput, xnew_d, xs_d,
                      theta_d);

        cublasStatus_t status =
            CUBLASGEMV(cublasHandle, CUBLAS_OP_N, xnew.n_cols, N, &alpha,
                       work_mat_d, xnew.n_cols, invCts_d, 1, &beta, result_d,
                       1);

        cudaDeviceSynchronize();

        cudaError_t err =
            cudaMemcpy(result.memptr(), result_d, result.n_rows*sizeof(REAL),
                       cudaMemcpyDeviceToHost);

        if (err != cudaSuccess)
        {
            printf("predict_batch: A CUDA Error occured: %s\n",
                   cudaGetErrorString(err));
        }
    }

    DenseGP_GPU(unsigned int N_, unsigned int Ninput_, const double *theta_,
                const double *xs_, const double *ts_, const double *Q_,
                const double *invQ_, const double *invQt_)
        :
        N(N_),
        Ninput(Ninput_),
        theta(theta_, Ninput_ + 1),
        xs(xs_, Ninput_, N_),
        ts(ts_, N_),
        C(Q_, N_, N_),
        invC(invQ_, N_, N_),
        invCts(invQt_, N_)
    {
        Col<REAL> theta_tmp (arma::conv_to<Col<REAL> >::from(theta));

        cublasStatus_t status;
        status = cublasCreate(&cublasHandle);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error("CUBLAS initialization error\n");
        }

        if (cudaMalloc((void**)(&invCts_d), N*sizeof(REAL)) != cudaSuccess)
        {
            throw std::runtime_error("Device allocation failure (invCts_d)");
        }

        if (cudaMalloc((void**)(&work_d), N*sizeof(REAL)) != cudaSuccess)
        {
            throw std::runtime_error("Device allocation failure (work_d)");
        }

        if (cudaMalloc((void**)(&theta_d), theta_tmp.n_rows*sizeof(REAL))
            != cudaSuccess)
        {
            throw std::runtime_error("Device allocation failure (theta_d)");
        }

        if (cudaMalloc((void**)(&xs_d), N*Ninput*sizeof(REAL))
            != cudaSuccess)
        {
            throw std::runtime_error("Device allocation failure (xs_d)");
        }

        if (cudaMalloc((void**)(&xnew_d), Ninput*xnew_size*sizeof(REAL))
            != cudaSuccess)
        {
            throw std::runtime_error("Device allocation failure (xnew_d)");
        }

        if (cudaMalloc((void**)(&work_mat_d), N*xnew_size*sizeof(REAL))
            != cudaSuccess)
        {
            throw std::runtime_error("Device allocation failure (work_mat_d)");
        }

        if (cudaMalloc((void**)(&result_d), xnew_size*sizeof(REAL))
            != cudaSuccess)
        {
            throw std::runtime_error("Device allocation failure (result_d)");
        }

        Col<REAL> RinvCts (arma::conv_to<Col<REAL> >::from(invCts));
        cudaMemcpy(invCts_d, RinvCts.memptr(), N*sizeof(REAL),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(theta_d, theta_tmp.memptr(), theta_tmp.n_rows*sizeof(REAL),
                   cudaMemcpyHostToDevice);

        // Mat<REAL> xs_transpose (arma::conv_to<Mat<REAL> >::from(xs.t()));
        Mat<REAL> xs_transpose (arma::conv_to<Mat<REAL> >::from(xs));
        cudaMemcpy(xs_d, xs_transpose.memptr(), N*Ninput*sizeof(REAL),
                   cudaMemcpyHostToDevice);
    }

    // write this: cuda free etc?
    // ~DenseGP_GPU()
    // {

    // }
};

#endif
