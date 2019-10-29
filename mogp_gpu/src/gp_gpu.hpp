#ifndef GP_GPU_HPP
#define GP_GPU_HPP

#include <iostream>

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
#include <thrust/copy.h>

#include "cov_gpu.hpp"
#include "identity.hpp"

template <typename T>
T *dev_ptr(thrust::device_vector<T>& dv)
{
    return dv.data().get();
}

inline void check_cusolver_status(cusolverStatus_t status, int info_h) {
    if (status || info_h) {
	std::string msg;
	std::stringstream smsg(msg);
	smsg << "Error in potrf: return code " << status << ", info " << info_h;
	throw std::runtime_error(smsg.str());
    }
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

    // handle for cuSOLVER calls
    cusolverDnHandle_t cusolverHandle;
	
    // inverse covariance matrix on device
    thrust::device_vector<REAL> invC_d;

    // hyperparameters on the device
    thrust::device_vector<REAL> theta_d;

    // xs, on the device, row major order
    thrust::device_vector<REAL> xs_d;

    // targets, on the device
    thrust::device_vector<REAL> ts_d;

    // precomputed product, used in the prediction
    thrust::device_vector<REAL> invCts_d;
    
    // preallocated work array (length N) on the CUDA device
    thrust::device_vector<REAL> work_d;

    // buffer for Cholesky factorization
    thrust::device_vector<REAL> potrf_buffer_d;

    // more work arrays
    thrust::device_vector<REAL> xnew_d, work_mat_d, result_d, kappa_d, invCk_d;

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

        REAL result = std::numeric_limits<double>::quiet_NaN();
        CUBLASDOT(cublasHandle, N, dev_ptr(work_d), 1, dev_ptr(invCts_d), 1,
                  &result);

        return double(result);
    }

    // xnew (in): input point, to predict
    // var (output): variance
    // returns: prediction of value
    double predict_variance(REAL *xnew, REAL *var)
    {
        // value prediction
        thrust::device_vector<REAL> xnew_d(xnew, xnew + Ninput);
        cov_all_gpu(dev_ptr(work_d), N, Ninput, dev_ptr(xnew_d), dev_ptr(xs_d),
                    dev_ptr(theta_d));

        REAL result = std::numeric_limits<double>::quiet_NaN();
        CUBLASDOT(cublasHandle, N, dev_ptr(work_d), 1, dev_ptr(invCts_d), 1,
                  &result);

        // variance prediction
        thrust::device_vector<REAL> kappa_d(1);
        cov_val_gpu(dev_ptr(kappa_d), Ninput, dev_ptr(xnew_d), dev_ptr(xnew_d),
                    dev_ptr(theta_d));

        double zero(0.0);
        double one(1.0);
        thrust::device_vector<REAL> invCk_d(N);

        cublasDgemv(cublasHandle, CUBLAS_OP_N, N, N, &one, dev_ptr(invC_d), N,
                    dev_ptr(work_d), 1, &zero, dev_ptr(invCk_d), 1);

        CUBLASDOT(cublasHandle, N, dev_ptr(work_d), 1, dev_ptr(invCts_d),
                  1, &result);
        CUBLASDOT(cublasHandle, N, dev_ptr(work_d), 1, dev_ptr(invCk_d),
                  1, var);

        double kappa;
        thrust::copy(kappa_d.begin(), kappa_d.end(), &kappa);

        cudaDeviceSynchronize();

        *var = kappa - *var;

        return REAL(result);
    }

    // assumes on input that xnew is Nbatch * Ninput, and that result
    // contains space for Nbatch result values
    void predict_batch(int Nbatch, REAL *xnew, REAL *result)
    {	    
        REAL zero(0.0);
        REAL one(1.0);
        thrust::device_vector<REAL> xnew_d(xnew, xnew + Nbatch * Ninput);
        thrust::device_vector<REAL> result_d(Nbatch);

        cov_batch_gpu(dev_ptr(work_mat_d), Nbatch, N, Ninput,
                      dev_ptr(xnew_d), dev_ptr(xs_d), dev_ptr(theta_d));

        cublasStatus_t status =
            cublasDgemv(cublasHandle, CUBLAS_OP_N, Nbatch, N, &one,
                        dev_ptr(work_mat_d), Nbatch, dev_ptr(invCts_d), 1, &zero,
                        dev_ptr(result_d), 1);

        cudaDeviceSynchronize();

        thrust::copy(result_d.begin(), result_d.end(), result);
    }

    void predict_variance_batch(int Nbatch, REAL *xnew, REAL *mean, REAL *var)
    {
        REAL zero(0.0);
        REAL one(1.0);
        REAL minus_one(-1.0);

        thrust::device_vector<REAL> xnew_d(xnew, xnew + Nbatch * Ninput);
        thrust::device_vector<REAL> mean_d(Nbatch);

        // compute predictive means for the batch
        cov_batch_gpu(dev_ptr(work_mat_d), Nbatch, N, Ninput, dev_ptr(xnew_d),
                      dev_ptr(xs_d), dev_ptr(theta_d));

        cublasStatus_t status =
            cublasDgemv(cublasHandle, CUBLAS_OP_N, Nbatch, N, &one,
                        dev_ptr(work_mat_d), Nbatch, dev_ptr(invCts_d), 1, &zero,
                        dev_ptr(mean_d), 1);

        // compute predictive variances for the batch
        cov_diag_gpu(dev_ptr(kappa_d), Nbatch, Ninput, dev_ptr(xnew_d),
                     dev_ptr(xnew_d), dev_ptr(theta_d));

        cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                    N,
                    Nbatch,
                    N,
                    &one,
                    dev_ptr(invC_d), N,
                    dev_ptr(work_mat_d), Nbatch,
                    &zero,
                    dev_ptr(invCk_d), N);
	
        // result accumulated into 'kappa'
        status = cublasDgemmStridedBatched(
            cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            1, // m
            1, // n
            N, // k
            // A (m x k), B (k x n), C (m x n)
            &minus_one, // alpha
            dev_ptr(work_mat_d), Nbatch, 1, // A, lda, strideA
            dev_ptr(invCk_d), N, N, // B, ldb, strideB (= covariances "k")
            &one,
            dev_ptr(kappa_d), 1, 1, // C, ldc, strideC
            Nbatch);

        cudaDeviceSynchronize();

        // copy back means
        thrust::copy(mean_d.begin(), mean_d.end(), mean);

        // copy back variances
        thrust::copy(kappa_d.begin(), kappa_d.begin() + Nbatch, var);
    }

    // Update the hyperparameters, and invQ and invQt which depend on them
    void update_theta(const double *theta)
    {
        thrust::device_vector<int> info_d(1);
        int info_h;
        cusolverStatus_t status;
        
        thrust::copy(theta, theta + N + 1, theta_d.begin());
        
        cov_batch_gpu(dev_ptr(invC_d), N, N, Ninput, dev_ptr(xs_d),
                      dev_ptr(xs_d), dev_ptr(theta_d));

        double mean_diag, jitter;
        const int max_tries = 5;
        int itry = 0;
        while (itry < max_tries) {
            // invC_d holds the covariance matrix - work with a copy
            // in work_mat_d, in case the factorization fails
            thrust::copy(invC_d.begin(), invC_d.end(), work_mat_d.begin());

            // if the first attempt at factorization failed, add a
            // small term to the diagonal, increasing each iteration
            // until the factorization succeeds
            if (itry >= 1) {
                if (itry == 1) {
                    // find mean of (absolute) diagonal elements (diagonal
                    // elements should all be positive)
                    cublasDasum(cublasHandle, N, dev_ptr(invC_d), N+1, &mean_diag);
                    mean_diag /= N;
                    jitter = 1e-6 * mean_diag;
                }
                add_diagonal(N, jitter, dev_ptr(work_mat_d));
                jitter *= 10;
            }

            // compute Cholesky factors
            status = cusolverDnDpotrf(cusolverHandle, CUBLAS_FILL_MODE_LOWER, N,
                                      dev_ptr(work_mat_d), N, dev_ptr(potrf_buffer_d),
                                      potrf_buffer_d.size(), dev_ptr(info_d));

            thrust::copy(info_d.begin(), info_d.end(), &info_h);

            if (status != CUSOLVER_STATUS_SUCCESS || info_h < 0) {
                std::string msg;
                std::stringstream smsg(msg);
                smsg << "Error in potrf: return code " << status << ", info " << info_h;
                throw std::runtime_error(smsg.str());

            } else if (info_h == 0) {    
                break;
            }

            itry++;
        }

        // if none of the factorization attempts succeeded:
        if (itry == max_tries) {
            std::string msg;
            std::stringstream smsg(msg);
            smsg << "All attempts at factorization failed. Last return code " << status << ", info " << info_h;
            throw std::runtime_error(smsg.str());
        }

        identity_device(N, dev_ptr(invC_d));

        // invQ
        status = cusolverDnDpotrs(cusolverHandle, CUBLAS_FILL_MODE_LOWER, N, N,
                                  dev_ptr(work_mat_d), N, dev_ptr(invC_d), N,
                                  dev_ptr(info_d));

        thrust::copy(info_d.begin(), info_d.end(), &info_h);
        check_cusolver_status(status, info_h);
        

        // invQt
        thrust::copy(ts_d.begin(), ts_d.end(), invCts_d.begin());
        status = cusolverDnDpotrs(cusolverHandle, CUBLAS_FILL_MODE_LOWER, N, 1,
                                  dev_ptr(work_mat_d), N, dev_ptr(invCts_d), N,
                                  dev_ptr(info_d));

        thrust::copy(info_d.begin(), info_d.end(), &info_h);
        check_cusolver_status(status, info_h);
    }

    DenseGP_GPU(unsigned int N_, unsigned int Ninput_, const double *theta_,
                const double *xs_, const double *ts_)
        : N(N_)
        , Ninput(Ninput_)
        , invC_d(N_ * N_, 0.0)
        , theta_d(theta_, theta_ + Ninput_ + 1)
        , xs_d(xs_, xs_ + Ninput_ * N_)
        , ts_d(ts_, ts_ + N_)
        , invCts_d(N_, 0.0)
        , xnew_d(Ninput_ * xnew_size, 0.0)
        , work_d(N_, 0.0)
        , work_mat_d(N_ * xnew_size, 0.0)
        , result_d(xnew_size, 0.0)
        , kappa_d(xnew_size, 0.0)
        , invCk_d(xnew_size * N_, 0.0)
    {
        cublasStatus_t cublas_status = cublasCreate(&cublasHandle);
        if (cublas_status != CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error("CUBLAS initialization error\n");
        }	
        cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolverHandle);
        if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
        {
            throw std::runtime_error("cuSolver initialization error\n");
        }

        int potrfBufferSize;
        cusolverDnDpotrf_bufferSize(cusolverHandle, CUBLAS_FILL_MODE_LOWER,
                                    N, dev_ptr(potrf_buffer_d), N,
                                    &potrfBufferSize);

        potrf_buffer_d.resize(potrfBufferSize);
    }
};

#endif
