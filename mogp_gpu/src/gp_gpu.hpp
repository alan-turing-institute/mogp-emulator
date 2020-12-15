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
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/copy.h>

#include <Eigen/Dense>

#include "strided_range.hpp"
#include "cov_gpu.hpp"
#include "util.hpp"

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

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

// see https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
__inline__ __device__
double warpReduceSum(double val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

// Invoke as
// sumLogDiag<<<1, N>>>(N, A, result);
//
__global__
void sumLogDiag(int N, double *A, double *result)
{
    // assumes a single block
    int i = threadIdx.x;

    static __shared__ double work[WARP_SIZE];
    double log_diag = 0.0;

    work[i] = 0.0;
    if (i < N) {
        log_diag = log(A[i * (N+1)]);
    }

    int laneIdx = i % WARP_SIZE;
    int warpIdx = i / WARP_SIZE;

    log_diag = warpReduceSum(log_diag);

    if (laneIdx == 0) work[warpIdx] = log_diag;

    __syncthreads();

    log_diag = work[laneIdx];

    if (warpIdx == 0) log_diag = warpReduceSum(log_diag);
    if (i == 0) *result = 2.0 * log_diag;
}

typedef int obs_kind;
// typedef ivec vec_obs_kind;
typedef typename Eigen::Matrix<REAL, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat;
typedef typename Eigen::Ref<Eigen::Matrix<REAL, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > mat_ref;
typedef typename Eigen::Matrix<REAL, Eigen::Dynamic, 1> vec;
typedef typename Eigen::Ref<Eigen::Matrix<REAL, Eigen::Dynamic, 1> > vec_ref;

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

    // log determinant of covariance matrix (on host)
    double logdetC;

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
    double predict(mat_ref xnew)
    {
        thrust::device_vector<REAL> xnew_d(xnew.data(), xnew.data() + Ninput);
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
    double predict_variance(mat_ref xnew, vec_ref var)
    {
        // value prediction
        thrust::device_vector<REAL> xnew_d(xnew.data(), xnew.data() + Ninput);
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
                  1, var.data());

        double kappa;
        thrust::copy(kappa_d.begin(), kappa_d.end(), &kappa);

        cudaDeviceSynchronize();

        var = kappa - var.array();

        return REAL(result);
    }

    // assumes on input that xnew is Nbatch * Ninput, and that result
    // contains space for Nbatch result values
    void predict_batch(mat_ref xnew, vec_ref result)
    {
        int Nbatch = result.size();

        // TODO
        // assert result.size() == xnew.rows()

        REAL zero(0.0);
        REAL one(1.0);
        thrust::device_vector<REAL> xnew_d(xnew.data(), xnew.data() + Nbatch * Ninput);
        thrust::device_vector<REAL> result_d(Nbatch);

        cov_batch_gpu(dev_ptr(work_mat_d), Nbatch, N, Ninput,
                      dev_ptr(xnew_d), dev_ptr(xs_d), dev_ptr(theta_d));

        cublasStatus_t status =
            cublasDgemv(cublasHandle, CUBLAS_OP_N, Nbatch, N, &one,
                        dev_ptr(work_mat_d), Nbatch, dev_ptr(invCts_d), 1, &zero,
                        dev_ptr(result_d), 1);

        cudaDeviceSynchronize();

        thrust::copy(result_d.begin(), result_d.end(), result.data());
    }

    void predict_variance_batch(int Nbatch, mat_ref xnew, vec_ref mean, vec_ref var)
    {
        REAL zero(0.0);
        REAL one(1.0);
        REAL minus_one(-1.0);

        thrust::device_vector<REAL> xnew_d(xnew.data(), xnew.data() + Nbatch * Ninput);
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
        thrust::copy(mean_d.begin(), mean_d.end(), mean.data());

        // copy back variances
        thrust::copy(kappa_d.begin(), kappa_d.begin() + Nbatch, var.data());
    }


    // Update the hyperparameters, and invQ and invQt which depend on them
    void update_theta(vec_ref theta)
    {
        thrust::device_vector<int> info_d(1);
        int info_h;
        cusolverStatus_t status;

        thrust::copy(theta.data(), theta.data() + Ninput + 1, theta_d.begin());

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

        // logdetQ
        thrust::device_vector<double> logdetC_d(1);

        sumLogDiag<<<1, N>>>(N, dev_ptr(work_mat_d), dev_ptr(logdetC_d));
        thrust::copy(logdetC_d.begin(), logdetC_d.end(), &logdetC);
    }

    void get_invQ(mat_ref invQ_h)
    {
      thrust::copy(invC_d.begin(), invC_d.end(), invQ_h.data());
    }

    void get_invQt(mat_ref invQt_h)
    {
      thrust::copy(invCts_d.begin(), invCts_d.end(), invQt_h.data());
    }

    double get_logdetQ(void)
    {
        return logdetC;
    }

    void dloglik_dtheta(mat_ref result_h)
    {
            // TODO: calculation of dloglik_dtheta

            // // dK{jk}_dtheta{i}
            // cov_deriv_batch_gpu(/* result_d */,
            //                     Ninput, N, N,
            //                     xs_d, xs_d,
            //                     theta_d);

            // // // compute intermediate matrix "C" (can overlap)
            // // ...

            // // // gemm (treat the matrix indices as a single 1-d index)
            // cublasDgemv(cublasHandle, CUBLAS_OP_N, N * N, Ninput, &one,
            //             dev_ptr("C"), N * N, dev_ptr(work_d),
            //             Ninput, &zero, dev_ptr(invCk_d), 1);
    }

  DenseGP_GPU(mat_ref xs_, vec_ref ts_)
        : N(xs_.rows())
	, Ninput(xs_.cols())
        , invC_d(N * N, 0.0)
        , theta_d(Ninput + 1)
        , xs_d(xs_.data(), xs_.data() + Ninput * N)
        , ts_d(ts_.data(), ts_.data() + N)
        , invCts_d(N, 0.0)
        , xnew_d(Ninput * xnew_size, 0.0)
        , work_d(N, 0.0)
        , work_mat_d(N * xnew_size, 0.0)
        , result_d(xnew_size, 0.0)
        , kappa_d(xnew_size, 0.0)
        , invCk_d(xnew_size * N, 0.0)
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
