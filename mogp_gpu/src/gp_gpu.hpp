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
#include <cub/cub.cuh>

#include <Eigen/Dense>

#include "strided_range.hpp"
#include "cov_gpu.hpp"
#include "util.hpp"

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

#define USE_SHUFFLE_SUM_IMPL 0


/// Extract the raw pointer from the
template <typename T>
T *dev_ptr(thrust::device_vector<T>& dv)
{
    return dv.data().get();
}


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
bool have_compatible_device(void);


#if USE_SHUFFLE_SUM_IMPL
// ----------------------------------------

// Implementation of sum_log_diag using warp reduction,
// based on https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
//
// Will fail for N > 1024


__inline__ __device__
double warp_reduce_sum(double val)
{
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

__global__
void sum_log_diag_kernel(int N, double *A, double *result)
{
    int i = threadIdx.x;

    static __shared__ double work[WARP_SIZE];
    double log_diag = 0.0;

    if (i < WARP_SIZE) work[i] = 0.0;
    if (i < N) {
        log_diag = log(A[i * (N+1)]);
    }

    int laneIdx = i % WARP_SIZE;
    int warpIdx = i / WARP_SIZE;

    __syncthreads();
    log_diag = warp_reduce_sum(log_diag);

    if (laneIdx == 0) work[warpIdx] = log_diag;

    __syncthreads();

    log_diag = work[laneIdx];

    if (warpIdx == 0) log_diag = warp_reduce_sum(log_diag);

    if (i == 0) *result = 2.0 * log_diag;
}

//
//
// The unused arguments are needed for the CUB implementation
void sum_log_diag(int N, double *A, double *result, double *, size_t)
{
    // The number of thread blocks *must* be 1
    // N can be at most 1024 (WARP_SIZE**2)
    int thread_blocks = 1;
    int threads_per_block = N;
    sum_log_diag_kernel<<<thread_blocks, threads_per_block>>>(N, A, result);
}

#else
// ----------------------------------------

// Implementation of sum_log_diag using cub::DeviceReduce

struct LogSq : public thrust::unary_function<double, double>
{
    __host__ __device__ double operator()(double x) const { return 2.0 * log(x); }
};

void sum_log_diag(int N, double *A, double *result, double *work, size_t work_size)
{
    auto transform_it = thrust::make_transform_iterator(A, LogSq());
    auto sr = make_strided_range(transform_it, transform_it + N*N, N+1);

    cub::DeviceReduce::Sum(work, work_size, sr.begin(), result, N);
}

#endif // USE_SHUFFLE_SUM_IMPL

typedef int obs_kind;
// typedef ivec vec_obs_kind;
typedef typename Eigen::Matrix<REAL, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat;
typedef typename Eigen::Ref<Eigen::Matrix<REAL, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > mat_ref;
typedef typename Eigen::Matrix<REAL, Eigen::Dynamic, 1> vec;
typedef typename Eigen::Ref<Eigen::Matrix<REAL, Eigen::Dynamic, 1> > vec_ref;

enum nugget_type {NUG_ADAPTIVE, NUG_FIT, NUG_FIXED};

// The GP class
// By convention, members ending "_d" are allocated on the device
class DenseGP_GPU {
    // batch size (for allocation)
    unsigned int testing_size;

    // Number of training points, dimension of input
    unsigned int n, D;

    // inputs
    mat inputs;

    // targets
    vec targets;

    // flag for whether we have fit hyperparameters
    bool theta_fitted;

    // handle for CUBLAS calls
    cublasHandle_t cublasHandle;

    // handle for cuSOLVER calls
    cusolverDnHandle_t cusolverHandle;

    // inverse covariance matrix on device
    thrust::device_vector<REAL> invQ_d;

    // lower triangular Cholesky factor on device
    thrust::device_vector<REAL> chol_lower_d;

    // log determinant of covariance matrix (on host)
    double logdetC;

    // adaptive nugget jitter
    double jitter;

    // hyperparameters on the device
    thrust::device_vector<REAL> theta_d;

    // inputs, on the device, row major order
    thrust::device_vector<REAL> inputs_d;

    // targets, on the device
    thrust::device_vector<REAL> targets_d;

    // precomputed product, used in the prediction
    thrust::device_vector<REAL> invQt_d;

    // preallocated work array (length n) on the CUDA device
    thrust::device_vector<REAL> work_d;

    // buffer for Cholesky factorization
    thrust::device_vector<REAL> potrf_buffer_d;

    // buffer for cub::DeviceReduce::Sum
    thrust::device_vector<REAL> sum_buffer_d;

    // size of sum_buffer_d
    size_t sum_buffer_size_bytes;

    // more work arrays
    thrust::device_vector<REAL> testing_d, work_mat_d, result_d, kappa_d, invQk_d;

public:
    int get_n(void) const
    {
        return n;
    }

    int get_D(void) const
    {
        return D;
    }

    mat get_inputs(void) const
    {
        return inputs;
    }

    vec get_targets(void) const
    {
        return targets;
    }

    int get_n_params(void) const
    {
        return D + 2;
    }

    void get_theta(vec_ref theta)
    {
        thrust::copy(theta_d.begin(), theta_d.end(), theta.data());
    }

    double get_jitter(void) const
    {
        return jitter;
    }

    double predict(mat_ref testing)
    {
        // On entry: the number of points to predict (number of rows
        // of testing) is assumed to be D

      thrust::copy(testing.data(), testing.data() + D, testing_d.begin());
        cov_all_gpu(dev_ptr(work_d), n, D, dev_ptr(testing_d), dev_ptr(inputs_d),
                    dev_ptr(theta_d));

        REAL result = std::numeric_limits<double>::quiet_NaN();
        CUBLASDOT(cublasHandle, n, dev_ptr(work_d), 1, dev_ptr(invQt_d), 1,
                  &result);

        return double(result);
    }

    double predict_variance(mat_ref testing, vec_ref var)
    {
        if (var.size() < testing.rows()) {
            throw std::runtime_error("predict_variance: the result buffer passed was "
                                     "too small to hold the variance");
        }

        // value prediction
        thrust::copy(testing.data(), testing.data() + D, testing_d.begin());

        cov_all_gpu(dev_ptr(work_d), n, D, dev_ptr(testing_d), dev_ptr(inputs_d),
                    dev_ptr(theta_d));

        REAL result = std::numeric_limits<double>::quiet_NaN();
        CUBLASDOT(cublasHandle, n, dev_ptr(work_d), 1, dev_ptr(invQt_d), 1,
                  &result);

        // variance prediction
        thrust::device_vector<REAL> kappa_d(1);
        cov_val_gpu(dev_ptr(kappa_d), D, dev_ptr(testing_d), dev_ptr(testing_d),
                    dev_ptr(theta_d));

        double zero(0.0);
        double one(1.0);
        thrust::device_vector<REAL> invQk_d(n);

        cublasDgemv(cublasHandle, CUBLAS_OP_N, n, n, &one, dev_ptr(invQ_d), n,
                    dev_ptr(work_d), 1, &zero, dev_ptr(invQk_d), 1);

        CUBLASDOT(cublasHandle, n, dev_ptr(work_d), 1, dev_ptr(invQt_d),
                  1, &result);
        CUBLASDOT(cublasHandle, n, dev_ptr(work_d), 1, dev_ptr(invQk_d),
                  1, var.data());

        double kappa;
        thrust::copy(kappa_d.begin(), kappa_d.end(), &kappa);

        cudaDeviceSynchronize();

        var = kappa - var.array();

        return REAL(result);
    }

    void predict_batch(mat_ref testing, vec_ref result)
    {
        // Assumes on entry that testing has shape (Nbatch, D), and that result
        // contains space for Nbatch result values

        int Nbatch = testing.rows();

        if (result.size() < testing.rows()) {
            throw std::runtime_error("predict_batch: the result buffer passed was "
                                     "too small to hold the result");
        }

        if (Nbatch > testing_size) {
            throw std::runtime_error("predict_variance_batch: More test points were passed "
                                     "than the maximum batch size");
        }

        REAL zero(0.0);
        REAL one(1.0);
        thrust::copy( testing.data(), testing.data() + Nbatch * D, testing_d.begin());

        cov_batch_gpu(dev_ptr(work_mat_d), Nbatch, n, D,
                      dev_ptr(testing_d), dev_ptr(inputs_d), dev_ptr(theta_d));

        cublasStatus_t status =
            cublasDgemv(cublasHandle, CUBLAS_OP_N, Nbatch, n, &one,
                        dev_ptr(work_mat_d), Nbatch, dev_ptr(invQt_d), 1, &zero,
                        dev_ptr(result_d), 1);

        cudaDeviceSynchronize();

        thrust::copy(result_d.begin(), result_d.end(), result.data());
    }

    void predict_variance_batch(mat_ref testing, vec_ref var)
    {
        REAL zero(0.0);
        REAL one(1.0);
        REAL minus_one(-1.0);
        int Nbatch = testing.rows();

        if (var.size() < testing.rows()) {
            throw std::runtime_error("predict_variance_batch: The result buffer passed was "
                                     "too small to hold the variance");
        }

        if (Nbatch > testing_size) {
            throw std::runtime_error("predict_variance_batch: More test points were passed "
                                     "than the maximum batch size");
        }

	thrust::copy( testing.data(), testing.data() + Nbatch * D, testing_d.begin());

        // compute predictive variances for the batch
        cov_diag_gpu(dev_ptr(kappa_d), Nbatch, D, dev_ptr(testing_d),
                     dev_ptr(testing_d), dev_ptr(theta_d));

        cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                    n,
                    Nbatch,
                    n,
                    &one,
                    dev_ptr(invQ_d), n,
                    dev_ptr(work_mat_d), Nbatch,
                    &zero,
                    dev_ptr(invQk_d), n);

        // result accumulated into 'kappa'
        status = cublasDgemmStridedBatched(
            cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            1, // m
            1, // n
            n, // k
            // A (m x k), B (k x n), C (m x n)
            &minus_one, // alpha
            dev_ptr(work_mat_d), Nbatch, 1, // A, lda, strideA
            dev_ptr(invQk_d), n, n, // B, ldb, strideB (= covariances "k")
            &one,
            dev_ptr(kappa_d), 1, 1, // C, ldc, strideC
            Nbatch);

        cudaDeviceSynchronize();

        // copy back variances
        thrust::copy(kappa_d.begin(), kappa_d.begin() + Nbatch, var.data());
    }

    void predict_deriv_batch(mat_ref testing, vec_ref mean, mat_ref deriv)
    {
        int Nbatch = testing.rows();

        if (result.rows() < testing.rows() || result.cols() != D) {
            throw std::runtime_error("predict_deriv: the result buffer passed was "
                                     "the wrong shape to hold the result");
        }

        if (Nbatch > testing_size) {
            throw std::runtime_error("predict_variance_batch: More test points were passed "
                                     "than the maximum batch size");
        }

        REAL zero(0.0);
        REAL one(1.0);

	// now computes means (in first part of work_mat_d) and variances (in following part)
        cov_deriv_x_batch_gpu(dev_ptr(work_mat_d),dev_ptr(work_mat_d)+Nbatch, D, Nbatch, n,
			      dev_ptr(testing_d), dev_ptr(inputs_d), dev_ptr(theta_d));

	// calculate mean
        cublasStatus_t status =
            cublasDgemv(cublasHandle, CUBLAS_OP_N, Nbatch, n, &one,
                        dev_ptr(work_mat_d), Nbatch, dev_ptr(invQt_d), 1, &zero,
                        dev_ptr(result_d), 1);


	// calculate deriv
        cublasStatus_t status =
            cublasDgemv(cublasHandle, CUBLAS_OP_N,
                        D * Nbatch, n, // nrows, ncols
                        &one, dev_ptr(work_mat_d)+Nbatch, D * Nbatch, // alpha, A, lda
                        dev_ptr(invQt_d), 1, // x, incx
                        &zero, dev_ptr(result_d)+Nbatch, 1); // beta, y, incy

        cudaDeviceSynchronize();
	// copy first part of result_d to mean vector
	thrust::copy(result_d.begin(), result_d.begin()+Nbatch, mean.data());
	// copy next bit of result_d to deriv
        thrust::copy(result_d.begin()+Nbatch, result_d.begin()+Nbatch+Nbatch*D, deriv.data());
    }

    int calc_cholesky_factors(void)
    {
        // On entry: assumes that work_mat_d contains the inverse covariance matrix invQ_d

        thrust::device_vector<int> info_d(1);
        int info_h;
        cusolverStatus_t status;

	// compute Cholesky factors
	status = cusolverDnDpotrf(cusolverHandle, CUBLAS_FILL_MODE_LOWER, n,
				  dev_ptr(work_mat_d), n, dev_ptr(potrf_buffer_d),
				  potrf_buffer_d.size(), dev_ptr(info_d));

	thrust::copy(info_d.begin(), info_d.end(), &info_h);

	if (status != CUSOLVER_STATUS_SUCCESS || info_h < 0) {
            std::string msg;
            std::stringstream smsg(msg);
            smsg << "Error in potrf: return code " << status << ", info " << info_h;
            throw std::runtime_error(smsg.str());

	}
	return info_h;
    }

    // return the lower triangular matrix (actually it will be the whole matrix
    // but only the lower triangle will be correct!)
    void get_cholesky_lower(mat_ref result)
    {
        thrust::copy(chol_lower_d.begin(), chol_lower_d.end(), result.data());
    }

    // Update the hyperparameters, and invQ and invQt which depend on them
    void fit(vec_ref theta, nugget_type nugget, double nugget_size=0.0)
    {
        thrust::copy(theta.data(), theta.data() + D + 2, theta_d.begin());

        cov_batch_gpu(dev_ptr(invQ_d), n, n, D, dev_ptr(inputs_d),
                      dev_ptr(inputs_d), dev_ptr(theta_d));

	/// copy the covariance matrix invQ_d into work_mat_d
	thrust::copy(invQ_d.begin(), invQ_d.end(), work_mat_d.begin());

        thrust::device_vector<int> info_d(1);
        int info_h;
        cusolverStatus_t status;
	int factorisation_status;

	if (nugget == NUG_ADAPTIVE) {
	    double mean_diag;
	    const int max_tries = 5;
	    int itry = 0;
	    while (itry < max_tries) {
		// invQ_d holds the covariance matrix - work with a copy
		// in work_mat_d, in case the factorization fails
		thrust::copy(invQ_d.begin(), invQ_d.end(), work_mat_d.begin());

		// if the first attempt at factorization failed, add a
		// small term to the diagonal, increasing each iteration
		// until the factorization succeeds
		if (itry >= 1) {
		    if (itry == 1) {
			// find mean of (absolute) diagonal elements (diagonal
			// elements should all be positive)
			cublasDasum(cublasHandle, n, dev_ptr(invQ_d), n+1, &mean_diag);
			mean_diag /= n;
			jitter = 1e-6 * mean_diag;
		    }
		    add_diagonal(n, jitter, dev_ptr(work_mat_d));
		}

		factorisation_status = calc_cholesky_factors();
		if (factorisation_status == 0) {
                    break;
		}
		jitter *= 10;
		itry++;
	    }

	    // if none of the factorization attempts succeeded:
	    if (itry == max_tries) {
		std::string msg;
		std::stringstream smsg(msg);
		smsg << "All attempts at factorization failed. Last return code " << factorisation_status;
		throw std::runtime_error(smsg.str());
	    }

	} else if (nugget == NUG_FIXED) {
	    add_diagonal(n, nugget_size, dev_ptr(work_mat_d));

	    factorisation_status = calc_cholesky_factors();
	    if (factorisation_status != 0) {
                throw std::runtime_error("Unable to factorize matrix using fixed nugget");
	    }

	} else { //nugget == "fit"
	    factorisation_status = calc_cholesky_factors();
	    if (factorisation_status != 0) {
                throw std::runtime_error("Unable to factorize matrix using fitted nugget");
	    }
	}

        identity_device(n, dev_ptr(invQ_d));

        // invQ
        status = cusolverDnDpotrs(cusolverHandle, CUBLAS_FILL_MODE_LOWER, n, n,
                                  dev_ptr(work_mat_d), n, dev_ptr(invQ_d), n,
                                  dev_ptr(info_d));

        thrust::copy(info_d.begin(), info_d.end(), &info_h);
        check_cusolver_status(status, info_h);


        // invQt
        thrust::copy(targets_d.begin(), targets_d.end(), invQt_d.begin());
        status = cusolverDnDpotrs(cusolverHandle, CUBLAS_FILL_MODE_LOWER, n, 1,
                                  dev_ptr(work_mat_d), n, dev_ptr(invQt_d), n,
                                  dev_ptr(info_d));

        thrust::copy(info_d.begin(), info_d.end(), &info_h);
        check_cusolver_status(status, info_h);

        // logdetC
        thrust::device_vector<double> logdetC_d(1);

        sum_log_diag(n, dev_ptr(work_mat_d), dev_ptr(logdetC_d), dev_ptr(sum_buffer_d), sum_buffer_size_bytes);

        thrust::copy(logdetC_d.begin(), logdetC_d.end(), &logdetC);

	//copy work_mat_d into the lower triangular Cholesky factor
	thrust::copy(work_mat_d.begin(), work_mat_d.begin()+n*n, chol_lower_d.begin());

	//set the flag to say we have fitted theta
	theta_fitted = true;
    }

    bool get_theta_fit_status(void)
    {
        return theta_fitted;
    }

    void reset_theta_fit_status(void)
    {
        theta_fitted = false;
    }

    void get_invQ(mat_ref invQ_h)
    {
        thrust::copy(invQ_d.begin(), invQ_d.end(), invQ_h.data());
    }

    void get_invQt(mat_ref invQt_h)
    {
        thrust::copy(invQt_d.begin(), invQt_d.end(), invQt_h.data());
    }

    double get_logpost(void)
    {
        double result;
        CUBLASDOT(cublasHandle, n, dev_ptr(targets_d), 1, dev_ptr(invQt_d), 1,
                  &result);

        result += logdetC + n * log(2.0 * M_PI);

        return 0.5 * result;
    }

    void logpost_deriv(vec_ref result)
    {
        double zero(0.0);
        double one(1.0);
        double half(0.5);
        double m_half(-0.5);

        const int Ntheta = D + 1;

        // Compute
        //   \pderiv{C_{jk}}{theta_i}
        //
        // The length of work_mat_d is n * testing_size
        // The derivative above has  n * n * Ntheta components
        // The following assumes that testing_size > Ntheta * n
        cov_deriv_theta_batch_gpu(dev_ptr(work_mat_d),
				  D, n, n,
				  dev_ptr(inputs_d), dev_ptr(inputs_d),
				  dev_ptr(theta_d));

        // Compute
        //   \deriv{logpost}{theta_i}
        //     =   0.5 Tr(inv(C)*\pderiv{C}{\theta_i}           (Term 1)
        //       - 0.5 (invQ_ts)^T \pderiv{C}{\theta_i} invQt  (Term 2)

        // Working memory on device:
        //   result_d:
        //       accumulated result
        //   work_mat_d (until Term 2 step 1):
        //       covariance derivatives
        //   work_mat_d (after Term 2 step 1):
        //       temporary product in term 2 (okay to overwrite derivatives at this point)

        // ***** Term 1 *****
        // Treat 'jk' as a new 1-d index in C_{jk} and \pderiv{C_{jk}}{\theta}, called 'l' below.
        // That is:
        //    R_i = \pderiv{vec(C)_l}{\theta_i} vec(inv(C))_l
        //
        // Compute with gemv (y = \alpha A x + \beta y)
        cublasDgemv(cublasHandle, CUBLAS_OP_T, // handle, op (transpose)
                    n * n, Ntheta, // nrows, ncols (N.B. column-major order!)
                    &half, dev_ptr(work_mat_d), n * n, // alpha, A, lda
                    dev_ptr(invQ_d), 1, // x, incx
                    &zero, // beta
                    dev_ptr(result_d), 1); // y, incy

        // ***** Term 2 *****
        // First step:
        //   S_ij = \pderiv{C_{jk}}{\theta_i} invQt_k
        //
        // Treat \pderiv{C_{jk}}{\theta_i} as (n * Ntheta) * n, treating 'ij' as a single index
        // and take mat-vec product with invQt:
        cublasDgemv(cublasHandle, CUBLAS_OP_T, // handle, op (transpose)
                    n, n * Ntheta, // nrows, ncols (N.B. column-major order!)
                    &one, dev_ptr(work_mat_d), n, // alpha, A, lda
                    dev_ptr(invQt_d), 1, // x, incx
                    &zero, // beta
                    dev_ptr(work_mat_d), 1); // y, incy

        // Second step:
        //   R_i += 0.5 S_ij invQt_j
        cublasDgemv(cublasHandle, CUBLAS_OP_T, // handle, op (transpose)
                    n, Ntheta, // nrows, ncols (N.B. column-major order!)
                    &m_half, dev_ptr(work_mat_d), n, // alpha, A, lda
                    dev_ptr(invQt_d), 1, // x, incx
                    &one, // beta
                    dev_ptr(result_d), 1); // y, incy

        thrust::copy(result_d.begin(), result_d.begin() + Ntheta, result.data());
    }

    DenseGP_GPU(mat_ref inputs_, vec_ref targets_, unsigned int testing_size_)
        : testing_size(testing_size_)
        , n(inputs_.rows())
        , D(inputs_.cols())
        , invQ_d(n * n, 0.0)
        , chol_lower_d(n * n, 0.0)
        , theta_d(D + 2)
        , inputs(inputs_)
        , targets(targets_)
        , theta_fitted(false)
        , inputs_d(inputs_.data(), inputs_.data() + D * n)
        , targets_d(targets_.data(), targets_.data() + n)
        , logdetC(0.0)
        , jitter(0.0)
        , invQt_d(n, 0.0)
        , testing_d(D * testing_size, 0.0)
        , work_d(n, 0.0)
        , work_mat_d(n * testing_size, 0.0)
        , sum_buffer_size_bytes(0)
        , result_d((D+1) * testing_size, 0.0)
        , kappa_d(testing_size, 0.0)
        , invQk_d(testing_size * n, 0.0)
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

        // CUBLAS potrf workspace
        int potrfBufferSize;
        cusolverDnDpotrf_bufferSize(cusolverHandle, CUBLAS_FILL_MODE_LOWER,
                                    n, dev_ptr(potrf_buffer_d), n,
                                    &potrfBufferSize);
        potrf_buffer_d.resize(potrfBufferSize);

        // The following call determines the size of the cub::DeviceReduce::Sum workspace:
        // sum_buffer_size_bytes is 0 before this call, and the size of sum_buffer_d afterwards.
        // The end iterators are supplied but are not used.
        cub::DeviceReduce::Sum(dev_ptr(sum_buffer_d), sum_buffer_size_bytes,
                               sum_buffer_d.end(), sum_buffer_d.end(), n);
        sum_buffer_d.resize(sum_buffer_size_bytes);
    }
};

#endif
