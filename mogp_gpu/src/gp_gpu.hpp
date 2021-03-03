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

    if (i < WARP_SIZE) work[i] = 0.0;
    if (i < N) {
        log_diag = log(A[i * (N+1)]);
    }

    int laneIdx = i % WARP_SIZE;
    int warpIdx = i / WARP_SIZE;

    __syncthreads();
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

enum nugget_type {NUG_ADAPTIVE, NUG_FIT, NUG_FIXED};

// The GP class
// By convention, members ending "_d" are allocated on the device
class DenseGP_GPU {
    // batch size (for allocation)
    unsigned int xnew_size;

    // Number of training points, dimension of input
    unsigned int N, Ninput;

    // inputs
    mat xs;

    // targets
    vec ts;

    // flag for whether we have fit hyperparameters
    bool theta_fitted;

    // handle for CUBLAS calls
    cublasHandle_t cublasHandle;

    // handle for cuSOLVER calls
    cusolverDnHandle_t cusolverHandle;

    // inverse covariance matrix on device
    thrust::device_vector<REAL> invC_d;

    // lower triangular Cholesky factor on device
    thrust::device_vector<REAL> chol_lower_d;

    // log determinant of covariance matrix (on host)
    double logdetC;

    // adaptive nugget jitter
    double jitter;

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

    int D(void) const
    {
        return Ninput;
    }

    mat inputs(void) const
    {
        return xs;
    }

    vec targets(void) const
    {
        return ts;
    }

    int n_params(void) const
    {
        return Ninput + 2;
    }

    void get_theta(vec_ref theta)
    {
      thrust::copy(theta_d.begin(), theta_d.end(), theta.data());
    }

    double get_jitter() const
    {
      return jitter;
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
        int Nbatch = xnew.rows();

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

    void predict_variance_batch(mat_ref xnew, vec_ref mean, vec_ref var)
    {
        REAL zero(0.0);
        REAL one(1.0);
        REAL minus_one(-1.0);
	int Nbatch = mean.size();

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

    void predict_deriv(mat_ref xnew, mat_ref result) {
        int Nbatch = xnew.rows();

        // TODO
        // assert result.size() == xnew.rows()

        REAL zero(0.0);
        REAL one(1.0);
        thrust::device_vector<REAL> xnew_d(xnew.data(), xnew.data() + Nbatch * Ninput);
        thrust::device_vector<REAL> result_d(Nbatch*Ninput);

        cov_deriv_x_batch_gpu(dev_ptr(work_mat_d), Ninput, Nbatch, N,
			      dev_ptr(xnew_d), dev_ptr(xs_d), dev_ptr(theta_d));

        cublasStatus_t status =
            cublasDgemv(cublasHandle, CUBLAS_OP_N,
                        Ninput * Nbatch, N, // nrows, ncols
                        &one, dev_ptr(work_mat_d), Ninput * Nbatch, // alpha, A, lda
                        dev_ptr(invCts_d), 1, // x, incx
                        &zero, dev_ptr(result_d), 1); // beta, y, incy

        cudaDeviceSynchronize();

        thrust::copy(result_d.begin(), result_d.end(), result.data());
    }


  // Assume at this point that work_mat_d contains the inverse covariance matrix invC_d
  int calc_Cholesky_factors()
    {
        thrust::device_vector<int> info_d(1);
        int info_h;
        cusolverStatus_t status;

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

	}
	return info_h;
    }

  // return the lower triangular matrix (actually it will be the whole matrix
  // but only the lower triangle will be correct!)
  void get_Cholesky_lower(mat_ref result) {

    thrust::copy(chol_lower_d.begin(), chol_lower_d.end(), result.data());

  }

    // Update the hyperparameters, and invQ and invQt which depend on them

  void update_theta(vec_ref theta, nugget_type nugget, double nugget_size=0.)
    {

        thrust::copy(theta.data(), theta.data() + Ninput + 2, theta_d.begin());

        cov_batch_gpu(dev_ptr(invC_d), N, N, Ninput, dev_ptr(xs_d),
                      dev_ptr(xs_d), dev_ptr(theta_d));

	/// copy the covariance matrix invC_d into work_mat_d
	thrust::copy(invC_d.begin(), invC_d.end(), work_mat_d.begin());

        thrust::device_vector<int> info_d(1);
        int info_h;
        cusolverStatus_t status;
	int factorisation_status;

	if (nugget == NUG_ADAPTIVE) {
	    double mean_diag;
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
		}

		factorisation_status = calc_Cholesky_factors();
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
	    add_diagonal(N, nugget_size, dev_ptr(work_mat_d));

	    factorisation_status = calc_Cholesky_factors();
	    if (factorisation_status != 0) {
	      throw std::runtime_error("Unable to factorize matrix using fixed nugget");
	    }

	} else { //nugget == "fit"

	    factorisation_status = calc_Cholesky_factors();
	    if (factorisation_status != 0) {
	      throw std::runtime_error("Unable to factorize matrix using fitted nugget");
	    }
	}

        identity_device(N, dev_ptr(invC_d));

        // invC
        status = cusolverDnDpotrs(cusolverHandle, CUBLAS_FILL_MODE_LOWER, N, N,
                                  dev_ptr(work_mat_d), N, dev_ptr(invC_d), N,
                                  dev_ptr(info_d));

        thrust::copy(info_d.begin(), info_d.end(), &info_h);
        check_cusolver_status(status, info_h);


        // invCt
        thrust::copy(ts_d.begin(), ts_d.end(), invCts_d.begin());
        status = cusolverDnDpotrs(cusolverHandle, CUBLAS_FILL_MODE_LOWER, N, 1,
	                         dev_ptr(work_mat_d), N, dev_ptr(invCts_d), N,
	                         dev_ptr(info_d));

        thrust::copy(info_d.begin(), info_d.end(), &info_h);
        check_cusolver_status(status, info_h);

        // logdetC
        thrust::device_vector<double> logdetC_d(1);

        sumLogDiag<<<1, N>>>(N, dev_ptr(work_mat_d), dev_ptr(logdetC_d));

        thrust::copy(logdetC_d.begin(), logdetC_d.end(), &logdetC);

	//copy work_mat_d into the lower triangular Cholesky factor
	thrust::copy(work_mat_d.begin(), work_mat_d.begin()+N*N, chol_lower_d.begin());

	//set the flag to say we have fitted theta

	theta_fitted = true;

    }

    bool theta_fit_status() {
        return theta_fitted;
    }

    void get_invQ(mat_ref invQ_h)
    {
        thrust::copy(invC_d.begin(), invC_d.end(), invQ_h.data());
    }

    void get_invQt(mat_ref invQt_h)
    {
        thrust::copy(invCts_d.begin(), invCts_d.end(), invQt_h.data());
    }

    double get_logpost(void)
    {
        double result;
        CUBLASDOT(cublasHandle, N, dev_ptr(ts_d), 1, dev_ptr(invCts_d), 1,
                  &result);

        result += logdetC + N * log(2.0 * M_PI);

        return 0.5 * result;
    }

    void dloglik_dtheta(vec_ref result)
    {
        double zero(0.0);
        double one(1.0);
        double half(0.5);
        double m_half(-0.5);

        const int Ntheta = Ninput + 1;

        // Compute
        //   \pderiv{C_{jk}}{theta_i}
        //
        // The length of work_mat_d is N * xnew_size
        // The derivative above has  N * N * Ntheta components
        // The following assumes that xnew_size > Ntheta * N
        cov_deriv_theta_batch_gpu(dev_ptr(work_mat_d),
				  Ninput, N, N,
				  dev_ptr(xs_d), dev_ptr(xs_d),
				  dev_ptr(theta_d));

        // Compute
        //   \deriv{logpost}{theta_i}
        //     =   0.5 Tr(inv(C)*\pderiv{C}{\theta_i}           (Term 1)
        //       - 0.5 (invC_ts)^T \pderiv{C}{\theta_i} invCts  (Term 2)

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
                    N * N, Ntheta, // nrows, ncols (N.B. column-major order!)
                    &half, dev_ptr(work_mat_d), N * N, // alpha, A, lda
                    dev_ptr(invC_d), 1, // x, incx
                    &zero, // beta
                    dev_ptr(result_d), 1); // y, incy

        // ***** Term 2 *****
        // First step:
        //   S_ij = \pderiv{C_{jk}}{\theta_i} invCts_k
        //
        // Treat \pderiv{C_{jk}}{\theta_i} as (N * Ntheta) * N, treating 'ij' as a single index
        // and take mat-vec product with invCts:
        cublasDgemv(cublasHandle, CUBLAS_OP_T, // handle, op (transpose)
                    N, N * Ntheta, // nrows, ncols (N.B. column-major order!)
                    &one, dev_ptr(work_mat_d), N, // alpha, A, lda
                    dev_ptr(invCts_d), 1, // x, incx
                    &zero, // beta
                    dev_ptr(work_mat_d), 1); // y, incy

        // Second step:
        //   R_i += 0.5 S_ij invCts_j
        cublasDgemv(cublasHandle, CUBLAS_OP_T, // handle, op (transpose)
                    N, Ntheta, // nrows, ncols (N.B. column-major order!)
                    &m_half, dev_ptr(work_mat_d), N, // alpha, A, lda
                    dev_ptr(invCts_d), 1, // x, incx
                    &one, // beta
                    dev_ptr(result_d), 1); // y, incy

        thrust::copy(result_d.begin(), result_d.begin() + Ntheta, result.data());
    }

    DenseGP_GPU(mat_ref xs_, vec_ref ts_, unsigned int xnew_size_)
        : xnew_size(xnew_size_)
        , N(xs_.rows())
	, Ninput(xs_.cols())
        , invC_d(N * N, 0.0)
	, chol_lower_d(N * N, 0.0)
        , theta_d(Ninput + 2)
	, xs(xs_)
	, ts(ts_)
	, theta_fitted(false)
        , xs_d(xs_.data(), xs_.data() + Ninput * N)
        , ts_d(ts_.data(), ts_.data() + N)
	, logdetC(0.0)
	, jitter(0.0)
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
