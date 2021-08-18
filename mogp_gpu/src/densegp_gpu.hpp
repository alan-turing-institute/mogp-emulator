#ifndef DENSEGP_GPU_HPP
#define DENSEGP_GPU_HPP

/*
This file contains the C++ implementation of the Gaussian Process class.
(The corresponding .cu file contains the pybind11 bindings allowing the functions
to be called from Python).

The key methods of the DenseGP_GPU class are:
  fit:  update the theta hyperparameters
  predict_batch:  make a prediction on a set of testing points
  predict_variance_batch: make a prediction on a set of testing points, including variance
  predict_deriv: get the derivative of prediction on a set of testing points.
These in turn use CUDA kernels defined in the file cov_gpu.cu
*/

#include <iostream>

#include <algorithm>
#include <string>
#include <sstream>
#include <assert.h>
#include <stdexcept>

#include "util.hpp"
#include "kernel.hpp"
#include "meanfunc.hpp"




//////////////////////////////////////////////////////////////////////////////////////
////////////////////////// The Gaussian Process  class ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

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

    // what Kernel are we using?
    kernel_type kern_type;

    // pointer to the Kernel
    BaseKernel* kernel;

    // pointer to the Mean Function
    BaseMeanFunc* meanfunc;

    // is the nugget adapted, fixed, or fitted?
    nugget_type nug_type;

    // hyperparameters
    vec current_theta;

    // flag for whether we have fit hyperparameters
    bool theta_fitted;

    // handle for CUBLAS calls
    cublasHandle_t cublasHandle;

    // handle for cuSOLVER calls
    cusolverDnHandle_t cusolverHandle;

    // covariance matrix on device
    thrust::device_vector<REAL> K_d;

    // inverse covariance matrix on device
    thrust::device_vector<REAL> invQ_d;

    // lower triangular Cholesky factor on device
    thrust::device_vector<REAL> chol_lower_d;

    // log determinant of covariance matrix (on host)
    double logdetC;

    // current value of log posterior (on host)
    double current_logpost;

    // adaptive nugget jitter, or specified fixed, or fitted, nugget size
    double nug_size;

    // kernel hyperparameters on the device
    thrust::device_vector<REAL> theta_d;

    // mean function hyperparameters
    vec meanfunc_params;

    // inputs, on the device, row major order
    thrust::device_vector<REAL> inputs_d;

    // targets, on the device
    thrust::device_vector<REAL> targets_d;

    // precomputed product, used in the prediction
    thrust::device_vector<REAL> invQt_d;

    // preallocated work array (length n) on the CUDA device
    thrust::device_vector<REAL> work_d;

    // device vector for derivative of mean function
    thrust::device_vector<REAL> meanfunc_deriv_d;

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

    vec get_theta(void)
    {
        return current_theta;
    }

    double get_nugget_size(void) const
    {
        return nug_size;
    }

    void set_nugget_size(double nugget_size) 
    {
        nug_size = nugget_size;
    }

    void set_nugget_type(nugget_type nugtype)
    {
        nug_type = nugtype;
    }

    nugget_type get_nugget_type(void)
    {
        return nug_type;
    }

    BaseMeanFunc* get_meanfunc(void)
    {
        return meanfunc;
    }

    BaseKernel* get_kernel(void)
    {
        return kernel;
    }

    kernel_type get_kernel_type(void)
    {
        return kern_type;
    }

    // make a single prediction (mainly for testing - most use-cases will use predict_batch or predict_deriv_batch)
    double predict(mat_ref testing)
    {
        // On entry: the number of points to predict (number of rows
        // of testing) is assumed to be D

        thrust::device_vector<REAL> testing_d(testing.data(), testing.data() + D);
        kernel->cov_all_gpu(dev_ptr(work_d), n, D, dev_ptr(testing_d), dev_ptr(inputs_d),
			    dev_ptr(theta_d));

        REAL result = std::numeric_limits<double>::quiet_NaN();
        CUBLASDOT(cublasHandle, n, dev_ptr(work_d), 1, dev_ptr(invQt_d), 1,
                  &result);
	// evaluate the mean function and add to the result
        vec meanfunc_vals = meanfunc->mean_f(testing, meanfunc_params);
        return double(result + meanfunc_vals(0));
    }

    // variance of a single prediction (mainly for testing - most use-cases will use predict_variance_batch)
    double predict_variance(mat_ref testing, vec_ref var)
    {
        if (var.size() < testing.rows()) {
            throw std::runtime_error("predict_variance: the result buffer passed was "
                                     "too small to hold the variance");
        }

        // value prediction
        thrust::device_vector<REAL> testing_d(testing.data(), testing.data() + D);
        kernel->cov_all_gpu(dev_ptr(work_d), n, D, dev_ptr(testing_d), dev_ptr(inputs_d),
                    dev_ptr(theta_d));

        REAL result = std::numeric_limits<double>::quiet_NaN();
        CUBLASDOT(cublasHandle, n, dev_ptr(work_d), 1, dev_ptr(invQt_d), 1,
                  &result);

        // variance prediction
        thrust::device_vector<REAL> kappa_d(1);
        kernel->cov_val_gpu(dev_ptr(kappa_d), D, dev_ptr(testing_d), dev_ptr(testing_d),
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
	    // evaluate the mean function and add to the result
	    vec meanfunc_vals = meanfunc->mean_f(testing, meanfunc_params);
        return REAL(result + meanfunc_vals(0));
    }

    // Use the GP emulator to calculate a prediction on testing points, without calculating variance or derivative
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
        thrust::device_vector<REAL> testing_d(testing.data(), testing.data() + Nbatch * D);
        thrust::device_vector<REAL> result_d(Nbatch);

        kernel->cov_batch_gpu(dev_ptr(work_mat_d), Nbatch, n, D,
			      dev_ptr(testing_d), dev_ptr(inputs_d), dev_ptr(theta_d));

        cublasStatus_t status =
            cublasDgemv(cublasHandle, CUBLAS_OP_N, Nbatch, n, &one,
                        dev_ptr(work_mat_d), Nbatch, dev_ptr(invQt_d), 1, &zero,
                        dev_ptr(result_d), 1);

        cudaDeviceSynchronize();

        thrust::copy(result_d.begin(), result_d.end(), result.data());

	// evaluate the mean function and add to the result
	vec meanfunc_vals = meanfunc->mean_f(testing, meanfunc_params);
	result += meanfunc_vals;
    }

    // Use the GP emulator to calculate a prediction on testing points, also calculating variance
    void predict_variance_batch(mat_ref testing, vec_ref mean, vec_ref var)
    {
        REAL zero(0.0);
        REAL one(1.0);
        REAL minus_one(-1.0);
        int Nbatch = testing.rows();

        if (var.size() < testing.rows() || mean.size() < testing.rows()) {
            throw std::runtime_error("predict_variance_batch: The result buffer passed was "
                                     "too small to hold the variance");
        }

        if (Nbatch > testing_size) {
            throw std::runtime_error("predict_variance_batch: More test points were passed "
                                     "than the maximum batch size");
        }


        thrust::device_vector<REAL> testing_d(testing.data(), testing.data() + Nbatch * D);
        thrust::device_vector<REAL> mean_d(Nbatch);

        // compute predictive means for the batch
        kernel->cov_batch_gpu(dev_ptr(work_mat_d), Nbatch, n, D, dev_ptr(testing_d),
                      dev_ptr(inputs_d), dev_ptr(theta_d));

        cublasStatus_t status =
            cublasDgemv(cublasHandle, CUBLAS_OP_N, Nbatch, n, &one,
                        dev_ptr(work_mat_d), Nbatch, dev_ptr(invQt_d), 1, &zero,
                        dev_ptr(mean_d), 1);

        // compute predictive variances for the batch
        kernel->cov_diag_gpu(dev_ptr(kappa_d), Nbatch, D, dev_ptr(testing_d),
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

        // copy back means
        thrust::copy(mean_d.begin(), mean_d.end(), mean.data());
	// evaluate the mean function and add to the mean
	vec meanfunc_vals = meanfunc->mean_f(testing, meanfunc_params);
	mean += meanfunc_vals;
        // copy back variances
        thrust::copy(kappa_d.begin(), kappa_d.begin() + Nbatch, var.data());
    }

    // Use the GP emulator to calculate derivative of  prediction on testing points
    void predict_deriv(mat_ref testing, mat_ref result)
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
        thrust::device_vector<REAL> testing_d(testing.data(), testing.data() + Nbatch * D);
        thrust::device_vector<REAL> result_d(Nbatch*D);

        kernel->cov_deriv_x_batch_gpu(dev_ptr(work_mat_d), D, Nbatch, n,
				      dev_ptr(testing_d), dev_ptr(inputs_d), dev_ptr(theta_d));

        cublasStatus_t status =
            cublasDgemv(cublasHandle, CUBLAS_OP_N,
                        D * Nbatch, n, // nrows, ncols
                        &one, dev_ptr(work_mat_d), D * Nbatch, // alpha, A, lda
                        dev_ptr(invQt_d), 1, // x, incx
                        &zero, dev_ptr(result_d), 1); // beta, y, incy

        cudaDeviceSynchronize();

        thrust::copy(result_d.begin(), result_d.end(), result.data());

	// evaluate deriv of meanfunc wrt test points, and add to result
	mat meanfunc_inputderiv = meanfunc->mean_inputderiv(testing, meanfunc_params);
	result += meanfunc_inputderiv.transpose();
    }

    // perform Cholesky factorization of the matrix currently stored in work_mat_d
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

    // Update the hyperparameters, and K, invQ and invQt which depend on them
    void fit(vec_ref theta)
    {

	    int param_switch_index = meanfunc->get_n_params(inputs);
	    meanfunc_params = theta.block(0,0,param_switch_index,1);

	    // evaluate the mean function at the input values and subtract from targets
	    vec new_targets = targets - meanfunc->mean_f(inputs, meanfunc_params);
	    thrust::copy(new_targets.data(), new_targets.data() + n, targets_d.begin());

	    // copy all the parameters _after_ those for the mean function to the device vector theta_d
        thrust::copy(theta.data() + param_switch_index,
		         theta.data() + D + 2 + param_switch_index,
		         theta_d.begin());

	    // calculate covariance matrix, put result into K_d
        kernel->cov_batch_gpu(dev_ptr(K_d), n, n, D, dev_ptr(inputs_d),
		    	      dev_ptr(inputs_d), dev_ptr(theta_d));
	    /// copy the covariance matrix K_d into work_mat_d
	    thrust::copy(K_d.begin(), K_d.end(), work_mat_d.begin());

        thrust::device_vector<int> info_d(1);
        int info_h;
        cusolverStatus_t status;
	    int factorisation_status;
	    // for adaptive nugget start with a nugget of zero and increase by small amount
	    // until we find a value where factorization succeeds.
	    if (nug_type == NUG_ADAPTIVE) {
	        double mean_diag;
	        const int max_tries = 5;
	        int itry = 0;
	        while (itry < max_tries) {
		        // K_d holds the covariance matrix - work with a copy
		        // in work_mat_d, in case the factorization fails
		        thrust::copy(K_d.begin(), K_d.end(), work_mat_d.begin());

		        // if the first attempt at factorization failed, add a
		        // small term to the diagonal, increasing each iteration
		        // until the factorization succeeds
		        if (itry >= 1) {
		            if (itry == 1) {
			            // find mean of (absolute) diagonal elements (diagonal
			            // elements should all be positive)
		    	        cublasDasum(cublasHandle, n, dev_ptr(K_d), n+1, &mean_diag);
			            mean_diag /= n;
			            nug_size = 1e-6 * mean_diag;
		            }
		            add_diagonal(n, nug_size, dev_ptr(work_mat_d));
		        }

		        factorisation_status = calc_cholesky_factors();
		        if (factorisation_status == 0) {
                    break;
		        }
		        nug_size *= 10;
		        itry++;
	        }

	        // if none of the factorization attempts succeeded:
	        if (itry == max_tries) {
		        std::string msg;
		        std::stringstream smsg(msg);
		        smsg << "All attempts at factorization failed. Last return code " << factorisation_status;
		        throw std::runtime_error(smsg.str());
	        }
	    // for fixed nugget, add "nugget_size" to the diagonal of the matrix.
	    } else if (nug_type == NUG_FIXED) {
	        add_diagonal(n, nug_size, dev_ptr(work_mat_d));
	        factorisation_status = calc_cholesky_factors();
	        if (factorisation_status != 0) {
                throw std::runtime_error("Unable to factorize matrix using fixed nugget");
	        }

	    } else if (nug_type == NUG_FIT) {
	        // set to exp(last-element-of-theta)
	        nug_size = exp( theta(theta.size()-1) );

	        add_diagonal(n, nug_size, dev_ptr(work_mat_d));
	        factorisation_status = calc_cholesky_factors();
	        if (factorisation_status != 0) {
                throw std::runtime_error("Unable to factorize matrix using fitted nugget");
	        }
	    } else throw std::runtime_error("Unrecognized nugget_type");


        // get the inverse covariance matrix invQ by solving the system of linear eqns
	    //    work_mat_d . invQ_d = I
	    // where work_mat_d is holding the current covariance matrix.

        identity_device(n, dev_ptr(invQ_d));

        status = cusolverDnDpotrs(cusolverHandle, CUBLAS_FILL_MODE_LOWER, n, n,
                                  dev_ptr(work_mat_d), n, dev_ptr(invQ_d), n,
                                  dev_ptr(info_d));

        thrust::copy(info_d.begin(), info_d.end(), &info_h);
        check_cusolver_status(status, info_h);


        // invQt - product of inverse covariance matrix with the target values
        thrust::copy(targets_d.begin(), targets_d.end(), invQt_d.begin());
        status = cusolverDnDpotrs(cusolverHandle, CUBLAS_FILL_MODE_LOWER, n, 1,
                                  dev_ptr(work_mat_d), n, dev_ptr(invQt_d), n,
                                  dev_ptr(info_d));

        thrust::copy(info_d.begin(), info_d.end(), &info_h);
        check_cusolver_status(status, info_h);

        // logdetC - sum the log of the diagonal elements of the Cholesky factor of covariance matrix (in work_mat_d)
        thrust::device_vector<double> logdetC_d(1);

        sum_log_diag(n, dev_ptr(work_mat_d), dev_ptr(logdetC_d), dev_ptr(sum_buffer_d), sum_buffer_size_bytes);

        thrust::copy(logdetC_d.begin(), logdetC_d.end(), &logdetC);

	    //copy work_mat_d into the lower triangular Cholesky factor
	    thrust::copy(work_mat_d.begin(), work_mat_d.begin()+n*n, chol_lower_d.begin());

	    //set the flag to say we have fitted theta
	    theta_fitted = true;
        // copy mean function params then kernel params into current_theta
	    current_theta.block(0,0,param_switch_index,1) = meanfunc_params;
        thrust::copy(theta_d.begin(), theta_d.end(), current_theta.data()+param_switch_index);
        
        // update the current_logpost
        double logpost;
        CUBLASDOT(cublasHandle, n, dev_ptr(targets_d), 1, dev_ptr(invQt_d), 1,
                  &logpost);

        logpost += logdetC + n * log(2.0 * M_PI);

        logpost = 0.5 * logpost;
        current_logpost = logpost;
    }

    bool get_theta_fit_status(void)
    {
        return theta_fitted;
    }

    void reset_theta_fit_status(void)
    {
        theta_fitted = false;
    }

    void get_K(mat_ref K_h)
    {
        thrust::copy(K_d.begin(), K_d.end(), K_h.data());
    }

    void get_invQ(mat_ref invQ_h)
    {
        thrust::copy(invQ_d.begin(), invQ_d.end(), invQ_h.data());
    }

    void get_invQt(mat_ref invQt_h)
    {
        thrust::copy(invQt_d.begin(), invQt_d.end(), invQt_h.data());
    }

    double get_logpost(vec_ref new_theta)
    {

        bool theta_close = (new_theta - current_theta).norm() < 1e-8;
	    if (theta_close) {
	        return current_logpost;
	    }
	    
        //theta has changed - refit
	    fit(new_theta);

	    return current_logpost;
    }

    void logpost_deriv(vec_ref result)
    {
        double zero(0.0);
        double one(1.0);
        double half(0.5);
        double m_half(-0.5);
	    double minusone(-1.0);

        const int Ntheta = D + 1;

        // Compute
        //   \pderiv{C_{jk}}{theta_i}
        //
        // The length of work_mat_d is n * testing_size
        // The derivative above has  n * n * Ntheta components
        // The following assumes that testing_size > Ntheta * n
        kernel->cov_deriv_theta_batch_gpu(dev_ptr(work_mat_d),
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

	    // first elements of result (up to meanfunc_nparam) will be from meanfunc,
	    // then the next (D+1) from the Kernel.
	    int meanfunc_nparam = meanfunc_params.rows();
	    if (meanfunc_nparam > 0) { // only copy data to device and do calculation if we need to.
	        mat meanfunc_deriv = meanfunc->mean_deriv(inputs, meanfunc_params);
	        thrust::copy(meanfunc_deriv.data(), meanfunc_deriv.data() + (n * meanfunc_nparam), meanfunc_deriv_d.begin());
	        // product of meanfunc_deriv with invQt
	        cublasDgemv(cublasHandle, CUBLAS_OP_T, // handle, op (transpose)
		            n, n * meanfunc_nparam, // nrows, ncols (N.B. column-major order!)
                    &minusone, dev_ptr(meanfunc_deriv_d), n, // alpha, A, lda
                    dev_ptr(invQt_d), 1, // x, incx
                    &zero, // beta
                    dev_ptr(meanfunc_deriv_d), 1); // y, incy
	        thrust::copy(meanfunc_deriv_d.begin(), meanfunc_deriv_d.begin() + (meanfunc_nparam), result.data());
	    }
        thrust::copy(result_d.begin(), result_d.begin() + Ntheta, result.data()+(meanfunc_nparam));

	    // fitted nugget - last element of theta is log(nugget)
	    // partial deriv is 0.5*nugget*(trace(invQ) - invQt.invQt)
	    if (nug_type == NUG_FIT) {
	        // trace of invQ
	        REAL tr_invQ = 0.;
	        thrust::device_vector<double> tr_invQ_d(1);
	        trace(n, dev_ptr(invQ_d), dev_ptr(tr_invQ_d), dev_ptr(sum_buffer_d), sum_buffer_size_bytes);
	        thrust::copy(tr_invQ_d.begin(), tr_invQ_d.end(), &tr_invQ);
	        // invQt dot invQt
	        REAL invQtSq = std::numeric_limits<double>::quiet_NaN();
	        CUBLASDOT(cublasHandle, n, dev_ptr(invQt_d), 1, dev_ptr(invQt_d), 1,
		        &invQtSq);

	        // set the last element of result, putting it all together
	        result(result.size()-1) = 0.5 * nug_size * (tr_invQ - invQtSq);
	    }
    }

    // destructor - just to see when it is being called
    //~DenseGP_GPU() {
    //    std::cout<<"In destructor of DenseGP_GPU"<<std::endl;
   // }

    // constructor
    DenseGP_GPU(mat_ref inputs_,
	      vec_ref targets_,
	      unsigned int testing_size_,
	      BaseMeanFunc* mean_ = NULL,
	      kernel_type kern_=SQUARED_EXPONENTIAL)
        : testing_size(testing_size_)
        , n(inputs_.rows())
        , D(inputs_.cols())
        , K_d(n * n, 0.0)
        , invQ_d(n * n, 0.0)
        , chol_lower_d(n * n, 0.0)
        , theta_d(D + 2)
	    , meanfunc_params(1) // resize later if necessary
        , inputs(inputs_)
        , targets(targets_)
	    , kern_type(kern_)
        , kernel(0)
	    , meanfunc(mean_)
	    , nug_type(NUG_ADAPTIVE)
	    , current_theta(1) // resize later
        , theta_fitted(false)
        , inputs_d(inputs_.data(), inputs_.data() + D * n)
        , targets_d(targets_.data(), targets_.data() + n)
        , logdetC(0.0)
	    , current_logpost(0.0)
        , nug_size(0.0)
        , invQt_d(n, 0.0)
        , testing_d(D * testing_size, 0.0)
        , work_d(n, 0.0)
        , work_mat_d(n * testing_size, 0.0)
        , sum_buffer_size_bytes(0)
        , result_d(testing_size, 0.0)
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

        // if mean function is not provided, assume zero everywhere.
        if (!meanfunc) meanfunc = new ZeroMeanFunc();

	    if (kern_type == SQUARED_EXPONENTIAL) {
	        kernel = new SquaredExponentialKernel();
	    } else if (kern_type == MATERN52) {
	        kernel = new Matern52Kernel();
	    } else throw std::runtime_error("Unrecognized kernel type\n");

	    // resize meanfunc_params vector here
	    meanfunc_params.resize(meanfunc->get_n_params(inputs),1);
	    // resize the device vector that will store derivative of mean function
	    meanfunc_deriv_d.resize(meanfunc->get_n_params(inputs) * inputs.rows());
        // resize current_theta vector
        current_theta.resize(meanfunc->get_n_params(inputs) + get_n_params(),1);
    }

};

#endif
