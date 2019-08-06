#ifndef GP_GPU_HPP
#define GP_GPU_HPP

#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "gp.hpp"
#include "cov_gpu.hpp"

// based on DenseGP in gp.hpp, but for the GPU
// Can only be initialized by reading the from a file
class DenseGP_GPU : public GP {

	// hyperparameters on the device
	REAL *theta_d;
	
	unsigned int N, Ninput;

	// training inputs (rows)
	mat xs;
	vec ts;
	vec_obs_kind Ts;

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

	const int xnew_size = 20000;
	REAL *xnew_d, *work_mat_d, *alpha_d, *result_d;

	// handle for CUBLAS calls
	cublasHandle_t cublasHandle;

	vec noise(void) const {
		return hypers.rows(0,3);//hypers.rows(0,Ninput);
	}
	
	// The remaining hyperparameters (e.g. lengthscales etc)
	vec theta(void) const {
		return hypers.rows(4,7); //hypers.rows(Ninput+1, hypers.n_rows-1);
	}

	virtual void update_matrices(void) { throw std::runtime_error("update_matrices not implemented"); }
	virtual double log_lik(void) const { throw std::runtime_error("log_lik not implemented"); }

public:
	virtual int data_length(void) const
	{
		return N;
	}

	virtual double predict(vec xnew, obs_kind Tnew, vec& work) const final
	{
		// work === k, of length N
		// for (unsigned i=0; i<N; i++) {
		// 	work(i) = cov(Tnew, Ts(i), xnew, xs.row(i).t(), theta());
		// }
		// cudaMemcpy(work_d, work.memptr(), N*sizeof(double), cudaMemcpyHostToDevice);
		// input for
		REAL *xnew_d;
		Col<REAL> xnewf = arma::conv_to<Col<REAL> >::from(xnew);
		if (cudaMalloc((void**)(&xnew_d), Ninput*sizeof(REAL)) != cudaSuccess) {
			throw std::runtime_error("Device allocation failure (xnew_d)");
		}
		
		cudaMemcpy(xnew_d, xnewf.memptr(), Ninput*sizeof(REAL), cudaMemcpyHostToDevice);
		cov_all_wrapper(work_d, N, xs.n_cols, xnew_d, xs_d, theta_d);
		
		REAL result;
		CUBLASDOT(cublasHandle, N, work_d, 1, invCts_d, 1, &result);
		return double(result);
	}

	virtual void predict_batch(Col<REAL> &result, Mat<REAL> xnew, obs_kind Tnew) const final
	{
		REAL alpha = 1.0, beta = 0.0;
		cudaMemcpy(xnew_d, xnew.memptr(), Ninput*xnew.n_cols*sizeof(REAL), cudaMemcpyHostToDevice);
		cov_batch_wrapper(work_mat_d, xnew.n_cols, N, Ninput, xnew_d, xs_d, theta_d);
		
		//cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);
		
		cublasStatus_t status = CUBLASGEMV(cublasHandle, CUBLAS_OP_N, xnew.n_cols, N, &alpha, work_mat_d, xnew.n_cols, invCts_d, 1, &beta, result_d, 1);
		
		//cublasStatus_t status = cublasDgemv(cublasHandle, CUBLAS_OP_N, 1, 1, &alpha, work_mat_d, 1, invCts_d, 1, &beta, result_d, 1);
		cudaDeviceSynchronize();
		//if (status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("cublas failure"); }
		
		// double result;
		// result = -2.0;

		cudaError_t err = cudaMemcpy(result.memptr(), result_d, result.n_rows*sizeof(REAL), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) { printf("%s line %d: CUDA Error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); }
	}

	virtual double predict(vec xnew, obs_kind Tnew) const final
	{
		vec k(N);
		return predict(xnew, Tnew, k);
	}

	DenseGP_GPU(std::string filename)
	{
		std::ifstream in(filename);
		in >> N >> Ninput;
		hypers.load(in, arma_binary);
		xs.load(in, arma_binary);
		ts.load(in, arma_binary);
		Ts.load(in, arma_binary);
		C.load(in, arma_binary);
		invC.load(in, arma_binary);
		invCts.load(in, arma_binary);		

		Col<REAL> theta_tmp (arma::conv_to<Col<REAL> >::from(theta()));

		cublasStatus_t status;
		status = cublasCreate(&cublasHandle);
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			throw std::runtime_error("CUBLAS initialization error\n");
		}

		if (cudaMalloc((void**)(&invCts_d), N*sizeof(REAL)) != cudaSuccess) {
			throw std::runtime_error("Device allocation failure (invCts_d)");
		}

		if (cudaMalloc((void**)(&work_d), N*sizeof(REAL)) != cudaSuccess) {
			throw std::runtime_error("Device allocation failure (work_d)");
		}

		if (cudaMalloc((void**)(&theta_d), theta_tmp.n_rows*sizeof(REAL)) != cudaSuccess) {
			throw std::runtime_error("Device allocation failure (theta_d)");
		}
		
		if (cudaMalloc((void**)(&xs_d), N*Ninput*sizeof(REAL)) != cudaSuccess) {
			throw std::runtime_error("Device allocation failure (xs_d)");
		}


		if (cudaMalloc((void**)(&xnew_d), Ninput*xnew_size*sizeof(REAL)) != cudaSuccess) {
			throw std::runtime_error("Device allocation failure (xnew_d)");
		}
		if (cudaMalloc((void**)(&work_mat_d), N*xnew_size*sizeof(REAL)) != cudaSuccess) {
			throw std::runtime_error("Device allocation failure (work_mat_d)");
		}
		if (cudaMalloc((void**)(&result_d), xnew_size*sizeof(REAL)) != cudaSuccess) {
			throw std::runtime_error("Device allocation failure (result_d)");
		}

		Col<REAL> RinvCts (arma::conv_to<Col<REAL> >::from(invCts));
		cudaMemcpy(invCts_d, RinvCts.memptr(), N*sizeof(REAL), cudaMemcpyHostToDevice);
		cudaMemcpy(theta_d, theta_tmp.memptr(), theta_tmp.n_rows*sizeof(REAL), cudaMemcpyHostToDevice);
		Mat<REAL> xs_transpose (arma::conv_to<Mat<REAL> >::from(xs.t()));
		cudaMemcpy(xs_d, xs_transpose.memptr(), N*Ninput*sizeof(REAL), cudaMemcpyHostToDevice);
	}

	// ~DenseGP_GPU()
	// {
		
	// }
};

#endif
