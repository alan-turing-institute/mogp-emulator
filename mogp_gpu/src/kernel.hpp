#ifndef KERNEL_HPP
#define KERNEL_HPP
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "types.hpp"

#define CUBLASDOT cublasDdot
#define CUBLASGEMV cublasDgemv


class BaseKernel {

public:

  virtual ~BaseKernel(){};

  // Performs a single evaluation of the covariance function, for inputs
  // x and y; that is, Cov(x(:), y(:); theta).
  virtual void cov_val_gpu(REAL *result_d, int Ninput, REAL *x_d, REAL *y_d,
			   REAL *theta_d) = 0;

// Computes the vector k_i = Cov(xnew(:), xs(i,:); theta).
  virtual void cov_all_gpu(REAL *result_d, int N, int Ninput, REAL *xnew_d, REAL *xs_d,
			   REAL *theta_d) = 0;

  // Computes the vector k_i = Cov(xnew(i,:), xs(i,:); theta).
  virtual void cov_diag_gpu(REAL *result_d, int N, int Ninput, REAL *xnew_d, REAL *xs_d,
		    REAL *theta_d) = 0;

  // Computes the submatrix K_ij = Cov(xsnew(i,:), xs(j,:); theta)
  virtual void cov_batch_gpu(REAL *result_d, int Nnew, int N, int Ninput, REAL *xsnew_d,
			     REAL *xs_d, REAL *theta_d) = 0;

  // Computes the matrix (dC/dtheta)(xs(i,:), ys(j,:); theta)
  virtual void cov_deriv_theta_batch_gpu(REAL *result_d, int Ninput, int Nx, int Ny,
					 const REAL *xs_d, const REAL *ys_d,
					 const REAL *theta_d) = 0;

  // Computes the matrix (dC/dx)(xs(i,:), ys(j,:); theta)
  virtual void cov_deriv_x_batch_gpu(REAL *result_d, int Ninput, int Nx, int Ny,
				     const REAL *xs_d, const REAL *ys_d,
				     const REAL *theta_d) = 0;

  // For pybind11, implement method to replicate python interface for evaluating cov matrix
  mat kernel_f(mat x1, mat x2, vec params) {
    int N = x1.rows();
    int Nnew = x2.rows();
    int D = x1.cols();
    assert (x2.cols() == D);
    thrust::device_vector<REAL> result_d(N*Nnew);
    thrust::device_vector<REAL> x1_d(x1.data(), x1.data()+N*D);
    thrust::device_vector<REAL> x2_d(x2.data(), x2.data()+Nnew*D);
    thrust::device_vector<REAL> params_d(params.data(), params.data()+D+1);

    cov_batch_gpu(dev_ptr(result_d), Nnew, N, D, dev_ptr(x2_d), dev_ptr(x1_d), dev_ptr(params_d));
    mat result(N,Nnew);
    thrust::copy(result_d.begin(), result_d.end(), result.data());
    return result;
  }

  // For pybind11, implement method to replicate python interface deriv of
  // cov matrix wrt first set of inputs
  // NOTE - this will return values as a 1D array of size N*Nnew*D, unlike
  // the pure Python version that will return a 3D array of size [D, N, Nnew]
  vec kernel_inputderiv(mat x1, mat x2, vec params) {
    int N = x1.rows();
    int Nnew = x2.rows();
    int D = x1.cols();
    assert (x2.cols() == D);
    thrust::device_vector<REAL> result_d(N * Nnew * D);
    thrust::device_vector<REAL> x1_d(x1.data(), x1.data()+N*D);
    thrust::device_vector<REAL> x2_d(x2.data(), x2.data()+Nnew*D);
    thrust::device_vector<REAL> params_d(params.data(), params.data()+D+1);
    cov_deriv_x_batch_gpu(dev_ptr(result_d), D, N, Nnew,
			      dev_ptr(x1_d), dev_ptr(x2_d), dev_ptr(params_d));
    vec result(N * Nnew * D);
    thrust::copy(result_d.begin(), result_d.end(), result.data());
    return result;
  }

  // For pybind11, implement method to replicate python interface deriv of
  //  covariance matrix wrt theta.
  // NOTE - this will return values as a 1D array of size N*Nnew*n_params, unlike
  // the pure Python version that will return a 3D array of size [n_params, N, Nnew]
  vec kernel_deriv(mat x1, mat x2, vec params) {
    int N = x1.rows();
    int Nnew = x2.rows();
    int D = x1.cols();
    assert (x2.cols() == D);
    int n_params = D+1;
    thrust::device_vector<REAL> result_d(N * Nnew * n_params);
    thrust::device_vector<REAL> x1_d(x1.data(), x1.data()+N*D);
    thrust::device_vector<REAL> x2_d(x2.data(), x2.data()+Nnew*D);
    thrust::device_vector<REAL> params_d(params.data(), params.data()+n_params);
    cov_deriv_theta_batch_gpu(dev_ptr(result_d), D, N, Nnew,
			      dev_ptr(x1_d), dev_ptr(x2_d), dev_ptr(params_d));
    vec result(N*Nnew*n_params);
    thrust::copy(result_d.begin(), result_d.end(), result.data());
    return result;
  }

};


class SquaredExponentialKernel : public BaseKernel {
public:

  // Squared exponential covariance function

  // Performs a single evaluation of the covariance function, for inputs
  // x and y; that is, Cov(x(:), y(:); theta).
  void cov_val_gpu(REAL *result_d, int Ninput, REAL *x_d, REAL *y_d,
		   REAL *theta_d);

  // Computes the vector k_i = Cov(xnew(:), xs(i,:); theta).
  void cov_all_gpu(REAL *result_d, int N, int Ninput, REAL *xnew_d, REAL *xs_d,
		   REAL *theta_d);

  // Computes the vector k_i = Cov(xnew(i,:), xs(i,:); theta).
  void cov_diag_gpu(REAL *result_d, int N, int Ninput, REAL *xnew_d, REAL *xs_d,
		    REAL *theta_d);

  // Computes the submatrix K_ij = Cov(xsnew(i,:), xs(j,:); theta)
  void cov_batch_gpu(REAL *result_d, int Nnew, int N, int Ninput, REAL *xsnew_d,
		     REAL *xs_d, REAL *theta_d);

  // Computes the matrix (dC/dtheta)(xs(i,:), ys(j,:); theta)
  void cov_deriv_theta_batch_gpu(REAL *result_d, int Ninput, int Nx, int Ny,
				 const REAL *xs_d, const REAL *ys_d,
				 const REAL *theta_d);

  // Computes the matrix (dC/dx)(xs(i,:), ys(j,:); theta)
  void cov_deriv_x_batch_gpu(REAL *result_d, int Ninput, int Nx, int Ny,
			     const REAL *xs_d, const REAL *ys_d,
			     const REAL *theta_d);

};


class Matern52Kernel : public BaseKernel {
public:

  // Squared exponential covariance function

  // Performs a single evaluation of the covariance function, for inputs
  // x and y; that is, Cov(x(:), y(:); theta).
  void cov_val_gpu(REAL *result_d, int Ninput, REAL *x_d, REAL *y_d,
		   REAL *theta_d);

  // Computes the vector k_i = Cov(xnew(:), xs(i,:); theta).
  void cov_all_gpu(REAL *result_d, int N, int Ninput, REAL *xnew_d, REAL *xs_d,
		   REAL *theta_d);

  // Computes the vector k_i = Cov(xnew(i,:), xs(i,:); theta).
  void cov_diag_gpu(REAL *result_d, int N, int Ninput, REAL *xnew_d, REAL *xs_d,
		    REAL *theta_d);

  // Computes the submatrix K_ij = Cov(xsnew(i,:), xs(j,:); theta)
  void cov_batch_gpu(REAL *result_d, int Nnew, int N, int Ninput, REAL *xsnew_d,
		     REAL *xs_d, REAL *theta_d);

  // Computes the matrix (dC/dtheta)(xs(i,:), ys(j,:); theta)
  void cov_deriv_theta_batch_gpu(REAL *result_d, int Ninput, int Nx, int Ny,
				 const REAL *xs_d, const REAL *ys_d,
				 const REAL *theta_d);

  // Computes the matrix (dC/dx)(xs(i,:), ys(j,:); theta)
  void cov_deriv_x_batch_gpu(REAL *result_d, int Ninput, int Nx, int Ny,
			     const REAL *xs_d, const REAL *ys_d,
			     const REAL *theta_d);

};

#endif
