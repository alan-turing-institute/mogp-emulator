#ifndef KERNEL_HPP
#define KERNEL_HPP

typedef double REAL;
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

/*
class Matern52Kernel : public Kernel {
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
*/
#endif
