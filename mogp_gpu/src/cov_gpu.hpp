#ifndef COV_GPU_HPP
#define COV_GPU_HPP

typedef double REAL;
#define CUBLASDOT cublasDdot
#define CUBLASGEMV cublasDgemv

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
void cov_deriv_batch_gpu(REAL *result_d, int Ninput, int Nx, int Ny,
                         const REAL *xs_d, const REAL *ys_d,
                         const REAL *theta_d);

#endif

