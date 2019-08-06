#ifndef COV_GPU_HPP
#define COV_GPU_HPP

typedef double REAL;
#define CUBLASDOT cublasDdot
#define CUBLASGEMV cublasDgemv

// squared exponential covariance function

void cov_all_wrapper(REAL *result, int N, int n_dim, REAL *xnew, REAL *xs, REAL *theta);

void cov_val_wrapper(REAL *result_d, int n_dim, REAL *x, REAL *y, REAL *hypers);

void cov_batch_wrapper(REAL *result, int Nnew, int N, int n_dim, REAL *xsnew, 
		       REAL *xs, REAL *theta);

#endif
