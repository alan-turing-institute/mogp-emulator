#include "cov_gpu.hpp"
#include <stdio.h>

//__device__ void cov_val_d(int n_dim, double *x, double *y, double *hypers_d);

__device__ REAL cov_val_d(int n_dim, REAL *x, REAL *y, REAL *hypers)
{
	REAL scale = hypers[0];
	//vec r(hypers.rows(1,hypers.n_rows-1));

	// bit of a hack.  Assumes a size on the vectors
	//vec r{hypers(1), hypers(1), hypers(1), hypers(2), hypers(2), hypers(2), hypers(3)};
	REAL r[] = {hypers[1], hypers[1], hypers[1], hypers[2], hypers[2], hypers[2], hypers[3]};

	REAL s = 0.0;
	for (unsigned i=0; i<n_dim; i++)
	{
		s+=pow(x[i]-y[i],REAL(2.0))/(r[i]*r[i]);
	}
	return scale * exp(-0.5*s);
}

__global__ void cov_val_d_wrapper(REAL *result_d, int n_dim, REAL *x, REAL *y, REAL *hypers)
{
	printf("hello from kernel!\n");
	*result_d = cov_val_d(n_dim,x,y,hypers);
	//result_d[0] = 5.0;
}

void cov_val_wrapper(REAL *result_d, int n_dim, REAL *x, REAL *y, REAL *hypers)
{
	cov_val_d_wrapper<<<1,1>>>(result_d, n_dim, x, y, hypers);
}
// static double dcov_x1(int n, vec x, vec y, vec hypers)
// {
// //	vec r(hypers.rows(1,hypers.n_rows-1));
// 	vec r{hypers(1), hypers(1), hypers(1), hypers(2), hypers(2), hypers(2), hypers(3)};

// 	return -(x(n)-y(n))/(r(n)*r(n)) * cov_val(x,y,hypers);
// }

// static double dcov_x2(int n, vec x, vec y, vec hypers)
// {
// 	return -dcov_x1(n,x,y,hypers);
// }

// static double d2cov_xx(int n, int m, vec x, vec y, vec hypers)
// {
// //	vec r(hypers.rows(1,hypers.n_rows-1));
// 	vec r{hypers(1), hypers(1), hypers(1), hypers(2), hypers(2), hypers(2), hypers(3)};

// 	return ((n==m)?(cov_val(x,y,hypers)/(r(n)*r(n))):0.0) 
// 		- (x(n)-y(n))/(r(n)*r(n)) * dcov_x2(m,x,y,hypers);
// }

// double cov(int n, int m, vec x, vec y, vec hypers)
// {
// 	if (n == 0 && m == 0) return cov_val(x,y,hypers);
// 	else if (n==0) return dcov_x2(m-1,x,y,hypers);
// 	else if (m==0) return dcov_x1(n-1,x,y,hypers);
// 	else return d2cov_xx(n-1,m-1,x,y,hypers);
// }

// Computes the vector of covariances with a new point with (the vector 'k' in the notation I have been using)
// could use thrust device vectors or similar
// just values for now -- fix when working
__global__ void cov_all_kernel(REAL *result, int N, int n_dim, REAL *xnew, REAL *xs, REAL *theta)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<N) {
		result[i] = cov_val_d(n_dim, xnew, xs + n_dim*i, theta);
	}
}

__global__ void cov_batch(REAL *result, int Nnew, int N, int n_dim, REAL *xsnew, 
			  REAL *xs, REAL *theta)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i<N && j<Nnew) {
		result[j+Nnew*i] = cov_val_d(n_dim, xsnew + n_dim*j, xs + n_dim*i, theta);
	}
}

// wrapper
void cov_all_wrapper(REAL *result, int N, int n_dim, REAL *xnew, REAL *xs, REAL *theta)
{
	const int threads_per_block = 256;
//	const int blocks =  (N + threads_per_block - 1) / threads_per_block; // round up
	cov_all_kernel <<< 10, threads_per_block >>> (result, N, n_dim, xnew, xs, theta);
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr,"GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

void cov_batch_wrapper(REAL *result, int Nnew, int N, int n_dim, REAL *xsnew, 
			  REAL *xs, REAL *theta)
{
	dim3 threads_per_block(8,32);
	dim3 blocks(250,625);
	cov_batch <<< blocks, threads_per_block >>> (result, Nnew, N, n_dim, xsnew, xs, theta);

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}
