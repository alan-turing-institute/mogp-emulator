#include "kernel.hpp"
#include <stdio.h>
#include <iostream>

/// Device functions, and global functions (also, confusingly called "kernels") preceded by
/// "sqexp" for the SquaredExponential kernel, and "mat" for the Matern52 kernel.


// Covariance device function
__device__ REAL sqexp_cov_val_d(int Ninput, REAL *x_d, REAL *y_d, REAL *theta_d)
{
    REAL s = 0.0;
    for (unsigned int i=0; i < Ninput; i++)
    {
        REAL d_i = x_d[i] - y_d[i];
        s += d_i * d_i * exp(theta_d[i]);
    }
    return exp(-0.5 * s + theta_d[Ninput]);
}

////////////////////
__global__ void sqexp_cov_val_kernel(REAL *result_d, int Ninput, REAL *x_d,
                               REAL *y_d, REAL *theta_d)
{
    *result_d = sqexp_cov_val_d(Ninput, x_d, y_d, theta_d);
}


////////////////////
__global__ void sqexp_cov_diag_kernel(REAL *result_d, int N, int Ninput, REAL *xnew_d,
                                REAL *xs, REAL *theta_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        result_d[i] = sqexp_cov_val_d(Ninput, xnew_d + Ninput * i, xs + Ninput * i,
                                theta_d);
    }
}



////////////////////
__global__ void sqexp_cov_all_kernel(REAL *result_d, int N, int Ninput, REAL *xnew_d,
                               REAL *xs_d, REAL *theta_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        result_d[i] = sqexp_cov_val_d(Ninput, xnew_d, xs_d + Ninput * i, theta_d);
    }
}



////////////////////
__global__ void sqexp_cov_batch_kernel(REAL *result_d, int Nnew, int N, int Ninput,
                                 REAL *xsnew_d, REAL *xs_d, REAL *theta_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < Nnew)
    {
        result_d[j + Nnew * i] =
            sqexp_cov_val_d(Ninput, xsnew_d + Ninput * j, xs_d + Ninput * i, theta_d);
    }
}


////////////////////
__device__ void sqexp_cov_deriv_x(REAL *result_d, int Ninput,
			    const REAL *x_d, const REAL *y_d,
			    const REAL *theta_d)
{
    REAL s = 0.0;
    for (unsigned int i=0; i < Ninput; i++)
    {
        REAL d_i = x_d[i] - y_d[i];
        REAL a = d_i * exp(theta_d[i]);
	result_d[i] = a;
        s += d_i * a;
    }
    REAL c = -exp(-0.5*s + theta_d[Ninput]);
    for (unsigned int i=0; i < Ninput; i++)
    {
        result_d[i] *= c;
    }
}

__global__ void sqexp_cov_deriv_x_batch_kernel(
    REAL *result_d, int Ninput, int Nx, int Ny, const REAL *xs_d,
    const REAL *ys_d, const REAL *theta_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Nx && j < Ny)
    {
        sqexp_cov_deriv_x(result_d + Ninput * (Nx * j + i),
		    Ninput, xs_d + Ninput * i, ys_d + Ninput * j,
		    theta_d);
    }
}




////////////////////
__device__ void sqexp_cov_deriv_theta(REAL *result_d, int result_stride, int Ninput,
                          const REAL *x_d, const REAL *y_d,
                          const REAL *theta_d)
{
    REAL s = 0.0;
    REAL exp_thetaN = exp(theta_d[Ninput]);
    for (unsigned int i=0; i < Ninput; i++)
    {
        REAL d_i = x_d[i] - y_d[i];
        REAL a = d_i * d_i * exp(theta_d[i]);
        s += a;
        result_d[i * result_stride] = -0.5 * a;
    }
    s = exp_thetaN * exp(-0.5 * s);
    for (unsigned int i=0; i < Ninput; i++)
    {
        result_d[i * result_stride] *= s;
    }
    result_d[Ninput * result_stride] = s;
}

__global__ void sqexp_cov_deriv_theta_batch_kernel(
    REAL *result_d, int Ninput, int Nx, int Ny, const REAL *xs_d,
    const REAL *ys_d, const REAL *theta_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int result_stride = Nx * Ny;
    if (i < Nx && j < Ny)
    {
        sqexp_cov_deriv_theta(result_d + Ny * i + j,
			      result_stride,
			      Ninput, xs_d + Ninput * j, ys_d + Ninput * i,
			      theta_d);
    }
}


// Implementation of the SquaredExponentialKernel

void SquaredExponentialKernel::cov_val_gpu(REAL *result_d, int Ninput, REAL *x_d, REAL *y_d,
                 REAL *theta_d)
{
    sqexp_cov_val_kernel<<<1,1>>>(result_d, Ninput, x_d, y_d, theta_d);
}

void SquaredExponentialKernel::cov_all_gpu(REAL *result_d, int N, int Ninput, REAL *xnew_d, REAL *xs_d,
                 REAL *theta_d)
{
    const int threads_per_block = 256;
    sqexp_cov_all_kernel<<<10, threads_per_block>>>(
        result_d, N, Ninput, xnew_d, xs_d, theta_d);
}


void SquaredExponentialKernel::cov_diag_gpu(REAL *result_d, int N, int Ninput, REAL *xnew_d, REAL *xs_d,
                  REAL *theta_d)
{
    const int threads_per_block = 256;
    sqexp_cov_diag_kernel<<<10, threads_per_block>>>(
        result_d, N, Ninput, xnew_d, xs_d, theta_d);
}


void SquaredExponentialKernel::cov_batch_gpu(REAL *result_d, int Nnew, int N, int Ninput, REAL *xsnew_d,
                   REAL *xs_d, REAL *theta_d)
{
    dim3 threads_per_block(8, 32);
    dim3 blocks(250, 625);
    sqexp_cov_batch_kernel<<<blocks, threads_per_block>>>(
	result_d, Nnew, N, Ninput, xsnew_d, xs_d, theta_d
							  );
}


void SquaredExponentialKernel::cov_deriv_x_batch_gpu(
    REAL *result_d, int Ninput, int Nx, int Ny, const REAL *xs_d,
    const REAL *ys_d, const REAL *theta_d)
{
    const int Bx = 16, By = 16;
    dim3 threads_per_block(Bx, By);
    dim3 blocks((Nx + Bx - 1)/Bx, (Ny + By - 1)/By);
    sqexp_cov_deriv_x_batch_kernel<<<blocks, threads_per_block>>>(
        result_d, Ninput, Nx, Ny, xs_d, ys_d, theta_d);
}


void SquaredExponentialKernel::cov_deriv_theta_batch_gpu(
    REAL *result_d, int Ninput, int Nx, int Ny, const REAL *xs_d,
    const REAL *ys_d, const REAL *theta_d)
{
    const int Bx = 16, By = 16;
    dim3 threads_per_block(Bx, By);
    dim3 blocks((Nx + Bx - 1)/Bx, (Ny + By - 1)/By);
    sqexp_cov_deriv_theta_batch_kernel<<<blocks, threads_per_block>>>(
        result_d, Ninput, Nx, Ny, xs_d, ys_d, theta_d);
}


////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Matern52 /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

// Covariance device function
__device__ REAL mat52_cov_val_d(int Ninput, REAL *x_d, REAL *y_d, REAL *theta_d)
{
    REAL s = 0.0;
    for (unsigned int i=0; i < Ninput; i++)
    {
        REAL d_i = x_d[i] - y_d[i];
        s += d_i * d_i * exp(theta_d[i]);
    }
    REAL r = sqrt(s);
    return (1 + r*sqrt(5.) + (5./3.)*s) *  exp(theta_d[Ninput] - sqrt(5.)*r);
}

////////////////////
__global__ void mat52_cov_val_kernel(REAL *result_d, int Ninput, REAL *x_d,
                               REAL *y_d, REAL *theta_d)
{
    *result_d = mat52_cov_val_d(Ninput, x_d, y_d, theta_d);
}

////////////////////
__global__ void mat52_cov_diag_kernel(REAL *result_d, int N, int Ninput, REAL *xnew_d,
                                REAL *xs, REAL *theta_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        result_d[i] = mat52_cov_val_d(Ninput, xnew_d + Ninput * i, xs + Ninput * i,
				    theta_d);
    }
}

////////////////////
__global__ void mat52_cov_all_kernel(REAL *result_d, int N, int Ninput, REAL *xnew_d,
				   REAL *xs_d, REAL *theta_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        result_d[i] = mat52_cov_val_d(Ninput, xnew_d, xs_d + Ninput * i, theta_d);
    }
}

////////////////////
__global__ void mat52_cov_batch_kernel(REAL *result_d, int Nnew, int N, int Ninput,
				     REAL *xsnew_d, REAL *xs_d, REAL *theta_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < Nnew)
    {
        result_d[j + Nnew * i] =
            mat52_cov_val_d(Ninput, xsnew_d + Ninput * j, xs_d + Ninput * i, theta_d);
    }
}

////////////////////
__device__ void mat52_cov_deriv_x(REAL *result_d, int Ninput,
			    const REAL *x_d, const REAL *y_d,
			    const REAL *theta_d)
{
  // dK/dx = dr/dx * dK/dr

  //drdx
    REAL s = 0.0;
    for (unsigned int i=0; i < Ninput; i++)
    {
        REAL d_i = x_d[i] - y_d[i];
        REAL a = d_i * exp(theta_d[i]);
	result_d[i] = a;
        s += d_i * a;
    }
    REAL r = sqrt(s);
    if (r > 0.) {
      REAL dkdr = -5./3.*r*(1+sqrt(5.)*r)*exp(-1.*sqrt(5.)*r);
      dkdr *= exp(theta_d[Ninput]);
      for (unsigned int i=0; i < Ninput; i++)
	{
	  result_d[i] = dkdr * result_d[i] / r;
	}
    }
}

__global__ void mat52_cov_deriv_x_batch_kernel(
    REAL *result_d, int Ninput, int Nx, int Ny, const REAL *xs_d,
    const REAL *ys_d, const REAL *theta_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Nx && j < Ny)
    {
        mat52_cov_deriv_x(result_d + Ninput * (Nx * j + i),
		    Ninput, xs_d + Ninput * i, ys_d + Ninput * j,
		    theta_d);
    }
}

////////////////////
__device__ void mat52_cov_deriv_theta(REAL *result_d, int result_stride, int Ninput,
                          const REAL *x_d, const REAL *y_d,
                          const REAL *theta_d)
{

    //dK/dtheta = exp(thetaN) * dK/dr * dr/dtheta
    REAL exp_thetaN = exp(theta_d[Ninput]);
    REAL s = 0.0;
    for (unsigned int i=0; i < Ninput; i++)
    {
        REAL d_i = x_d[i] - y_d[i];
        REAL a = d_i * d_i * exp(theta_d[i]);
	result_d[i * result_stride] = 0.5 * a;
        s += a;
    }
    REAL r = sqrt(s);
    REAL dkdr = 0.;
    if (r > 0.) {
      dkdr = -5./3.*r*(1+sqrt(5.)*r)*exp(-1.*sqrt(5.)*r);
      dkdr *= exp_thetaN;
    }

    for (unsigned int i=0; i < Ninput; i++)
    {
      if (r > 0.) result_d[i * result_stride] = dkdr * result_d[i * result_stride] / r;
    }
    result_d[Ninput * result_stride] = (1 + r*sqrt(5.) + (5./3.)*s) *  exp(theta_d[Ninput] - sqrt(5.)*r);
}

__global__ void mat52_cov_deriv_theta_batch_kernel(
    REAL *result_d, int Ninput, int Nx, int Ny, const REAL *xs_d,
    const REAL *ys_d, const REAL *theta_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int result_stride = Nx * Ny;
    if (i < Nx && j < Ny)
    {
        mat52_cov_deriv_theta(result_d + Ny * i + j,
			      result_stride,
			      Ninput, xs_d + Ninput * j, ys_d + Ninput * i,
			      theta_d);
    }
}

/// implementation of Matern52 kernel

void Matern52Kernel::cov_val_gpu(REAL *result_d, int Ninput, REAL *x_d, REAL *y_d,
                 REAL *theta_d)
{
    mat52_cov_val_kernel<<<1,1>>>(result_d, Ninput, x_d, y_d, theta_d);
}

void Matern52Kernel::cov_all_gpu(REAL *result_d, int N, int Ninput, REAL *xnew_d, REAL *xs_d,
                 REAL *theta_d)
{
    const int threads_per_block = 256;
    mat52_cov_all_kernel<<<10, threads_per_block>>>(
        result_d, N, Ninput, xnew_d, xs_d, theta_d);
}


void Matern52Kernel::cov_diag_gpu(REAL *result_d, int N, int Ninput, REAL *xnew_d, REAL *xs_d,
                  REAL *theta_d)
{
    const int threads_per_block = 256;
    mat52_cov_diag_kernel<<<10, threads_per_block>>>(
        result_d, N, Ninput, xnew_d, xs_d, theta_d);
}


void Matern52Kernel::cov_batch_gpu(REAL *result_d, int Nnew, int N, int Ninput, REAL *xsnew_d,
                   REAL *xs_d, REAL *theta_d)
{
    dim3 threads_per_block(8, 32);
    dim3 blocks(250, 625);
    mat52_cov_batch_kernel<<<blocks, threads_per_block>>>(
	result_d, Nnew, N, Ninput, xsnew_d, xs_d, theta_d
							  );
}


void Matern52Kernel::cov_deriv_x_batch_gpu(
    REAL *result_d, int Ninput, int Nx, int Ny, const REAL *xs_d,
    const REAL *ys_d, const REAL *theta_d)
{
    const int Bx = 16, By = 16;
    dim3 threads_per_block(Bx, By);
    dim3 blocks((Nx + Bx - 1)/Bx, (Ny + By - 1)/By);
    mat52_cov_deriv_x_batch_kernel<<<blocks, threads_per_block>>>(
        result_d, Ninput, Nx, Ny, xs_d, ys_d, theta_d);
}


void Matern52Kernel::cov_deriv_theta_batch_gpu(
    REAL *result_d, int Ninput, int Nx, int Ny, const REAL *xs_d,
    const REAL *ys_d, const REAL *theta_d)
{
    const int Bx = 16, By = 16;
    dim3 threads_per_block(Bx, By);
    dim3 blocks((Nx + Bx - 1)/Bx, (Ny + By - 1)/By);
    mat52_cov_deriv_theta_batch_kernel<<<blocks, threads_per_block>>>(
        result_d, Ninput, Nx, Ny, xs_d, ys_d, theta_d);
}