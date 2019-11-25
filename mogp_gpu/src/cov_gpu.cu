#include "cov_gpu.hpp"
#include <stdio.h>

// Covariance device function
__device__ REAL cov_val_d(int Ninput, REAL *x_d, REAL *y_d, REAL *theta_d)
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
__global__ void cov_val_kernel(REAL *result_d, int Ninput, REAL *x_d,
                               REAL *y_d, REAL *theta_d)
{
    *result_d = cov_val_d(Ninput, x_d, y_d, theta_d);
}

void cov_val_gpu(REAL *result_d, int Ninput, REAL *x_d, REAL *y_d,
                 REAL *theta_d)
{
    cov_val_kernel<<<1,1>>>(result_d, Ninput, x_d, y_d, theta_d);
}

////////////////////
__global__ void cov_diag_kernel(REAL *result_d, int N, int Ninput, REAL *xnew_d,
                                REAL *xs, REAL *theta_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        result_d[i] = cov_val_d(Ninput, xnew_d + Ninput * i, xs + Ninput * i,
                                theta_d);
    }
}

void cov_diag_gpu(REAL *result_d, int N, int Ninput, REAL *xnew_d, REAL *xs_d,
                  REAL *theta_d)
{
    const int threads_per_block = 256;
    cov_diag_kernel<<<10, threads_per_block>>>(
        result_d, N, Ninput, xnew_d, xs_d, theta_d);
}

////////////////////
__global__ void cov_all_kernel(REAL *result_d, int N, int Ninput, REAL *xnew_d,
                               REAL *xs_d, REAL *theta_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        result_d[i] = cov_val_d(Ninput, xnew_d, xs_d + Ninput * i, theta_d);
    }
}

void cov_all_gpu(REAL *result_d, int N, int Ninput, REAL *xnew_d, REAL *xs_d,
                 REAL *theta_d)
{
    const int threads_per_block = 256;
    cov_all_kernel<<<10, threads_per_block>>>(
        result_d, N, Ninput, xnew_d, xs_d, theta_d);
}

////////////////////
__global__ void cov_batch_kernel(REAL *result_d, int Nnew, int N, int Ninput,
                                 REAL *xsnew_d, REAL *xs_d, REAL *theta_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < Nnew)
    {
        result_d[j + Nnew * i] =
            cov_val_d(Ninput, xsnew_d + Ninput * j, xs_d + Ninput * i, theta_d);
    }
}

void cov_batch_gpu(REAL *result_d, int Nnew, int N, int Ninput, REAL *xsnew_d,
                   REAL *xs_d, REAL *theta_d)
{
    dim3 threads_per_block(8, 32);
    dim3 blocks(250, 625);
    cov_batch_kernel<<<blocks, threads_per_block>>>(
        result_d, Nnew, N, Ninput, xsnew_d, xs_d, theta_d);
}

////////////////////
__device__ void cov_deriv(REAL *result_d, int Ninput, const REAL *x_d,
                          const REAL *y_d, const REAL *theta_d)
{
    REAL s = 0.0;
    REAL exp_thetaN = exp(theta_d[Ninput]);
    for (unsigned int i=0; i < Ninput; i++)
    {
        REAL d_i = x_d[i] - y_d[i];
        REAL a = d_i * d_i * exp(theta_d[i]);
        s += a;
        result_d[i] = -0.5 * exp_thetaN * a;
    }
    for (unsigned int i=0; i < Ninput; i++)
    {
        result_d[i] *= s;
    }
    result_d[Ninput] = exp_thetaN * exp(-0.5 * s);   
}

__global__ void cov_deriv_batch_kernel(
    REAL *result_d, int Ninput, int Nx, int Ny, const REAL *xs_d,
    const REAL *ys_d, const REAL *theta_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Nx && j < Ny)
    {
        cov_deriv(result_d + (Ninput + 1) * (j + Nx * j), Ninput,
                  xs_d + Ninput * j, ys_d + Ninput * i, theta_d);
    }
}

void cov_deriv_batch_gpu(
    REAL *result_d, int Ninput, int Nx, int Ny, const REAL *xs_d,
    const REAL *ys_d, const REAL *theta_d)
{
    const int Bx = 16, By = 16;
    dim3 threads_per_block(Bx, By);
    dim3 blocks(Nx / Bx, Ny / By);
    cov_deriv_batch_kernel<<<blocks, threads_per_block>>>(
        result_d, Ninput, Nx, Ny, xs_d, ys_d, theta_d);
}
