#include "util.hpp"

// Create an NxN Identity matrix A on the device
__global__ void identity_kernel(int N, double *A)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i < N && j < N) A[i + N * j] = (i == j)?1.0:0.0;
}

void identity_device(int N, double *A)
{
    const int Nx = 8, Ny = 8;
    dim3 threads_per_block(Nx, Ny);
    dim3 blocks((N + Nx - 1)/Nx, (N + Ny - 1)/Ny);
    identity_kernel<<<blocks, threads_per_block>>>(N, A);
}

// Add b to the diagonal of NxN matrix A
__global__ void add_diagonal_kernel(int N, double b, double *A)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i < N && j < N && i == j) A[i + N * j] += b;
}

void add_diagonal(int N, double b, double *A)
{
    const int Nx = 8, Ny = 8;
    dim3 threads_per_block(Nx, Ny);
    dim3 blocks((N + Nx - 1)/Nx, (N + Ny - 1)/Ny);
    add_diagonal_kernel<<<blocks, threads_per_block>>>(N, b, A);
}

// CUDA kernel for summing log diagonal elements of a matrix,
// using cub::DeviceReduce

struct LogSq : public thrust::unary_function<double, double>
{
    __host__ __device__ double operator()(double x) const { return 2.0 * log(x); }
};

void sum_log_diag(int N, double *A, double *result, double *work, size_t work_size)
{
    auto transform_it = thrust::make_transform_iterator(A, LogSq());
    auto sr = make_strided_range(transform_it, transform_it + N*N, N+1);

    cub::DeviceReduce::Sum(work, work_size, sr.begin(), result, N);
}

// Implementation of trace using cub::DeviceReduce

void trace(int N, double *A, double *result, double *work, size_t work_size)
{
    auto sr = make_strided_range(A, A + N*N, N+1);
    cub::DeviceReduce::Sum(work, work_size, sr.begin(), result, N);
}