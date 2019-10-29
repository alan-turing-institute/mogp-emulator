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
