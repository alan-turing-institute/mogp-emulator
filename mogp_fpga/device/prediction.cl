#define MAX_DIM 64
#define MAX_M 128
#define MAX_N 128

// Determine the squared euclidean distance between each pair of vectors in x
// and y scaled by the array l
//
// x is an (nx,dim) array, y is an (ny,dim) array both representing n vectors
// of length dim
// l is a (dim) array or scaling parameters. The difference between dimension
// i, for each pair of x and y vectors are divided by l[i]
//
// The output is a (nx,ny) array which is written one row at a time to the pipe
// r
//
// nx, ny are the number of vectors in x and y respectively
// dim is the length of each of the vectors in x and y
kernel void distance(global float* restrict x, global float* restrict y,
                     write_only pipe float __attribute__((blocking)) r,
                     global float* restrict l, int nx, int ny, int dim){
    // Cache the scaling factors
    float l_cache[MAX_DIM];
    for(unsigned i=0; i<MAX_DIM; i++){
        l_cache[i] = 1;
    }
    for(unsigned i=0; i<dim; i++){
        l_cache[i] = exp(-1.0f * l[i]);
    }

    // Calculate one row of r at a time
    for(unsigned row=0; row<nx; row++){
        int row_stride = row*dim;
        // Cache the corresponding vector from x
        float x_cache[MAX_DIM];
        for(unsigned i=0; i<dim; i++){
            x_cache[i] = x[row_stride+i];
        }

        // Declare local array for row of r
        float temp[MAX_N];
        for(unsigned col=0; col<ny; col++){
            int col_stride = col*dim;
            // Cache the corresponding vector from y
            float y_cache[MAX_DIM];
            for(unsigned i=0; i<dim; i++){
                y_cache[i] = y[col_stride+i];
            }

            // Determine the value of r[row,col], the squared euclidean
            // distance between x[row] and y[col]
            float value = 0;
            #pragma unroll
            for(unsigned i=0; i<MAX_DIM; i++){
                float x = x_cache[i];
                float y = y_cache[i];
                float difference = x - y;
                value += (difference * difference) / l_cache[i];
            }

            // Store the value in the temporary row of r
            temp[col] = value;
        }

        // Copy row of r to the pipe
        int r_stride = row*ny;
        for(unsigned i=0; i<ny; i++){
            write_pipe(r, &temp[i]);
        }
    }
}

// Determine K(r) for the squared exponential Kernel
//
// r is a pipe from which the (m,n) matrix of squared distances is read
// k is a (m,n) matrix
// sigma is a prefactor, by which each element of k is multiplied
kernel void sq_exp(read_only pipe float __attribute__((blocking)) r,
                   global float* restrict k, float sigma, int m, int n){
    // Declare caches for r and k
    local float r_cache[MAX_N];
    local float k_cache[MAX_N];

    // Determine exponential of sigma
    local float exp_sigma;
    exp_sigma = exp(sigma);

    // Calculate K(r) one row at a time
    for (unsigned row=0; row<m; row++){
        unsigned offset = row*n;
        // Cache one row of r
        for (unsigned col=0; col<n; col++){
            read_pipe(r, &r_cache[col]);
        }

        // Calculate one row of K(r)
        #pragma unroll
        for (unsigned col=0; col<MAX_N; col++){
            float temp = r_cache[col];
            k_cache[col] = exp_sigma*exp(-0.5f * temp);
        }

        // Send one row of K(r) to the host
        for (unsigned col=0; col<n; col++){
            k[offset+col] = k_cache[col];
        }
    }
}

// Determine the product c=A^Tb where A is a (m,n) matrix, and b and c are (m)
// vectors (A^T means the transpose of the matrix A).
kernel void matrix_vector_product(global float* restrict a,
                                  global float* restrict b,
                                  global float* restrict c, int m, int n){
    // Copy b to local memory
    float b_cache[MAX_M];
    for (unsigned i=0; i<m; i++){
        b_cache[i] = b[i];
    }

    // Calculate one element of c at at time by finding the 'dot product' of
    // one column of a with b
    for (unsigned col=0; col<n; col++){
        float sum = 0;

        // Cache column of a
        float a_cache[MAX_M];
        for (unsigned i=0; i<m; i++){
            a_cache[i] = a[i*n+col];
        }

        #pragma unroll
        for (unsigned i=0; i<MAX_M; i++){
            sum += a_cache[i] * b_cache[i];
        }
        c[col] = sum;
    }
}
