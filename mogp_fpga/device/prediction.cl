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
// The output is a (nx,ny) array which is written one column at a time to the
// pipe r
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

    // Calculate one column of r at a time
    for(unsigned col=0; col<ny; col++){
        int col_stride = col*dim;
        // Cache the corresponing vector from y
        float y_cache[MAX_DIM];
        for(unsigned i=0; i<dim; i++){
            y_cache[i] = y[col_stride+i];
        }

        // Declare local array for column of r
        float temp[MAX_M];
        for(unsigned row=0; row<nx; row++){
            int row_stride = row*dim;
            // Cache the corresponding vector from x
            float x_cache[MAX_DIM];
            for(unsigned i=0; i<dim; i++){
                x_cache[i] = x[row_stride+i];
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

            // Store the value in the temporary column of r
            temp[row] = value;
        }

        // Copy column of r to the pipe
        for(unsigned i=0; i<nx; i++){
            write_pipe(r, &temp[i]);
        }
    }
}

// Determine K(r) for the squared exponential Kernel
//
// r is a pipe from which the (m,n) matrix of squared distances is read one
// column at a time
// k is a (m,n) matrix
// sigma is a prefactor, by which each element of k is multiplied
kernel void sq_exp(read_only pipe float __attribute__((blocking)) r,
                   write_only pipe float __attribute__((blocking)) k,
                   float sigma, int m, int n){
    // Declare caches for r and k
    local float r_cache[MAX_M];
    local float k_cache[MAX_M];

    // Determine exponential of sigma
    local float exp_sigma;
    exp_sigma = exp(sigma);

    // Calculate K(r) one column at a time
    for (unsigned col=0; col<n; col++){
        // Cache one column of r
        for (unsigned row=0; row<m; row++){
            read_pipe(r, &r_cache[row]);
        }

        // Calculate one column of K(r)
        #pragma unroll
        for (unsigned row=0; row<MAX_M; row++){
            float temp = r_cache[row];
            k_cache[row] = exp_sigma*exp(-0.5f * temp);
        }

        // Send one column of K(r) to the pipe
        for (unsigned row=0; row<m; row++){
            write_pipe(k, &k_cache[row]);
        }
    }
}

// Determine the product c=K^Tb where K is a (m,n) matrix, and b and c are (m)
// vectors (K^T means the transpose of the matrix K).
kernel void matrix_vector_product(
    read_only pipe float __attribute__((blocking)) k,
    global float* restrict b, global float* restrict c, int m, int n
    ){
    // Copy b to local memory
    float b_cache[MAX_M];
    for (unsigned i=0; i<m; i++){
        b_cache[i] = b[i];
    }

    // Calculate one element of c at at time by finding the 'dot product' of
    // one column of k with b
    for (unsigned col=0; col<n; col++){
        float sum = 0;

        // Cache column of k
        float k_cache[MAX_M];
        for (unsigned i=0; i<m; i++){
            read_pipe(k, &k_cache[i]);
        }

        #pragma unroll
        for (unsigned i=0; i<MAX_M; i++){
            sum += k_cache[i] * b_cache[i];
        }
        c[col] = sum;
    }
}
