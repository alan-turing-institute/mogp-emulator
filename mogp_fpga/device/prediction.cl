#define MAX_DIM 64
#define MAX_M 128
#define MAX_N 128

// Determine K(X,Y) for the squared exponential Kernel
//
// x is an (nx,dim) array, y is an (ny,dim) array both representing n vectors
// of length dim
//
// l is a (dim) array or scaling parameters. The difference between dimension
// i, for each pair of x and y vectors are divided by l[i]
//
// sigma is a prefactor, each element of K is multiplied by exp(sigma)
//
// The output is a (nx,ny) array which is written one column at a time to the
// pipe k
//
// nx, ny are the number of vectors in x and y respectively
// dim is the length of each of the vectors in x and y
kernel void sq_exp(global float* restrict x, global float* restrict y,
                   write_only pipe float __attribute__((blocking)) k,
                   global float* restrict l, float sigma, int nx, int ny,
                   int dim){
    // Cache the scaling factors
    float l_cache[MAX_DIM];
    for(unsigned i=0; i<MAX_DIM; i++){
        l_cache[i] = 1;
    }
    for(unsigned i=0; i<dim; i++){
        l_cache[i] = exp(-1.0f * l[i]);
    }

    // Determine exponential of sigma
    local float exp_sigma;
    exp_sigma = exp(sigma);

    // Calculate one column of k at a time
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
            temp[row] = exp_sigma*exp(-0.5f * value);
        }

        // Copy column of r to the pipe
        for(unsigned i=0; i<nx; i++){
            write_pipe(k, &temp[i]);
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
