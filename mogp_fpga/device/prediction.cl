#define MAX_DIM 64
#define MAX_M 128
#define MAX_N 128

#define MODE_EXPECTATION 1
#define MODE_VARIANCE 2

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
// pipe k1 or k1 and k2 depending on the mode
//
// nx, ny are the number of vectors in x and y respectively
// dim is the length of each of the vectors in x and y
//
// mode sets whether to calculate only the expectation values (mode=1) or
// expectation values and variance (mode=2)
kernel void sq_exp(global float* restrict x, global float* restrict y,
                   write_only pipe float __attribute__((blocking)) k1,
                   write_only pipe float __attribute__((blocking)) k2,
                   global float* restrict l, float sigma, int nx, int ny,
                   int dim, int mode){
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
        for(unsigned row=0; row<nx; row++){
            int row_stride = row*dim;
            // Cache the corresponding vector from x
            float x_cache[MAX_DIM];
            for(unsigned i=0; i<dim; i++){
                x_cache[i] = x[row_stride+i];
            }

            // Determine the element r[row,col], the squared euclidean
            // distance between x[row] and y[col]
            float elem = 0;
            #pragma unroll
            for(unsigned i=0; i<MAX_DIM; i++){
                float x = x_cache[i];
                float y = y_cache[i];
                float difference = x - y;
                elem += (difference * difference) / l_cache[i];
            }

            // Calculate the element k[row,col] of the square exponential kernel
            elem = exp_sigma*exp(-0.5f * elem);
            // Push the element to the pipe
            if (mode == MODE_EXPECTATION){
                write_pipe(k1, &elem);
            } else if (mode == MODE_VARIANCE){
                write_pipe(k1, &elem);
                write_pipe(k2, &elem);
            }
        }
    }
}

// Determine the expectation values of predictions
kernel void expectation(
    read_only pipe float __attribute__((blocking)) k1,
    global float* restrict invqt,
    global float* restrict expectation,
    int m, int n
    ){
    // Copy invqt to local memory
    float invqt_cache[MAX_M];
    for (unsigned i=0; i<m; i++){
        invqt_cache[i] = invqt[i];
    }

    // Calculate one element of c at at time by finding the 'dot product' of
    // one column of k with invqt
    for (unsigned col=0; col<n; col++){
        float sum = 0;

        // Cache column of k
        float k_cache[MAX_M];
        for (unsigned i=0; i<m; i++){
            read_pipe(k1, &k_cache[i]);
        }

        #pragma unroll
        for (unsigned i=0; i<MAX_M; i++){
            sum += k_cache[i] * invqt_cache[i];
        }
        expectation[col] = sum;
    }
}

// Determine the variance of predictions
kernel void variance(
    read_only pipe float __attribute__((blocking)) k2,
    global float* restrict variance,
    global float* restrict invq, float sigma, int m, int n
    ){
    local float invq_cache[MAX_M*MAX_M];
    for (unsigned i=0; i<m; i++){
        int offset1 = i*m;
        int offset2 = i*MAX_M;
        for (unsigned j=0; j<m; j++){
            invq_cache[offset2+j] = invq[offset1+j];
        }
    }
    // Take exponential of sigma
    local float exp_sigma;
    exp_sigma = exp(sigma);

    for (unsigned col=0; col<n; col++){
        float sum = 0;

        // Cache column of k
        float k_cache[MAX_M];
        for (unsigned i=0; i<m; i++){
            read_pipe(k2, &k_cache[i]);
        }

        // Variance calculation
        // dot products of the column of k with each row of invQ
        // multiplied elementwise by the column of k
        // sum over
        float var = 0;
        for (unsigned row=0; row<m; row++){
            int offset = row*MAX_M;
            float dot_product = 0;
            #pragma unroll
            for (unsigned i=0; i<MAX_M; i++){
                dot_product += k_cache[i] * invq_cache[offset+i];
            }
            var += dot_product * k_cache[row];
        }
        // Subtract from hyperparameter
        var = exp_sigma - var;
        variance[col] = max(var, 0.0f);
    }
}
