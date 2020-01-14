#define MAX_DIM 64
#define MAX_NX 128
#define MAX_NXSTAR 128

// Determine K(X,X*) for the squared exponential Kernel
//
// x is an (nx,dim) array, xstar is an (nxstar,dim) array both representing n
// vectors of length dim
//
// scale is a (dim) array or scaling parameters. The difference between
// dimension i, for each pair of x and y vectors are divided by scale[i]
//
// sigma is a prefactor, each element of K is multiplied by exp(sigma)
//
// The output is a (nxstar,nx) array which is written one column at a time to
// the pipe k1, or k1 and k2  when var=1
//
// Optionally the pairwise distances may also be sent to the pipe r (for
// derivative calculation) when deriv=1
//
// nx, nxstar are the number of vectors in x and xstar respectively
// dim is the number of dimensions in each of the vectors of x and xstar
kernel void sq_exp(global float* restrict x, global float* restrict xstar,
                   write_only pipe float __attribute__((blocking)) k1,
                   write_only pipe float __attribute__((blocking)) k2,
                   write_only pipe float __attribute__((blocking)) r,
                   global float* restrict scale, float sigma,
                   int nx, int nxstar, int dim, int var, int deriv){
    // Cache the scaling factors
    local float l_cache[MAX_DIM];
    for(unsigned i=0; i<MAX_DIM; i++){
        l_cache[i] = 1;
    }
    for(unsigned i=0; i<dim; i++){
        l_cache[i] = exp(-1.0f * scale[i]);
    }

    // Determine exponential of sigma
    local float exp_sigma;
    exp_sigma = exp(sigma);

    // Calculate one column of k at a time
    for(unsigned col=0; col<nxstar; col++){
        int col_stride = col*dim;
        // Cache the corresponing vector from xstar
        float xstar_cache[MAX_DIM];
        for(unsigned i=0; i<dim; i++){
            xstar_cache[i] = xstar[col_stride+i];
        }

        // Declare local array for column of r
        for(unsigned row=0; row<nx; row++){
            int row_stride = row*dim;
            // Cache the corresponding vector from x
            float x_cache[MAX_DIM];
            for(unsigned i=0; i<dim; i++){
                x_cache[i] = x[row_stride+i];
            }

            // Determine the element k[row,col], the squared euclidean
            // distance between x[row] and xstar[col]
            float elem = 0;
            #pragma unroll
            for(unsigned i=0; i<MAX_DIM; i++){
                float x = x_cache[i];
                float xstar = xstar_cache[i];
                float difference = x - xstar;
                elem += (difference * difference) / l_cache[i];
            }

            // Calculate pairwise distance if needed for derivatives
            if (deriv){
                float dist = sqrt(elem);
                write_pipe(r, &dist);
            }

            // Calculate the element k[row,col] of the square exponential kernel
            elem = exp_sigma*exp(-0.5f * elem);
            // Push the element to the pipe
            write_pipe(k1, &elem);
            if (var){
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
    int nx, int nxstar
    ){
    // Copy invqt to local memory
    local float invqt_cache[MAX_NX];
    for (unsigned i=0; i<nx; i++){
        invqt_cache[i] = invqt[i];
    }

    // Calculate one element of c at at time by finding the 'dot product' of
    // one column of k with invqt
    for (unsigned col=0; col<nxstar; col++){
        float sum = 0;

        // Cache column of k
        float k_cache[MAX_NX];
        for (unsigned i=0; i<nx; i++){
            read_pipe(k1, &k_cache[i]);
        }

        #pragma unroll
        for (unsigned i=0; i<MAX_NX; i++){
            sum += k_cache[i] * invqt_cache[i];
        }
        expectation[col] = sum;
    }
}

// Determine the variance of predictions
kernel void variance(
    read_only pipe float __attribute__((blocking)) k2,
    global float* restrict variance,
    global float* restrict invq, float sigma, int nx, int nxstar
    ){
    local float invq_cache[MAX_NX*MAX_NX];
    for (unsigned i=0; i<nx; i++){
        int offset1 = i*nx;
        int offset2 = i*MAX_NX;
        for (unsigned j=0; j<nx; j++){
            invq_cache[offset2+j] = invq[offset1+j];
        }
    }
    // Take exponential of sigma
    local float exp_sigma;
    exp_sigma = exp(sigma);

    for (unsigned col=0; col<nxstar; col++){
        float sum = 0;

        // Cache column of k
        float k_cache[MAX_NX];
        for (unsigned i=0; i<nx; i++){
            read_pipe(k2, &k_cache[i]);
        }

        // Variance calculation
        // dot products of the column of k with each row of invQ
        // multiplied elementwise by the column of k
        // sum over
        float var = 0;
        for (unsigned row=0; row<nx; row++){
            int offset = row*MAX_NX;
            float dot_product = 0;
            #pragma unroll
            for (unsigned i=0; i<MAX_NX; i++){
                dot_product += k_cache[i] * invq_cache[offset+i];
            }
            var += dot_product * k_cache[row];
        }
        // Subtract from hyperparameter
        var = exp_sigma - var;
        variance[col] = max(var, 0.0f);
    }
}

// Determine the derivatives
//
// The pairwise distance matrix is passed, one column at a time, through the
// pipe r.
//
// The entire matrix is needed for parts of the calculation (?) so not all of
// this kernel can run concurrently with the others. However, the derivative of
// the kernel matrix with respect to r can be calculated as r values are read
// in.
kernel void derivatives(
        read_only pipe float __attribute__((blocking)) r,
        global float* restrict deriv,
        global float* restrict x, global float* restrict xstar,
        global float* restrict invqt, global float* restrict scale,
        float sigma, int nx, int nxstar, int dim){

    // Copy invqt to local memory
    local float invqt_cache[MAX_NX];
    for (unsigned i=0; i<nx; i++){
        invqt_cache[i] = invqt[i];
    }

    // Determine exponential of sigma
    local float exp_sigma;
    exp_sigma = exp(sigma);

    // Cache entire r matrix and calculate dK/dr
    local float r_cache[MAX_NX*MAX_NXSTAR];
    local float dKdr[MAX_NX*MAX_NXSTAR];
    local float dist;
    for (int i=0; i<nx*nxstar; i++){
        read_pipe(r, &r_cache[i]);
        dist = r_cache[i];
        dKdr[i] = -1.0 * dist * exp(-0.5 * dist * dist);
    }
    // At this point the exp_sq kernel has finished and determined all pairwise
    // distances

    // Calculate dr/dx
    // Set 0 values to 1
    #pragma unroll
    for (int i=0; i<MAX_NX*MAX_NXSTAR; i++){
        if (r_cache[i] == 0.0f){
            r_cache[i] = 1.0f;
        }
    }

    // For each dimension...
    for (int d=0; d<dim; d++){
        // Cache scaling parameter
        float scale_d = exp(scale[d]);

        // Cache dimension dim of all testing inputs x_star
        float xstar_cache[MAX_NXSTAR];
        for (int i=0; i<nxstar; i++){
            xstar_cache[i] = xstar[i*dim+d];
        }
        // Cache dimension dim of all training inputs x
        float x_cache[MAX_NX];
        for (int i=0; i<nx; i++){
            x_cache[i] = x[i*dim+d];
        }

        // Calculate drdx for dimension dim
        float drdx[MAX_NX*MAX_NXSTAR];
        for (int row=0; row<nxstar; row++){
            int row_stride = row*nx;

            for (int col=0; col<nx; col++){
                int index = row_stride + col;
                float value;
                // xstar - x in dimension dim only
                value = xstar_cache[row] - x_cache[col];
                value *= scale_d;
                value /= r_cache[index];
                drdx[index] = value;
            }
        }

        // Calculate dK/dx for dimension dim as the elementwise product of dK/dr
        // and dr/dx
        float dKdx[MAX_NX*MAX_NXSTAR];
        #pragma unroll
        for (int i=0; i<MAX_NX*MAX_NXSTAR; i++){
            dKdx[i] = exp_sigma * dKdr[i] * drdx[i];
        }

        // Calculate d(expectation)/dx for each expectation value for dimension
        // dim as the dot product of dK/dx and InvQt
        // These vectors are the columns of deriv
        for (int row=0; row<nxstar; row++){
            float sum = 0;
            int row_stride = row*nx;
            for (int col=0; col<nx; col++){
                sum += dKdx[row_stride+col] * invqt_cache[col];
            }
            deriv[row_stride+d] = sum;
        }
    }
}
