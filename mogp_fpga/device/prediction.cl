#define MAX_DIM 64
#define MAX_NX 128
#define MAX_NXSTAR 128

// Determine K(X,X*) for the squared exponential Kernel
//
// x is an (nx,dim) array representing nx training inputs of dimension dim
//
// xstar is an (nxstar,dim) array representing nxstar prediction inputs of
// dimension dim
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
//
// dim is the number of dimensions in each of the vectors of x and xstar
kernel void sq_exp(global float* restrict x, global float* restrict xstar,
                   write_only pipe float __attribute__((blocking)) k1,
                   write_only pipe float __attribute__((blocking)) k2,
                   write_only pipe float __attribute__((blocking)) r,
                   global float* restrict scale, float sigma,
                   int nx, int nxstar, int dim, int var, int deriv){

    // Determine exponential of sigma
    local float exp_sigma;
    exp_sigma = exp(sigma);
    
    local float x_;
    local float xstar_;
  
    // Calculate one column of k at a time
    for(unsigned col=0; col<nxstar; col++){
        int col_stride = col*dim;
        // Declare local array for column of r
        for(unsigned row=0; row<nx; row++){
            int row_stride = row*dim;
            // Determine the element k[row,col], the squared euclidean
            // distance between x[row] and xstar[col]
            float elem = 0;
            for(unsigned i=0; i<dim; i++){ 
                x_ = x[row_stride+i];
                xstar_ = xstar[col_stride+i];
                float difference = x_ - xstar_;
                elem += (difference * difference) * exp(scale[i]);
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
//
// k1 is a pipe which provides the square exponential kernel of X and X* in
// column major order
//
// invqt is an array of length nx calculated during training which is used to
// determine expectation values of the predictions.
// Q = K(X,X), invqt = Q^-1 * Y
//
// expectation is an array of length nxstar where the predictions' expectation
// values are written
//
// nx and nxstar, as before, are the number of inputs in X and X* respectively
kernel void expectation(
    read_only pipe float __attribute__((blocking)) k1,
    global float* restrict invqt,
    global float* restrict expectation,
    int nx, int nxstar
    ){
    // Calculate one predictions expectation value at time by finding the 'dot
    // product' of one column of k with invqt
    for (unsigned col=0; col<nxstar; col++){
        float sum = 0;
        for (unsigned i=0; i<nx; i++){ 
            float temp;
            read_pipe(k1, &temp); /*k_cache is only initialized for the first nx values. 
                                Assuming that MAX_NX is larger than nx, it means that the remaining addresses have no value. 
                                Also we don't need to have a separate array for k (no need for k_cache, because we use the values as soon as we read them and we don't need them later on.
                                So we reduced the number of for loops to 2 and got rid of the separate array. 
                                But the compiler is smart enough to see that k_cache was not used after so it wasn't occupying extra resources
                                in the final hardware.*/
            sum += temp * invqt[i];
        }
        expectation[col] = sum;
    }
}

// Determine the variance of predictions
//
// k2 is a pipe which provides the square exponential kernel of X and X* in
// column major order
//
// variance is an array of length nxstar where the predictions' variance
// values are written
//
// cholesky_factor is the (nx,nx) lower triangular cholesky factor of K(X,X)
//
// sigma, as before, is the prefactor in the squared exponential kernel
//
// nx and nxstar, as before, are the number of inputs in X and X* respectively
kernel void variance(
    read_only pipe float __attribute__((blocking)) k2,
    global float* restrict variance,
    global float* restrict cholesky_factor, float sigma, int nx, int nxstar
    ){
    local float chol_cache[MAX_NX*MAX_NX];
    for (unsigned i=0; i<nx; i++){
        int offset1 = i*nx;
        int offset2 = i*MAX_NX;
        for (unsigned j=0; j<nx; j++){
            chol_cache[offset2+j] = cholesky_factor[offset1+j];
        }
    }
    // Take exponential of sigma
    local float exp_sigma;
    exp_sigma = exp(sigma);

    // Calculate variance for one prediction per iteration
    for (unsigned col=0; col<nxstar; col++){
        // Cache column of k
        float k_cache[MAX_NX];
        float temp;
        for (unsigned i=0; i<nx; i++){
            read_pipe(k2, &k_cache[i]);
        }

        // Cholesky solve for one column of k
        //
        // Forward substitution
        float y[MAX_NX];
        float x[MAX_NX]; 
        float var = 0;        
        //#pragma unroll // Use of #pragma unroll, results in excessive use of the on-chip resources which prohibits the compile to finish
        for (unsigned i=0; i<MAX_NX; i++){
            y[i] = 0.0;
            x[i] = 0.0;
        }
        for (unsigned i=0; i<nx; i++){
            int offset = i*MAX_NX;
            //temp = k_cache[i]; //  the initialization cannot take place within the DSP block, therefore the register is placed outside, which results in a longer latency  
            // Solution
            float temp;
            temp = 0.0;
            for (unsigned t=0; t<MAX_NX; t++){
                int j = MAX_NX-t-1;
                //temp -= chol_cache[offset+j] * y[j];
                temp += chol_cache[offset+j] * y[j];
            }
            temp = k_cache[i] - temp;
            y[i] = temp / chol_cache[offset+i];
        }
        for (unsigned t=0; t<nx; t++){
            int i = nx-t-1;
            //temp = y[i];
            float temp;
            temp = 0.0;
            //#pragma unroll
            for (unsigned j=0; j<MAX_NX; j++){
                temp += chol_cache[j*MAX_NX+i] * x[j];
            }
            temp = y[i] - temp;
            x[i] = temp / chol_cache[i*MAX_NX+i];
        }

        for (unsigned i=0; i<MAX_NX; i++){
            var += k_cache[i] * x[i];
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
//
// r is a pipe which supplies the distance between each pair of (X, X*) in
// column major order (as with the k pipes)
//
// deriv is an (nxstar,dim) array giving the derivative of each prediction with
// respect to each dimension of the inputs X*
//
// x is an (nx,dim) array representing nx training inputs of dimension dim
//
// xstar is an (nxstar,dim) array representing nxstar prediction inputs of
// dimension dim
//
// invqt is an array of length nx calculated during training which is used to
// determine expectation values of the predictions.
// Q = K(X,X), invqt = Q^-1 * Y
//
// scale is a (dim) array or scaling parameters. The difference between
// dimension i, for each pair of x and y vectors are divided by scale[i]
//
// sigma, as before, is the prefactor in the squared exponential kernel
//
// nx and nxstar, as before, are the number of inputs in X and X* respectively
//
// dim is the number of dimensions in each of the vectors of x and xstar
kernel void derivatives(
        read_only pipe float __attribute__((blocking)) r,
        global float* restrict deriv,
        global float* restrict x, global float* restrict xstar,
        global float* restrict invqt, global float* restrict scale,
        float sigma, int nx, int nxstar, int dim){

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
    // For each dimension...
    #pragma max_concurrency 1  //used to limit the concurrency of the loop to 1, in order to reduce the amount of RAMs used. The max_concurrency pragma enables you to control the on-chip memory resources required to pipeline your loop. 
    for (int d=0; d<dim; d++){
        // Cache scaling parameter
        float scale_d = exp(scale[d]);
        // Calculate dr/dx for dimension dim
        float drdx[MAX_NX*MAX_NXSTAR];
        for (int row=0; row<nxstar; row++){
            int row_stride = row*nx;
            
            for (int col=0; col<nx; col++){
                int index = row_stride + col;
                float value;
                float x_;
                float xstar_;
                x_ = x[col*dim+d];
                xstar_= xstar[row*dim+d];
                // xstar - x in dimension dim only
                value = xstar_ - x_;
                value *= scale_d;
                value /= r_cache[index];
                drdx[index] = value;
            }
        }

        // Calculate dK/dx for dimension dim using the chain rule as the
        // elementwise product of dK/dr and dr/dx
        float dKdx[MAX_NX*MAX_NXSTAR];
        //#pragma unroll
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
                sum += dKdx[row_stride+col] * invqt[col];
            }
            deriv[row_stride+d] = sum;
        }
    }
}