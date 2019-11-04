// Determine K(r) for the squared exponential Kernel
//
// r is a (m,n) matrix of squared distances
// k is a (m,n) matrix

#define MAX_N 1024

kernel void sq_exp(global const float* restrict r, global float* restrict k,
                   int m, int n){
    // Declare caches for r and k
    local float r_cache[MAX_N];
    local float k_cache[MAX_N];

    // Calculate K(r) one row at a time
    for (unsigned row=0; row<m; row++){
        unsigned offset = row*n;
        // Cache one row of r
        for (unsigned col=0; col<n; col++){
            r_cache[col] = r[offset+col];
        }

        // Calculate one row of K(r)
        #pragma unroll
        for (unsigned col=0; col<MAX_N; col++){
            float temp = r_cache[col];
            k_cache[col] = exp(-0.5f * temp * temp);
        }

        // Send one row of K(r) to the host
        for (unsigned col=0; col<n; col++){
            k[offset+col] = r[offset+col];
        }
    }
}
