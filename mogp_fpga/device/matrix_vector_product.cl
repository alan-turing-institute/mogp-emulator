// Determine the product c=Ab where A is a (m,n) matrix, and b and c are (n)
// vectors.

#define MAX_N 128

kernel void matrix_vector_product(global float* restrict a,
                                  global float* restrict b,
                                  global float* restrict c, int m, int n){

    // Copy b to local memory
    float b_cache[MAX_N];
    for (unsigned i=0; i<n; i++){
        b_cache[i] = b[i];
    }

    // Calculate one element of c at at time by finding the 'dot product' of
    // one row of a with b
    for (unsigned row=0; row<m; row++){
        float sum = 0;
        int offset = row*n;

        // Cache row of a
        float a_cache[MAX_N];
        for (unsigned i=0; i<n; i++){
            a_cache[i] = a[offset + i];
        }

        #pragma unroll
        for (unsigned i=0; i<MAX_N; i++){
            sum += a_cache[i] * b_cache[i];
        }
        c[row] = sum;
    }
}
