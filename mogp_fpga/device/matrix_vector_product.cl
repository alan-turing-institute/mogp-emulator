// Determine the product c=A^Tb where A is a (m,n) matrix, and b and c are (m)
// vectors (A^T means the transpose of the matrix A).

#define MAX_M 128

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
