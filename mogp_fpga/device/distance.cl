// Determine the squared euclidean distance between each pair of vectors in x
// and y
//
// x is an (nx,dim) array, y is an (ny,dim) array
// Returns r, a (nx,ny) array
// nx, ny are the number of vectors in x and y respectively
// dim is the length of each of the vectors in x and y

#define MAX_DIM 64
#define MAX_N 1024

kernel void distance(global float* restrict x, global float* restrict y,
        global float* restrict r, int nx, int ny, int dim){

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
                value += difference * difference;
            }

            // Store the value in the temporary row of r
            temp[col] = value;
        }

        // Copy row of r back to global memory
        for(unsigned i=0; i<ny; i++){
            r[row_stride+i] = temp[i];
        }
    }
}
