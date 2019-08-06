#include <iostream>
#include <cuda_runtime.h>
#include "gp_gpu.hpp"

extern "C" {

double gplib_hello_world(void) { return 0.1134; }

void *gplib_make_gp(unsigned int N, unsigned int Ninput, const double *theta,
                    const double *xs, const double *ts, const double *Q,
                    const double *invQ, const double *invQt)
{
    return new DenseGP_GPU(N, Ninput, theta, xs, ts, Q, invQ, invQt);
}
    
void gplib_destroy_gp(void *handle)
{
    delete static_cast<DenseGP_GPU *>(handle);
}

double gplib_predict(void *handle, double *xs)
{
    std::cout << "Calling gplib_predict" << std::endl;
    DenseGP_GPU *gp = static_cast<DenseGP_GPU *>(handle);
    return gp->predict(xs);
}

} // extern "C"
