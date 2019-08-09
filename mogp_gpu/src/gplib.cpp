#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gp_gpu.hpp"

struct gplib_handle {
    int status;
    DenseGP_GPU *gp;
    std::string message;
};

extern "C" {

    double gplib_hello_world(void)
    {
        return 0.1134;
    }
    
    // int gplib_check_cublas(void) {
    //     cublasHandle_t cublasHandle;
    //     cublasCreate(&cublasHandle);
    //     cublasStatus_t status;
        
    // }
    
    // Return an error code from the last operation performed
    int gplib_status(void *handle) {
        return static_cast<gplib_handle *>(handle)->status;
    }

    // if the status was non-zero, return a pointer to a c-string
    // corresponding to the error
    const char *gplib_error_string(void *handle) {
        return static_cast<gplib_handle *>(handle)->message.c_str();
    }

    void *gplib_make_gp(unsigned int N, unsigned int Ninput,
                        const double *theta, const double *xs, const double *ts,
                        const double *Q, const double *invQ,
                        const double *invQt)
    {
        auto handle = new gplib_handle;
        try {
            auto gp = new DenseGP_GPU(N, Ninput, theta, xs, ts, Q, invQ, invQt);
            handle->gp = gp;
            handle->status = 0;
        } catch(std::runtime_error& e) {
            handle->status = 1;
            handle->gp = nullptr;
            handle->message = e.what();
        }
        return handle;
    }
    
    void gplib_destroy_gp(void *handle)
    {
        gplib_handle *h = static_cast<gplib_handle *>(handle);
        if (h) {
            if (h->gp) delete h->gp;
            delete h;
        }
    }
    
    double gplib_predict(void *handle, double *xs)
    {
        DenseGP_GPU *gp = static_cast<gplib_handle *>(handle)->gp;
        return gp->predict(xs);
    }
    
// double gplib_predict_batch(void *handle, int Nnew, double *xs, double *result)
// {
//     return gp->predict
// }

} // extern "C"
