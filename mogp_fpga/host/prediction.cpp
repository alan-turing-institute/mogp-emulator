#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include "CL/cl2.hpp"
#include "CL/cl.h"
#include <cmath>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <vector>

#define MAX_NX 128
#define MAX_NXSTAR 128

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<float>);

void compare_results(std::vector<float> expected, std::vector<float> actual,
                     std::string kernel_name){
    float TOL = 1.0e-6;
    int length = expected.size();
    bool discrepency = false;

    std::cout << "Comparing expected and actual results for " << kernel_name << std::endl;
    for (int i=0; i<length; i++){
        if (std::abs(expected[i] - actual[i]) > TOL){
            std::cout << "Element " << i << " of expected and actual results do not agree: " << expected[i] << " " << actual[i] << std::endl;
            discrepency = true;
        }
    }

    if (!discrepency){
        std::cout << "Expected and actual results agree" << std::endl;
    }
}


struct CLContainer{
    cl::Context context;
    std::vector<cl::Device> devices;
    cl::Program program;
};


CLContainer default_container(const char* path){
        // Create context using default device
        cl::Context context(CL_DEVICE_TYPE_DEFAULT);

        // Get devices
        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        // Create queue from binary
        FILE* fp;
        fp = fopen(path, "rb");
        if(fp == 0) {
            throw cl::Error(0, "can't open kernel binary file");
        }
        // Get size of binary
        fseek(fp, 0, SEEK_END);
        size_t binary_size = ftell(fp);
        // Read binary as void*
        std::vector<unsigned char> binary(binary_size);
        rewind(fp);
        if (fread(&(binary[0]), binary_size, 1, fp) == 0) {
            fclose(fp);
            throw cl::Error(0, "error while reading kernel binary");
        }
        cl::Program::Binaries binaries(1, binary);

        // Create program
        cl::Program program(context, devices, binaries);

        auto container = CLContainer();
        container.context = context;
        container.devices = devices;
        container.program = program;

        return container;
}


// Conduct a single prediction using the square exponetial kernel
// X - Training inputs
// nx - Number of training inputs
// dim - The number of input dimensions (length of X and Xstar records)
// Xstar - Testing inputs
// nx_star - Number of testing inputs
// scale - Per dimension scaling factors for pairwise distances)
// sigma - Kernel scaling parameter
// InvQt - Vector, calculated during training, used for expecation value
//         calculation (Q = K(X,X), InvQt = Q^-1 * Y)
// Ystar - Training prediction expectation values
// context - The OpenCL Context
// program - The OpenCL Program
void predict_single_expectation(
        std::vector<float> &X, int nx, int dim,
        std::vector<float> &Xstar, int nxstar,
        std::vector<float> &scale, float sigma,
        std::vector<float> &InvQt,
        std::vector<float> &Ystar,
        CLContainer &container){
    cl::Context context = container.context;
    cl::Program program = container.program;

    // Create queues
    cl::CommandQueue queue1(context);
    cl::CommandQueue queue2(context);

    // Create kernel functor for square exponential kernel
    auto square_exponential = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Pipe&, cl::Pipe&, cl::Pipe&, cl::Buffer, float, int, int, int, int, int>(program, "sq_exp");
    // Create kernel functor for expectation kernel
    auto expectation = cl::KernelFunctor<cl::Pipe&, cl::Buffer, cl::Buffer, int, int>(program, "expectation");

    // Create device variables
    cl::Buffer d_X(X.begin(), X.end(), true);
    cl::Buffer d_Xstar(Xstar.begin(), Xstar.end(), true);
    cl::Buffer d_scale(scale.begin(), scale.end(), true);
    cl::Buffer d_InvQt(InvQt.begin(), InvQt.end(), true);
    cl_int status;
    cl_mem pipe_k = clCreatePipe(context(), 0, sizeof(cl_float), MAX_NX, NULL, &status);
    cl::Pipe k(pipe_k);
    cl_mem pipe_dummy = clCreatePipe(context(), 0, sizeof(int), 0, NULL, &status);
    cl::Pipe dummy(pipe_dummy);
    cl::Buffer d_Ystar(Ystar.begin(), Ystar.end(), false);

    // Prediction
    square_exponential(cl::EnqueueArgs(queue1, cl::NDRange(1)), d_X,
                       d_Xstar, k, dummy, dummy, d_scale, sigma, nx, nxstar,
                       dim, 0, 0);
    expectation(cl::EnqueueArgs(queue2, cl::NDRange(1)), k, d_InvQt, d_Ystar, nx,
                nxstar);
    queue1.finish();
    queue2.finish();

    // Retreive expectation values
    cl::copy(d_Ystar, Ystar.begin(), Ystar.end());
}


// Conduct a single prediction using the square exponetial kernel
// X - Training inputs
// nx - Number of training inputs
// dim - The number of input dimensions (length of X and Xstar records)
// Xstar - Testing inputs
// nx_star - Number of testing inputs
// scale - Per dimension scaling factors for pairwise distances)
// sigma - Kernel scaling parameter
// InvQt - Vector, calculated during training, used for expecation value
//         calculation (Q = K(X,X), InvQt = Q^-1 * Y)
// Q_chol - Lower triangular cholesky factor of K(X,X), used for variance
//          calculation
// Ystar - Training prediction expectation values
// Ystarvar - Training prediction variances
// context - The OpenCL Context
// program - The OpenCL Program
void predict_single_variance(
        std::vector<float> &X, int nx, int dim,
        std::vector<float> &Xstar, int nxstar,
        std::vector<float> &scale, float sigma,
        std::vector<float> &InvQt, std::vector<float> &Q_chol,
        std::vector<float> &Ystar, std::vector<float> &Ystarvar,
        CLContainer &container){
    cl::Context context = container.context;
    cl::Program program = container.program;

    // Create queues
    cl::CommandQueue queue1(context);
    cl::CommandQueue queue2(context);
    cl::CommandQueue queue3(context);

    // Create kernel functor for square exponential kernel
    auto square_exponential = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Pipe&, cl::Pipe&, cl::Pipe&, cl::Buffer, float, int, int, int, int, int>(program, "sq_exp");
    // Create kernel functor for expectation kernel
    auto expectation = cl::KernelFunctor<cl::Pipe&, cl::Buffer, cl::Buffer, int, int>(program, "expectation");
    // Create kernel functor for variance kernel
    auto variance = cl::KernelFunctor<cl::Pipe&, cl::Buffer, cl::Buffer, float, int, int>(program, "variance");

    // Create device variables
    cl::Buffer d_X(X.begin(), X.end(), true);
    cl::Buffer d_Xstar(Xstar.begin(), Xstar.end(), true);
    cl::Buffer d_scale(scale.begin(), scale.end(), true);
    cl::Buffer d_InvQt(InvQt.begin(), InvQt.end(), true);
    cl::Buffer d_Q_chol(Q_chol.begin(), Q_chol.end(), true);
    //cl::Pipe pipe(context, sizeof(cl_float), MAX_NX);
    cl_int status;
    cl_mem pipe_k = clCreatePipe(context(), 0, sizeof(cl_float), MAX_NX, NULL, &status);
    cl::Pipe k(pipe_k);
    cl_mem pipe_k2 = clCreatePipe(context(), 0, sizeof(cl_float), MAX_NX, NULL, &status);
    cl::Pipe k2(pipe_k2);
    cl_mem pipe_dummy = clCreatePipe(context(), 0, sizeof(int), 0, NULL, &status);
    cl::Pipe dummy(pipe_dummy);
    cl::Buffer d_Ystar(Ystar.begin(), Ystar.end(), false);
    cl::Buffer d_Ystarvar(Ystarvar.begin(), Ystarvar.end(), false);

    // Prediction
    square_exponential(cl::EnqueueArgs(queue1, cl::NDRange(1)), d_X,
                       d_Xstar, k, k2, dummy, d_scale, sigma, nx, nxstar, dim,
                       1, 0);
    expectation(cl::EnqueueArgs(queue2, cl::NDRange(1)), k, d_InvQt, d_Ystar, nx,
                nxstar);
    variance(cl::EnqueueArgs(queue3, cl::NDRange(1)), k2, d_Ystarvar, d_Q_chol,
             sigma, nx, nxstar);
    queue1.finish();
    queue2.finish();
    queue3.finish();

    // Retreive expectation values
    cl::copy(d_Ystar, Ystar.begin(), Ystar.end());
    cl::copy(d_Ystarvar, Ystarvar.begin(), Ystarvar.end());
}

// Conduct a single prediction using the square exponetial kernel
// X - Training inputs
// nx - Number of training inputs
// dim - The number of input dimensions (length of X and Xstar records)
// Xstar - Testing inputs
// nx_star - Number of testing inputs
// scale - Per dimension scaling factors for pairwise distances)
// sigma - Kernel scaling parameter
// InvQt - Vector, calculated during training, used for expecation value
//         calculation (Q = K(X,X), InvQt = Q^-1 * Y)
// Q_chol - Lower triangular cholesky factor of K(X,X), used for variance
//          calculation
// Ystar - Training prediction expectation values
// Ystarvar - Training prediction variances
// Ystarderiv - Prediction derivatives
// context - The OpenCL Context
// program - The OpenCL Program
void predict_single_deriv(
        std::vector<float> &X, int nx, int dim,
        std::vector<float> &Xstar, int nxstar,
        std::vector<float> &scale, float sigma,
        std::vector<float> &InvQt, std::vector<float> &Q_chol,
        std::vector<float> &Ystar, std::vector<float> &Ystarvar,
        std::vector<float> &Ystarderiv,
        CLContainer &container){
    cl::Context context = container.context;
    cl::Program program = container.program;

    // Create queues
    cl::CommandQueue queue1(context);
    cl::CommandQueue queue2(context);
    cl::CommandQueue queue3(context);
    cl::CommandQueue queue4(context);

    // Create kernel functor for square exponential kernel
    auto square_exponential = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Pipe&, cl::Pipe&, cl::Pipe&, cl::Buffer, float, int, int, int, int, int>(program, "sq_exp");
    // Create kernel functor for expectation kernel
    auto expectation = cl::KernelFunctor<cl::Pipe&, cl::Buffer, cl::Buffer, int, int>(program, "expectation");
    // Create kernel functor for variance kernel
    auto variance = cl::KernelFunctor<cl::Pipe&, cl::Buffer, cl::Buffer, float, int, int>(program, "variance");
    // Create kernel functor for derivative kernel
    auto derivatives = cl::KernelFunctor<cl::Pipe&, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, float, int, int, int>(program, "derivatives");

    // Create device variables
    cl::Buffer d_X(X.begin(), X.end(), true);
    cl::Buffer d_Xstar(Xstar.begin(), Xstar.end(), true);
    cl::Buffer d_scale(scale.begin(), scale.end(), true);
    cl::Buffer d_InvQt(InvQt.begin(), InvQt.end(), true);
    cl::Buffer d_Q_chol(Q_chol.begin(), Q_chol.end(), true);
    //cl::Pipe pipe(context, sizeof(cl_float), MAX_NX);
    cl_int status;
    cl_mem pipe_k = clCreatePipe(context(), 0, sizeof(cl_float), MAX_NX, NULL, &status);
    cl::Pipe k(pipe_k);
    cl_mem pipe_k2 = clCreatePipe(context(), 0, sizeof(cl_float), MAX_NX, NULL, &status);
    cl::Pipe k2(pipe_k2);
    cl_mem pipe_r = clCreatePipe(context(), 0, sizeof(cl_float), MAX_NX*MAX_NXSTAR, NULL, &status);
    cl::Pipe r(pipe_r);
    cl::Buffer d_Ystar(Ystar.begin(), Ystar.end(), false);
    cl::Buffer d_Ystarvar(Ystarvar.begin(), Ystarvar.end(), false);
    cl::Buffer d_Ystarderiv(Ystarderiv.begin(), Ystarderiv.end(), false);

    // Prediction
    square_exponential(cl::EnqueueArgs(queue1, cl::NDRange(1)), d_X,
                       d_Xstar, k, k2, r, d_scale, sigma, nx, nxstar, dim,
                       1, 1);
    expectation(cl::EnqueueArgs(queue2, cl::NDRange(1)), k, d_InvQt, d_Ystar, nx,
                nxstar);
    variance(cl::EnqueueArgs(queue3, cl::NDRange(1)), k2, d_Ystarvar, d_Q_chol,
             sigma, nx, nxstar);
    derivatives(cl::EnqueueArgs(queue4, cl::NDRange(1)), r, d_Ystarderiv, d_X, d_Xstar, d_InvQt, d_scale, sigma, nx, nxstar, dim);
    queue1.finish();
    queue2.finish();
    queue3.finish();
    queue4.finish();

    // Retreive expectation values
    cl::copy(d_Ystar, Ystar.begin(), Ystar.end());
    cl::copy(d_Ystarvar, Ystarvar.begin(), Ystarvar.end());
    cl::copy(d_Ystarderiv, Ystarderiv.begin(), Ystarderiv.end());
}

PYBIND11_MODULE(prediction, m){
    py::class_<CLContainer>(m, "CLContainer");
    py::bind_vector<std::vector<float>>(m, "VectorFloat");
    m.def("default_container", &default_container);
    m.def("predict_single_expectation", &predict_single_expectation);
    m.def("predict_single_variance", &predict_single_variance);
    m.def("predict_single_deriv", &predict_single_deriv);
}


int main(){
    try{
        const char* binary_file_name = "../device/prediction.aocx";
        auto container = default_container(binary_file_name);

        // Test prediction case
        // Based on the Python package test 'test_GaussianProcess_predict_single'
        // X = [[1,2,3],[2,4,1],[4,2,2]]
        // Y = [2,3,4]
        // Hyperparameters = [0,0,0,0]
        //
        // After training InvQt = [1.9407565,2.93451157,3.95432381]
        //
        // X* = [[1,3,2],[3,2,1]]
        // Expected Y* = [1.39538648,1.73114001]

        // Create host variables
        // Training inputs X
        std::vector<float> h_X = {1.0, 2.0, 3.0,
                                  2.0, 4.0, 1.0,
                                  4.0, 2.0, 2.0};
        // Prediction inputs X*
        std::vector<float> h_Xstar = {1.0, 3.0, 2.0,
                                      3.0, 2.0, 1.0};
        // Number of training inputs and prediction inputs
        int nx = 3; int nxstar = 2;
        // Dimension of inputs
        int dim = 3;
        // InvQt vector, a product of training
        std::vector<float> h_InvQt = {1.9407565, 2.93451157, 3.95432381};
        // InvQ matrix, a product of training
        std::vector<float> h_Q_chol = {1.00000000, 0.00000000, 0.00000000,
                                       0.01110900, 0.99993829, 0.00000000,
                                       0.00673795, 0.01103483, 0.99991641};
        // Hyperparameter used to scale predictions
        float sigma = 0.0f;
        // Hyperparameters to set length scale of distances between inputs
        std::vector<float> h_scale = {0.0, 0.0, 0.0};
        // Prediction result
        std::vector<float> h_Ystar(nxstar, 0);
        // Prediction variance
        std::vector<float> h_Ystarvar(nxstar, 0);
        // Prediction derivatives
        std::vector<float> h_Ystarderiv(nxstar*dim, 0);

        // Expected results
        std::vector<float> expected_Ystar = {1.39538648, 1.73114001};
        std::vector<float> expected_Ystarvar = {0.81667540, 0.858355920};
        std::vector<float> expected_Ystarderiv = {
            0.73471011, -0.0858304,  0.05918638,
            1.14274266,  0.48175876,  1.52580682
            };

        predict_single_deriv(h_X, nx, dim, h_Xstar, nxstar, h_scale, sigma,
                             h_InvQt, h_Q_chol, h_Ystar, h_Ystarvar,
                             h_Ystarderiv, container);
        compare_results(expected_Ystar, h_Ystar, "predict expectation values");
        for (auto const& i : h_Ystar)
            std::cout << i << ' ';
        std::cout << std::endl;
        compare_results(expected_Ystarvar, h_Ystarvar, "predict variance");
        for (auto const& i : h_Ystarvar)
            std::cout << i << ' ';
        std::cout << std::endl;
        compare_results(expected_Ystarderiv, h_Ystarderiv, "predict derivatives");
        for (auto const& i : h_Ystarderiv)
            std::cout << i << ' ';
        std::cout << std::endl;
    }
    catch (cl::Error err){
        std::cout << "OpenCL Error: " << err.what() << " code " << err.err() << std::endl;
        exit(-1);
    }

    return 0;
}
