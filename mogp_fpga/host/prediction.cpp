#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include "CL/cl2.hpp"
#include "CL/cl.h"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <vector>

#define MAX_NX 128
#define MAX_NXSTAR 128

struct CLContainer{
    cl::Context context;
    std::vector<cl::Device> devices;
    cl::Program program;
};


CLContainer create_cl_container(const char* path){
    try{
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
    catch (cl::Error err){
        std::cout << "OpenCL Error: " << err.what() << " code " << err.err() << std::endl;
        exit(-1);
    }
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
void predict_single(
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
void predict_single(
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
void predict_single(
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


namespace py = pybind11;
PYBIND11_MAKE_OPAQUE(std::vector<float>);


PYBIND11_MODULE(prediction, m){
    py::class_<CLContainer>(m, "CLContainer");
    py::bind_vector<std::vector<float>>(m, "VectorFloat");
    m.def("create_cl_container", &create_cl_container);
    m.def("predict_single",
          py::overload_cast<std::vector<float> &, int, int,
                            std::vector<float> &, int, std::vector<float> &,
                            float, std::vector<float> &, std::vector<float> &,
                            CLContainer &>(&predict_single));
    m.def("predict_single",
          py::overload_cast<std::vector<float> &, int, int,
                            std::vector<float> &, int, std::vector<float> &,
                            float, std::vector<float> &, std::vector<float> &,
                            std::vector<float> &, std::vector<float> &,
                            CLContainer &>(&predict_single));
    m.def("predict_single",
          py::overload_cast<std::vector<float> &, int, int,
                            std::vector<float> &, int, std::vector<float> &,
                            float, std::vector<float> &, std::vector<float> &,
                            std::vector<float> &, std::vector<float> &,
                            std::vector<float> &,
                            CLContainer &>(&predict_single));
}
