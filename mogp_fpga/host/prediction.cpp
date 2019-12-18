#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include "CL/cl2.hpp"
#include "CL/cl.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#define MAX_M 128
#define MAX_N 128

#define MODE_EXPECTATION 1
#define MODE_VARIANCE 2

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

// Conduct a single prediction using the square exponetial kernel
// X - Training inputs
// nx - Number of training inputs
// dim - The number of input dimensions (length of X and Xstar records)
// Xstar - Testing inputs
// nx_star - Number of testing inputs
// theta - Hyperparameters (used to scale pairwise distances)
// sigma - Kernel scaling parameter
// InvQt - Vector calculated during training (invQ*Y) used for expecation value
//         calculation
// Ystar - Training prediction expectation values
// context - The OpenCL Context
// program - The OpenCL Program
void predict_single(std::vector<float> &X, int nx, int dim,
                    std::vector<float> &Xstar, int nxstar,
                    std::vector<float> &theta, float sigma,
                    std::vector<float> &InvQt,
                    std::vector<float> &Ystar,
                    cl::Context &context, cl::Program &program){

    // Create queues
    cl::CommandQueue queue1(context);
    cl::CommandQueue queue2(context);

    // Create kernel functor for square exponential kernel
    auto square_exponential = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Pipe&, cl::Pipe&, cl::Buffer, float, int, int, int, int>(program, "sq_exp");
    // Create kernel functor for expectation kernel
    auto expectation = cl::KernelFunctor<cl::Pipe&, cl::Buffer, cl::Buffer, int, int>(program, "expectation");

    // Create device variables
    cl::Buffer d_X(X.begin(), X.end(), true);
    cl::Buffer d_Xstar(Xstar.begin(), Xstar.end(), true);
    cl::Buffer d_theta(theta.begin(), theta.end(), true);
    cl::Buffer d_InvQt(InvQt.begin(), InvQt.end(), true);
    cl_int status;
    cl_mem pipe_k = clCreatePipe(context(), 0, sizeof(cl_float), MAX_M, NULL, &status);
    cl::Pipe k(pipe_k);
    cl_mem pipe_dummy = clCreatePipe(context(), 0, sizeof(int), 0, NULL, &status);
    cl::Pipe dummy(pipe_dummy);
    cl::Buffer d_Ystar(Ystar.begin(), Ystar.end(), false);

    // Prediction
    square_exponential(cl::EnqueueArgs(queue1, cl::NDRange(1)), d_X,
                       d_Xstar, k, dummy, d_theta, sigma, nx, nxstar, dim,
                       MODE_EXPECTATION);
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
// theta - Hyperparameters (used to scale pairwise distances)
// sigma - Kernel scaling parameter
// InvQt - Vector calculated during training (invQ*Y) used for expecation value
//         calculation
// invQ - Matrix calculated during training ((K(X,X) + nugget)^-1) used for
//        variance calculation
// Ystar - Training prediction expectation values
// Ystarvar - Training prediction variances
// context - The OpenCL Context
// program - The OpenCL Program
void predict_single(std::vector<float> &X, int nx, int dim,
                    std::vector<float> &Xstar, int nxstar,
                    std::vector<float> &theta, float sigma,
                    std::vector<float> &InvQt, std::vector<float> &InvQ,
                    std::vector<float> &Ystar, std::vector<float> &Ystarvar,
                    cl::Context &context, cl::Program &program){

    // Create queues
    cl::CommandQueue queue1(context);
    cl::CommandQueue queue2(context);
    cl::CommandQueue queue3(context);

    // Create kernel functor for square exponential kernel
    auto square_exponential = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Pipe&, cl::Pipe&, cl::Buffer, float, int, int, int, int>(program, "sq_exp");
    // Create kernel functor for expectation kernel
    auto expectation = cl::KernelFunctor<cl::Pipe&, cl::Buffer, cl::Buffer, int, int>(program, "expectation");
    // Create kernel functor for variance kernel
    auto variance = cl::KernelFunctor<cl::Pipe&, cl::Buffer, cl::Buffer, float, int, int>(program, "variance");

    // Create device variables
    cl::Buffer d_X(X.begin(), X.end(), true);
    cl::Buffer d_Xstar(Xstar.begin(), Xstar.end(), true);
    cl::Buffer d_theta(theta.begin(), theta.end(), true);
    cl::Buffer d_InvQt(InvQt.begin(), InvQt.end(), true);
    cl::Buffer d_InvQ(InvQ.begin(), InvQ.end(), true);
    //cl::Pipe pipe(context, sizeof(cl_float), MAX_M);
    cl_int status;
    cl_mem pipe_k = clCreatePipe(context(), 0, sizeof(cl_float), MAX_M, NULL, &status);
    cl::Pipe k(pipe_k);
    cl_mem pipe_k2 = clCreatePipe(context(), 0, sizeof(cl_float), MAX_M, NULL, &status);
    cl::Pipe k2(pipe_k2);
    cl::Buffer d_Ystar(Ystar.begin(), Ystar.end(), false);
    cl::Buffer d_Ystarvar(Ystarvar.begin(), Ystarvar.end(), false);

    // Prediction
    square_exponential(cl::EnqueueArgs(queue1, cl::NDRange(1)), d_X,
                       d_Xstar, k, k2, d_theta, sigma, nx, nxstar, dim,
                       MODE_VARIANCE);
    expectation(cl::EnqueueArgs(queue2, cl::NDRange(1)), k, d_InvQt, d_Ystar, nx,
                nxstar);
    variance(cl::EnqueueArgs(queue3, cl::NDRange(1)), k2, d_Ystarvar, d_InvQ,
             sigma, nx, nxstar);
    queue1.finish();
    queue2.finish();
    queue3.finish();

    // Retreive expectation values
    cl::copy(d_Ystar, Ystar.begin(), Ystar.end());
    cl::copy(d_Ystarvar, Ystarvar.begin(), Ystarvar.end());
}

int main(){
    try{
        // Create context using default device
        cl::Context context(CL_DEVICE_TYPE_DEFAULT);

        // Get devices
        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);

        // Create queue from binary
        const char* binary_file_name = "../device/prediction.aocx";
        FILE* fp;
        fp = fopen(binary_file_name, "rb");
        if(fp == 0) {
            throw cl::Error(0, "can't open aocx file");
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
        std::vector<float> h_InvQ = { 1.0001672 , -0.01103735, -0.00661646,
                                     -0.01103735,  1.00024523, -0.01103735,
                                     -0.00661646, -0.01103735,  1.0001672};
        // Hyperparameter used to scale predictions
        float sigma = 0.0f;
        // Hyperparameters to set length scale of distances between inputs
        std::vector<float> h_l = {0.0, 0.0, 0.0};
        // Prediction result
        std::vector<float> h_Ystar(nxstar, 0);
        // Prediction variance
        std::vector<float> h_Ystarvar(nxstar, 0);

        // Expected results
        std::vector<float> expected_Ystar = {1.39538648, 1.73114001};
        std::vector<float> expected_Ystarvar = {0.81667540, 0.858355920};

        predict_single(h_X, nx, dim, h_Xstar, nxstar, h_l, sigma, h_InvQt,
                       h_Ystar, context, program);
        compare_results(expected_Ystar, h_Ystar, "predict expectation values");
        for (auto const& i : h_Ystar)
            std::cout << i << ' ';
        std::cout << std::endl;

        predict_single(h_X, nx, dim, h_Xstar, nxstar, h_l, sigma, h_InvQt,
                       h_InvQ, h_Ystar, h_Ystarvar, context, program);
        compare_results(expected_Ystar, h_Ystar, "predict expectation values");
        for (auto const& i : h_Ystar)
            std::cout << i << ' ';
        std::cout << std::endl;
        compare_results(expected_Ystarvar, h_Ystarvar, "predict variance");
        for (auto const& i : h_Ystarvar)
            std::cout << i << ' ';
        std::cout << std::endl;

        // Based on the Python package test 'test_GaussianProcess_predict_single'
        // X = [[1,2,3],[2,4,1],[4,2,2]]
        // Y = [2,3,4]
        // Hyperparameters = [1,1,1,01
        //
        // After training InvQt = [0.73575167,1.10362757,1.47151147]
        //
        // X* = [4,0,2]
        // Expected Y* = [0.01741762]
        nxstar = 1;
        h_Xstar.resize(nxstar*dim);
        h_Xstar = {4.0, 0.0, 2.0};
        h_Ystar.resize(nxstar);
        h_Ystarvar.resize(nxstar);
        h_InvQt = {0.73575167, 1.10362757, 1.47151147};
        h_InvQ = { 3.67879441e-01, -1.79183651e-06, -4.60281258e-07,
                  -1.79183651e-06,  3.67879441e-01, -1.79183651e-06,
                  -4.60281258e-07, -1.79183651e-06,  3.67879441e-01};
        h_l = {1.0, 1.0, 1.0};
        sigma = 1.0;
        expected_Ystar = {0.01741762};
        expected_Ystarvar = {2.718230287};

        predict_single(h_X, nx, dim, h_Xstar, nxstar, h_l, sigma, h_InvQt,
                       h_Ystar, context, program);
        compare_results(expected_Ystar, h_Ystar, "predict expectation values");
        for (auto const& i : h_Ystar)
            std::cout << i << ' ';
        std::cout << std::endl;

        predict_single(h_X, nx, dim, h_Xstar, nxstar, h_l, sigma, h_InvQt,
                       h_InvQ, h_Ystar, h_Ystarvar, context, program);
        compare_results(expected_Ystar, h_Ystar, "predict expectation values");
        for (auto const& i : h_Ystar)
            std::cout << i << ' ';
        std::cout << std::endl;
        compare_results(expected_Ystarvar, h_Ystarvar, "predict variance");
        for (auto const& i : h_Ystarvar)
            std::cout << i << ' ';
        std::cout << std::endl;
    }
    catch (cl::Error err){
        std::cout << "OpenCL Error: " << err.what() << " code " << err.err() << std::endl;
        exit(-1);
    }

    return 0;
}
