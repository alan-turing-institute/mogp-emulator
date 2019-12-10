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

void square_exponential_native(std::vector<float> r, std::vector<float> &k,
                               float sigma){
    std::transform(r.begin(), r.end(), k.begin(),
                   [](float x) -> float { return exp(-0.5f * x); });
    float exp_sigma = exp(sigma);
    for (auto i=k.begin(); i!=k.end(); i++){
        *i *= exp_sigma;
    }
}

void distance_native(std::vector<float> x, std::vector<float> y,
                     std::vector<float> &r, std::vector<float> l,
                     int nx, int ny, int dim){
    for (int row=0; row<nx; row++){
        int row_stride = row*dim;
        for (int col=0; col<ny; col++){
            int col_stride = col*dim;

            float sum = 0;
            for (int i=0; i<dim; i++){
                float difference = x[row_stride+i] - y[col_stride+i];
                sum += (difference * difference) / exp(-1.0f*l[i]);
            }
            r[row*ny+col] = sum;
        }
    }
}

void matrix_vector_product_native(std::vector<float> a, std::vector<float> b,
                                  std::vector<float> &c, int m, int n){
    for (int col=0; col<n; col++){
        float sum = 0.0f;
        for (int row=0; row<m; row++){
            sum += a[row*n+col] * b[row];
        }
        c[col] = sum;
    }
}

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
// InvQt - Vector which K(X,X*) is multiplied by to obtain expectation values
std::vector<float> predict_single(std::vector<float> &X, int nx, int dim,
                                  std::vector<float> &Xstar, int nxstar,
                                  std::vector<float> &theta, float sigma,
                                  std::vector<float> &InvQt,
                                  cl::Context &context,
                                  cl::Program &program){
    // Prediction results array
    std::vector<float> Ystar(nxstar, 0);

    // Create queues
    cl::CommandQueue queue1(context);
    cl::CommandQueue queue2(context);

    // Create kernel functor for square exponential kernel
    auto square_exponential = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Pipe&, cl::Buffer, float, int, int, int>(program, "sq_exp");
    // Create kernel functor for matrix vector product kernel
    auto matrix_vector_product = cl::KernelFunctor<cl::Pipe&, cl::Buffer, cl::Buffer, int, int>(program, "matrix_vector_product");

    // Create device variables
    cl::Buffer d_X(X.begin(), X.end(), true);
    cl::Buffer d_Xstar(Xstar.begin(), Xstar.end(), true);
    cl::Buffer d_theta(theta.begin(), theta.end(), true);
    cl::Buffer d_InvQt(InvQt.begin(), InvQt.end(), true);
    //cl::Pipe pipe(context, sizeof(cl_float), MAX_M);
    cl_int status;
    cl_mem pipe_k = clCreatePipe(context(), 0, sizeof(cl_float), MAX_N, NULL, &status);
    cl::Pipe k(pipe_k);
    cl::Buffer d_Ystar(Ystar.begin(), Ystar.end(), false);

    // Prediction
    // Determine square exponential kernel matrix
    square_exponential(cl::EnqueueArgs(queue1, cl::NDRange(1)), d_X,
                       d_Xstar, k, d_theta, sigma, nx, nxstar, dim);
    // Columns of the kernel matrix are sent by a pipe to the matrix vector
    // product matrix
    matrix_vector_product(cl::EnqueueArgs(queue2, cl::NDRange(1)), k,
                          d_InvQt, d_Ystar, nx, nxstar);
    queue2.finish();

    // Retreive expectation values
    cl::copy(d_Ystar, Ystar.begin(), Ystar.end());
    return Ystar;
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
        // Square distances between X and X*
        std::vector<float> r_native(nx*nxstar, 0);
        // InvQt vector, a product of training
        std::vector<float> h_InvQt = {1.9407565, 2.93451157, 3.95432381};
        // Hyperparameter used to scale predictions
        float sigma = 0.0f;
        // Hyperparameters to set length scale of distances between inputs
        std::vector<float> h_l = {0.0, 0.0, 0.0};
        // Kernel matrix
        std::vector<float> k_native(nx*nxstar, 0);
        // Prediction result
        std::vector<float> h_Ystar_native(nxstar, 0);

        // Expected results
        std::vector<float> expected_Ystar = {1.39538648, 1.73114001};

        distance_native(h_X, h_Xstar, r_native, h_l, nx, nxstar, dim);
        square_exponential_native(r_native, k_native, sigma);
        matrix_vector_product_native(k_native, h_InvQt, h_Ystar_native, nx, nxstar);

        auto h_Ystar = predict_single(h_X, nx, dim, h_Xstar, nxstar, h_l,
                                      sigma, h_InvQt, context, program);

        compare_results(expected_Ystar, h_Ystar, "matrix_vector_product");
        compare_results(expected_Ystar, h_Ystar_native, "matrix_vector_product_native");
        for (auto const& i : h_Ystar)
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
        h_Xstar = {4.0, 0.0, 2.0};
        nxstar = 1;
        h_InvQt = {0.73575167, 1.10362757, 1.47151147};
        h_l = {1.0, 1.0, 1.0};
        sigma = 1.0;
        expected_Ystar = {0.01741762};
        h_Ystar = predict_single(h_X, nx, dim, h_Xstar, nxstar, h_l,
                                 sigma, h_InvQt, context, program);
        compare_results(expected_Ystar, h_Ystar, "matrix_vector_product");
        for (auto const& i : h_Ystar)
            std::cout << i << ' ';
        std::cout << std::endl;
    }
    catch (cl::Error err){
        std::cout << "OpenCL Error: " << err.what() << " code " << err.err() << std::endl;
        exit(-1);
    }

    return 0;
}
