#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include "CL/cl2.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

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

int main(){
    try{
        // Create context using default device
        cl::Context context(CL_DEVICE_TYPE_DEFAULT);
        // Create queue
        cl::CommandQueue queue(context);

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
        // Create kernel functor for square exponential kernel
        auto square_exponential = cl::KernelFunctor<cl::Buffer, cl::Buffer, float, int, int>(program, "sq_exp");
        // Create kernel functor for distance kernel
        auto distance = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int, int>(program, "distance");
        // Create kernel functor for matrix vector product kernel
        auto matrix_vector_product = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int>(program, "matrix_vector_product");

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
        std::vector<float> h_r(nx*nxstar, 0);
        std::vector<float> h_r_native(nx*nxstar, 0);
        // InvQt vector, a product of training
        std::vector<float> h_InvQt = {1.9407565, 2.93451157, 3.95432381};
        // Hyperparameter used to scale predictions
        float sigma = 0.0f;
        // Hyperparameters to set length scale of distances between inputs
        std::vector<float> h_l = {0.0, 0.0, 0.0};
        // Kernel matrix
        std::vector<float> h_k(nx*nxstar, 0);
        std::vector<float> h_k_native(nx*nxstar, 0);
        // Prediction result
        std::vector<float> h_Ystar(nxstar, 0);
        std::vector<float> h_Ystar_native(nxstar, 0);

        // Create device variables
        cl::Buffer d_X(h_X.begin(), h_X.end(), true);
        cl::Buffer d_Xstar(h_Xstar.begin(), h_Xstar.end(), true);
        cl::Buffer d_r(h_r.begin(), h_r.end(), false);
        cl::Buffer d_InvQt(h_InvQt.begin(), h_InvQt.end(), true);
        cl::Buffer d_l(h_l.begin(), h_l.end(), true);
        cl::Buffer d_k(h_k.begin(), h_k.end(), false);
        cl::Buffer d_Ystar(h_Ystar.begin(), h_Ystar.end(), false);

        // Expected results
        std::vector<float> expected_distances = {2.0, 8.0,
                                                 3.0, 5.0,
                                                 10.0, 2.0};
        std::vector<float> expected_k = {0.36787944, 0.01831564,
                                         0.22313016, 0.082085,
                                         0.00673795, 0.36787944};
        std::vector<float> expected_Ystar = {1.39538648, 1.73114001};

        // Prediction
        // Determine SQUARED distances between training and test inputs
        distance(cl::EnqueueArgs(queue, cl::NDRange(1)), d_X, d_Xstar, d_r, d_l,
                 nx, nxstar, dim);
        queue.finish();
        cl::copy(d_r, h_r.begin(), h_r.end());
        distance_native(h_X, h_Xstar, h_r_native, h_l, nx, nxstar, dim);
        compare_results(expected_distances, h_r, "distance");
        compare_results(expected_distances, h_r_native, "distance_native");

        // Determine kernel matrix of distances
        square_exponential(cl::EnqueueArgs(queue, cl::NDRange(1)), d_r, d_k,
                           sigma, nx, nxstar);
        queue.finish();
        cl::copy(d_k, h_k.begin(), h_k.end());
        square_exponential_native(h_r, h_k_native, sigma);
        compare_results(expected_k, h_k, "square_exponential");
        compare_results(expected_k, h_k_native, "square_exponential_native");

        // Get prediction result
        matrix_vector_product(cl::EnqueueArgs(queue, cl::NDRange(1)), d_k,
                              d_InvQt, d_Ystar, nx, nxstar);
        queue.finish();
        cl::copy(d_Ystar, h_Ystar.begin(), h_Ystar.end());
        matrix_vector_product_native(h_k, h_InvQt, h_Ystar_native, nx, nxstar);
        compare_results(expected_Ystar, h_Ystar, "matrix_vector_product");
        compare_results(expected_Ystar, h_Ystar_native, "matrix_vector_product_native");
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
