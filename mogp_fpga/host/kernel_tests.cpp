#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include "CL/cl2.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

void square_exponential_native(std::vector<float> r, std::vector<float>* k){
    std::transform(r.begin(), r.end(), k->begin(),
                   [](float x) -> float { return exp(-0.5f * x * x); });
}

void distance_native(std::vector<float> x, std::vector<float> y,
                     std::vector<float> &r, int nx, int ny, int dim){
    for (int row=0; row<nx; row++){
        int row_stride = row*dim;
        for (int col=0; col<ny; col++){
            int col_stride = col*dim;

            float sum = 0;
            for (int i=0; i<dim; i++){
                float difference = x[row_stride+i] - y[col_stride+i];
                sum += difference*difference;
            }
            r[row*ny+col] = sum;
        }
    }
}

void compare_results(std::vector<float> host, std::vector<float> device,
                     std::string kernel_name){
    float TOL = 1.0e-6;
    int length = host.size();
    bool discrepency = false;

    std::cout << "Comparing host and device results for " << kernel_name << std::endl;
    // Ensure FPGA and native implementation agree within a tolerance
    for (int i=0; i<length; i++){
        if (std::abs(host[i] - device[i]) > TOL){
            std::cout << "Element " << i << " of host and FPGA implementations do not agree: " << host[i] << " " << device[i] << std::endl;
            discrepency = true;
        }
    }

    if (!discrepency){
        std::cout << "Host and device implementations agree" << std::endl;
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
        auto square_exponential = cl::KernelFunctor<cl::Buffer, cl::Buffer, int, int>(program, "sq_exp");
        // Create kernel functor for distance kernel
        auto distance = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int>(program, "distance");
        // Create kernel functor for matrix vector product kernel
        auto matrix_vector_product = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int>(program, "matrix_vector_product");

        // Create host variables
        std::vector<float> h_r = {1.0, 2.0, 4.0, 8.0};
        std::vector<float> h_k(4, 0.0);
        std::vector<float> native_k(4, 0.0);
        int m=2, n=2;
        // Create device objects
        cl::Buffer d_r(h_r.begin(), h_r.end(), true);
        cl::Buffer d_k(h_k.begin(), h_k.end(), false);

        for (auto const& i : h_r)
            std::cout << i << ' ';
        std::cout << std::endl;

        // Call square exponential kernel
        square_exponential(cl::EnqueueArgs(queue, cl::NDRange(1)), d_r, d_k, m, n);
        queue.finish();

        // Copy result from buffer to host
        cl::copy(d_k, h_k.begin(), h_k.end());
        for (auto const& i : h_k)
            std::cout << i << ' ';
        std::cout << std::endl;

        // Run native implementation
        square_exponential_native(h_r, &native_k);
        for (auto const& i : native_k)
            std::cout << i << ' ';
        std::cout << std::endl;

        // Ensure FPGA and native implementation agree within a tolerance
        compare_results(native_k, h_k, "square exponential");

        std::vector<float> h_a = {0,1,0,2};
        std::vector<float> h_b = {0,0,0,1};
        std::vector<float> h_c(4,0);
        int nx = 2;
        int ny = 2;
        int dim = 2;
        // Create device objects
        cl::Buffer d_a(h_a.begin(), h_a.end(), true);
        cl::Buffer d_b(h_b.begin(), h_b.end(), false);
        cl::Buffer d_c(h_c.begin(), h_c.end(), false);

        distance(cl::EnqueueArgs(queue, cl::NDRange(1)), d_a, d_b, d_c, nx, ny, dim);
        queue.finish();

        cl::copy(d_c, h_c.begin(), h_c.end());
        for (auto const& i : h_c)
            std::cout << i << ' ';
        std::cout << std::endl;

        std::vector<float> native_c(4,0);
        distance_native(h_a, h_b, native_c, nx, ny, dim);
        for (auto const& i : native_c)
            std::cout << i << ' ';
        std::cout << std::endl;

        compare_results(native_c, h_c, "distance");
    }
    catch (cl::Error err){
        std::cout << "OpenCL Error: " << err.what() << " code " << err.err() << std::endl;
        exit(-1);
    }

    return 0;
}
