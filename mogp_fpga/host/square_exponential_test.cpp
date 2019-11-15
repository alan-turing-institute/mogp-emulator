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

        // Call kernel
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
        float TOL = 1.0e-6;
        for (int i=0; i<4; i++){
            if (std::abs(h_k[i] - native_k[i]) > TOL)
                std::cout << "Element " << i << " of native and FPGA implementations do not agree: " << h_k[i] << " " << native_k[i] << std::endl;
                exit(-1);
        }
    }
    catch (cl::Error err){
        std::cout << "OpenCL Error: " << err.what() << " code " << err.err() << std::endl;
        exit(-1);
    }
    return 0;
}
