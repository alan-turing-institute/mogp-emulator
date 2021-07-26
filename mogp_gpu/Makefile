# Makefile for standalone C++ tests of CUDA functionality.
# Note that this Makefile is NOT used when building mogp_emulator.
# (The logic in setup.py that builds the Extension takes care of that).
# The targets in this Makefile are mainly standalone tests of low-level functionality.
#
# If running on CSD3, run the following commands to ensure dependencies are in place:
#
# module load anaconda  #(not needed here, but needed for mogp_emulator)
# module load py-pybind11-2.2.4-gcc-5.4.0-tdtz6iq #(only needed for mogp_emulator)
# module load cuda/11.2
# module load gcc/7
# module load eigen
#

CXX=g++
CXXFLAGS=-std=c++14 -O3
NVCCFLAGS=--compiler-options -Wall,-Wextra -arch=sm_60 --generate-code arch=compute_60,code=sm_60 --generate-code arch=compute_37,code=sm_37
EXTRAFLAGS=-Xcompiler -fPIC

CUDA_INC=/usr/local/cuda/include
PYBIND_INC=$(shell python3 -m pybind11 --includes)

all: lib/libgpgpu.so

clean:
	rm -rf obj lib bin

lib/libgpgpu.so:  src/gp_gpu.cu obj/cov_gpu.o obj/util.o
	mkdir -p lib
	nvcc $(CXXFLAGS) $(NVCCFLAGS) $(EXTRAFLAGS) -shared $(PYBIND_INC) $< obj/cov_gpu.o obj/util.o -o $@ -lcusolver -lcublas -lcudart

obj/%.o: src/%.cu src/%.hpp
	mkdir -p obj
	nvcc $(CXXFLAGS) $(NVCCFLAGS) $(EXTRAFLAGS) -c $< -o $@

bin/test/test_gpu_basic: test/test_gpu_basic.cu  obj/util.o
	mkdir -p bin/test
	nvcc $(CXXFLAGS) $(NVCCFLAGS) $<  obj/util.o -o $@

bin/test/test_gpu_kernel: test/test_gpu_kernel.cu  obj/kernel.o obj/util.o
	mkdir -p bin/test
	nvcc $(CXXFLAGS) $(NVCCFLAGS) $<  obj/kernel.o obj/util.o -o $@

bin/test/test_gpu_utils: test/test_gpu_utils.cu obj/util.o
	mkdir -p bin/test
	nvcc $(CXXFLAGS) $(NVCCFLAGS) $< obj/util.o -o $@

bin/test/test_gpu_cholesky: test/test_gpu_cholesky.cu
	mkdir -p bin/test
	nvcc $(CXXFLAGS) $(NVCCFLAGS) -lcusolver -lcublas $< -o $@

bin/test/test_basic_fitting: test/test_basic_fitting.cu
	mkdir -p bin/test
	nvcc $(CXXFLAGS) $(NVCCFLAGS) -lnlopt $< -o $@

bin/test/test_meanfunc: test/test_meanfunc.cu obj/util.o
	mkdir -p bin/test
	nvcc $(CXXFLAGS) $(NVCCFLAGS) $<  obj/util.o -o $@

bin/test/test_gp_gpu: test/test_gp_gpu.cu obj/util.o obj/kernel.o
	mkdir -p bin/test
	nvcc $(CXXFLAGS) $(NVCCFLAGS) -lcublas -lcusolver $< obj/kernel.o  obj/util.o -o $@

bin/test/test_gp_fitting: test/test_gp_fitting.cu obj/util.o obj/kernel.o
	mkdir -p bin/test
	nvcc $(CXXFLAGS) $(NVCCFLAGS) -lnlopt -lcublas -lcusolver $< obj/kernel.o  obj/util.o -o $@



.PHONY: all clean
