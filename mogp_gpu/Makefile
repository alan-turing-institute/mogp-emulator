CXX=g++
CXXFLAGS=-std=c++11 -O3
NVCCFLAGS=--compiler-options -Wall,-Wextra --generate-code arch=compute_60,code=sm_60
CUDA_INC=/usr/local/cuda/include
PYBIND_INC=$(shell python3 -m pybind11 --includes)

all: lib/libgpgpu.so

clean:
	rm -rf obj lib bin

lib/libgpgpu.so:  src/gp_gpu.cu obj/cov_gpu.o obj/util.o
	mkdir -p lib
	nvcc $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler -fPIC -shared $(PYBIND_INC) $< obj/cov_gpu.o obj/util.o -o $@ -lcusolver -lcublas -lcudart

obj/cov_gpu.o: src/cov_gpu.cu src/cov_gpu.hpp
	mkdir -p obj
	nvcc $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler -fPIC -c src/cov_gpu.cu -o obj/cov_gpu.o

obj/util.o: src/util.cu src/util.hpp
	mkdir -p obj
	nvcc $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler -fPIC -c src/util.cu -o obj/util.o

bin/test/gpu_basic: test/gpu_basic.cu obj/cov_gpu.o obj/util.o
	mkdir -p bin/test
	nvcc $(CXXFLAGS) $(NVCCFLAGS) $< obj/cov_gpu.o obj/util.o -o $@

bin/test/gpu_cholesky: test/gpu_cholesky.cu
	mkdir -p bin/test
	nvcc $(CXXFLAGS) $(NVCCFLAGS) -lcusolver -lcublas $< -o $@

.PHONY: all clean
