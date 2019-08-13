CXX=g++
CXXFLAGS=-std=c++11 -O3
CUDA_INC=/usr/local/cuda/include

all: lib/libgpgpu.so

clean:
	rm -rf obj lib

lib/libgpgpu.so: src/gplib.cu obj/cov_gpu.o src/gp_gpu.hpp 
	mkdir -p lib
	nvcc $(CXXFLAGS) -Xcompiler -fPIC -Xcompiler -Wall -Xcompiler -Wextra -shared $< obj/cov_gpu.o -o $@ -lcublas -lcudart

obj/cov_gpu.o: src/cov_gpu.cu src/cov_gpu.hpp
	mkdir -p obj
	nvcc $(CXXFLAGS) -Xcompiler -fPIC -Xcompiler -Wall -Xcompiler -Wextra -c src/cov_gpu.cu -o obj/cov_gpu.o

.PHONY: all clean
