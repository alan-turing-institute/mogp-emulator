CXX=g++
CXXFLAGS=-std=c++11 -O3
CUDA_INC=/usr/local/cuda/include

all: lib/libgpgpu.so

clean:
	rm -rf obj lib

lib/libgpgpu.so: src/gplib.cpp obj/cov_gpu.o src/gp_gpu.hpp 
	mkdir -p lib
	$(CXX) $(CXXFLAGS) -I$(CUDA_INC) -fPIC -shared $< obj/cov_gpu.o -o $@ -larmadillo -lcublas -lcudart

obj/cov_gpu.o: src/cov_gpu.cu src/cov_gpu.hpp
	mkdir -p obj
	nvcc $(CXXFLAGS) -Xcompiler -fPIC -c src/cov_gpu.cu -o obj/cov_gpu.o

.PHONY: all clean
