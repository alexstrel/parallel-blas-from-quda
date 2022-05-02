nvc++ -O3 -cuda -std=c++17 -I../../build/include -I../../include -I../../build/include/targets/cuda -I../../core/cuda/include -o transform_reduce_test.o -c transform_reduce_test.cpp
nvc++ -o transform_reduce_test.exe transform_reduce_test.o -L/scratch/astrel/benchmarks/parallel-blas-from-quda/build/lib -lquda  -L/usr/local/openmpi-4.0.5-gcc-9.3-cuda-11.2/lib -lmpi

