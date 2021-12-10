nvcc -O3 -std=c++17 --expt-relaxed-constexpr -I../include -I../include/targets/cuda -I../../parallel-blas-from-quda-upd/core/cuda/include -o transform_reduce_test.o -c transform_reduce_test.cu
nvcc -o transform_reduce_test.exe transform_reduce_test.o -L../lib -lquda -lmpi

