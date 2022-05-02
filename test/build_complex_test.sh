nvcc -O3 -std=c++17 --expt-relaxed-constexpr -I../include -I../include/targets/cuda -I../../parallel-blas-from-quda-upd4/core/cuda/include -o transform_reduce_test_complex.o -c transform_reduce_test_complex.cu
nvcc -o transform_complex_reduce_test.exe transform_reduce_test_complex.o -L../lib -lquda -lmpi

