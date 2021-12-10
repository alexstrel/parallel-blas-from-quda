nvc++ -O3 -std=c++17 -cuda -I../include -I../include/targets/cuda -I../../parallel-blas-from-quda-upd/core/generic/include -o transform_reduce_test_complex.o -c transform_reduce_test_complex.cpp
nvc++ -o transform_complex_reduce_test.exe transform_reduce_test_complex.o -L../lib -lquda -lcudanvhpc -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda/lib64 -lcudart -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/comm_libs/mpi/lib -lmpi -lmpi_cxx

