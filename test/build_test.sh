nvc++ -O3 -Mfma -cuda -gpu=cc75 -gpu=fastmath -gpu=fma -gpu=lineinfo -gpu=autocollapse -gpu=unroll -I../common/inc -I/usr/local/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/include  -I../build/include -I../build/include/targets/cuda -I../core/cuda/include -o transform_reduce_test.o -c transform_reduce_test.cpp
nvc++ -o transform_reduce_test.exe transform_reduce_test.o -L/home/astrel/Work/TR/parallel-blas-from-quda-v4/build/lib -lquda -L/usr/local/nvidia/hpc_sdk/Linux_x86_64/21.7/comm_libs/openmpi/openmpi-3.1.5/lib -lmpi_cxx -L/usr/local/nvidia/hpc_sdk/Linux_x86_64/21.7/comm_libs/mpi/lib -lmpi -L/usr/local/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/lib64 -lcudart -L/usr/local/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/lib64/stubs -lcuda -L/usr/local/nvidia/hpc_sdk/Linux_x86_64/21.7/compilers/lib -lcudanvhpc

