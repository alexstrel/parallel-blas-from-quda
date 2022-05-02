#!/bin/bash

export QUDA_GPU_ARCH=sm_70

export CXX=nvc++
export CC=nvc

cmake .. \
        -DQUDA_BUILD_SHAREDLIB=ON \
	-DCMAKE_CUDA_COMPILER=nvc++ \
       	-DCMAKE_CXX_COMPILER=nvc++ \
	-DCMAKE_C_COMPILER=nvc \
        -DQUDA_MPI=ON \
        -DCMAKE_EXE_LINKER_FLAGS_SANITIZE="-fsanitize=address,undefined" \
        -DCMAKE_BUILD_TYPE=DEVEL


