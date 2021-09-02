#!/bin/bash

export QUDA_GPU_ARCH=sm_75

export CXX=/usr/local/nvidia/hpc_sdk/Linux_x86_64/21.7/compilers/bin/nvc++
export CC=/usr/local/nvidia/hpc_sdk/Linux_x86_64/21.7/compilers/bin/nvc

cmake .. \
        -DQUDA_BUILD_SHAREDLIB=ON \
        -DQUDA_MPI=ON \
        -DCMAKE_EXE_LINKER_FLAGS_SANITIZE="-fsanitize=address,undefined" \
	-DCMAKE_CXX_COMPILER=nvc++ \
	-DQUDA_CXX_STANDARD=17 \
	-DCMAKE_C_COMPILER=nvc \
	-DCUDAToolkit_INCLUDE_DIR= /usr/local/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/include \
	-DCMAKE_CXX_FLAGS="-O3 -Mfma -cuda -gpu=cc75 -gpu=fastmath -gpu=fma -gpu=lineinfo -gpu=autocollapse -gpu=unroll -I/usr/local/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/include" \
	-DCMAKE_BUILD_TYPE=HOSTDEBUG


