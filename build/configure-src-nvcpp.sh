#!/bin/bash

export QUDA_GPU_ARCH=sm_86

export CXX=nvc++
export CC=nvc

cmake ../parallel-blas-from-quda-upd \
        -DQUDA_BUILD_SHAREDLIB=ON \
	-DQUDA_HETEROGENEOUS_ATOMIC=OFF \
        -DQUDA_MPI=ON \
        -DCMAKE_EXE_LINKER_FLAGS_SANITIZE="-fsanitize=address,undefined" \
        -DCMAKE_BUILD_TYPE=DEVEL


