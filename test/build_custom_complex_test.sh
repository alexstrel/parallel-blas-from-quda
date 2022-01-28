
BUILD_DIR=../../build-upd4

nvcc -forward-unknown-to-host-compiler -I${BUILD_DIR}/include -I${BUILD_DIR}/include/targets/cuda -I${BUILD_DIR}/include/targets/generic -I../../parallel-blas-from-quda-upd4/core/cuda/include -I../../parallel-blas-from-quda-upd4 -O3  --generate-code=arch=compute_86,code=[compute_86,sm_86] -Xcompiler=-fPIC -ftz=true -prec-div=false -prec-sqrt=false -Wno-deprecated-gpu-targets -arch=sm_86 --expt-relaxed-constexpr -Xfatbin=-compress-all -Xcompiler=-g -Xcompiler=-O3 -Xcompiler=-Wall -Xcompiler=-Wextra -Wreorder -Xcompiler=-Wno-unknown-pragmas -Xptxas -warn-lmem-usage,-warn-spills -lineinfo -Xcompiler -pthread -std=c++17 -c custom_transform_reduce_test_complex.cu -o custom_transform_reduce_test_complex.o


nvcc -o custom_transform_complex_reduce_test.exe custom_transform_reduce_test_complex.o -L../lib -lquda -lmpi
