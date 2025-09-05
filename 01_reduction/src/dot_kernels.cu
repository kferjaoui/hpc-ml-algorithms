#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include "dot_kernels.cuh"

__global__
void kernel_small_dot_product(const double* x,
                              const double* y,
                              size_t n, 
                              double* result)
{
    // NOTE: this “small” kernel assumes a single block launch (gridDim.x == 1).
    // It also assumes that blockDim.x is a power of two and at least 64.

    // Per-thread sums
    double sum = 0.0;
    for (size_t idx = threadIdx.x; idx<n; idx+=blockDim.x){
        sum = std::fma(x[idx], y[idx], sum);
    }

    extern __shared__ double sh[];
    sh[threadIdx.x] = sum;
    __syncthreads();

    // block-level reduction
    for(size_t stride=blockDim.x >> 1; stride>= warpSize; stride>>= 1){       //warpSize = 32
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride]; 
        __syncthreads();
    }

    // warp-level reduction
    double value = sh[threadIdx.x]; // value in threads's register, not in shared memory

    if(threadIdx.x < warpSize){    // threads with indices: 0..31
        unsigned mask = __activemask();  // lanes currently active at this instruction
        #pragma unroll
        for(size_t offset=warpSize/2; offset>0; offset>>= 1 ){ //offset starts as 16 then 8, 4, 2 and 1 
            if (threadIdx.x < offset) value += __shfl_down_sync(mask, value, offset) ; 
        }
    }
    if (threadIdx.x == 0) *result = value;
}