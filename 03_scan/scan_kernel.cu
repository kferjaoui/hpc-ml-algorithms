#include <cuda_runtime.h>

#include <cuda/std/bit>
using cuda::std::bit_ceil;
using cuda::std::countr_zero;

__device__ inline int      ilog2_pow2(uint32_t n) { return countr_zero(n); } 

__device__ inline uint32_t next_pow2_u32(uint32_t n) { return bit_ceil(n); } // ceil to next power of two

__global__
void blelloch_exclusive_scan_singleblock(const int* __restrict__ in_dev,
                                        int* __restrict__ out_dev,
                                        uint32_t N)
{
    // Let's assume one block i.e. gridDim.x is 1 and blockIdx.x is 0
    const uint32_t nextPow2_N = next_pow2_u32(N ? N : 1u);
    const uint32_t depth = ilog2_pow2_u32(nextPow2_N);
    const uint32_t nThreads = blockDim.x; 
    const uint32_t tid = threadIdx.x;  // works as both local and globak index as blockIdx.x = 0

    extern __shared__ int sh[]; //assume that the kernel is launched with enough shared memory allocated i.e. size >= nextPow2_N * sizeof(int)
    // load input array into shared memory (strided here unecessary when nThreads >= nextPow2_N)
    for(size_t idx = tid; idx < nextPow2_N; idx += nThreads){
        sh[idx] = (idx < N) ? in_dev[idx] : 0;
    }
    __syncthreads();

    // upsweep phase
    for(uint32_t d = 0; d<depth-1; d++){
        // only active each second thread (d=0), then each 4th thread (d=1), then each 8th (d=2) and so on...
        // Assumes nThreads >= nextPow2_N
        if( ((tid+1) % (1u<<(d+1)) == 0) && (tid < nextPow2_N) ){
            sh[tid] = sh[tid] + sh[tid-(1u<<d)];
        }

        __syncthreads();
    }

    // clear last element
    if(tid == 0) sh[nextPow2_N-1] = 0;
    __syncthreads();

    // downsweep phase
    for(int d = depth-1; d>=0; d--){
        // E.g N = 16 -> only active each 16th thread (d=depth-1), then each 8th thread (d=depth-2), then each 4th (d=depth-3) and so on...
        if( ((tid+1) % (1u<<(d+1)) == 0) && (tid < nextPow2_N) ){
            int t = sh[tid-(1u<<d)];
            sh[tid-(1u<<d)] = sh[tid];
            sh[tid] = sh[tid] + t;
        }
        __syncthreads();
    }

    // write results to output array
    for(uint32_t idx = tid; idx < N; idx += nThreads){
        out_dev[idx] = sh[idx]; 
    }

}