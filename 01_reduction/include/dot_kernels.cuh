#pragma once
#include <cstddef>

__global__ void dot64_singleblock_warp_downsweep(const double* __restrict__ x,
                                                 const double* __restrict__ y,
                                                 size_t n, 
                                                 double* __restrict__ result);

__global__ void dot64_multiblock_warp_downsweep(const double* __restrict__ x,
                                                const double* __restrict__ y,
                                                size_t n, 
                                                double* __restrict__ result);
