#pragma once
#include <cstddef>

__global__ void kernel_small_dot_product(const double* x,
                                         const double* y,
                                         size_t n, 
                                         double* result);