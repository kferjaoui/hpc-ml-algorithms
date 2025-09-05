#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(stmt)                                     \
    do {                                                     \
        cudaError_t err = (stmt);                            \
        if (err != cudaSuccess) {                            \
            std::fprintf(stderr, "[%s:%d] CUDA error: %s\n", \
                         __FILE__, __LINE__,                 \
                         cudaGetErrorString(err));           \
            std::exit(1);                                    \
        }                                                    \
    } while (0)