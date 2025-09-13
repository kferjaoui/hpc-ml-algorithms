#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>

#include "helpers.h"
#include "dot_kernels.cuh"

int main() {
    const size_t n = 1u << 20; // 1,048,576 elements

    // host data
    std::vector<double> hx(n), hy(n);
    for (size_t i = 0; i < n; ++i) {
        // deterministic values (not too large)
        hx[i] = 1.0 / double(i + 1);
        hy[i] = std::sin(0.001 * double(i));
    }

    // CPU reference
    double ref = 0.0;
    for (size_t i = 0; i < n; ++i) ref += hx[i] * hy[i];

    // device memory
    double *dx = nullptr, *dy = nullptr, *dout = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dout, sizeof(double)));

    CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dout, 0, sizeof(double)));

    // launch config
    cudaDeviceProp prop{};
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    const int block = 256;
    const dim3 grid(4, 1, 1);
    const size_t shmemBytes = block * sizeof(double);

    // launch
    if (grid.x * grid.y * grid.z == 1){
        dot64_singleblock_warp_downsweep<<<grid, block, shmemBytes>>>(dx, dy, n, dout);
    } else{
        dot64_multiblock_warp_downsweep<<<grid, block, shmemBytes>>>(dx, dy, n, dout);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // fetch result
    double gpu = 0.0;
    CUDA_CHECK(cudaMemcpy(&gpu, dout, sizeof(double), cudaMemcpyDeviceToHost));

    // report
    double abs_err = std::abs(gpu - ref);
    double rel_err = abs_err / (std::abs(ref) + 1e-18);

    std::printf("CPU ref : %.17g\n", ref);
    std::printf("GPU dot : %.17g\n", gpu);
    std::printf("abs err : %.3e, rel err : %.3e\n", abs_err, rel_err);

    // cleanup
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dout);
    return 0;
}
