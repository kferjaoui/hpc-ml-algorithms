# High-Performance Computing & Machine Learning Kernels

This repository collects a series of high-performance implementations of foundational algorithms in numerical linear algebra, scientific computing, and machine learning.  
The focus is on **hardware-aware design**, **numerical correctness**, and **performance portability** across modern CPU and GPU architectures.

Each module pairs a clear reference implementation with progressively optimized variants (SIMD, cache-blocked, multi-threaded, and CUDA) together with reproducible benchmarks and lightweight analytical notes.  
The intent is to provide a concise, technically rigorous set of kernels that illustrate practical approaches to:

- dense and sparse linear algebra
- memory hierarchy utilization  
- parallelism (thread-, block-, warp-level)  
- GPU shared-memory tiling and register blocking  
- algorithmic intensity and roofline modeling  
- numerical stability and reproducibility

The codebase is organized as a lightweight “mx-core” library plus 14 focused modules, each demonstrating a specific computational pattern or algorithmic idea.

---

## Contents

The portfolio currently targets the following 14 core algorithms:

| #  | Project                                      | Description |
|----|----------------------------------------------|-----------------------------------------------------------------------------------------|
| 01 | Parallel Reduction (Sum / Max)               | Block / grid level reductions, shared memory, thread-level and warp-level aggregation. |
| 02 | Numerical Integration (Trapezoidal Rule)     | 1D numerical integration implemented as a weighted dot product built on top of the reduction primitives. |
| 03 | Prefix-Sum (Scan)                            | Work-efficient parallel prefix-sum (inclusive/exclusive), e.g. Blelloch / upsweep-downsweep variants. |
| 04 | Dense Matrix–Matrix Multiplication (GEMM)    | Naive, cache-blocked, vectorized and (where applicable) CUDA-optimized matrix multiplication. |
| 05 | K-Means Clustering                           | Lloyd’s algorithm with CPU multi-threading and GPU kernels for distance calculation and centroid updates. |
| 06 | LU Factorization & Gaussian Elimination      | In-place LU factorization with partial pivoting and linear system solve (Ax = b), validated against Eigen. |
| 07 | Gradient Descent for Optimization            | Mini-batch gradient descent with configurable step rules and vectorized updates. |
| 08 | Logistic Regression (Mini-Batch SGD)         | Binary classifier implemented with numerically stable forward/backward passes and SGD optimization. |
| 09 | Sparse Matrix–Vector Multiply (SpMV)         | CSR-based SpMV, exploring memory access patterns and performance for different sparsity structures. |
| 10 | Eigenvalue Computation (Power Method)        | Power iteration for dominant eigenpair, using dense or sparse mat-vec back-ends. |
| 11 | Principal Component Analysis (PCA)           | Covariance estimation via GEMM and eigen decomposition / power iterations for top-k components. |
| 12 | Fast Fourier Transform (FFT)                 | 1D FFT (Cooley–Tukey style) and variants optimized for GPU memory hierarchy. |
| 13 | 2D Convolution (Image Filtering / CNN)       | Direct spatial convolution with shared-memory tiling; optional im2col → GEMM pipeline. |
| 14 | Two-Layer Neural Network (MLP)               | Fully-connected network (forward + backprop), reusing GEMM and elementwise kernels. |

Not all modules are at the exact same level of maturity; the intent is that each project eventually ships with reference implementations, benchmarks, and a concise technical write-up.

---

## Repository Structure

The repository is organized as follows (names may slightly evolve as the project grows):

```text
.
├── 01_reduction/             # Parallel reduction kernels and tests
├── 02_trapezoid/             # Numerical integration via trapezoidal rule
├── 03_scan/                  # Prefix-sum (scan)
├── 04_gemm/                  # Dense GEMM (CPU + CUDA)
├── 05_K_means/               # K-means clustering
├── 06_lu_factorization/      # LU factorization + Gaussian elimination (Ax = b)
├── 07_gradient_descent/      # (planned) Gradient descent algorithms
├── 08_logistic_regression/   # (planned) Logistic regression with mini-batch SGD
├── 09_spmv/                  # (planned) Sparse matrix–vector multiplication
├── 10_power_method/          # (planned) Power iteration
├── 11_pca/                   # (planned) Principal Component Analysis
├── 12_fft/                   # (planned) Fast Fourier Transform
├── 13_conv2d/                # (planned) 2D convolution
├── 14_mlp/                   # (planned) Two-layer MLP
│
├── common/                   # Shared "mx" core library (data structures, utilities)
│   ├── 00_mx/
│   │   ├── include/mx/       # Dense, DenseView, types, layout helpers, etc.
│   │   └── tests/            # Unit tests for core components
│   ├── CycleTimer.h          # Timing utilities
│   └── Makefile              # Build rules for common components
│
├── third_party/
│   └── eigen/                # Eigen as a git submodule (for validation and comparison)
│
├── .gitignore
├── .gitmodules
└── README.md                 # This file
