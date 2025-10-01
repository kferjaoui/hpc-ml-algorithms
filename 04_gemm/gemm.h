#pragma once
#include<cassert>
#include<thread>
#include"mx/dense.h"
#include"mx/dense_view.h"

namespace mx{

template<typename T>
void gemm(const Dense<T>& A, const Dense<T>& B, Dense<T>& C){
    gemm(A.view(), B.view(), C.view());
}

template<typename T>
void gemm_cpu_threads_row_cyclic(const Dense<T>& A, const Dense<T>& B, Dense<T>& C, size_t numThreads){
    gemm_cpu_threads_row_cyclic(A.view(), B.view(), C.view(), numThreads);
}

template<typename T>
void gemm(DenseView<const T> A, DenseView<const T> B, DenseView<T> C) {
    size_t N = A.rows();
    size_t K = A.cols();
    size_t M = B.cols();

    assert(K == B.rows() && N == C.rows() && M == C.cols());

    for(size_t i=0; i<N; i++){
        for(size_t j=0; j<M; j++){
            T sum{};
            for(size_t k=0; k<K; k++){
                sum += A(i,k)*B(k,j); 
            }
            C(i,j) = sum;
        }
    }
}

template<typename T>
void gemm_cpu_threads_row_cyclic(DenseView<const T> A, DenseView<const T> B, DenseView<T> C, size_t numThreads = 8){
    size_t N = A.rows();
    size_t K = A.cols();
    size_t M = B.cols();

    assert(K == B.rows() && N == C.rows() && M == C.cols());
    if (N == 0 || M == 0 || K == 0) return;

    numThreads = numThreads? std::min(numThreads, N) : 1;

    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    // Materialize the transposed of B for better locality
    Dense<T> BT(M, K);
    for(size_t r=0; r<K; r++){
        for(size_t c=0; c<M; c++){
            BT(c,r) = B(r,c); 
        }
    }

    auto workFunction = [&, N, M, K, numThreads](size_t tid){
        for(size_t i = tid; i<N; i+= numThreads){
            for(size_t j=0; j<M; j++){
                T sum{};
                for(size_t k=0; k<K; k++){
                    sum += A(i,k)*BT(j,k); // cache-friendly for the pass on BT too 
                }
                C(i,j) = sum;
            }
        }
    };

    for(size_t tid=0; tid<numThreads; tid++){
        threads.emplace_back(workFunction, tid);
    }

    for(auto& t:threads){
        if(t.joinable()) t.join();
    }
    
}

}