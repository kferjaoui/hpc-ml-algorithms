#pragma once
#include<cassert>
#include<thread>
#include<cmath>
#include"mx/dense.h"
#include"mx/dense_view.h"

namespace mx{

template<typename T>
void gemm_cpu_threads_row_cyclic(const Dense<T>& A, const Dense<T>& B, Dense<T>& C, index_t numThreads){
    gemm_cpu_threads_row_cyclic(A.view(), B.view(), C.view(), numThreads);
}

template<typename T>
void gemm_cpu_threads_row_block(const Dense<T>& A, const Dense<T>& B, Dense<T>& C, index_t numThreads){
    gemm_cpu_threads_row_block(A.view(), B.view(), C.view(), numThreads);
}



template<typename T>
void gemm_cpu_threads_row_cyclic(DenseView<const T> A, DenseView<const T> B, DenseView<T> C, index_t numThreads = 8){
    const index_t N = A.rows();
    const index_t K = A.cols();
    const index_t M = B.cols();

    assert(K == B.rows() && N == C.rows() && M == C.cols());
    if (N == 0 || M == 0 || K == 0) return;

    numThreads = numThreads? std::min(numThreads, N) : 1;

    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    // Materialize the transposed of B for better locality
    Dense<T> BT(M, K);
    for(index_t r=0; r<K; r++){
        for(index_t c=0; c<M; c++){
            BT(c,r) = B(r,c); 
        }
    }

    auto workFunction = [&, N, M, K, numThreads](index_t tid){
        for(index_t i = tid; i<N; i+= numThreads){
            for(index_t j=0; j<M; j++){
                T sum{};
                for(index_t k=0; k<K; k++){
                    sum += A(i,k)*BT(j,k); // cache-friendly for the pass on BT too 
                }
                C(i,j) = sum;
            }
        }
    };

    for(index_t tid=0; tid<numThreads; tid++){
        threads.emplace_back(workFunction, tid);
    }

    for(auto& t:threads){
        if(t.joinable()) t.join();
    }
    
}


template<typename T>
void gemm_cpu_threads_row_block(DenseView<const T> A, DenseView<const T> B, DenseView<T> C, index_t numThreads = 8){
    const index_t N = A.rows();
    const index_t K = A.cols();
    const index_t M = B.cols();

    assert(K == B.rows() && N == C.rows() && M == C.cols());
    if (N == 0 || M == 0 || K == 0) return;

    numThreads = numThreads? std::min(numThreads, N) : 1;

    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    // Materialize the transposed of B for better locality
    Dense<T> BT(M, K);
    for(index_t r=0; r<K; r++){
        for(index_t c=0; c<M; c++){
            BT(c,r) = B(r,c); 
        }
    }

    auto baseWork  = N / numThreads;
    auto remainder = N % numThreads;

    auto workFunction = [&, N, M, K, numThreads](index_t tid){
        auto workChunk = baseWork + (tid<remainder? 1:0); 
        auto start = tid * baseWork + std::min(tid,remainder);
        auto end = start + workChunk;
        for(index_t i = start ; i < end; i+= 1){
            for(index_t j=0; j<M; j++){
                T sum{};
                for(index_t k=0; k<K; k++){
                    sum += A(i,k)*BT(j,k); // cache-friendly for the pass on BT too 
                }
                C(i,j) = sum;
            }
        }
    };

    for(index_t tid=0; tid<numThreads; tid++){
        threads.emplace_back(workFunction, tid);
    }

    for(auto& t:threads){
        if(t.joinable()) t.join();
    }
    
}

}