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
void gemm_cpu_threads_row_block(const Dense<T>& A, const Dense<T>& B, Dense<T>& C, size_t numThreads){
    gemm_cpu_threads_row_block(A.view(), B.view(), C.view(), numThreads);
}

template<typename T>
void gemm_cpu_cache_blocked(const Dense<T>& A, const Dense<T>& B, Dense<T>& C){
    gemm_cpu_cache_blocked(A.view(), B.view(), C.view());
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


template<typename T>
void gemm_cpu_threads_row_block(DenseView<const T> A, DenseView<const T> B, DenseView<T> C, size_t numThreads = 8){
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

    auto baseWork  = N / numThreads;
    auto remainder = N % numThreads;

    auto workFunction = [&, N, M, K, numThreads](size_t tid){
        auto workChunk = baseWork + (tid<remainder? 1:0); 
        auto start = tid * baseWork + std::min(tid,remainder);
        auto end = start + workChunk;
        for(size_t i = start ; i < end; i+= 1){
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


template<typename T>
void gemm_cpu_cache_blocked(DenseView<const T> A, DenseView<const T> B, DenseView<T> C){ //, size_t numThreads = 8){
    size_t N = A.rows();
    size_t K = A.cols();
    size_t M = B.cols();

    assert(K == B.rows() && N == C.rows() && M == C.cols());
    if (N == 0 || M == 0 || K == 0) return;

    Dense<T> BT(M, K);
    for(size_t r=0; r<K; r++){
        for(size_t c=0; c<M; c++){
            BT(c,r) = B(r,c); 
        }
    }

    const size_t nc = 256; // rows of A/C per block
    const size_t kc = 256; // depth of A/B per block
    const size_t mc = 256; // columns of B/C per block

    size_t Nb = (N + nc - 1) / nc; // number of row blocks 
    size_t Kb = (K + kc - 1) / kc; // number of depth blocks
    size_t Mb = (M + mc - 1) / mc; // number of column blocks

    // numThreads = numThreads? std::min(numThreads, N) : 1;

    // std::vector<std::thread> threads;
    // threads.reserve(numThreads);
    
    for(size_t Mi = 0; Mi<Mb; Mi++){                        // loop over column blocks of C
        const size_t jc = Mi*mc;
        const size_t jend = std::min(Mi*mc + mc, M);

        for(size_t Ki = 0; Ki<Kb; Ki++){                    // loop over depth blocks of A/B   
            const size_t pc = Ki*kc;
            const size_t pend = std::min(Ki*kc + kc, K);
            
            for(size_t Ni = 0; Ni<Nb; Ni++){                // loop over row blocks of C
                const size_t ic = Ni*nc;
                const size_t iend = std::min(Ni*nc + nc, N);

                // C_block(Ni,Mi) += A_block(Ni,Ki) * B_block(Ki,Mi) 
                for(size_t i=ic; i<iend; i++){
                    for(size_t j=jc; j<jend; j++){
                        T sum{}; // register accumulator
                        for(size_t p=pc; p<pend; p++){
                            sum += A(i,p) * BT(j,p);
                        }
                        C(i,j) += sum; // accumulate the current K-block
                    }
                }
            }
        }
    }     

}

}