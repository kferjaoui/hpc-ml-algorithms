#pragma once
#include<cassert>
#include<thread>
#include<cmath>
#include"mx/dense.h"
#include"mx/dense_view.h"

namespace mx{

template<typename T>
void gemm_cpu_threads_microtiles(const Dense<T>& A, const Dense<T>& B, Dense<T>& C, size_t numThreads){
    gemm_cpu_threads_microtiles(A.view(), B.view(), C.view(), numThreads);
}

template<typename T>
void gemm_cpu_threads_microtiles(DenseView<const T> A, DenseView<const T> B, DenseView<T> C, size_t numThreads = 8){
    size_t N = A.rows();
    size_t K = A.cols();
    size_t M = B.cols();

    assert(K == B.rows() && N == C.rows() && M == C.cols());
    if (N == 0 || M == 0 || K == 0) return;

    const size_t nc = 256; // rows of A/C per block
    const size_t kc = 256; // depth of A/B per block
    const size_t mc = 128; // columns of B/C per block

    size_t Nb = (N + nc - 1) / nc; // number of row blocks 
    size_t Kb = (K + kc - 1) / kc; // number of depth blocks
    size_t Mb = (M + mc - 1) / mc; // number of column blocks

    numThreads = numThreads? std::min(numThreads, Mb) : 1;

    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    auto baseWork  = Mb / numThreads;
    auto remainder = Mb % numThreads;

    auto workFunction = [&, M, K, N, numThreads](size_t tid){
        auto workChunk = baseWork + (tid<remainder? 1:0); 
        auto C_col_start = tid * baseWork + std::min(tid,remainder);
        auto C_col_end = C_col_start + workChunk;
    
        for(size_t Mi = C_col_start; Mi<C_col_end; Mi++){       // loop over column blocks of C allocated to the running thread
            const size_t jc = Mi*mc;
            const size_t jend = std::min(Mi*mc + mc, M);

            for(size_t Ki = 0; Ki<Kb; Ki++){                    // loop over depth blocks of A/B   
                const size_t pc = Ki*kc;
                const size_t pend = std::min(Ki*kc + kc, K);
                
                for(size_t Ni = 0; Ni<Nb; Ni++){                // loop over row blocks of C
                    const size_t ic = Ni*nc;
                    const size_t iend = std::min(Ni*nc + nc, N);
                    
                    constexpr size_t nr = 4; // micro-block rows
                    constexpr size_t mr = 8; // micro-block columns

                    size_t Nnr = (iend-ic + nr -1)/nr; // number of micro-tiles in a row block
                    size_t Nmr = (jend-jc + mr -1)/mr; // number of micro-tiles in a column block
                    
                    // Compute the micro-tile C_micro(N_micro, M_micro) = A_micro(N_micro, :) * B_micro(:, M_micro)
                    // where A_micro is (nr x K), B_micro is (K x mr) and C_micro is (nr x mr)
                    for(size_t N_micro=0; N_micro<Nnr; N_micro++){
                        const size_t i0_micro = ic + N_micro*nr;       // global starting row index of the micro-tile
                        const size_t i_valid = std::min(nr, iend - i0_micro); // number of valid rows in the micro-tile

                        for(size_t M_micro=0; M_micro<Nmr; M_micro++){
                            const size_t j0_micro = jc + M_micro*mr;   // global starting column index of the micro-tile
                            const size_t j_valid = std::min(mr, jend - j0_micro); // number of valid columns in the micro-tile
                            
                            // Register vector accumulator (per micro-tile)
                            T sum[nr*mr]; 
                            for(size_t idx=0; idx<nr*mr; idx++) sum[idx] = T{}; // set to zero basically

                            // Unroll the k-loop over the micro-tile
                            size_t p=pc;
                            const size_t pend4 = pc + ((pend - pc)/4)*4;
                            for(; p<pend4; p+=4){
                                const T* Bp0 = B.at(p,j0_micro);  //contiguous in memory across j
                                const T* Bp1 = B.at(p+1,j0_micro);
                                const T* Bp2 = B.at(p+2,j0_micro);
                                const T* Bp3 = B.at(p+3,j0_micro);

                                for(size_t i=0; i<i_valid; i++){
                                    // broadcast A(i+i0_micro,p)
                                    const T a0 = A(i0_micro + i,p); 
                                    const T a1 = A(i0_micro + i,p+1); 
                                    const T a2 = A(i0_micro + i,p+2); 
                                    const T a3 = A(i0_micro + i,p+3); 
                                    
                                    for(size_t j=0; j<j_valid; j++){
                                        T acc_ij{};
                                        acc_ij = std::fma(a0, Bp0[j], acc_ij);
                                        acc_ij = std::fma(a1, Bp1[j], acc_ij);
                                        acc_ij = std::fma(a2, Bp2[j], acc_ij);
                                        acc_ij = std::fma(a3, Bp3[j], acc_ij);

                                        sum[j+mr*i] +=  acc_ij;
                                    }
                                }
                            }

                            // loop over the remainder number of elements in K-block i.e. pend % 4
                            p = pend4;
                            for(; p<pend; p++){
                                const T* Bp = B.at(p,j0_micro);
                                for(size_t i=0; i<i_valid; i++){
                                    T a = A(i0_micro + i,p);
                                    for(size_t j=0; j<j_valid; j++){
                                        sum[j+mr*i] =  std::fma(a, Bp[j], sum[j+mr*i]);
                                    }
                                }
                            }

                            // Store the micro-tile back to C
                            for(size_t i=0; i<i_valid; i++){
                                for(size_t j=0; j<j_valid; j++){
                                    C(i0_micro + i, j0_micro + j) += sum[j+mr*i];
                                }
                            }

                        }
                    }
                }                    
           }     
        }   
    };

    std::cout << "Spawning "<< numThreads << " concurrent threads...\n";

    for(size_t tid=0; tid<numThreads; tid++){
        threads.emplace_back(workFunction, tid);
    }

    for(auto& t:threads){
        if(t.joinable()) t.join();
    }
}


}