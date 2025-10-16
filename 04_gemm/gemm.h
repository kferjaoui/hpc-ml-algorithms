#pragma once
#include<cassert>
#include<thread>
#include<cmath>
#include"mx/dense.h"
#include"mx/dense_view.h"

#include <experimental/simd>

namespace stdx = std::experimental;

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
void gemm_cpu_threads_cache_blocked(const Dense<T>& A, const Dense<T>& B, Dense<T>& C, size_t numThreads){
    gemm_cpu_threads_cache_blocked(A.view(), B.view(), C.view(), numThreads);
}

template<typename T>
void gemm_cpu_threads_microtiles(const Dense<T>& A, const Dense<T>& B, Dense<T>& C, size_t numThreads){
    gemm_cpu_threads_microtiles(A.view(), B.view(), C.view(), numThreads);
}


template<typename T>
void gemm_cpu_threads_vectorized(const Dense<T>& A, const Dense<T>& B, Dense<T>& C, size_t numThreads){
    gemm_cpu_threads_vectorized(A.view(), B.view(), C.view(), numThreads);
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
void gemm_cpu_cache_blocked(DenseView<const T> A, DenseView<const T> B, DenseView<T> C){
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

template<typename T>
void gemm_cpu_threads_cache_blocked(DenseView<const T> A, DenseView<const T> B, DenseView<T> C, size_t numThreads = 8){
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
    };

    std::cout << "Spawning "<< numThreads << " concurrent threads...\n";

    for(size_t tid=0; tid<numThreads; tid++){
        threads.emplace_back(workFunction, tid);
    }

    for(auto& t:threads){
        if(t.joinable()) t.join();
    }
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

template<typename T>
void gemm_cpu_threads_vectorized(DenseView<const T> A, DenseView<const T> B, DenseView<T> C, size_t numThreads = 8){
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
                            
                            // SIMD vector of elements of type T with width W corresponding to the
                            // native vector register width for the current target architecture
                            using vT = stdx::native_simd<T>; 
                            const size_t width = 8; // vT::size();
                            
                            // vectorized loop over the 'j_valid/width' full J-blocks of size 'width' 
                            size_t j=0;
                            for(; j+width <= j_valid; j+=width){

                                vT acc[nr];
                                for(size_t i=0; i<nr; i++) acc[i] = vT{};

                                // Unroll the K-loop by 4 
                                size_t p=pc;
                                const size_t pend4 = pc + ((pend - pc)/4)*4;
                                for(; p<pend; p+=4){
                                    const T* Bp0 = B.at(p, j+j0_micro);
                                    vT vB0(Bp0, stdx::element_aligned);
                                    const T* Bp1 = B.at(p+1, j+j0_micro);
                                    vT vB1(Bp1, stdx::element_aligned);
                                    const T* Bp2 = B.at(p+2, j+j0_micro);
                                    vT vB2(Bp2, stdx::element_aligned);
                                    const T* Bp3 = B.at(p+3, j+j0_micro);
                                    vT vB3(Bp3, stdx::element_aligned);
                               
                                    for(size_t i=0; i<i_valid; i++){
                                        vT vA0(A(i0_micro + i,p));
                                        vT vA1(A(i0_micro + i,p+1));
                                        vT vA2(A(i0_micro + i,p+2));
                                        vT vA3(A(i0_micro + i,p+3));
                                    
                                        acc[i] += vA0 * vB0; //std::fma(vA0, vB0, acc[i]);
                                        acc[i] += vA1 * vB1; //std::fma(vA1, vB1, acc[i]);
                                        acc[i] += vA2 * vB2; //std::fma(vA2, vB2, acc[i]);
                                        acc[i] += vA3 * vB3; //std::fma(vA3, vB3, acc[i]);
                                    }
                                }

                                // K-tail (0to 3 leftover)
                                p = pend4;
                                for(; p<pend4; p++){
                                    const T* Bp = B.at(p, j+j0_micro);  //contiguous in memory across j
                                    vT vB(Bp, stdx::element_aligned);
                                    for(size_t i=0; i<i_valid; i++){
                                        vT vA(A(i0_micro + i,p));
                                        acc[i] = vA * vB;  //std::fma(vA, vB, acc[i]);
                                    }
                                }

                                // Store back to C lane by lane
                                for(size_t i=0; i<i_valid; i++){
                                    for(size_t lane=0; lane<width; lane++){
                                        C(i0_micro + i, j0_micro + j + lane) += acc[i][lane];
                                    }
                                }
                            }

                            // TODO: Handle Tail of the J-loop i.e. remainder of 'j_valid % width' when 'j_valid' is not multiple of 'width'

                        } //M_micro
                    }     //N_micro

                }//Ni                  
            }    //Ki
        }        //Mi
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