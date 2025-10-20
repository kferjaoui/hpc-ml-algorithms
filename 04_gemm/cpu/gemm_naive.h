#pragma once
#include<cassert>
#include<cmath>
#include"mx/dense.h"
#include"mx/dense_view.h"

namespace mx{

template<typename T>
void gemm(const Dense<T>& A, const Dense<T>& B, Dense<T>& C){
    gemm(A.view(), B.view(), C.view());
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

}