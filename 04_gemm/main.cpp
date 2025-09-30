#include<iostream>
#include<cstdio>
#include"gemm.h"
#include"mx/utils/ostream.h"
#include"mx/dense_view.h"

#include"CycleTimer.h"

template<typename T>
mx::DenseView<const T> as_const(mx::DenseView<T> view) noexcept{
    return mx::DenseView<const T>(view.begin(), view.rows(), view.cols(), view.row_stride(), view.col_stride());
}

int main(){
    
    mx::Dense<double> A(1000, 1000, 1.0);
    mx::Dense<double> B(1000, 1000, 2.0);
    mx::Dense<double> C(1000, 1000, 0.0);
    mx::Dense<double> D(1000, 1000, 0.0);

    // Naive sequential GEMM 
    double startTime = CycleTimer::currentSeconds();
    mx::gemm(A, B, C);
    double endTime = CycleTimer::currentSeconds();
    printf("[Naive GEMM]: %.3f ms\n", (endTime - startTime) * 1000);

    // Naive strided parallel GEMM over rows
    startTime = CycleTimer::currentSeconds();
    mx::gemm_cpu_threads_row_cyclic(A, B, D, 32);
    endTime = CycleTimer::currentSeconds();

    if(C == D){
        printf("[GEMM Parallel Row-cyclic]: %.3f ms\n", (endTime - startTime) * 1000);
    } else std::cout << "Mismatch in values...";

    // mx::gemm(as_const(A.view()), as_const(B.view()), C.view());

    // std::cout << "C = A . B =\n" << C <<std::endl; 
    
    return 0;
}