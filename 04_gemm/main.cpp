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
    const size_t Nattempts = 1;

    mx::Dense<double> A(2000, 2000, 1.0);
    mx::Dense<double> B(2000, 2000, 2.0);
    mx::Dense<double> C(2000, 2000, 0.0);
    mx::Dense<double> D(2000, 2000, 0.0);
    mx::Dense<double> E(2000, 2000, 0.0);
    mx::Dense<double> F(2000, 2000, 0.0);


    // ++++++++++++ SEQUENTIAL GEMMs +++++++++++++++

    // Naive  GEMM 
    double startTime = CycleTimer::currentSeconds();
    mx::gemm(A, B, C);
    double endTime = CycleTimer::currentSeconds();
    auto seq_time = endTime - startTime;
    printf("[Naive GEMM]: %.3f ms\n", (endTime - startTime) * 1000);

    // GEMM with L1/L2/L3 cache blocking 
    auto min_time = seq_time;
    
    for(size_t i=0; i<Nattempts; i++){
        startTime = CycleTimer::currentSeconds();
        mx::gemm_cpu_cache_blocked(A, B, D);
        endTime = CycleTimer::currentSeconds();
        min_time = std::min(min_time, endTime - startTime);
    }

    if(C == D){
        printf("[GEMM Cache Blocked]: %.3f ms\n", (min_time) * 1000);
    } else std::cout << "Mismatch in values...";

    // ++++++++++++ PARALLEL GEMMs +++++++++++++++

    size_t Nthreads = std::thread::hardware_concurrency();
    std::cout << Nthreads << " concurrent threads are supported.\n";

    // Naive strided parallel GEMM over rows 
    min_time = seq_time;
    
    for(size_t i=0; i<Nattempts; i++){
        startTime = CycleTimer::currentSeconds();
        mx::gemm_cpu_threads_row_cyclic(A, B, E, Nthreads);
        endTime = CycleTimer::currentSeconds();
        min_time = std::min(min_time, endTime - startTime);
    }

    if(C == E){
        printf("[GEMM // Row-cyclic]: %.3f ms\n", (min_time) * 1000);
    } else std::cout << "Mismatch in values...";

    // Naive partionned parallel GEMM over rows
    min_time = seq_time;
    
    for(size_t i=0; i<Nattempts; i++){
        startTime = CycleTimer::currentSeconds();
        mx::gemm_cpu_threads_row_block(A, B, F, Nthreads);
        endTime = CycleTimer::currentSeconds();
        min_time = std::min(min_time, endTime - startTime);  
    }

    if(C == F){
        printf("[GEMM // Partionned]: %.3f ms\n", (min_time) * 1000);
    } else std::cout << "Mismatch in values...";
    
    return 0;
}