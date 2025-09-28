#include<iostream>
#include"gemm.h"
#include"mx/utils/ostream.h"
#include"mx/dense_view.h"

template<typename T>
mx::DenseView<const T> as_const(mx::DenseView<T> view) noexcept{
    return mx::DenseView<const T>(view.begin(), view.rows(), view.cols(), view.row_stride(), view.col_stride());
}

int main(){
    
    mx::Dense<double> A(4, 2, 1.0);
    mx::Dense<double> B(2, 4, 2.0);
    mx::Dense<double> C(4, 4, 0.0);

    mx::gemm(A, B, C);
    
    // mx::gemm(as_const(A.view()), as_const(B.view()), C.view());

    std::cout << "C = A . B =\n" << C <<std::endl; 
    
    return 0;
}