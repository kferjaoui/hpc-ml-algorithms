#include<iostream>
#include"mx/dense.h"
#include"mx/utils/fill.h"
#include"mx/utils/display.h"
#include"mx/utils/ostream.h"

int main()
{
    mx::Dense<double> A(2,4), B(8,4);
    mx::fill(A, 1.0);
    mx::fill(B, 2.0);

    A(0,1) = 3.141;
    A(1,3) = 24.95;

    B(3,2) = 2.345;
    B(7,3) = 4.618;

    std::cout << "A = \n" << A << "\n";
    std::cout << "B = \n" << B << "\n";
    
    mx::DenseView<double> A_view(A.begin(), A.rows(), A.cols());
    mx::DenseView<double> B_view(B.begin(), B.rows(), B.cols());
    
    std::cout << "Transposed of A = \n" << A_view.transposed() << "\n";
    std::cout << "Subview of B from element (3,2) of size [5,2] = \n" << B_view.subview(3,2,5,2) << "\n";

    return 0;
}