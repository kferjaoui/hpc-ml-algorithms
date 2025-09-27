#include<iostream>
#include"mx/dense.h"
#include"mx/utils/fill.h"
#include"mx/utils/ostream.h"

int main()
{
    mx::Dense<double> A(4,4), B(8,4);
    mx::fill(A, 1.0);
    mx::fill(B, 2.0);

    A(0,0) = 3.141;
    B(7,3) = 1.618;

    std::cout << "A = \n" << A << "\n";
    std::cout << "B = \n" << B << "\n";

    return 0;
}