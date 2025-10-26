#include<iostream>
#include"mx/dense.h"
#include"mx/dense_view.h"
#include"mx/utils/ostream.h"
#include "lu_factor.h"

int main(){
    mx::Dense<double> A(3, 3, {1, 2, 2, 4, 4, 12, 4, 8, 12});
    std::vector<int> piv(3, 0);
    mx::lu_factor(A.view(), piv);

    std::cout << "The pivot vector is: [ ";
    for(const auto& p : piv) std::cout << p << " ";
    std::cout << "]\n";
    
    std::cout << "The in-place LU after factorization:\n" << A << std::endl;

    return 0;
}