#include <iostream>
#include "mx/dense.h"
#include "mx/dense_view.h"
#include "mx/utils/ostream.h"
#include "lu_unblocked.h"

int main(){
    // A is a full-rank matrix
    mx::Dense<double> A(3, 3, {1, 2, 2, 4, 4, 12, 4, 8, 12});
    std::cout << "Original matrix A:\n" << A << std::endl;
    std::vector<mx::index_t> piv_A(2);
    mx::LUInfo info = mx::lu_factor_unblocked(A.view(), piv_A);

    std::cout << "LU factorization info: " << info << std::endl;

    std::cout << "The pivot vector is: [ ";
    for(const auto& p : piv_A) std::cout << p << " ";
    std::cout << "]\n";

    std::cout << "The in-place LU after factorization:\n" << A << std::endl;

    // B is a singular matrix i.e. rank deficient
    mx::Dense<double> B(3, 3, {1, 2, 3, 2, 4, 6, 1, 0, 1});
    std::cout << "Original matrix B:\n" << B << std::endl;

    std::vector<mx::index_t> piv_B(3);
    info = mx::lu_factor_unblocked(B.view(), piv_B);

    std::cout << "LU factorization info: " << info << std::endl;

    std::cout << "The pivot vector is: [ ";
    for(const auto& p : piv_B) std::cout << p << " ";
    std::cout << "]\n";

    std::cout << "The in-place LU after factorization:\n" << B << std::endl;

    return 0;
}