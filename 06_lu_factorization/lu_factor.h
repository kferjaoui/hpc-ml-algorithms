#pragma once
#include "mx/dense.h"
#include "mx/dense_view.h"
#include<algorithm>
#include <cmath>
#include <limits>

namespace mx{

template<typename T>
void row_swap_strided(DenseView<T> A, size_t i, size_t j) noexcept {
    if (i == j) return;
    const size_t cols = A.cols();
    for (size_t c = 0; c < cols; ++c) {
        using std::swap;
        swap(A(i, c), A(j, c));
    }
}

template<typename T>
void lu_factor(DenseView<T> LU, std::vector<int>& piv){
    const size_t n = LU.rows();
    const size_t m = LU.cols(); 
    
    // IMPORTANT: let's assume that the matrix is square i.e. n=m

    for(size_t k = 0; k<m; k++){

        // 1. find the pivot and save the index of its row
        T pivot = LU(k,k);
        size_t i_pivot = k;
        for(size_t ii=k+1; ii<n; ii++){
            if(std::abs(LU(ii,k)) > std::abs(pivot)){
                pivot = LU(ii,k);
                i_pivot = ii;
            }
        }

        piv[k] = static_cast<int>(i_pivot);

        // 2. swap the two rows
        if (i_pivot != k) row_swap_strided(LU, k, i_pivot);

        // 3. Eliminating the column k
        for(size_t i=k+1; i<n; i++){
            T m_ik = LU(i,k) / LU(k,k);
            // Update multiplier in lower section of LU which is L
            LU(i,k) = m_ik;
            // Update the whole the row i
            for(size_t j = k+1; j<m; j++) LU(i,j) = LU(i,j) - m_ik * LU(k,j);
        }

    }

    // At the end;
    // L: All elements in the strict lower part of LU (not inclusing diagonal) are the elements of L which also has an all ones diagonal
    // U: All the elements in the upper part of LU including the diagonal 

}

}