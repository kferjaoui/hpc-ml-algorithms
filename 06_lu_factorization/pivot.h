#pragma once
#include <cstdint>
#include "mx/dense_view.h"

namespace mx{

template<typename T>
void row_swap_full(DenseView<T> A, index_t i, index_t j) noexcept {
    if (i == j) return;
    const index_t cols = A.cols();
    for (index_t c = 0; c < cols; ++c) {
        std::swap(A(i, c), A(j, c));
    }
}



}