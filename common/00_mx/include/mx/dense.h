#pragma once
#include<vector>
#include<cassert>
#include"dense_view.h"

namespace mx{

template<typename T>
class Dense {
    size_t _rows{0};
    size_t _cols{0};
    std::vector<T> _data;
    
    [[nodiscard]] size_t rm_idx(size_t i, size_t j) const noexcept{
        assert(i<_rows && j<_cols);
        return j + _cols*i;
    }

    [[nodiscard]] size_t cm_idx(size_t i, size_t j) const noexcept{
        assert(i<_rows && j<_cols);
        return i + _rows*j;
    }

public:
    Dense() = default;
    Dense(size_t rows, size_t cols):
        _rows(rows), _cols(cols), _data(rows*cols){}
    
    Dense(size_t rows, size_t cols, const T& init):
        _rows(rows), _cols(cols), _data(rows*cols, init){}
    
    [[nodiscard]] T& operator()(size_t i, size_t j) noexcept {
        assert(i<_rows && j<_cols);
        return _data[rm_idx(i,j)];
    }
    
    [[nodiscard]] const T& operator()(size_t i, size_t j) const noexcept {
        assert(i<_rows && j<_cols);
        return _data[rm_idx(i,j)];
    }

    [[nodiscard]] size_t rows() const noexcept { return _rows; }
    [[nodiscard]] size_t cols() const noexcept { return _cols; }
    [[nodiscard]] size_t size() const noexcept { return _data.size(); }

    // expose a contiguous row-major view of the dense matrix
    DenseView<T>       view() noexcept       { return DenseView<T>(_data.data(), _rows, _cols); }
    DenseView<const T> view() const noexcept { return DenseView<const T>(_data.data(), _rows, _cols); }

    T*       begin() noexcept { return _data.data(); }
    const T* begin() const noexcept { return _data.data(); }
    
    T*       end() noexcept { return _data.data() + _data.size(); }
    const T* end() const noexcept { return _data.data() + _data.size(); }

};

}
