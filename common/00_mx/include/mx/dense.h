#pragma once
#include<vector>
#include <initializer_list>
#include<cassert>
#include"dense_view.h"

namespace mx{

template<typename T>
class Dense {
    size_t _rows{0};
    size_t _cols{0};
    size_t _size{0}; 
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
        _rows(rows), _cols(cols), _size(rows*cols), _data(rows*cols){}
    
    Dense(size_t rows, size_t cols, const T& init):
        _rows(rows), _cols(cols), _size(rows*cols), _data(rows*cols, init){}

    Dense(size_t rows, size_t cols, std::initializer_list<T> init):
    _rows(rows), _cols(cols), _size(rows*cols), _data(init)
    {
        assert(init.size() == rows * cols && "Initializer size must match rows*cols");
    }
    
    [[nodiscard]] T& operator()(size_t i, size_t j) noexcept {
        assert(i<_rows && j<_cols);
        return _data[rm_idx(i,j)];
    }
    
    [[nodiscard]] const T& operator()(size_t i, size_t j) const noexcept {
        assert(i<_rows && j<_cols);
        return _data[rm_idx(i,j)];
    }

    bool operator==(const Dense<T>& other) const noexcept{
        if(_size != other._size) return false;

        for(size_t idx; idx<_size; idx++){
            if(_data[idx] != other._data[idx]) return false;
        }

        return true;
    }

    [[nodiscard]] size_t rows() const noexcept { return _rows; }
    [[nodiscard]] size_t cols() const noexcept { return _cols; }
    [[nodiscard]] size_t size() const noexcept { return _data.size(); }

    // expose a contiguous row-major view of the dense matrix
    DenseView<T>       view() noexcept       { return DenseView<T>(_data.data(), _rows, _cols); }
    DenseView<const T> view() const noexcept { return DenseView<const T>(_data.data(), _rows, _cols); }

    T*       begin() noexcept { return _data.data(); }
    const T* begin() const noexcept { return _data.data(); }

    T*       at(size_t i, size_t j) noexcept       { return _data.data() + rm_idx(i,j); }
    const T* at(size_t i, size_t j) const noexcept { return _data.data() + rm_idx(i,j); }
    
    T*       end() noexcept { return _data.data() + _data.size(); }
    const T* end() const noexcept { return _data.data() + _data.size(); }

};

}
