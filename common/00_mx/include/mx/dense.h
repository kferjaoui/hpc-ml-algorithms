#pragma once
#include<vector>
#include<cassert>
#include<string>
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

public:
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

    [[nodiscard]] T*       begin() noexcept { return _data.data(); }
    [[nodiscard]] const T* begin() const noexcept { return _data.data(); }
    
    [[nodiscard]] T*       end() noexcept { return _data.data() + _data.size(); }
    [[nodiscard]] const T* end() const noexcept { return _data.data() + _data.size(); }

};

}
