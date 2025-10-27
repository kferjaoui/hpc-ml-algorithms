#pragma once
#include <array>
#include <cassert>
#include <cstdint>
#include "types.h"

namespace mx {
    
template<typename T>
class DenseView {
    index_t _rows{0};
    index_t _cols{0};
    index_t _size{0};
    std::array<index_t, 2> _strides{0, 0};
    T* _buffer{nullptr};

public:
    DenseView() = default;

    // Contiguous row-major layout (standard)
    DenseView(T* ptr, index_t rows, index_t cols): 
        _buffer(ptr), _rows(rows), _cols(cols), _size(rows*cols), _strides{cols, 1} {
        assert(rows >= 0 && cols >= 0);
        assert(ptr != nullptr || (rows == 0 && cols == 0));
    }

    // General strided layout
    DenseView(T* ptr, index_t rows, index_t cols, index_t row_stride, index_t col_stride): 
        _buffer(ptr), _rows(rows), _cols(cols), _size(rows*cols), 
        _strides{row_stride, col_stride} {
        assert(rows >= 0 && cols >= 0);
        assert(row_stride >= 0 && col_stride >= 0);
        assert(ptr != nullptr || (rows == 0 && cols == 0));
    }

    T& operator()(index_t i, index_t j) noexcept {
        assert(i >= 0 && i < _rows && j >= 0 && j < _cols);
        return _buffer[_strides[0] * i + _strides[1] * j];
    }

    const T& operator()(index_t i, index_t j) const noexcept {
        assert(i >= 0 && i < _rows && j >= 0 && j < _cols);
        return _buffer[_strides[0] * i + _strides[1] * j];
    }

    // Create subview maintaining parent strides
    DenseView subview(index_t i0, index_t j0, index_t n_rows, index_t n_cols) const noexcept {
        assert(i0 >= 0 && j0 >= 0);
        assert(i0 + n_rows <= _rows && j0 + n_cols <= _cols);
        return DenseView(_buffer + _strides[0] * i0 + _strides[1] * j0, 
                        n_rows, n_cols, _strides[0], _strides[1]); 
    }
    
    // Transposed view (swap dimensions and strides)
    DenseView transposed() const noexcept { 
        return DenseView(_buffer, _cols, _rows, _strides[1], _strides[0]); 
    }

    // Accessors
    [[nodiscard]] index_t rows() const noexcept { return _rows; }
    [[nodiscard]] index_t cols() const noexcept { return _cols; }
    [[nodiscard]] index_t size() const noexcept { return _size; }
    [[nodiscard]] index_t row_stride() const noexcept { return _strides[0]; }
    [[nodiscard]] index_t col_stride() const noexcept { return _strides[1]; }
    
    // Leading dimension
    // For row-major: LDA = row_stride
    // For col-major: LDA = col_stride
    [[nodiscard]] index_t leading_dim() const noexcept { 
        return _strides[0];  // Row-major assumption
    }

    // Direct buffer access
    T*       data() noexcept       { return _buffer; }
    const T* data() const noexcept { return _buffer; }

    // Iterator support (only valid for contiguous views)
    T*       begin() noexcept { return _buffer; }
    const T* begin() const noexcept { return _buffer; }
    
    T*       end() noexcept { return _buffer + _size; }
    const T* end() const noexcept { return _buffer + _size; }

    // Element pointer access
    T* at(index_t i, index_t j) noexcept {
        assert(i >= 0 && i < _rows && j >= 0 && j < _cols);
        return _buffer + _strides[0] * i + _strides[1] * j;
    }
    
    const T* at(index_t i, index_t j) const noexcept {
        assert(i >= 0 && i < _rows && j >= 0 && j < _cols);
        return _buffer + _strides[0] * i + _strides[1] * j;
    }

    // Check if view is contiguous in memory
    [[nodiscard]] bool is_contiguous() const noexcept {
        return (_strides[0] == _cols && _strides[1] == 1);  // Row-major contiguous
    }
};

}