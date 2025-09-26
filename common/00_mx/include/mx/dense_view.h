#pragma once
#include<array>
namespace mx
{
    
template<typename T>
class DenseView{
        size_t _rows{0};
        size_t _cols{0};
        size_t _size{0};
        std::array<size_t, 2> _strides{0,0};
        T* _buffer{nullptr};

    public:
        DenseView() = default;

        // Assumes contiguous and row-major
        DenseView(T* ptr, size_t rows, size_t cols): 
            _buffer(ptr), _rows(rows), _cols(cols), _size{rows*cols}, _strides{_cols, 1}{
        }

        // Assumes contiguous
        DenseView(T* ptr, size_t rows, size_t _cols, size_t row_strides, size_t col_strides): 
            _buffer(ptr), _rows(rows), _cols(cols), _size{rows*cols}, _strides{row_strides, col_strides} {} 

        T& operator()(size_t i, size_t j) noexcept{
            assert(i<_rows && j<_cols);
            return _buffer[_strides[1] * j + _strides[0] * i];
        }

        const T& operator()(size_t i, size_t j) const noexcept{
            assert(i<_rows && j<_cols);
            return _buffer[_strides[1] * j + _strides[0] * i];
        }

        DenseView subview(size_t i0, size_t j0, size_t n_rows, size_t n_cols) const noexcept{
            // TODO
        }

        DenseView transposed() const noexcept {return DenseView(_buffer, _rows, _cols, 1, _rows)}

        [[nodiscard]] size_t rows() const noexcept {return _rows;}
        [[nodiscard]] size_t cols() const noexcept {return _cols;}
        [[nodiscard]] size_t size() const noexcept {return _size;}
        [[nodiscard]] size_t row_strides() const noexcept {return _strides[0];}
        [[nodiscard]] size_t col_strides() const noexcept {return _strides[1];}

        // Assumes contiguous views; TODO: Generalize
        T*       begin() noexcept { return _buffer; }
        const T* begin() const noexcept { return _buffer; }
        
        T*       end() noexcept { return _buffer + _size; }
        const T* end() const noexcept { return _buffer + _size; }


};

}

