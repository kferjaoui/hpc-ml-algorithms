#pragma once

namespace mx
{
    
template<typename T>
class DenseView{
        size_t _rows{0};
        size_t _cols{0}; 
        T* buffer_{nullptr};

    public:
        DenseView() = default;

};

}

