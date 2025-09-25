#pragma once
#include<ostream>
#include"mx/dense.h"

namespace mx{

template<typename T>
std::ostream& operator<<(std::ostream& os, const Dense<T>& Matrix){
    const auto r = Matrix.rows(), c = Matrix.cols();
    os << "[\n";
    for(size_t i = 0; i < r; ++i){   
        os << " [";
        for(size_t j = 0; j < c; ++j){
            if (j) os << ", ";
            os << Matrix(i,j);
        }
        os << "]";
        if (i+1 < r) os << ",";
        os << "\n";
    }
    os << "]";
    return os;
}

}