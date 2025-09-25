#pragma once
#include<algorithm>
#include "mx/dense.h"

namespace mx{

template<typename T>
void fill(Dense<T>& A, const T& value){
    std::fill(A.begin(), A.end(), value);
}

}