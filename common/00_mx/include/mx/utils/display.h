#pragma once
#include<cstdio>
#include"mx/dense.h"

namespace mx{

template<typename T>
void display(const Dense<T>& A){
    const auto r = A.rows(), c = A.cols();
    printf("[\n");
    for(size_t i = 0; i < r; i++)
    {   
        printf(" [");
        for(size_t j = 0; j < c; j++)
        {
            if constexpr (std::is_floating_point_v<T>)
                printf("%g%s", A(i,j), (j+1==c? "": ","));
            else
                printf("%lld%s", static_cast<long long>(A(i,j)), (j+1==c? "": ", "));
        }
        printf("]%s\n", (i+1==r? "" :","));
    }
    printf("]\n");
}

}