#include"mx/dense.h"
#include"mx/alg/fill.h"
#include"mx/alg/display.h"

int main()
{
    mx::Dense<double> A(4,4), B(8,4);
    mx::fill(A, 1.0);
    mx::fill(B, 2.0);

    mx::display(A);
    mx::display(B);

    return 0;
}