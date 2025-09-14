#include<cstdio>

void cpu_inclusive_scan(const int* in, const int N, int* out) {
    if (N <= 0) return;
    out[0] = in[0];
    for (int i = 1; i < N; i++) {
        out[i] = out[i - 1] + in[i];
    }
}

int main() {
    const int N = 10;
    int* inputArray = new int[N];
    int* outputArray = new int[N];

    for (int i=0; i<N; i++) inputArray[i] = i;

    // Print input array
    printf("Input: ");
    for (int i = 0; i < N; i++) {
        printf("%i", inputArray[i]);
        if (i < N - 1) printf(", "); // Avoid trailing comma
    }
    printf("\n");

    cpu_inclusive_scan(inputArray, N, outputArray);

    // Print output array
    printf("Output: ");
    for (int i = 0; i < N; i++) {
        printf("%i", outputArray[i]);
        if (i < N - 1) printf(", "); // Avoid trailing comma
    }
    printf("\n");

    delete[] inputArray;
    delete[] outputArray;

    return 0;
}