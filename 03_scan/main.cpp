#include<cstdio>
#include<bit>
#include<cstdint>
#include<thread>
#include<barrier>

#include "../common/CycleTimer.h"

using namespace std;

void cpu_inclusive_scan_serial(const int* in, const uint32_t N, int* out) {
    if (N <= 0) return;
    out[0] = in[0];
    for (int i = 1; i < N; i++) {
        out[i] = out[i - 1] + in[i];
    }
} 

void cpu_exclusive_scan_serial(const int* in, const uint32_t N, int* out) {
    if (N <= 0) return;
    out[0] = 0;
    for (int i = 1; i < N; i++) {
        out[i] = out[i - 1] + in[i-1];
    }
}

int ilog2_pow2_u32(uint32_t n) { return countr_zero(n); }  // n is power of two, n != 0
uint32_t next_pow2_u32(uint32_t n) { return bit_ceil(n); } // ceil to next power of two

void cpu_exclusive_scan_parallel(const int* in, const uint32_t N, int* out) {
    if (N == 0) return;

    int depth = ilog2_pow2_u32(N);
    // printf("Depth: %d\n", depth);

    int numThreads = 2;
    thread* workers = new thread[numThreads];
    barrier sync_point(numThreads);

    auto workFunction = [depth, &sync_point, in, N, numThreads](int threadIdx, int* out){
        for(int i=threadIdx; i<N; i+=numThreads) out[i] = in[i]; //TODO: get rid of false sharing
        sync_point.arrive_and_wait();

        int stride = 0;
        int offset = 0;
        int jump = 0;
        // up-sweep step
        // printf("---------------- Up-sweep phase------------------\n");
        for(int d=0; d<depth-1; d++){
            // printf("D: %d\n", d);
            stride = numThreads * (1u << (d+1) ); //function of number of threads and depth
            offset = ( 1u << (d+1) ) - 1;         //function of depth only
            jump   = ( 1u << (d+1) );
            for(int idx = offset + threadIdx * jump;
                idx<(int)N;
                idx+=stride)
            {
                    // printf("Thread %d updating index %d\n", threadIdx, idx);
                    out[idx] = out[idx] + out[idx-(1u<<d)]; 
            }
            sync_point.arrive_and_wait();
        }
        
        if (threadIdx == 0) out[N-1] = 0; //any thread can take of this
        sync_point.arrive_and_wait();

        int tmp;
        // down-sweep step
        // printf("---------------- Down-sweep phase------------------\n");
        for(int d=depth-1; d>=0; d--){
            // printf("D: %d\n", d);
            stride = numThreads * ( 1u << (d+1) );
            jump   = ( 1u << (d+1) );
            for(int idx=(int)(N-1-threadIdx*jump);
                idx>=0;
                idx-=stride)
            {
                    tmp = out[idx];
                    out[idx] = out[idx] + out[idx-(1u<<d)]; 
                    out[idx-(1u<<d)] = tmp;
                    // printf("Thread %d updating index (%d, %d)\n", threadIdx, idx, idx-(1u<<d));
            }
            sync_point.arrive_and_wait();
        }
    };

    for(int tid=0; tid<numThreads; tid++){
        workers[tid] = std::thread(workFunction, tid, out);
    }

    for(int t=0; t<numThreads; t++){
        if (workers[t].joinable()) workers[t].join();
    }

    delete[] workers;
} 

int main() {
    const uint32_t N = 32;
    int* inputArray = new int[N];
    int* incScanOutputArray = new int[N];
    int* excScanOutputArray = new int[N];
    int* excScanOutputArray2 = new int[N];

    for (int i=0; i<N; i++) inputArray[i] = i;

    // Print input array
    printf("Input: ");
    for (int i = 0; i < N; i++) {
        printf("%i", inputArray[i]);
        if (i < N - 1) printf(", "); // Avoid trailing comma
    }
    printf("\n");

    cpu_inclusive_scan_serial(inputArray, N, incScanOutputArray);
    cpu_exclusive_scan_serial(inputArray, N, excScanOutputArray);
    cpu_exclusive_scan_parallel(inputArray, N, excScanOutputArray2);

    // Print result of inclusive scan
    printf("Inclusive scan: ");
    for (int i = 0; i < N; i++) {
        printf("%i", incScanOutputArray[i]);
        if (i < N - 1) printf(", ");
    }
    printf("\n");

    printf("Exclusive scan: ");
    for (int i = 0; i < N; i++) {
        printf("%i", excScanOutputArray[i]);
        if (i < N - 1) printf(", ");
    }
    printf("\n");

    printf("Exclusive Parl: ");
    for (int i = 0; i < N; i++) {
        printf("%i", excScanOutputArray2[i]);
        if (i < N - 1) printf(", ");
    }
    printf("\n");

    delete[] inputArray;
    delete[] incScanOutputArray;
    delete[] excScanOutputArray;
    delete[] excScanOutputArray2;

    return 0;
}