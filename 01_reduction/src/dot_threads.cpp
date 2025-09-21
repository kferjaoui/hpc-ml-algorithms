#include<thread>
#include<barrier>
#include<atomic>
#include<vector>
#include<cmath>
#include<cstdio>
#include "../../common/CycleTimer.h"

void dotThreadWorker(int threadIdx,
                    const double* x, 
                    const double* y, 
                    size_t numThreads, 
                    size_t n,
                    std::barrier<>& sync_point,
                    std::vector<double>& vResult){

    for(size_t idx=threadIdx; idx<n; idx+=numThreads){
        vResult[threadIdx] += x[idx]*y[idx];
    }

    sync_point.arrive_and_wait();

    for(size_t stride=numThreads>>1; stride>0; stride>>=1){
        if (threadIdx < stride) {
            vResult[threadIdx] += vResult[threadIdx+stride]; 
        }
        sync_point.arrive_and_wait();
    }
}

double dotThreads(const double* hx, const double* hy, size_t n){

    size_t numThreads = 32; 
    std::vector<std::thread> T;
    T.reserve(numThreads);
    std::barrier<> sync_point(numThreads);
    std::vector<double> vResult(numThreads, 0.0);

    for (size_t idx=0; idx<numThreads; idx++) T.emplace_back(dotThreadWorker, idx,
                                                                            hx, 
                                                                            hy, 
                                                                            numThreads, 
                                                                            n, 
                                                                            std::ref(sync_point), 
                                                                            std::ref(vResult));

    for(auto& t: T) t.join();

    return vResult[0];

}

int main(){
    const size_t n = 1u << 20; // 1,048,576 elements

    // host data
    std::vector<double> hx(n), hy(n);
    for (size_t i = 0; i < n; ++i) {
        // deterministic values (not too large)
        hx[i] = 1.0 / double(i + 1);
        hy[i] = std::sin(0.001 * double(i));
    }

    // CPU reference
    double startTime = CycleTimer::currentSeconds();
    double ref = 0.0;
    for (size_t i = 0; i < n; ++i) ref += hx[i] * hy[i];
    double endTime = CycleTimer::currentSeconds();
    std::printf("[Total Time Serial]: %.3f ms\n", (endTime - startTime) * 1000);
    std::printf("Serial ref : %.17g\n", ref);
    
    startTime = CycleTimer::currentSeconds();
    double resultThreads = dotThreads(hx.data(), hy.data(), n);
    endTime = CycleTimer::currentSeconds();
    std::printf("[Total Time Threads]: %.3f ms\n", (endTime - startTime) * 1000);
    std::printf("Thread dot : %.17g\n", resultThreads);

    return 0;
}