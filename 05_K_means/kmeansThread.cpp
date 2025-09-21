#include <iostream>
#include <thread>
#include <cstring>
#include <cmath>
#include <atomic>
#include <barrier>
#include "utils.h"
#include "../common/CycleTimer.h"

void kmeansThreads(int numThreads, int N, const int K, const int D, const double epsilon,
                const double* data, double*& clusterCentroids, int*& assignementClusters){

    // Array to store the new cluster centroids after re-assignemnt of the points
    double* newClusterCentroids = new double[K*D];
    int* globalCount = new int[K];       // Track the number of points in each cluster
    int* globalCountThreads = new int[K]; // Track the number of points in each cluster
    bool converged = false;
    int step{0};
    
    double startTime, endTime;
    
    static constexpr int MAX_THREADS = 32;
    
    std::thread workers[MAX_THREADS];
    
    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    int** globalCount_partial = new int*[numThreads];
    for (int t = 0; t < numThreads; ++t) {
        globalCount_partial[t] = new int[K];
        std::fill(globalCount_partial[t], globalCount_partial[t] + K, 0);
    }

    std::atomic<int> next{0};
    int chunk_count = 4;
    
    // Thread parameters
    int chunkLength[MAX_THREADS];
    int startIdx[MAX_THREADS];
    int base = N / numThreads;
    int remainder = N % numThreads;
    int current = 0;
    for (int t = 0; t < numThreads; ++t) {
        chunkLength[t] = base + ((t<remainder)? 1:0);
        startIdx[t] = current;
        current +=  chunkLength[t];
    }
    
    int** counts = new int*[numThreads];
    double** newClusterCentroids_local= new double*[numThreads];
    
    // VERSION 1: with PADDING
    // Determine the needed buffer size of counts[t] for each thread 't' to avoid false sharing
    
    static constexpr int CACHE_LINE_SIZE = 64;
    const int ints_per_line   = CACHE_LINE_SIZE / int(sizeof(int));
    const int doubles_per_line= CACHE_LINE_SIZE / int(sizeof(double));

    auto round_up = [](int x, int q) { return ((x + q - 1) / q) * q; }; // [(x + q - 1) / q] is equivalent to ceil(x/q)

    int counts_stride = round_up(K, ints_per_line); // needed count of ints to allocate full cache lines for each thread
    int sums_stride   = round_up(K*D, doubles_per_line);
    
    int* counts_buffer = new int[numThreads * counts_stride];
    double* sums_buffer = new double[numThreads * sums_stride];
    
    for (int t = 0; t < numThreads; ++t) {
        counts[t] = counts_buffer + t * counts_stride;
        std::memset(counts[t], 0, counts_stride * sizeof(int));
        
        newClusterCentroids_local[t] = sums_buffer + t * sums_stride;
        std::fill(newClusterCentroids_local[t], newClusterCentroids_local[t] + sums_stride, 0.0);
    }

    enum phase { 
    PH_IDLE = 0,
    PH_ASSIGN_AND_ACCUMULATE = 1,
    PH_REDUCE_COUNTS = 2,
    PH_REDUCE_SUM = 3,
    PH_NORMALIZE = 4,
    PH_EXIT = 9
    };

    std::barrier sync_point(numThreads+1);
    std::atomic<int> phase{PH_IDLE};

    auto workFn = [&](int t){
        for (;;){
            sync_point.arrive_and_wait(); // Start of phase as defined by the main thread
            int ph = phase.load();
            if (ph == PH_EXIT) break;
            else if (ph == PH_ASSIGN_AND_ACCUMULATE){
                // reinitialize from last run
                std::fill(newClusterCentroids_local[t], newClusterCentroids_local[t] + K * D, 0.0);
                std::memset(counts[t], 0, K * sizeof(int)); // Reset counts for this thread
                // ASSIGN
                int endIdx = startIdx[t] + chunkLength[t];
                for(int i=startIdx[t]; i<endIdx; i++){
                    int best = 0;
                    double bestDist = l2Distance(data, i*D, clusterCentroids, 0*D, D);
                    for(int k=1; k<K; k++){
                        double dist = l2Distance(data, i*D, clusterCentroids, k*D, D);
                        if ( dist < bestDist){
                            bestDist = dist;
                            best = k;
                        }
                    }
                    assignementClusters[i] = best;
                    counts[t][best]++;
                    // ACCUMULATE
                    for(int d=0; d<D; d++){
                        newClusterCentroids_local[t][best*D + d] += data[i*D + d];
                    }
                }
            }
             else if (ph == PH_REDUCE_COUNTS){
                int i = next.fetch_add(chunk_count);
                int e = std::min(i + chunk_count, numThreads);
                if (i <= numThreads){
                    // REDUCE_COUNTS
                    int s = i / chunk_count; // s is the worker index
                    for(int k=0; k<K; k++){
                        for(int t=i; t<e; t++){
                            globalCount_partial[s][k] += counts[t][k];
                        }
                    }
                }
                // else: no work but still proceed to the finish barrier
            } else if (ph == PH_REDUCE_SUM){
                int total = K*D;
                int lo = (total * t) / numThreads;
                int hi = (total * (t+1)) / numThreads;
                for (int off=lo; off<hi; ++off) {
                    double s=0.0;
                    for (int tt=0; tt<numThreads; ++tt) s += sums_buffer[tt*sums_stride + off];
                    newClusterCentroids[off] = s;
                }
            } else if (ph == PH_NORMALIZE) {
                // each thread gets a disjoint [lo, hi) over d
                int lo = (D * t) / numThreads;
                int hi = (D * (t + 1)) / numThreads;

                // compute inv per k here
                for (int k = 0; k < K; ++k) {
                    int c = globalCount[k];
                    if (c == 0) continue;                // policy: leave centroid as-is
                    double inv = 1.0 / double(c);
                    double* mu = newClusterCentroids + k*D;
                    for (int d = lo; d < hi; ++d) {
                        mu[d] *= inv;                    // contiguous writes for this thread
                    }
                }
            }

            sync_point.arrive_and_wait(); // End
        }
    };
    // Start all threads
    for(int t=0; t<numThreads; t++){
        workers[t] = std::thread(workFn, t);    
    }

    while(!converged){
        step++;
        std::cout<< "Step " << step << std::endl;
        
        // zero global accumulators
        std::fill(newClusterCentroids, newClusterCentroids + K * D, 0.0);                      
        std::fill(globalCount, globalCount + K, 0);
        std::fill(globalCountThreads, globalCountThreads + K, 0);

        for (int t = 0; t < numThreads; ++t)
        std::fill(globalCount_partial[t], globalCount_partial[t] + K, 0);
        
        // ******************************
        // PHASE 1: ASSIGN_AND_ACCUMULATE
        // ******************************
        startTime = CycleTimer::currentSeconds();
        phase.store(PH_ASSIGN_AND_ACCUMULATE);
        sync_point.arrive_and_wait(); // Start
        sync_point.arrive_and_wait(); // END of PH_ASSIGN_AND_ACCUMULATE
        endTime = CycleTimer::currentSeconds();
        printf("[Fused Assign + Accumulate Threads]: %.6f ms | ", (endTime - startTime) * 1000);
        
        // ************************
        // PHASE 2: COUNT REDUCTION
        // ************************
        // THREADED version
        // startTime = CycleTimer::currentSeconds();
        // next.store(0); // Reset next index for counting
        // phase.store(PH_REDUCE_COUNTS);
        // sync_point.arrive_and_wait(); // Start
        // sync_point.arrive_and_wait(); // END of PH_REDUCE_COUNTS
        
        // int nWorkers = (numThreads + chunk_count - 1) / chunk_count;
        
        // for (int k = 0; k < K; ++k) {
        //     for (int t = 0; t < nWorkers; ++t) {
        //         globalCountThreads[k] += globalCount_partial[t][k];
        //     }
        // }
        // endTime = CycleTimer::currentSeconds();
        // printf("[Count Reduc Threads]: %.6f ms | ", (endTime - startTime) * 1000); 
        
        // SERIAL version
        startTime = CycleTimer::currentSeconds();
        for(int k=0; k<K; k++){
            for(int t=0; t<numThreads; t++){
                globalCount[k] += counts[t][k];
            }
        }
        endTime = CycleTimer::currentSeconds();
        printf("[Count Reduction Serial]: %.6f ms | ", (endTime - startTime) * 1000);

        // Check if the counts match
        // if (std::memcmp(globalCount, globalCountThreads, K * sizeof(int)) != 0) {
        //     std::cerr << "Error: Global counts do not match after reduction!" << std::endl;
            // exit(EXIT_FAILURE);
        // }

        // *********************
        // PHASE 3: REDUCE_SUM
        // *********************
        // THREADED version
        // startTime = CycleTimer::currentSeconds();
        // phase.store(PH_REDUCE_SUM);
        // sync_point.arrive_and_wait(); // Start
        // sync_point.arrive_and_wait(); // END of PH_REDUCE_SUM  
        // endTime = CycleTimer::currentSeconds();
        // printf("[Sum Reduction Threads]: %.6f ms | ", (endTime - startTime) * 1000);
        
        // SERIAL version
        startTime = CycleTimer::currentSeconds();
        for(int d=0; d<K*D; d++){
            for(int t=0; t<numThreads; t++){
                newClusterCentroids[d] += newClusterCentroids_local[t][d];
            }
        }
        endTime = CycleTimer::currentSeconds();
        printf("[Sum Reduction Serial]: %.6f ms | ", (endTime - startTime) * 1000);
        
        // *********************
        // NORMALIZE
        // *********************
        // THREADED version
        // startTime = CycleTimer::currentSeconds();
        // phase.store(PH_NORMALIZE);
        // sync_point.arrive_and_wait(); // Start
        // sync_point.arrive_and_wait(); // END of PH_NORMALIZE
        // endTime = CycleTimer::currentSeconds();
        // printf("[Norm Threads]: %.6f ms | ", (endTime - startTime) * 1000);
        
        // SERIAL version
        startTime = CycleTimer::currentSeconds();
        for (int k = 0; k < K; ++k) {
            if (globalCount[k] > 0) {
                for (int d = 0; d < D; ++d)
                newClusterCentroids[k*D + d] /= globalCount[k];
            }
        }
        endTime = CycleTimer::currentSeconds();
        printf("[Norm Serial]: %.6f ms\n", (endTime - startTime) * 1000);
        
        // *********************
        // CHECK CONVERGENCE
        // *********************
        converged = true;
        for(int k=0; k<K; k++){
            double d = l2Distance(clusterCentroids, k*D, newClusterCentroids, k*D, D);
            converged &= (d < epsilon);
        }
        std::size_t bytes = static_cast<std::size_t>(K) * D * sizeof(double);
        std::memcpy(clusterCentroids,    // destination
            newClusterCentroids,         // source
            bytes);                      // byte count        
    }

    // End phase
    phase.store(PH_EXIT);
    sync_point.arrive_and_wait(); // Start of end phase

    for (int t = 0; t < numThreads; ++t)
        if (workers[t].joinable()) workers[t].join();


    delete[] newClusterCentroids;
    delete[] globalCount;
    for (int t = 0; t < numThreads; ++t) delete[] globalCount_partial[t];
    delete[] globalCount_partial;
    delete[] globalCountThreads;

    delete[] counts;
    delete[] newClusterCentroids_local;
    
    // V1: PADDING
    delete[] counts_buffer;
    delete[] sums_buffer;
    
}