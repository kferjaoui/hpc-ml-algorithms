#include <iostream>
#include <thread>
#include <cstring>
#include <cmath>
#include "utils.h"
#include "CycleTimer.h"

extern void normalizeClusterSums(int K, int D, double* newClusterCentroids, int* count);

void updateAssignmentLists_with_threads(int startIdx, int chunkLength, int K, int D,
                          const double* data, double* clusterCentroids,
                          int* assignementClusters, int* countPerThread){

    int endIdx = startIdx + chunkLength;
    for(int i=startIdx; i<endIdx; i++){
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
        countPerThread[best]++;
    }
}

void accumulateClusterSums_with_threads(int startIdx, int chunkLength, int D,
                         const double* data, double* newClusterCentroids_local, int* assignementClusters){
        int endIdx = startIdx + chunkLength;
        for (int n=startIdx; n<endIdx; n++){
            int k = assignementClusters[n];
            for(int d=0; d<D; d++){
                newClusterCentroids_local[k*D + d] += data[n*D + d];
            }
    }
}

void kmeansThreads(int numThreads, int N, const int K, const int D, const double epsilon,
                const double* data, double*& clusterCentroids, int*& assignementClusters){

    // Array to store the new cluster centroids after re-assignemnt of the points
    double* newClusterCentroids = new double[K*D];
    int* globalCount = new int[K]; // Track the number of points in each cluster
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
    
    // ********************
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
   
    
    // VERSION 2: no PADDING
    // for (int t = 0; t < numThreads; ++t) {
    //     counts[t] = new int[K];
    //     std::memset(counts[t], 0, K * sizeof(int));

    //     newClusterCentroids_local[t] = new double[K * D];
    //     std::fill(newClusterCentroids_local[t], newClusterCentroids_local[t] + K * D, 0.0);
    // }
    // ********************

    while(!converged){
        step++;
        std::cout<< "Step " << step << std::endl;
        
        // sets every element to zero
        std::fill(newClusterCentroids, newClusterCentroids + K * D, 0.0);                      
        std::fill(globalCount, globalCount + K, 0);
        for (int t = 0; t < numThreads; ++t)
        std::memset(counts[t], 0, K * sizeof(int));                     
        
        // A. Update the assigned Cluster to each data point 
        // (Also re-initialize the per-thread newClusterCentroids_local[t] from last iteration)
        startTime = CycleTimer::currentSeconds();
        for(int t=1; t<numThreads; t++){
            workers[t] = std::thread(updateAssignmentLists_with_threads, startIdx[t], chunkLength[t], K, D, data, clusterCentroids, assignementClusters, counts[t]);
            // re-initialize to zero the per-thread newClusterCentroids_local[t] for accumulation step
            std::fill(newClusterCentroids_local[t], newClusterCentroids_local[t] + K * D, 0.0);
        }
        
        updateAssignmentLists_with_threads(0, chunkLength[0], K, D, data, clusterCentroids, assignementClusters, counts[0]);
        std::fill(newClusterCentroids_local[0], newClusterCentroids_local[0] + K * D, 0.0);
        
        for(int t=1; t<numThreads; t++) workers[t].join();
        
        // Reduce counts[t] to count
        for(int k=0; k<K; k++){
            for(int t=0; t<numThreads; t++){
                globalCount[k] += counts[t][k];
            }
        }
        endTime = CycleTimer::currentSeconds();
        printf("[Assignement Threads]: %.3f ms  | ", (endTime - startTime) * 1000);
        
        // B. Compute the new coordinates of the cluster centroids 
        // 1. Sum all
        startTime = CycleTimer::currentSeconds();
        for(int t=1; t<numThreads; t++){
            workers[t] = std::thread(accumulateClusterSums_with_threads, startIdx[t], chunkLength[t], D, data, newClusterCentroids_local[t], assignementClusters);
        }
        
        accumulateClusterSums_with_threads(startIdx[0], chunkLength[0], D, data, newClusterCentroids_local[0], assignementClusters);
        
        for(int t=1; t<numThreads; t++) workers[t].join();

        // Reduce in newClusterCentroids
        for(int d=0; d<K*D; d++){
            for(int t=0; t<numThreads; t++){
                newClusterCentroids[d] += newClusterCentroids_local[t][d];
            }
        }
        endTime = CycleTimer::currentSeconds();
        printf("[Sum Threads]: %.3f ms\n", (endTime - startTime) * 1000);
        // ***************
        // SERIAL CHUNK
        // ***************
        normalizeClusterSums(K, D, newClusterCentroids, globalCount);
        
        // C. Check convergence of the K-means
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

    delete[] newClusterCentroids;
    delete[] globalCount;
    
    // V2
    // for (int t = 0; t<numThreads; ++t) delete[] counts[t];
    // for (int t = 0; t<numThreads; ++t) delete[] newClusterCentroids_local[t];

    delete[] counts;
    delete[] newClusterCentroids_local;
    
    // V1: PADDING
    delete[] counts_buffer;
    delete[] sums_buffer;
    
}