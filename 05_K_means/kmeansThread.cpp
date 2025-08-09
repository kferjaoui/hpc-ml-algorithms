#include <iostream>
#include <thread>
#include <cstring>
#include "utils.h"
#include "CycleTimer.h"

extern void normalizeClusterSums(int K, int D, double*& newClusterCentroids, int*& count);

void updateAssignmentLists_with_threads(int start, int chunkLength, int K, int D,
                          const double* data, double* clusterCentroids,
                          int* assignementClusters, int* countPerThread){

    int end = start + chunkLength;
    for(int i=start; i<end; i++){
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


void accumulateClusterSums_with_threads(int start, int chunkLength, int D,
                         const double* data, double* newClusterCentroids_local, int* assignementClusters){
        int end = start + chunkLength;
        for (int n=start; n<end; n++){
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
    int start[MAX_THREADS];
    int n = N/numThreads;
    int remainder = N%numThreads;
    // std::barrier sync_point(numThreads);
    
    int** counts = new int*[numThreads];
    double** newClusterCentroids_local= new double*[numThreads];
    int current = 0;
    for (int t = 0; t < numThreads; ++t) {
        chunkLength[t] = n + ((t<remainder)? 1:0);
        start[t] = current;
        current +=  chunkLength[t];
        
        counts[t] = new int[K];
        std::memset(counts[t], 0, K * sizeof(int));

        newClusterCentroids_local[t] = new double[K*D];
        std::fill(newClusterCentroids_local[t], newClusterCentroids_local[t] + K * D, 0.0);
    }
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
            workers[t] = std::thread(updateAssignmentLists_with_threads, start[t], chunkLength[t], K, D, data, clusterCentroids, assignementClusters, counts[t]);
            // re-initialize the local cluster coordinates for each thread for accumulation step
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
            workers[t] = std::thread(accumulateClusterSums_with_threads, start[t], chunkLength[t], D, data, newClusterCentroids_local[t], assignementClusters);
        }
        
        accumulateClusterSums_with_threads(start[0], chunkLength[0], D, data, newClusterCentroids_local[0], assignementClusters);
        
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
    for (int t = 0; t<numThreads; ++t) delete[] counts[t];
    for (int t = 0; t<numThreads; ++t) delete[] newClusterCentroids_local[t];
    delete[] counts;
    delete[] newClusterCentroids_local;
    delete[] globalCount;
}