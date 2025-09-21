#include <iostream>
#include <cstring>
#include "utils.h"
#include "../common/CycleTimer.h"

void updateAssignmentLists(int start, int N, int K, int D,
                          const double* data, double*& clusterCentroids,
                          int* assignementClusters, int* count){
    for(int i=0; i<N; i++){
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
        count[best]++;
    }
}

void accumulateClusterSums(int N, int D, const double* data, double* newClusterCentroids, int* assignementClusters){
    for (int n=0; n<N; n++){
        int k = assignementClusters[n];
        for(int d=0; d<D; d++){
            newClusterCentroids[k*D + d] += data[n*D + d];
        }
    }
}

void normalizeClusterSums(int K, int D, double* newClusterCentroids, int* count){
    for (int k = 0; k < K; ++k) {
        if (count[k] > 0) {
            for (int d = 0; d < D; ++d)
            newClusterCentroids[k*D + d] /= count[k];
        }
    }
}

void kmeansSerial(const int N, const int K, const int D, const double epsilon,
                const double* data, double*& clusterCentroids, int*& assignementClusters){

    // Array to store the new cluster centroids after re-assignemnt of the points
    double* newClusterCentroids = new double[K*D];
    int* count = new int[K]; // Track the number of points in each cluster
    bool converged = false;
    int step{0};
    double startTime, endTime;

    while(!converged){
        step++;
        std::cout<< "Step " << step << std::endl;
        
        // sets every element to zero
        std::fill(newClusterCentroids, newClusterCentroids + K * D, 0.0);                      
        std::fill(count, count + K, 0);                      
    
        // A. Update the assigned Cluster to each data point 
        startTime = CycleTimer::currentSeconds();
        updateAssignmentLists(0, N, K, D, data, clusterCentroids, assignementClusters, count);
        endTime = CycleTimer::currentSeconds();
        printf("[Assignement]: %.3f ms  | ", (endTime - startTime) * 1000);
        
        // B. Compute the new coordinates of the cluster centroids 
        // 1. Sum all
        startTime = CycleTimer::currentSeconds();
        accumulateClusterSums(N, D, data, newClusterCentroids, assignementClusters);
        endTime = CycleTimer::currentSeconds();
        printf("[Accumulation]: %.3f ms  | ", (endTime - startTime) * 1000);
        
        // 2. Divide by normalizing factor
        startTime = CycleTimer::currentSeconds();
        normalizeClusterSums(K, D, newClusterCentroids, count);
        endTime = CycleTimer::currentSeconds();
        printf("[Normalization]: %.3f ms  | ", (endTime - startTime) * 1000);
        
        // C. Check convergence of the K-means
        startTime = CycleTimer::currentSeconds();
        converged = true;
        for(int k=0; k<K; k++){
            double d = l2Distance(clusterCentroids, k*D, newClusterCentroids, k*D, D);
            converged &= (d < epsilon);
        }
        endTime = CycleTimer::currentSeconds();
        printf("[Convergence Check]: %.3f ms\n", (endTime - startTime) * 1000);
        std::size_t bytes = static_cast<std::size_t>(K) * D * sizeof(double);
        std::memcpy(clusterCentroids,    // destination
            newClusterCentroids,         // source
            bytes);                      // byte count 
    }

    delete[] newClusterCentroids;
    delete[] count;
}
