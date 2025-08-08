#include <iostream>
#include <cstring>
#include "utils.h"


void kmeansSerial(const int N, const int K, const int D, const double epsilon,
                const double* data, double*& clusterCentroids, int*& assignementClusters){

    // Array to store the new cluster centroids after re-assignemnt of the points
    double* newClusterCentroids = new double[K*D];
    int* count = new int[K]; // Track the number of points in each cluster
    bool converged = false;
    int step{0};

    while(!converged){
        step++;
        std::cout<< "Step " << step << std::endl;
        
        // sets every element to zero
        std::fill(newClusterCentroids, newClusterCentroids + K * D, 0.0);                      
        std::fill(count, count + K, 0);                      
    
        // A. Update the assigned Cluster to each point (i) 
        for(int i=0; i<N; i++){
            int best = 0;
            double bestDist = sqDist(data, i*D, clusterCentroids, 0*D, D);
            for(int k=1; k<K; k++){
                double dist = sqDist(data, i*D, clusterCentroids, k*D, D);
                if ( dist < bestDist){
                    bestDist = dist;
                    best = k;
                }
            }
            assignementClusters[i] = best;
            count[best]++;
        }
        
        // B. Compute the new coordinates of the cluster centroids 
        // 1. Sum all
        for (int n=0; n<N; n++){
            int k = assignementClusters[n];
            for(int d=0; d<D; d++){
                newClusterCentroids[k*D + d] += data[n*D + d];
            }
        }
        // 2. Divide by normalizing factor
        for (int k = 0; k < K; ++k) {
            if (count[k] > 0) {
                for (int d = 0; d < D; ++d)
                newClusterCentroids[k*D + d] /= count[k];
            }
        }
        
        // C. Check convergence of the K-means
        converged = true;
        for(int k=0; k<K; k++){
            double d = sqDist(clusterCentroids, k*D, newClusterCentroids, k*D, D);
            converged &= (d < epsilon);
        }
        
        std::size_t bytes = static_cast<std::size_t>(K) * D * sizeof(double);
        std::memcpy(clusterCentroids,    // destination
            newClusterCentroids,         // source
            bytes);                      // byte count 
    }

    delete[] newClusterCentroids;
    delete[] count;
}
