#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "utils.h"

#define SEED 7

double sqDist(double* data, int offsetData, double* clusterCentroids, int offsetCentroids, int D){
    double distance{0.0};
    for(int d=0; d<D; d++){
        double diff = data[offsetData+d] - clusterCentroids[offsetCentroids+d];
        distance += diff*diff;
    }
    return std::sqrt(distance);
}

int main(){
    std::srand(SEED);

    int N = 1e2;        // Number of data points
    int D = 2;          // Dimesionality of the problem i.e. Number of features per data point
    int K = 3;          // Clustering parameter
    double epsilon = 0.1;

    double* data = new double[N*D];
    double* clusterCentroids = new double[K*D];
    int* assignementClusters = new int[N];

    std::string filename = "./data.bin";

    // Generate synthetic data
    std::ifstream infile(filename, std::ios::binary);
    if (!infile){
        // If no file to read, write the data
        std::cout << "No data found!\n";
        std::cout << "Writing synthetic data now..." << std::endl;
        
        double mean = 0.0;
        double stddev = 0.5;
        int Nc = 5;        // Number of initial centroids around which data points are generated
        double* initialCentroids = new double[Nc*D];
        
        generateCentroids(Nc, D, initialCentroids);                     // generates initial_centroids
        generateInitialData(N, K, Nc, D, data, initialCentroids, assignementClusters, mean, stddev, SEED); // generates data for the start
        
        // Randomly generate the K-means K centroids
        generateCentroids(K, D, clusterCentroids);
        
        writeData(filename, data, clusterCentroids, assignementClusters, &N, &D, &K, &epsilon); 
        std::cout << "Data written in '" << filename << "'" << std::endl;
        
        delete[] initialCentroids;
    } else{
        // Read data from binary
        std::cout << "Reading data..." << std::endl;
        readData(filename, data, clusterCentroids, assignementClusters, &N, &D, &K, &epsilon);
    }
    
    // Print the first 3 points
    // for(int p=0; p<3; p++){
    //     std::cout << "Point " << p << " : ";
    //     std::cout << "[ ";
    //     for(int k=0; k<D; k++){
    //         std::cout << data[p*D+k] << "; "; 
    //     }
    //     std::cout << "]" << std::endl;
    // }

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
    
        // Update the assigned Cluster to each point (i) 
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
        
        // Compute the new coordinates of the cluster centroids 
        // How? It is the gravity center of all the points within it 
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
        
        // Check convergence of the K-means
        converged = true;
        for(int k=0; k<K && converged; k++){
            double d = sqDist(clusterCentroids, k*D, newClusterCentroids, k*D, D);
            converged &= (d <= epsilon);
        }

        std::size_t bytes = static_cast<std::size_t>(K) * D * sizeof(double);
        std::memcpy(clusterCentroids,            // destination
                    newClusterCentroids,         // source
                    bytes);                      // byte count

    }
    
    delete[] data;
    delete[] clusterCentroids;
    delete[] assignementClusters;
    delete[] newClusterCentroids;
    delete[] count;

    return 0;
}