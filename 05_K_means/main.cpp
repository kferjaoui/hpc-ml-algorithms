#include <iostream>
#include <cstdlib>
#include "utils.h"

#define SEED 7

int main(){
    std::srand(SEED);

    int N = 1e2;        // Number of data points
    int D = 2;          // Dimesionality of the problem i.e. Number of features per data point
    int K = 3;          // Clustering parameter
    double epsilon = 0.000001;

    double* data = new double[N*D];

    // Generate synthetic data
    double stddev = 0.5;
    int Nc = 5;        // Number of initial centroids around which data points are generated
    double* initialCentroids = new double[Nc*D];
    
    generateCentroids(Nc, D, initialCentroids);                     // generates initial_centroids
    generateInitialData(N, Nc, D, data, initialCentroids, stddev, SEED);   // generates data for the start
    
    double* clusterCentroids = new double[K*D];
    // Randomly generate the K-means K centroids
    generateCentroids(K, D, clusterCentroids);

    readData("./data.bin", data, clusterCentroids, &N, &D, &K, &epsilon);
    // writeData("./data.bin", data, clusterCentroids, &N, &D, &K, &epsilon);

    // Print the first 3 points
    for(int p=0; p<3; p++){
        std::cout << "Point " << p << " : ";
        std::cout << "[ ";
        for(int k=0; k<D; k++){
            std::cout << data[p*D+k] << "; "; 
        }
        std::cout << "]" << std::endl;
    }

    delete[] data;
    delete[] initialCentroids;
    delete[] clusterCentroids;

    return 0;
}