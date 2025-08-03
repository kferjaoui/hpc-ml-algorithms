#include <iostream>
#include <cstdlib>
#include "utils.h"

#define SEED 7

int main(){
    std::srand(SEED);

    int N = 1e4;        // Number of data points
    int D = 2;          // Dimesionality of the problem i.e. Number of features per data point
    int K = 3;          // Clustering parameter

    double stddev = 0.5;

    double* data = new double[N*D];
    
    int Nc = 5;        // Number of initial centroids around which data points are generated

    double* initial_centroids = new double[Nc*D];

    generateInitialCentroids(Nc, D, initial_centroids);                     // generates initial_centroids
    generateInitialData(N, Nc, D, data, initial_centroids, stddev, SEED);   // generates data for the start

    // Print the first 3 points
    for(int p=0; p<3; p++){
        std::cout << "Point " << p << " :" << std::endl;
        std::cout << "[" << std::endl;
        for(int k=0; k<D; k++){
            std::cout << data[p*D+k] << '\n'; 
        }
        std::cout << "]" << std::endl;
    }

    delete[] data;
    delete[] initial_centroids;

    return 0;
}