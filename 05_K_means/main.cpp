#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "utils.h"

#include "CycleTimer.h"

#define SEED 7
#define SAMPLE_RATE 1e-2

double sqDist(double* data, int offsetData, double* clusterCentroids, int offsetCentroids, int D){
    double distance{0.0};
    for(int d=0; d<D; d++){
        double diff = data[offsetData+d] - clusterCentroids[offsetCentroids+d];
        distance += diff*diff;
    }
    return std::sqrt(distance);
}

// logToFile() copied from cs149/asst1  [https://github.com/stanford-cs149/asst1/tree/master]
void logToFile(std::string filename, double sampleRate, double *data,
               int *clusterAssignments, double *clusterCentroids, int M, int N,
               int K) {
  std::ofstream logFile;
  logFile.open(filename);

  // Write header
  logFile << M << "," << N << "," << K << std::endl;

  // Log data points
  for (int m = 0; m < M; m++) {
    if (static_cast<double>(rand()) / static_cast<double>(RAND_MAX) <
        sampleRate) {
      logFile << "Example " << m << ", cluster " << clusterAssignments[m]
              << ": ";
      for (int n = 0; n < N; n++) {
        logFile << data[m * N + n] << " ";
      }
      logFile << "\n";
    }
  }

  // Log centroids
  for (int k = 0; k < K; k++) {
    logFile << "Centroid " << k << ": ";
    for (int n = 0; n < N; n++) {
      logFile << clusterCentroids[k * N + n] << " ";
    }
    logFile << "\n";
  }

  logFile.close();
}

int main(){
    std::srand(SEED);

    int N;        // Number of data points
    int D;        // Dimesionality of the problem i.e. Number of features per data point
    int K;          // Clustering parameter
    double epsilon;
    
    double* data;
    double* clusterCentroids;
    int* assignementClusters;

    std::string filename = "./data.dat";

    // Generate synthetic data
    std::ifstream infile(filename, std::ios::binary);
    if (!infile){
        // If no file to read, write the data
        std::cout << "No data found!\n";
        std::cout << "Writing synthetic data now..." << std::endl;
        
        N = 1e6;
        D = 100;
        K = 3;
        epsilon = 0.1;

        data = new double[N*D];
        clusterCentroids = new double[K*D];
        assignementClusters = new int[N];

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
        
        std::cout << "Number of data points (N): " << N << std::endl;
        std::cout << "Dimensionality of the problem (D): " << D << std::endl;
        std::cout << "Number of final clusters (K): " << K << std::endl;
        epsilon *= epsilon;
        std::cout << "Epsilon: " << epsilon << std::endl;
    }

    // Array to store the new cluster centroids after re-assignemnt of the points
    double* newClusterCentroids = new double[K*D];
    int* count = new int[K]; // Track the number of points in each cluster
    bool converged = false;
    int step{0};

    // Log the starting state of the algorithm
    logToFile("./start.log", SAMPLE_RATE, data, assignementClusters,
                clusterCentroids, N, D, K);

    double startTime = CycleTimer::currentSeconds();
                
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
        for(int k=0; k<K; k++){
            double d = sqDist(clusterCentroids, k*D, newClusterCentroids, k*D, D);
            converged &= (d < epsilon);
        }

        std::size_t bytes = static_cast<std::size_t>(K) * D * sizeof(double);
        std::memcpy(clusterCentroids,            // destination
                    newClusterCentroids,         // source
                    bytes);                      // byte count

    }

    double endTime = CycleTimer::currentSeconds();
    printf("[Total Time]: %.3f ms\n", (endTime - startTime) * 1000);

    // Log the end state of the algorithm
    logToFile("./end.log", SAMPLE_RATE, data, assignementClusters,
                clusterCentroids, N, D, K); 
    
    delete[] data;
    delete[] clusterCentroids;
    delete[] assignementClusters;
    delete[] newClusterCentroids;
    delete[] count;

    return 0;
}