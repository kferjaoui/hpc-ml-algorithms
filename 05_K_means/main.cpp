#include <iostream>
#include <fstream>
#include <cstdlib>
#include <thread>
#include <cstring>
#include "utils.h"

#include "CycleTimer.h"

#define SEED 7
#define SAMPLE_RATE 1e-2

#define MAX_THREADS 128

extern void kmeansSerial(const int N, const int K, const int D, const double epsilon,
                const double* data, double*& clusterCentroids, int*& assignementClusters);

extern void kmeansThreads(int numThreads, int N, const int K, const int D, const double epsilon,
                const double* data, double*& clusterCentroids, int*& assignementClusters);

int main(){
    std::srand(SEED);

    int N;        // Number of data points
    int D;        // Dimesionality of the problem i.e. Number of features per data point
    int K;          // Clustering parameter
    double epsilon;
    
    double* data;
    double* clusterCentroids;
    double* clusterCentroidsThreads;
    int* assignementClusters;
    int* assignementClustersThreads;
    
    std::string filename = "./data.dat";
    // *********************
    // Read or Write data
    // *********************
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

    clusterCentroidsThreads = new double[K*D];
    assignementClustersThreads = new int[N];

    std::size_t bytes_d = static_cast<std::size_t>(K) * D * sizeof(double);
    std::memcpy(clusterCentroidsThreads, clusterCentroids, bytes_d);
    std::size_t bytes_i = static_cast<std::size_t>(N) * sizeof(int);
    std::memcpy(assignementClustersThreads, assignementClusters, bytes_i);
    
    // Log the starting state of the algorithm
    logToFile("./start.log", SAMPLE_RATE, data, assignementClusters,
        clusterCentroids, N, D, K);
    
    // *********************
    // K-means (Serial)
    // *********************
    printf("Running Kmeans in Serial...\n");
    double startTime = CycleTimer::currentSeconds();
    kmeansSerial(N, K, D, epsilon, data, clusterCentroids, assignementClusters);
    double endTime = CycleTimer::currentSeconds();
    printf("[Total Time Serial]: %.3f ms\n", (endTime - startTime) * 1000);
    
    logToFile("./end.log", SAMPLE_RATE, data, assignementClusters,
                clusterCentroids, N, D, K);
    // *********************
    // K-means (Threads)
    // *********************
    int numThreads = std::thread::hardware_concurrency();

    printf("Running Kmeans in Parallel...\n");
    startTime = CycleTimer::currentSeconds();
    kmeansThreads(numThreads, N, K, D, epsilon, data, clusterCentroidsThreads, assignementClustersThreads);
    endTime = CycleTimer::currentSeconds();
    printf("[Total Time Threads]: %.3f ms\n", (endTime - startTime) * 1000);

    // Log the end state of the algorithm
    logToFile("./end_threads.log", SAMPLE_RATE, data, assignementClustersThreads,
                clusterCentroidsThreads, N, D, K); 
    
    delete[] data;
    delete[] clusterCentroids;
    delete[] assignementClusters;

    return 0;
}