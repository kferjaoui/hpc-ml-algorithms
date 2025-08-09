#include <cstdlib>
#include <iostream>
#include <random>
#include "utils.h"
#include <string>
#include <fstream>
#include <cmath>


// Generates random double precision number in [0,1]
double randDouble(){
    return static_cast<double>(std::rand()) / RAND_MAX;
}

// Compute the square root distance between two D-dimension points
double l2Distance(const double* data, int offsetData, double* clusterCentroids, int offsetCentroids, int D){
    double distance{0.0};
    for(int d=0; d<D; d++){
        double diff = data[offsetData+d] - clusterCentroids[offsetCentroids+d];
        distance += diff*diff;
    }
    return std::sqrt(distance);
}

// Randomly generates Nc centroids with coordinates between [0,1]
void generateCentroids(int Nc, int D, double* initialCentroids){
    for(int i=0; i<Nc; i++){
        for(int j=0; j<D; j++){
            initialCentroids[i*D+j] = randDouble(); 
        }
    }
}

// Randomly generates N data points around Nc centroids
void generateInitialData(int N, int K, int Nc, int D, double* data, double* initialCentroids, int* assignementClusters, double mean, double stddev, unsigned int seed){
    std::mt19937 generator(seed);
    std::normal_distribution<double> noise{mean, stddev};   
    std::uniform_int_distribution<> dist(0, Nc-1);
    std::uniform_int_distribution<> distK(0, K-1);

    for(int i=0; i<N; i++){
        int random_centroid_id = dist(generator); //std::experimental::randint(0,Nc-1);
        for(int k=0; k<D; k++){
            data[i*D+k] = initialCentroids[random_centroid_id*D+k] + noise(generator);
        }
        assignementClusters[i] = distK(generator); // Randomly assign the generated point (i) to a one of the K clusters 
    }
}

// Write data in binary file
void writeData(const std::string& filename, double* data, double* clusterCentroids, int* assignementClusters, int* N_p, int* D_p, int* K_p, double* epsilon_p){
    std::ofstream file(filename, std::ios::binary);

    int N = *N_p;
    int D = *D_p;
    int K = *K_p;

    file.write(reinterpret_cast<const char *>(N_p), sizeof(int));
    file.write(reinterpret_cast<const char *>(D_p), sizeof(int));
    file.write(reinterpret_cast<const char *>(K_p), sizeof(int));
    file.write(reinterpret_cast<const char *>(epsilon_p), sizeof(double));
    file.write(reinterpret_cast<const char *>(data), sizeof(double)*N*D);
    file.write(reinterpret_cast<const char *>(clusterCentroids), sizeof(double)*K*D);
    file.write(reinterpret_cast<const char *>(assignementClusters), sizeof(int)*N);
    
    file.close();
}

// Read data from binary file
void readData(const std::string& filename, double*& data, double*& clusterCentroids, int*& assignementClusters, int* N_p, int* D_p, int* K_p, double* epsilon_p){
    std::ifstream infile(filename,  std::ios::in | std::ios::binary);
    if (!infile) { std::cerr << "Cannot open " << filename << '\n'; std::exit(EXIT_FAILURE); }
    
    infile.read(reinterpret_cast<char *>(N_p), sizeof(int));
    infile.read(reinterpret_cast<char *>(D_p), sizeof(int));
    infile.read(reinterpret_cast<char *>(K_p), sizeof(int));
    infile.read(reinterpret_cast<char *>(epsilon_p), sizeof(double));
    
    int N = *N_p;
    int D = *D_p;
    int K = *K_p;

    data = new double[N*D];
    clusterCentroids = new double[K*D];
    assignementClusters = new int[N];
    
    infile.read(reinterpret_cast<char *>(data), sizeof(double)*N*D);
    infile.read(reinterpret_cast<char *>(clusterCentroids), sizeof(double)*K*D);
    infile.read(reinterpret_cast<char *>(assignementClusters), sizeof(int)*N);

    infile.close();
}

// logToFile() copied from cs149/asst1  [https://github.com/stanford-cs149/asst1/tree/master]
// Log a portion of data (using sampleRate) into a human readable formatted file
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
