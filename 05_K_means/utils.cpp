#include <cstdlib>
#include <iostream>
#include <random>
#include "utils.h"
#include <string>
#include <fstream>


// Generates random double precision number in [0,1]
double randDouble(){
    return static_cast<double>(std::rand()) / RAND_MAX;
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

void readData(const std::string& filename, double*& data, double*& clusterCentroids, int*& assignementClusters, int* N_p, int* D_p, int* K_p, double* epsilon_p){
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) { std::cerr << "Cannot open " << filename << '\n'; std::exit(EXIT_FAILURE); }
    
    infile.read(reinterpret_cast<char *>(N_p), sizeof(int));
    infile.read(reinterpret_cast<char *>(D_p), sizeof(int));
    infile.read(reinterpret_cast<char *>(K_p), sizeof(int));
    infile.read(reinterpret_cast<char *>(epsilon_p), sizeof(double));
    
    int N = *N_p;
    int D = *D_p;
    int K = *K_p;
    
    infile.read(reinterpret_cast<char *>(data), sizeof(double)*N*D);
    infile.read(reinterpret_cast<char *>(clusterCentroids), sizeof(double)*K*D);
    infile.read(reinterpret_cast<char *>(assignementClusters), sizeof(int)*N);

    infile.close();
}
