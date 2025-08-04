#ifndef UTILS_H
#define UTILS_H

#include <string>

double randDouble();
void generateCentroids(int Nc, int D, double* initial_centroids);
void generateInitialData(int N, int K, int Nc, int D, double* data, double* initialCentroids, int* assignementClusters, double mean, double stddev, unsigned int seed);
void writeData(const std::string& filename, double* data, double* clusterCentroids, int* assignementClusters, int* N_p, int* D_p, int* K_p, double* epsilon_p);
void readData(const std::string& filename, double*& data, double*& clusterCentroids, int*& assignementClusters, int* N_p, int* D_p, int* K_p, double* epsilon_p);

#endif
